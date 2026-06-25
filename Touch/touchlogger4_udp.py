import ctypes
from ctypes import wintypes
import math
import traceback
import socket
import struct
import argparse
import time
import win32con
import win32gui

user32 = ctypes.windll.user32

WM_TOUCH = 0x0240
WM_HOTKEY = 0x0312

TOUCHEVENTF_MOVE = 0x0001
TOUCHEVENTF_DOWN = 0x0002
TOUCHEVENTF_UP = 0x0004

HOTKEY_ID = 1
MIN_DIAMETER = 6
LINE_DISTANCE_THRESHOLD = 380  # render-window pixels

# UDP output: <B count> then repeated <H id, f nx, f ny, f rotation>
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
UDP_ENABLED = True

# UDP send cadence. Default is event-driven: one packet per WM_TOUCH bundle.
# --uncapped repeats the latest active_touches as fast as the message pump allows.
# --hz N repeats at a fixed rate.
UDP_UNCAPPED = False
UDP_HZ = 0.0
SEND_UDP_ON_TOUCH_EVENTS = True

# Physical remap. Units can be mm, cm, inches, whatever — just be consistent.
# If the touch frame is larger than the visible screen and top-left aligned,
# corrected_norm = windows_norm * frame_size / screen_size.
PHYSICAL_REMAP_ENABLED = False
SCREEN_PHYS_W = 1.0
SCREEN_PHYS_H = 1.0
FRAME_PHYS_W = 1.0
FRAME_PHYS_H = 1.0

# Optional legacy squeeze correction applied after physical remap.
# Normally leave these at 1.0 when using physical remap.
UDP_X_SQUEEZE = 1.0
UDP_Y_SQUEEZE = 1.0

udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

TOUCH_COORD_TO_PIXEL = lambda l: l // 100

active_touches = {}
capture_hwnd = None
render_hwnd = None
capture_enabled = True


class TOUCHINPUT(ctypes.Structure):
    _fields_ = [
        ("x", wintypes.LONG),
        ("y", wintypes.LONG),
        ("hSource", wintypes.HANDLE),
        ("dwID", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("dwMask", wintypes.DWORD),
        ("dwTime", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ("cxContact", wintypes.DWORD),
        ("cyContact", wintypes.DWORD),
    ]


RegisterTouchWindow = user32.RegisterTouchWindow
RegisterTouchWindow.argtypes = [wintypes.HWND, wintypes.ULONG]
RegisterTouchWindow.restype = wintypes.BOOL

GetTouchInputInfo = user32.GetTouchInputInfo
GetTouchInputInfo.argtypes = [
    wintypes.HANDLE,
    wintypes.UINT,
    ctypes.POINTER(TOUCHINPUT),
    ctypes.c_int,
]
GetTouchInputInfo.restype = wintypes.BOOL

CloseTouchInputHandle = user32.CloseTouchInputHandle
CloseTouchInputHandle.argtypes = [wintypes.HANDLE]
CloseTouchInputHandle.restype = wintypes.BOOL

InvalidateRect = user32.InvalidateRect
InvalidateRect.argtypes = [wintypes.HWND, ctypes.c_void_p, wintypes.BOOL]
InvalidateRect.restype = wintypes.BOOL

RegisterHotKey = user32.RegisterHotKey
RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
RegisterHotKey.restype = wintypes.BOOL

UnregisterHotKey = user32.UnregisterHotKey
UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
UnregisterHotKey.restype = wintypes.BOOL


def squeeze01(v, amount):
    return 0.5 + (v - 0.5) * amount


def apply_coordinate_correction(nx, ny):
    """Convert Windows-normalized coords to physical-screen-normalized coords.

    Windows is assuming the whole touch frame maps to the active display.
    If the real visible screen is smaller/larger than the frame, and both
    top-left corners are aligned, multiply by frame/screen.
    """
    if PHYSICAL_REMAP_ENABLED:
        nx = nx * (FRAME_PHYS_W / max(1e-9, SCREEN_PHYS_W))
        ny = ny * (FRAME_PHYS_H / max(1e-9, SCREEN_PHYS_H))

    nx = squeeze01(nx, UDP_X_SQUEEZE)
    ny = squeeze01(ny, UDP_Y_SQUEEZE)
    return nx, ny


def send_udp_objects():
    if not UDP_ENABLED:
        return

    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)

    touches = list(active_touches.items())[:255]
    packet = struct.pack("<B", len(touches))

    for tid, t in touches:
        # active_touches stores Windows screen-space pixel coords as "x" and "y".
        nx = t["x"] / max(1, screen_w)
        ny = t["y"] / max(1, screen_h)
        nx, ny = apply_coordinate_correction(nx, ny)

        rotation = 0.0

        packet += struct.pack(
            "<H f f f",
            tid % 65535,
            float(nx),
            float(ny),
            float(rotation),
        )

    try:
        udp_sock.sendto(packet, (UDP_IP, UDP_PORT))
    except OSError as e:
        print(f"UDP send failed: {e}")

def set_capture_enabled(enabled):
    global capture_enabled
    capture_enabled = enabled

    if capture_hwnd:
        if enabled:
            win32gui.ShowWindow(capture_hwnd, win32con.SW_SHOW)
            # keep debug window above it
            if render_hwnd:
                win32gui.SetWindowPos(
                    render_hwnd,
                    win32con.HWND_TOPMOST,
                    0, 0, 0, 0,
                    win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE
                )
            print("Capture ON")
        else:
            win32gui.ShowWindow(capture_hwnd, win32con.SW_HIDE)
            print("Capture OFF")


def update_touch(ti):
    screen_x = TOUCH_COORD_TO_PIXEL(ti.x)
    screen_y = TOUCH_COORD_TO_PIXEL(ti.y)

    width = TOUCH_COORD_TO_PIXEL(ti.cxContact)
    height = TOUCH_COORD_TO_PIXEL(ti.cyContact)

    diameter = max(width, height)
    diameter = max(MIN_DIAMETER, diameter)

    tid = ti.dwID

    if ti.dwFlags & (TOUCHEVENTF_DOWN | TOUCHEVENTF_MOVE):
        active_touches[tid] = {
            "x": screen_x,
            "y": screen_y,
            "d": diameter,
        }
    elif ti.dwFlags & TOUCHEVENTF_UP:
        active_touches.pop(tid, None)

    if render_hwnd:
        InvalidateRect(render_hwnd, None, False)


def get_render_dots(hwnd):
    client_rect = win32gui.GetClientRect(hwnd)
    client_w = client_rect[2] - client_rect[0]
    client_h = client_rect[3] - client_rect[1]

    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)

    scale_x = client_w / max(1, screen_w)
    scale_y = client_h / max(1, screen_h)
    scale = min(scale_x, scale_y)

    dots = []

    for tid, t in active_touches.items():
        nx = t["x"] / max(1, screen_w)
        ny = t["y"] / max(1, screen_h)
        nx, ny = apply_coordinate_correction(nx, ny)

        # Intentionally unclamped: if the physical frame extends beyond the screen,
        # touches outside the screen area can go outside the debug window too.
        x = nx * client_w
        y = ny * client_h

        d = max(4, t["d"] * scale)

        dots.append({
            "id": tid,
            "x": x,
            "y": y,
            "d": d,
        })

    return dots, client_rect


def capture_wnd_proc(hwnd, msg, wparam, lparam):
    try:
        if msg == WM_TOUCH:
            count = wparam & 0xFFFF
            inputs = (TOUCHINPUT * count)()

            ok = GetTouchInputInfo(lparam, count, inputs, ctypes.sizeof(TOUCHINPUT))
            if ok:
                # Update the whole Windows touch bundle first, then send exactly
                # one UDP packet for the completed state. This preserves simultaneity.
                for ti in inputs:
                    update_touch(ti)

                if SEND_UDP_ON_TOUCH_EVENTS:
                    send_udp_objects()

            CloseTouchInputHandle(lparam)
            return 0

        return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)

    except Exception:
        traceback.print_exc()
        return 0


def render_wnd_proc(hwnd, msg, wparam, lparam):
    try:
        if msg == WM_HOTKEY:
            if wparam == HOTKEY_ID:
                set_capture_enabled(not capture_enabled)
            return 0

        elif msg == win32con.WM_KEYDOWN:
            if wparam == win32con.VK_ESCAPE:
                set_capture_enabled(False)
                return 0

        elif msg == win32con.WM_PAINT:
            hdc, ps = win32gui.BeginPaint(hwnd)
            try:
                dots, client_rect = get_render_dots(hwnd)

                win32gui.FillRect(hdc, client_rect, win32gui.GetStockObject(win32con.WHITE_BRUSH))
                win32gui.SetBkMode(hdc, win32con.TRANSPARENT)

                pen = win32gui.CreatePen(win32con.PS_SOLID, 1, 0)
                brush = win32gui.CreateSolidBrush(0)

                old_pen = win32gui.SelectObject(hdc, pen)
                old_brush = win32gui.SelectObject(hdc, brush)

                # draw lines and distance labels
                for i in range(len(dots)):
                    for j in range(i + 1, len(dots)):
                        a = dots[i]
                        b = dots[j]

                        dx = b["x"] - a["x"]
                        dy = b["y"] - a["y"]
                        dist = math.hypot(dx, dy)

                        if dist <= LINE_DISTANCE_THRESHOLD:
                            x1 = int(a["x"])
                            y1 = int(a["y"])
                            x2 = int(b["x"])
                            y2 = int(b["y"])

                            win32gui.MoveToEx(hdc, x1, y1)
                            win32gui.LineTo(hdc, x2, y2)

                            mx = int((a["x"] + b["x"]) / 2)
                            my = int((a["y"] + b["y"]) / 2)

                            label = f"{dist:.1f}"
                            text_rect = (mx + 4, my + 4, mx + 80, my + 24)
                            win32gui.DrawText(
                                hdc,
                                label,
                                -1,
                                text_rect,
                                win32con.DT_LEFT | win32con.DT_TOP | win32con.DT_SINGLELINE
                            )

                # draw dots
                for t in dots:
                    x = t["x"]
                    y = t["y"]
                    r = t["d"] / 2.0

                    win32gui.Ellipse(
                        hdc,
                        int(x - r),
                        int(y - r),
                        int(x + r),
                        int(y + r),
                    )

                win32gui.SelectObject(hdc, old_pen)
                win32gui.SelectObject(hdc, old_brush)
                win32gui.DeleteObject(pen)
                win32gui.DeleteObject(brush)

            finally:
                win32gui.EndPaint(hwnd, ps)

            return 0

        elif msg == win32con.WM_DESTROY:
            UnregisterHotKey(hwnd, HOTKEY_ID)
            win32gui.PostQuitMessage(0)
            return 0

        return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)

    except Exception:
        traceback.print_exc()
        return 0


def create_capture_window():
    wc = win32gui.WNDCLASS()
    wc.lpfnWndProc = capture_wnd_proc
    wc.lpszClassName = "TouchCaptureWindow"

    atom = win32gui.RegisterClass(wc)

    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)

    hwnd = win32gui.CreateWindowEx(
        win32con.WS_EX_TOPMOST | win32con.WS_EX_TOOLWINDOW,
        atom,
        "",
        win32con.WS_POPUP,
        0,
        0,
        screen_w,
        screen_h,
        0,
        0,
        0,
        None,
    )

    if not hwnd:
        raise RuntimeError("Failed to create capture window")

    ok = RegisterTouchWindow(hwnd, 0)
    print(f"RegisterTouchWindow(capture): {bool(ok)}")

    win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
    return hwnd


def create_render_window():
    wc = win32gui.WNDCLASS()
    wc.lpfnWndProc = render_wnd_proc
    wc.lpszClassName = "TouchRenderWindow"
    wc.hCursor = win32gui.LoadCursor(0, win32con.IDC_ARROW)

    atom = win32gui.RegisterClass(wc)

    hwnd = win32gui.CreateWindow(
        atom,
        "Touch Debug",
        win32con.WS_OVERLAPPEDWINDOW | win32con.WS_VISIBLE,
        100,
        100,
        900,
        700,
        0,
        0,
        0,
        None,
    )

    if not hwnd:
        raise RuntimeError("Failed to create render window")

    ok = RegisterHotKey(
        hwnd,
        HOTKEY_ID,
        win32con.MOD_CONTROL | win32con.MOD_SHIFT,
        ord("T")
    )
    print(f"Hotkey registered: {bool(ok)} (Ctrl+Shift+T)")

    # explicitly keep debug window above the hidden capture layer
    win32gui.SetWindowPos(
        hwnd,
        win32con.HWND_TOPMOST,
        0, 0, 0, 0,
        win32con.SWP_NOMOVE | win32con.SWP_NOSIZE
    )

    return hwnd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fullscreen Windows touch capture/debugger with UDP output."
    )
    parser.add_argument("--udp-ip", default=UDP_IP)
    parser.add_argument("--udp-port", type=int, default=UDP_PORT)
    parser.add_argument("--no-udp", action="store_true")
    parser.add_argument(
        "--uncapped", action="store_true",
        help="Send UDP packets continuously as fast as possible using the latest touch state."
    )
    parser.add_argument(
        "--hz", type=float, default=0.0,
        help="Send UDP continuously at a fixed rate, e.g. --hz 1000. Overrides event-only sending."
    )

    parser.add_argument(
        "--screen-phys", nargs=2, type=float, metavar=("W", "H"),
        help="Physical visible screen size, e.g. --screen-phys 344 194"
    )
    parser.add_argument(
        "--frame-phys", nargs=2, type=float, metavar=("W", "H"),
        help="Physical touch-frame active size, same units as --screen-phys"
    )
    parser.add_argument(
        "--x-squeeze", type=float, default=UDP_X_SQUEEZE,
        help="Legacy center-based X squeeze applied after physical remap. Default 1.0"
    )
    parser.add_argument(
        "--y-squeeze", type=float, default=UDP_Y_SQUEEZE,
        help="Legacy center-based Y squeeze applied after physical remap. Default 1.0"
    )
    return parser.parse_args()


def apply_args(args):
    global UDP_IP, UDP_PORT, UDP_ENABLED
    global UDP_UNCAPPED, UDP_HZ, SEND_UDP_ON_TOUCH_EVENTS
    global PHYSICAL_REMAP_ENABLED, SCREEN_PHYS_W, SCREEN_PHYS_H, FRAME_PHYS_W, FRAME_PHYS_H
    global UDP_X_SQUEEZE, UDP_Y_SQUEEZE

    UDP_IP = args.udp_ip
    UDP_PORT = args.udp_port
    UDP_ENABLED = not args.no_udp
    UDP_UNCAPPED = bool(args.uncapped)
    UDP_HZ = max(0.0, float(args.hz))

    # In continuous modes, the loop sends packets. Do not also send inside WM_TOUCH.
    SEND_UDP_ON_TOUCH_EVENTS = not (UDP_UNCAPPED or UDP_HZ > 0.0)

    UDP_X_SQUEEZE = args.x_squeeze
    UDP_Y_SQUEEZE = args.y_squeeze

    if args.screen_phys or args.frame_phys:
        if not (args.screen_phys and args.frame_phys):
            raise SystemExit("Use --screen-phys W H and --frame-phys W H together.")
        SCREEN_PHYS_W, SCREEN_PHYS_H = args.screen_phys
        FRAME_PHYS_W, FRAME_PHYS_H = args.frame_phys
        PHYSICAL_REMAP_ENABLED = True



def run_message_loop():
    """Run either the normal blocking Windows message pump or a UDP send loop.

    Default mode uses PumpMessages(), so UDP sends happen only when WM_TOUCH arrives.
    --uncapped and --hz use PumpWaitingMessages() so we can keep transmitting the
    latest active touch state even when Windows has no new touch message.
    """
    if not UDP_UNCAPPED and UDP_HZ <= 0.0:
        run_message_loop()
        return

    if UDP_UNCAPPED:
        print("UDP continuous send mode: UNCAPPED")
        while True:
            if win32gui.PumpWaitingMessages():
                break
            send_udp_objects()
            # Yield to the OS without intentionally rate-limiting. Remove this only
            # if you truly want to peg a CPU core.
            time.sleep(0)
        return

    interval = 1.0 / UDP_HZ
    next_send = time.perf_counter()
    print(f"UDP continuous send mode: {UDP_HZ:.3f} Hz")

    while True:
        if win32gui.PumpWaitingMessages():
            break

        now = time.perf_counter()
        if now >= next_send:
            send_udp_objects()
            # Avoid death-spiral if the app is paused/stalled.
            if now - next_send > interval * 4:
                next_send = now + interval
            else:
                next_send += interval
        else:
            sleep_for = min(0.001, next_send - now)
            if sleep_for > 0:
                time.sleep(sleep_for)


def main():
    global capture_hwnd, render_hwnd

    args = parse_args()
    apply_args(args)

    capture_hwnd = create_capture_window()
    render_hwnd = create_render_window()

    print("Fullscreen touch capture active")
    print("Visible debug window remaps touches into its own bounds")
    print(f"Drawing connection lines for dots within {LINE_DISTANCE_THRESHOLD}px")
    print(f"UDP output: {UDP_IP}:{UDP_PORT} enabled={UDP_ENABLED}")
    if UDP_UNCAPPED:
        print("UDP cadence: uncapped continuous latest-state packets")
    elif UDP_HZ > 0.0:
        print(f"UDP cadence: fixed continuous latest-state packets at {UDP_HZ:.3f} Hz")
    else:
        print("UDP cadence: event-driven, one packet per WM_TOUCH bundle")
    if PHYSICAL_REMAP_ENABLED:
        print(
            "Physical remap enabled: "
            f"screen={SCREEN_PHYS_W}x{SCREEN_PHYS_H}, "
            f"frame={FRAME_PHYS_W}x{FRAME_PHYS_H}, "
            f"scale=({FRAME_PHYS_W / SCREEN_PHYS_W:.6f}, {FRAME_PHYS_H / SCREEN_PHYS_H:.6f})"
        )
    else:
        print("Physical remap disabled")
    print(f"Legacy squeeze: x={UDP_X_SQUEEZE}, y={UDP_Y_SQUEEZE}")
    print("Ctrl+Shift+T toggles capture ON/OFF")
    print("Esc in the debug window turns capture OFF")

    run_message_loop()


if __name__ == "__main__":
    main()