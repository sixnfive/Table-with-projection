import ctypes
from ctypes import wintypes
import math
import traceback
import socket
import struct
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

# Optional coordinate correction applied to UDP only.
# 1.0 = no correction. Use 16/9 if you need the same horizontal unsqueeze.
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


def send_udp_objects():
    if not UDP_ENABLED:
        return

    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)

    touches = list(active_touches.items())[:255]
    packet = struct.pack("<B", len(touches))

    for tid, t in touches:
        # active_touches stores screen-space pixel coords as "x" and "y"
        nx = t["x"] / max(1, screen_w)
        ny = t["y"] / max(1, screen_h)

        nx = squeeze01(nx, UDP_X_SQUEEZE)
        ny = squeeze01(ny, UDP_Y_SQUEEZE)

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

    send_udp_objects()


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
        nx = max(0.0, min(1.0, t["x"] / max(1, screen_w)))
        ny = max(0.0, min(1.0, t["y"] / max(1, screen_h)))

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
                for ti in inputs:
                    update_touch(ti)

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


def main():
    global capture_hwnd, render_hwnd

    capture_hwnd = create_capture_window()
    render_hwnd = create_render_window()

    print("Fullscreen touch capture active")
    print("Visible debug window remaps touches into its own bounds")
    print(f"Drawing connection lines for dots within {LINE_DISTANCE_THRESHOLD}px")
    print(f"UDP output: {UDP_IP}:{UDP_PORT} enabled={UDP_ENABLED}")
    print("Ctrl+Shift+T toggles capture ON/OFF")
    print("Esc in the debug window turns capture OFF")

    win32gui.PumpMessages()


if __name__ == "__main__":
    main()