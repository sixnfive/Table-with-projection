"""
tracker_procrustes_v2.py

v2 changes over tracker_procrustes.py:
  1. CAMERA SELECTION  — probe available cameras at startup and pick one with a
     live preview (N/P or number keys to switch, ENTER to confirm).
  2. MOUSE ROI         — draw a polygon over the area you actually care about.
     Everything outside is masked out so the tracker ignores it. The frame is
     also cropped to the polygon's bounding box, so the heavy ops (contours,
     morphology, optical-flow pyramids) run on a smaller image → faster.
  3. The ROI is saved to roi_config.json and reloaded next run (per camera/res).
  4. UDP normalization and MAX_AREA now use the real resolution, not 1920x1080.
  5. SPEED (tracking logic + UDP output unchanged):
       - MJPG + CAP_DSHOW so the camera honors 1080p instead of streaming 4K.
       - The preview window is downscaled (display only; processing is full-res).
       - ROI crop (#2) shrinks the pixels every per-frame op has to chew through.

Setup controls
  Camera picker : N / P  switch camera   |  0-9 pick by index  |  ENTER select  |  Q quit
  ROI editor    : LEFT-CLICK add point   |  RIGHT-CLICK / Z undo  |  C clear
                  R reload saved  |  F use full frame  |  ENTER confirm  |  Q quit
Runtime controls (tracker window, unchanged)
  SPACE calibrate   |   Q quit
"""

import cv2
import numpy as np
import socket
import struct
import time
import threading
import queue
import os
import json
import sys

# ── Capture config ──────────────────────────────────────────────────────────
CAMERA_BACKEND = cv2.CAP_DSHOW          # DSHOW is fast/reliable for probing on Windows
REQ_WIDTH      = 1920
REQ_HEIGHT     = 1080
REQ_FPS        = 60
USE_MJPG       = True                   # needed for 1080p60 over USB; set False if cam misbehaves
MAX_CAM_INDEX  = 6                      # probe indices 0..MAX_CAM_INDEX
CROP_TO_ROI    = True                   # crop processing to the ROI bounding box for speed
ROI_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "roi_config.json")

# setup windows are scaled down to fit the screen; mouse coords are mapped back to full-res
DISPLAY_MAX_W  = 1280
DISPLAY_MAX_H  = 720

FONT = cv2.FONT_HERSHEY_SIMPLEX


def fit_display(img, max_w=DISPLAY_MAX_W, max_h=DISPLAY_MAX_H):
    """Downscale an image to fit on screen. Returns (resized_img, scale)."""
    h, w = img.shape[:2]
    scale = min(1.0, max_w / w, max_h / h)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img, scale


def open_capture(src, backend=CAMERA_BACKEND, w=REQ_WIDTH, h=REQ_HEIGHT, fps=REQ_FPS):
    cap = cv2.VideoCapture(src, backend)
    if USE_MJPG:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS,          fps)
    return cap


class CameraThread:
    def __init__(self, src=0, backend=CAMERA_BACKEND, w=REQ_WIDTH, h=REQ_HEIGHT, fps=REQ_FPS):
        self.cap = open_capture(src, backend, w, h, fps)
        print(f"Camera FPS: {self.cap.get(cv2.CAP_PROP_FPS)}")
        print(f"Camera resolution: "
              f"{self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        self.frame   = None
        self.lock    = threading.Lock()
        self.running = True
        self.thread  = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.cap.release()


def wait_for_frame(cam, timeout=5.0):
    """Block until the camera thread delivers a frame (or give up)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        f = cam.read()
        if f is not None:
            return f
        time.sleep(0.02)
    return None


# ── Camera selection ──────────────────────────────────────────────────────────

def list_available_cameras(max_index=MAX_CAM_INDEX, backend=CAMERA_BACKEND):
    found = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i, backend)
        if cap is not None and cap.isOpened():
            ret, _ = cap.read()
            if ret:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                found.append((i, w, h))
        cap.release()
    return found


def select_camera(backend=CAMERA_BACKEND):
    """Live-preview picker. Returns the chosen device index."""
    print("Probing cameras...")
    cams = list_available_cameras(backend=backend)
    if not cams:
        print("No cameras found! Check connections / backend.")
        sys.exit(1)

    print("Available cameras:")
    for i, w, h in cams:
        print(f"  [{i}]  native {w}x{h}")

    if len(cams) == 1:
        print(f"Only one camera ([{cams[0][0]}]) — selecting it.")
        return cams[0][0]

    idxs = [c[0] for c in cams]
    sel  = 0
    cap  = open_capture(idxs[sel], backend)
    win  = "Select camera"
    cv2.namedWindow(win)

    def reopen(new_sel):
        nonlocal cap, sel
        cap.release()
        sel = new_sel % len(idxs)
        cap = open_capture(idxs[sel], backend)

    chosen = None
    while True:
        ret, frame = cap.read()
        base = frame if (ret and frame is not None) else np.zeros((540, 960, 3), np.uint8)
        disp, _ = fit_display(base)
        cur = idxs[sel]
        cv2.putText(disp, f"Camera [{cur}]   ({sel + 1}/{len(idxs)})", (20, 40),
                    FONT, 1.0, (0, 255, 255), 2)
        cv2.putText(disp, "N / P switch  |  0-9 index  |  ENTER select  |  Q quit",
                    (20, 80), FONT, 0.6, (0, 255, 255), 2)
        cv2.imshow(win, disp)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord('n'), ord('d')):
            reopen(sel + 1)
        elif key in (ord('p'), ord('a')):
            reopen(sel - 1)
        elif ord('0') <= key <= ord('9') and (key - ord('0')) in idxs:
            reopen(idxs.index(key - ord('0')))
        elif key == 13:                       # ENTER
            chosen = idxs[sel]
            break
        elif key in (ord('q'), 27):           # Q / ESC
            cap.release()
            cv2.destroyWindow(win)
            sys.exit(0)

    cap.release()
    cv2.destroyWindow(win)
    cv2.waitKey(1)
    print(f"Selected camera [{chosen}]")
    return chosen


# ── ROI (region of interest) ────────────────────────────────────────────────

def load_roi_config():
    if os.path.exists(ROI_CONFIG_FILE):
        try:
            with open(ROI_CONFIG_FILE) as f:
                return json.load(f)
        except Exception as e:
            print(f"Could not read {ROI_CONFIG_FILE}: {e}")
    return None


def save_roi_config(camera, w, h, points):
    cfg = {"camera": camera, "width": w, "height": h, "points": points}
    try:
        with open(ROI_CONFIG_FILE, 'w') as f:
            json.dump(cfg, f, indent=2)
        print(f"ROI saved to {ROI_CONFIG_FILE}")
    except Exception as e:
        print(f"Could not save ROI: {e}")


def draw_roi(cam, camera_idx, frame_w, frame_h):
    """
    Interactive polygon editor. Returns a list of (x,y) points (image coords).
    Fewer than 3 points means 'use the full frame' (no masking).
    """
    saved   = load_roi_config()
    points  = []
    if saved and saved.get("camera") == camera_idx \
            and saved.get("width") == frame_w and saved.get("height") == frame_h:
        points = [tuple(p) for p in saved.get("points", [])]
        if points:
            print(f"Loaded saved ROI with {len(points)} points (press C to clear, R to reload).")

    scale = min(1.0, DISPLAY_MAX_W / frame_w, DISPLAY_MAX_H / frame_h)
    state = {'points': points, 'cursor': None}

    def on_mouse(event, x, y, flags, param):
        fx, fy = int(x / scale), int(y / scale)         # display coords -> full-res coords
        if event == cv2.EVENT_LBUTTONDOWN:
            state['points'].append((fx, fy))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if state['points']:
                state['points'].pop()
        elif event == cv2.EVENT_MOUSEMOVE:
            state['cursor'] = (fx, fy)

    win = "Draw ROI"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        frame = cam.read()
        if frame is None:
            continue
        disp = frame.copy()
        pts  = state['points']

        # darken everything outside the polygon so excluded zones are obvious
        if len(pts) >= 3:
            mask = np.zeros((frame_h, frame_w), np.uint8)
            cv2.fillPoly(mask, [np.array(pts, np.int32)], 255)
            dark = (disp * 0.30).astype(np.uint8)
            disp = np.where(mask[:, :, None] == 255, disp, dark)
            cv2.polylines(disp, [np.array(pts, np.int32)], True, (0, 255, 0), 2)

        # vertices + edges so far
        for i, p in enumerate(pts):
            cv2.circle(disp, p, 5, (0, 0, 255), -1)
            if i > 0:
                cv2.line(disp, pts[i - 1], p, (0, 255, 255), 1)
        # rubber-band line to the cursor
        if pts and state['cursor'] is not None:
            cv2.line(disp, pts[-1], state['cursor'], (0, 200, 200), 1)

        disp, _ = fit_display(disp)
        cv2.putText(disp, "LEFT-CLICK add  |  RIGHT/Z undo  |  C clear  |  R reload",
                    (15, 28), FONT, 0.55, (0, 255, 255), 2)
        cv2.putText(disp, "F full frame  |  ENTER confirm  |  Q quit    points: %d" % len(pts),
                    (15, 54), FONT, 0.55, (0, 255, 255), 2)
        cv2.imshow(win, disp)
        key = cv2.waitKey(20) & 0xFF

        if key == 13:                                   # ENTER confirm
            break
        elif key == ord('z'):                           # undo
            if pts:
                pts.pop()
        elif key == ord('c'):                           # clear
            pts.clear()
        elif key == ord('f'):                           # full frame
            pts.clear()
            break
        elif key == ord('r'):                           # reload saved
            if saved and saved.get("points"):
                state['points'] = [tuple(p) for p in saved["points"]]
        elif key in (ord('q'), 27):
            cv2.destroyWindow(win)
            cam.stop()
            sys.exit(0)

    cv2.destroyWindow(win)
    cv2.waitKey(1)
    final = state['points'] if len(state['points']) >= 3 else []
    save_roi_config(camera_idx, frame_w, frame_h, final)
    if final:
        print(f"ROI confirmed with {len(final)} points.")
    else:
        print("No ROI — using full frame.")
    return final


# ── UDP ───────────────────────────────────────────────────────────────────────
UDP_IP   = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

DIFF_THRESHOLD      = 30
MIN_AREA            = 500
MAX_AREA            = REQ_WIDTH * REQ_HEIGHT * 0.8   # recomputed once we know the proc size
MIN_POINTS_PROCRUST = 4
MIN_POINTS_HEALTHY  = 35
MAX_VELOCITY        = 20
MAX_LOST            = 60
HU_THRESHOLD        = 30.0
MATCH_RADIUS        = 60    # max pixels to match new point to cal_ref entry

lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

feature_params = dict(
    maxCorners=40,
    qualityLevel=0.01,
    minDistance=5,
    blockSize=7
)

objects   = {}
prev_gray = None

# set during setup, in processed (possibly cropped) coordinates
ROI_MASK_PROC = None      # uint8 mask the size of the processed frame, or None
ROI_POLYGON   = []        # polygon in FULL-frame coords (for visualization)
OX, OY        = 0, 0      # crop offset (full = processed + offset)
FRAME_W_FULL  = REQ_WIDTH
FRAME_H_FULL  = REQ_HEIGHT
PROC_W        = REQ_WIDTH
PROC_H        = REQ_HEIGHT


def crop_frame(full):
    """Slice the processed region out of a full frame."""
    if CROP_TO_ROI:
        return full[OY:OY + PROC_H, OX:OX + PROC_W]
    return full

# ── Core helpers ──────────────────────────────────────────────────────────────

def get_fg_mask(frame, background, roi_mask=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, background)
    _, fg = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    if roi_mask is not None:
        fg = cv2.bitwise_and(fg, roi_mask)          # kill everything outside the ROI
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel_close)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  kernel_open)
    return fg

def get_hu(contour):
    hu = cv2.HuMoments(cv2.moments(contour)).flatten()
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

def find_feature_points(gray, contour):
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    pts = cv2.goodFeaturesToTrack(gray, mask=mask, **feature_params)
    return pts

def get_contours(frame, background, roi_mask=None):
    fg = get_fg_mask(frame, background, roi_mask)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA or area > MAX_AREA:
            continue
        M = cv2.moments(c)
        if M['m00'] == 0:
            continue
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        valid.append((c, cx, cy, area))
    return valid

def procrustes(ref_pts, obs_pts):
    """Fit obs to ref. Returns (cx, cy, angle)."""
    obs_mean = np.mean(obs_pts, axis=0)
    obs_c    = obs_pts - obs_mean
    ref_c    = ref_pts - np.mean(ref_pts, axis=0)
    H        = ref_c.T @ obs_c
    U, _, Vt = np.linalg.svd(H)
    d        = np.linalg.det(Vt.T @ U.T)
    R        = Vt.T @ np.diag([1, d]) @ U.T
    angle    = np.arctan2(R[1, 0], R[0, 0])
    t        = np.mean(obs_pts - (R @ ref_pts.T).T, axis=0)
    return float(t[0]), float(t[1]), float(angle)

def fix_angle(new_angle, prev_angle, alpha=0.4):
    flipped = new_angle + np.pi if new_angle < 0 else new_angle - np.pi
    if abs(flipped - prev_angle) < abs(new_angle - prev_angle):
        new_angle = flipped
    return alpha * prev_angle + (1 - alpha) * new_angle

def project_cal_ref(cal_ref, cx, cy, angle):
    """Project calibrated offsets into world space given position and rotation."""
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    R     = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return (R @ cal_ref.T).T + np.array([cx, cy])

def match_points_to_cal_ref(new_pts, cal_ref, cx, cy, angle):
    """
    Spatially match new feature points to their nearest cal_ref entry.
    Projects cal_ref into world space, then nearest-neighbour match.
    Returns (matched_obs, matched_ref) or (None, None) if too few matches.
    """
    cal_world   = project_cal_ref(cal_ref, cx, cy, angle)
    p           = new_pts.reshape(-1, 2).astype(np.float32)
    matched_obs = []
    matched_ref = []
    used        = set()

    for pt in p:
        dists = np.linalg.norm(cal_world - pt, axis=1)
        idx   = int(np.argmin(dists))
        if idx not in used and dists[idx] < MATCH_RADIUS:
            matched_obs.append(pt)
            matched_ref.append(cal_ref[idx])
            used.add(idx)

    if len(matched_obs) < MIN_POINTS_PROCRUST:
        return None, None

    return (np.array(matched_obs, dtype=np.float32),
            np.array(matched_ref, dtype=np.float32))

# ── Reacquisition ─────────────────────────────────────────────────────────────

def attempt_reacquire(gray, all_contours, obj, oid):
    """
    Search whole frame by Hu shape match.
    On match: spatially map new points to cal_ref, run Procrustes immediately.
    cal_ref is never modified.
    """
    best_c     = None
    best_cx    = None
    best_cy    = None
    best_score = HU_THRESHOLD

    for c, cx, cy, area in all_contours:
        if area > obj['cal_area'] * 3.0 or area < obj['cal_area'] * 0.3:
            continue
        score = float(np.linalg.norm(get_hu(c) - obj['cal_hu']))
        if score < best_score:
            best_score = score
            best_c     = c
            best_cx    = cx
            best_cy    = cy

    if best_c is None:
        scores = [(round(float(np.linalg.norm(get_hu(c) - obj['cal_hu'])), 2), round(area))
                  for c, cx, cy, area in all_contours
                  if obj['cal_area'] * 0.3 < area < obj['cal_area'] * 3.0]
        if scores:
            print(f"Obj {oid} no match — best: {sorted(scores)[:3]} thresh={HU_THRESHOLD}")
        return False

    new_pts = find_feature_points(gray, best_c)
    if new_pts is None or len(new_pts) < MIN_POINTS_PROCRUST:
        return False

    # spatially match new points to cal_ref using contour centroid as anchor
    obs, ref = match_points_to_cal_ref(
        new_pts, obj['cal_ref'], best_cx, best_cy, obj['rotation'])

    if obs is None:
        # fallback — just use contour centroid directly
        print(f"Obj {oid} reacquired (fallback) at ({best_cx:.0f},{best_cy:.0f})")
        n = min(len(new_pts), len(obj['cal_ref']))
        obs = new_pts[:n].reshape(-1, 2).astype(np.float32)
        ref = obj['cal_ref'][:n]

    # run Procrustes immediately for accurate centroid
    new_cx, new_cy, new_angle = procrustes(ref, obs)
    new_angle = fix_angle(new_angle, obj['rotation'])

    obj['pts_obs']     = obs
    obj['pts_ref']     = ref
    obj['cx']          = float(new_cx)
    obj['cy']          = float(new_cy)
    obj['rotation']    = float(new_angle)
    obj['lost']        = False
    obj['lost_frames'] = 0
    print(f"Obj {oid} reacquired at ({new_cx:.0f},{new_cy:.0f}) score={best_score:.2f} pts={len(obs)}")
    return True

# ── Topup ─────────────────────────────────────────────────────────────────────

def topup(gray, all_contours, obj):
    """
    Add more points when sparse, only if clean unoccluded contour exists.
    Spatially maps new points to cal_ref.
    Never updates cx/cy — Procrustes owns that.
    """
    # find nearby contour with area close to calibrated
    best_c  = None
    best_d  = 200
    for c, cx, cy, area in all_contours:
        if area > obj['cal_area'] * 1.5 or area < obj['cal_area'] * 0.5:
            continue  # reject merged/partial blobs
        d = np.hypot(cx - obj['cx'], cy - obj['cy'])
        if d < best_d:
            best_d = d
            best_c = c

    if best_c is None:
        return  # no clean contour — don't topup during occlusion

    new_pts = find_feature_points(gray, best_c)
    if new_pts is None or len(new_pts) < MIN_POINTS_PROCRUST:
        return

    # spatially match to cal_ref using current known position/rotation
    obs, ref = match_points_to_cal_ref(
        new_pts, obj['cal_ref'], obj['cx'], obj['cy'], obj['rotation'])

    if obs is None:
        return

    obj['pts_obs'] = obs
    obj['pts_ref'] = ref

# ── Pack UDP ──────────────────────────────────────────────────────────────────

def pack_objects(objects):
    active = {oid: o for oid, o in objects.items() if not o['lost']}
    data   = struct.pack('<B', len(active))
    for oid, o in active.items():
        # map processed-frame coords back into full-frame normalized space
        data += struct.pack('<H f f f',
            oid % 65535,
            (o['cx'] + OX) / FRAME_W_FULL,
            (o['cy'] + OY) / FRAME_H_FULL,
            o['rotation'])
    return data

# ── Viz thread ────────────────────────────────────────────────────────────────

viz_queue     = queue.Queue(maxsize=1)
space_pressed = False

def viz_thread_fn():
    global space_pressed
    while True:
        try:
            frame = viz_queue.get(timeout=1)
            if frame is None:
                break
            cv2.imshow("tracker", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                space_pressed = True
            elif key == ord('q'):
                os._exit(0)
        except queue.Empty:
            continue

# ── Startup / setup ─────────────────────────────────────────────────────────

def main():
    global ROI_MASK_PROC, ROI_POLYGON, OX, OY, PROC_W, PROC_H
    global FRAME_W_FULL, FRAME_H_FULL, MAX_AREA, prev_gray

    # 1) pick the camera ------------------------------------------------------
    src = select_camera(CAMERA_BACKEND)

    cam = CameraThread(src, CAMERA_BACKEND, REQ_WIDTH, REQ_HEIGHT, REQ_FPS)
    time.sleep(0.5)
    probe = wait_for_frame(cam)
    if probe is None:
        print("Camera produced no frames — aborting.")
        cam.stop()
        sys.exit(1)
    FRAME_H_FULL, FRAME_W_FULL = probe.shape[:2]
    print(f"Working resolution: {FRAME_W_FULL}x{FRAME_H_FULL}")
    if FRAME_W_FULL * FRAME_H_FULL > REQ_WIDTH * REQ_HEIGHT * 1.3:
        print(f"  !! Camera is delivering {FRAME_W_FULL}x{FRAME_H_FULL} instead of "
              f"{REQ_WIDTH}x{REQ_HEIGHT}.")
        print( "  !! Processing that many pixels is the #1 cause of low FPS. "
               "Lower REQ_WIDTH/REQ_HEIGHT, or the camera mode isn't honoring the request.")

    # 2) draw the ROI ---------------------------------------------------------
    print("Draw the area to watch. Keep the scene clear of objects for this step.")
    ROI_POLYGON = draw_roi(cam, src, FRAME_W_FULL, FRAME_H_FULL)

    # build full-frame mask, then derive crop box + processed-size mask
    if ROI_POLYGON:
        full_mask = np.zeros((FRAME_H_FULL, FRAME_W_FULL), np.uint8)
        cv2.fillPoly(full_mask, [np.array(ROI_POLYGON, np.int32)], 255)
        if CROP_TO_ROI:
            x, y, bw, bh = cv2.boundingRect(np.array(ROI_POLYGON, np.int32))
            OX, OY       = x, y
            PROC_W, PROC_H = bw, bh
            ROI_MASK_PROC  = full_mask[OY:OY + PROC_H, OX:OX + PROC_W]
        else:
            OX, OY         = 0, 0
            PROC_W, PROC_H = FRAME_W_FULL, FRAME_H_FULL
            ROI_MASK_PROC  = full_mask
    else:
        OX, OY         = 0, 0
        PROC_W, PROC_H = FRAME_W_FULL, FRAME_H_FULL
        ROI_MASK_PROC  = None

    MAX_AREA = PROC_W * PROC_H * 0.8
    print(f"Processing region: {PROC_W}x{PROC_H} at offset ({OX},{OY})  "
          f"{'(cropped)' if (PROC_W, PROC_H) != (FRAME_W_FULL, FRAME_H_FULL) else '(full)'}")

    # 3) capture background (on the processed region) -------------------------
    print("Capturing background... keep frame clear for 3 seconds")
    buf, t0 = [], time.time()
    while time.time() - t0 < 3.0:
        f = cam.read()
        if f is not None:
            buf.append(cv2.cvtColor(crop_frame(f), cv2.COLOR_BGR2GRAY))
    background = np.median(np.stack(buf), axis=0).astype(np.uint8)
    print("Background captured! Place objects then press SPACE.")

    # start the viz/key thread now that interactive setup windows are gone
    threading.Thread(target=viz_thread_fn, daemon=True).start()

    # 4) calibration ----------------------------------------------------------
    calibrated = False
    global space_pressed
    while not calibrated:
        full = cam.read()
        if full is None:
            continue
        frame = crop_frame(full)

        if space_pressed:
            space_pressed = False
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray  = cv2.GaussianBlur(gray, (5, 5), 0)
            valid = get_contours(frame, background, ROI_MASK_PROC)

            for i, (contour, cx, cy, area) in enumerate(valid):
                pts = find_feature_points(gray, contour)
                if pts is None or len(pts) < MIN_POINTS_PROCRUST:
                    print(f"Warning: object {i} too few points")
                    continue

                p       = pts.reshape(-1, 2).astype(np.float32)
                cal_ref = p - np.array([cx, cy], dtype=np.float32)

                _, evecs      = cv2.PCACompute(cal_ref, mean=None)
                initial_angle = float(np.arctan2(evecs[0, 1], evecs[0, 0]))
                x, y, bw, bh  = cv2.boundingRect(contour)

                objects[i] = {
                    # sacred calibration data
                    'cal_ref':  cal_ref,
                    'cal_hu':   get_hu(contour),
                    'cal_area': float(area),
                    # active state — pts_obs and pts_ref always parallel
                    'pts_obs':     p.copy(),
                    'pts_ref':     cal_ref.copy(),
                    'cx':          float(cx),
                    'cy':          float(cy),
                    'rotation':    initial_angle,
                    'w':           float(bw),
                    'h':           float(bh),
                    'lost':        False,
                    'lost_frames': 0,
                }
                print(f"Object {i}: {len(p)} pts  "
                      f"centroid=({cx:.0f},{cy:.0f})  "
                      f"angle={np.degrees(initial_angle):.1f}°")

            if objects:
                prev_gray  = cv2.GaussianBlur(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
                calibrated = True
                print(f"Calibrated {len(objects)} objects.")
            else:
                print("No objects detected — try again")

        else:
            valid = get_contours(frame, background, ROI_MASK_PROC)
            vis   = full.copy()
            draw_roi_overlay(vis)
            for c, cx, cy, area in valid:
                cc = shift_contour(c)
                cv2.drawContours(vis, [cc], -1, (0, 255, 255), 2)
                cv2.putText(vis, f"{int(area)}", (int(cx) + OX, int(cy) + OY),
                            FONT, 0.5, (0, 255, 255), 1)
            cv2.putText(vis, f"Objects: {len(valid)} | SPACE to calibrate",
                        (20, 40), FONT, 0.8, (0, 255, 255), 2)
            if not viz_queue.full():
                viz_small, _ = fit_display(vis)
                viz_queue.put_nowait(viz_small)

    # 5) main loop ------------------------------------------------------------
    fps_counter = 0
    fps_start   = time.time()

    while True:
        full = cam.read()
        if full is None:
            continue
        frame = crop_frame(full)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if prev_gray is None:
            prev_gray = gray.copy()
            continue

        all_contours = get_contours(frame, background, ROI_MASK_PROC)

        for oid, obj in objects.items():

            obs = obj['pts_obs']
            ref = obj['pts_ref']

            # ── no points ───────────────────────────────────────────────────
            if obs is None or len(obs) < MIN_POINTS_PROCRUST:
                obj['lost_frames'] += 1
                obj['lost'] = obj['lost_frames'] > MAX_LOST
                attempt_reacquire(gray, all_contours, obj, oid)
                continue

            # ── optical flow ────────────────────────────────────────────────
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray,
                obs.reshape(-1, 1, 2).astype(np.float32),
                None, **lk_params)

            if new_pts is None or status is None:
                obj['lost_frames'] += 1
                obj['lost'] = obj['lost_frames'] > MAX_LOST
                attempt_reacquire(gray, all_contours, obj, oid)
                continue

            good     = status.flatten() == 1
            surv_obs = new_pts.reshape(-1, 2)[good]
            surv_ref = ref[good]

            # relative velocity filter
            prev_p     = obs[good]
            velocities = np.linalg.norm(surv_obs - prev_p, axis=1)
            if len(velocities) > 0:
                median_vel = np.median(velocities)
                vel_ok     = velocities < max(MAX_VELOCITY, median_vel * 3.0)
                surv_obs   = surv_obs[vel_ok]
                surv_ref   = surv_ref[vel_ok]

            n = len(surv_obs)

            if n >= MIN_POINTS_PROCRUST:
                # ── Procrustes ────────────────────────────────────────────────
                new_cx, new_cy, new_angle = procrustes(surv_ref, surv_obs)
                new_angle = fix_angle(new_angle, obj['rotation'])

                alpha           = 0.4
                obj['cx']       = alpha * new_cx    + (1 - alpha) * obj['cx']
                obj['cy']       = alpha * new_cy    + (1 - alpha) * obj['cy']
                obj['rotation'] = new_angle

                obj['pts_obs']     = surv_obs
                obj['pts_ref']     = surv_ref
                obj['lost']        = False
                obj['lost_frames'] = max(0, obj['lost_frames'] - 1)

                # topup only if clean contour available
                if n < MIN_POINTS_HEALTHY:
                    topup(gray, all_contours, obj)

            else:
                # ── too few — reacquire ───────────────────────────────────────
                obj['lost_frames'] += 1
                obj['lost']   = obj['lost_frames'] > MAX_LOST
                obj['pts_obs'] = surv_obs if n > 0 else None
                obj['pts_ref'] = surv_ref if n > 0 else None
                attempt_reacquire(gray, all_contours, obj, oid)

        prev_gray = gray.copy()

        sock.sendto(pack_objects(objects), (UDP_IP, UDP_PORT))

        # ── Visualize (drawn on the FULL frame, with crop offset) ────────────
        vis = full.copy()
        draw_roi_overlay(vis)
        for oid, obj in objects.items():
            cx    = int(obj['cx']) + OX
            cy    = int(obj['cy']) + OY
            angle = obj['rotation']
            n_pts = len(obj['pts_obs']) if obj['pts_obs'] is not None else 0
            state = 'lost'   if obj['lost'] else \
                    'sparse' if n_pts < MIN_POINTS_HEALTHY else \
                    'tracking'
            color = (0, 255, 0)   if state == 'tracking' else \
                    (0, 165, 255) if state == 'sparse'   else \
                    (0, 0, 255)

            if obj['pts_obs'] is not None:
                for pt in obj['pts_obs']:
                    cv2.circle(vis, (int(pt[0]) + OX, int(pt[1]) + OY), 3, color, -1)

            cv2.circle(vis, (cx, cy), 6, color, -1)
            dx = int(np.cos(angle) * 60)
            dy = int(np.sin(angle) * 60)
            cv2.arrowedLine(vis, (cx - dx, cy - dy), (cx + dx, cy + dy),
                            (0, 165, 255), 2, tipLength=0.3)
            cv2.putText(vis, f"{oid} {state} {np.degrees(angle):.0f}° p:{n_pts}",
                        (cx + 10, cy), FONT, 0.7, color, 2)

        if not viz_queue.full():
            viz_small, _ = fit_display(vis)
            viz_queue.put_nowait(viz_small)

        fps_counter += 1
        if fps_counter % 60 == 0:
            elapsed = time.time() - fps_start
            print(f"FPS: {fps_counter / elapsed:.1f}  "
                  f"pts: {[len(o['pts_obs']) if o['pts_obs'] is not None else 0 for o in objects.values()]}")
            fps_counter = 0
            fps_start   = time.time()

    cam.stop()
    cv2.destroyAllWindows()
    sock.close()


def shift_contour(c):
    """Shift a processed-space contour into full-frame coords for drawing."""
    if OX == 0 and OY == 0:
        return c
    return c + np.array([[OX, OY]], dtype=c.dtype)


def draw_roi_overlay(vis):
    """Draw the ROI polygon + crop box on a full-frame visualization."""
    if ROI_POLYGON:
        cv2.polylines(vis, [np.array(ROI_POLYGON, np.int32)], True, (0, 255, 0), 2)
    if CROP_TO_ROI and (PROC_W, PROC_H) != (FRAME_W_FULL, FRAME_H_FULL):
        cv2.rectangle(vis, (OX, OY), (OX + PROC_W, OY + PROC_H), (80, 80, 80), 1)


if __name__ == "__main__":
    main()
