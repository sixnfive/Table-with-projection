import cv2
import numpy as np
import socket
import struct
import time
import threading
import queue

class CameraThread:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        print(f"Camera FPS: {self.cap.get(cv2.CAP_PROP_FPS)}")
        print(f"Camera resolution: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
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

# UDP
UDP_IP   = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

DIFF_THRESHOLD      = 30
MIN_AREA            = 500
MAX_AREA            = 1920 * 1080 * 0.8
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

# ── Core helpers ──────────────────────────────────────────────────────────────

def get_fg_mask(frame, background):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, background)
    _, fg = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
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

def get_contours(frame, background):
    fg = get_fg_mask(frame, background)
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
        data += struct.pack('<H f f f',
            oid % 65535,
            o['cx'] / 1920,
            o['cy'] / 1080,
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
                import os; os._exit(0)
        except queue.Empty:
            continue

threading.Thread(target=viz_thread_fn, daemon=True).start()

# ── Startup ───────────────────────────────────────────────────────────────────

cam = CameraThread(0)
time.sleep(1)

print("Capturing background... keep frame clear for 3 seconds")
buf, t0 = [], time.time()
while time.time() - t0 < 3.0:
    f = cam.read()
    if f is not None:
        buf.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
background = np.median(np.stack(buf), axis=0).astype(np.uint8)
print("Background captured! Place objects then press SPACE.")

# ── Calibration ───────────────────────────────────────────────────────────────

calibrated = False
while not calibrated:
    frame = cam.read()
    if frame is None:
        continue

    if space_pressed:
        space_pressed = False
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray  = cv2.GaussianBlur(gray, (5, 5), 0)
        valid = get_contours(frame, background)

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
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        valid = get_contours(frame, background)
        vis   = frame.copy()
        for c, cx, cy, area in valid:
            cv2.drawContours(vis, [c], -1, (0, 255, 255), 2)
            cv2.putText(vis, f"{int(area)}", (int(cx), int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(vis, f"Objects: {len(valid)} | SPACE to calibrate",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if not viz_queue.full():
            viz_queue.put_nowait(vis)

# ── Main loop ─────────────────────────────────────────────────────────────────

fps_counter = 0
fps_start   = time.time()

while True:
    frame = cam.read()
    if frame is None:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    if prev_gray is None:
        prev_gray = gray.copy()
        continue

    all_contours = get_contours(frame, background)

    for oid, obj in objects.items():

        obs = obj['pts_obs']
        ref = obj['pts_ref']

        # ── no points ─────────────────────────────────────────────────────────
        if obs is None or len(obs) < MIN_POINTS_PROCRUST:
            obj['lost_frames'] += 1
            obj['lost'] = obj['lost_frames'] > MAX_LOST
            attempt_reacquire(gray, all_contours, obj, oid)
            continue

        # ── optical flow ──────────────────────────────────────────────────────
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
            # ── Procrustes ────────────────────────────────────────────────────
            new_cx, new_cy, new_angle = procrustes(surv_ref, surv_obs)
            new_angle = fix_angle(new_angle, obj['rotation'])

            alpha           = 0.4
            obj['cx']       = alpha * new_cx    + (1-alpha) * obj['cx']
            obj['cy']       = alpha * new_cy    + (1-alpha) * obj['cy']
            obj['rotation'] = new_angle

            obj['pts_obs']     = surv_obs
            obj['pts_ref']     = surv_ref
            obj['lost']        = False
            obj['lost_frames'] = max(0, obj['lost_frames'] - 1)

            # topup only if clean contour available
            if n < MIN_POINTS_HEALTHY:
                topup(gray, all_contours, obj)

        else:
            # ── too few — reacquire ───────────────────────────────────────────
            obj['lost_frames'] += 1
            obj['lost']   = obj['lost_frames'] > MAX_LOST
            obj['pts_obs'] = surv_obs if n > 0 else None
            obj['pts_ref'] = surv_ref if n > 0 else None
            attempt_reacquire(gray, all_contours, obj, oid)

    prev_gray = gray.copy()

    sock.sendto(pack_objects(objects), (UDP_IP, UDP_PORT))

    # ── Visualize ─────────────────────────────────────────────────────────────
    vis = frame.copy()
    for oid, obj in objects.items():
        cx    = int(obj['cx'])
        cy    = int(obj['cy'])
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
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 3, color, -1)

        cv2.circle(vis, (cx, cy), 6, color, -1)
        dx = int(np.cos(angle) * 60)
        dy = int(np.sin(angle) * 60)
        cv2.arrowedLine(vis, (cx-dx, cy-dy), (cx+dx, cy+dy),
                        (0, 165, 255), 2, tipLength=0.3)
        cv2.putText(vis, f"{oid} {state} {np.degrees(angle):.0f}° p:{n_pts}",
                    (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if not viz_queue.full():
        viz_queue.put_nowait(vis)

    fps_counter += 1
    if fps_counter % 60 == 0:
        elapsed = time.time() - fps_start
        print(f"FPS: {fps_counter/elapsed:.1f}  "
              f"pts: {[len(o['pts_obs']) if o['pts_obs'] is not None else 0 for o in objects.values()]}")
        fps_counter = 0
        fps_start   = time.time()

cam.stop()
cv2.destroyAllWindows()
sock.close()