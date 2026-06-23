# Rock & Water Tracker

Real-time computer-vision tracker for an interactive art installation.

## What it does

A long table (~6 m × 1.65 m) holds large real rocks. Laser projectors cast a moving water animation onto the table. This program watches the table with a camera, detects each rock's position, silhouette and rotation in real time, and streams that pose over UDP so a real-time engine (TouchDesigner / Unreal Engine / a web app) can make the projected digital water react to the real rocks — flowing around them and rippling at their edges.

> Status: R&D prototype, under active development.

## How it works

1. Capture a clean background of the empty table.

2. Background subtraction -> foreground blobs (the rocks).

3. Feature points + Lucas-Kanade optical flow track each rock frame to frame.

4. Procrustes (Kabsch) fit recovers centroid + rotation; Hu-moment matching re-acquires a lost rock.

5. Pack and send a UDP packet per frame to 127.0.0.1:5005.

### UDP packet format

Little-endian. One header byte = number of active objects, then per object: id (uint16), x (float32, 0–1), y (float32, 0–1), rotation (float32, radians). Struct: header <B, per object <H f f f. Coordinates are normalized over the full camera frame.

## Files

- tracker_procrustes.py — v1, original tracker (camera index 0, full-frame processing).

- tracker_procrustes_v2.py — v2. Adds, without changing the tracking algorithm or the UDP format:

  - Camera selection at startup (live preview).

  - Mouse-drawn ROI: draw a polygon over the area to watch; everything outside is ignored and the frame is cropped to it. Saved to roi_config.json.

  - Performance: forces 1080p via MJPG + DSHOW (a 4K wide-angle webcam otherwise streams 4K and lags); downscaled preview.

## Requirements

Python 3.9+. Then:

```
pip install -r requirements.txt

python tracker_procrustes_v2.py
```

## Controls

- Camera picker: NP switch · 09 pick by index · ENTER select · Q quit

- ROI editor: left-click add point · right-click / Z undo · C clear · R reload saved · F full frame · ENTER confirm · Q quit

- Tracker: SPACE calibrate · Q quit

## Roadmap / known issues

The core open problem is sensing the rocks while a bright dynamic projection is on the same surface — plain RGB background subtraction breaks under the projected water. Planned directions (see REFERENCES.md):

- IR sensing (IR illumination + IR-pass filter) or a depth camera (projection-immune).

- Projector↔camera calibration so the water aligns to the real rock edges.

- Multi-camera coverage for the full 6 m table, merged in table coordinates.

## License

Proprietary — © Six N. Five. All rights reserved.
