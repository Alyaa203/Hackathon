"""The tracker on the real cameras (USB) or on image files: the script for the competition day.

    cd simulator
    uv run --with opencv-contrib-python python ../perception/live.py 0 1                  # two USB cameras
    uv run --with opencv-contrib-python python ../perception/live.py left.png right.png --once

Sources, left camera first: a camera index (0, 1, 2...: a RealSense D435i shows up as several video devices, try
them until the window shows the colour image) or an image file.  Only the colour (RGB) image is used, as the rulebook
asks.

Intrinsics: each camera's calibration from --cameras cams.json (the camera editor's file, ``climbing-sim-cameras``:
fx, fy, cx, cy, and optionally "dist", OpenCV's distortion coefficients), else the D435i's nominal values at the
image size.  The camera poses are not needed: they come from the four wall markers, at the start (key c: again).

The robot's marker: its size and where it is on the trunk are CODE, R_TRUNK_MARKER and P_TRUNK_MARKER in
robot_marker.py; measure them on your robot.

Window keys: c recalibrate the cameras, s save the annotated images to out/, q quit.
"""
import argparse
import json
import pathlib

import cv2
import numpy as np

from perception import ROBOT_ID, WALL_IDS, Tracker, annotate, detect
from robot_marker import CODE, P_TRUNK_MARKER, R_TRUNK_MARKER
from climbing_sim.cameras import D435I_HFOV, d435i_intrinsics

OUT = pathlib.Path(__file__).resolve().parent / "out"


def open_source(src: str):
    """A function returning the source's next RGB image (None when it fails)."""
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            raise SystemExit(f"camera {src} does not open")

        def read():
            ok, frame = cap.read()
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None
        return read
    img = cv2.imread(src)
    if img is None:
        raise SystemExit(f"cannot read the image {src}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return lambda: img


def intrinsics(name: str, image, cameras: dict):
    """(K, dist) of a camera: from the camera file when it has it, else the D435i's nominal values."""
    h, w = image.shape[:2]
    if name in cameras:
        c = cameras[name]
        if (c["width"], c["height"]) != (w, h):
            raise SystemExit(f"{name}: the camera file is for {c['width']} x {c['height']}, the images are {w} x {h}")
        return np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1.0]]), c.get("dist")
    try:
        c = d435i_intrinsics(w, h)
    except ValueError:              # not a D435i mode: the D435i's field of view, square pixels
        f = (w / 2) / np.tan(np.radians(D435I_HFOV / 2))
        c = dict(fx=f, fy=f, cx=w / 2, cy=h / 2)
    print(f"{name}: no calibration, nominal D435i intrinsics at {w} x {h} (less accurate)")
    return np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1.0]]), None


def calibrate(tracker, images) -> bool:
    """The cameras' poses from the wall markers; True when every camera has one."""
    tracker.extrinsics.clear()
    tracker.calibrate(images)
    for n, img in images.items():
        seen = sorted(i for i in detect(img) if i in WALL_IDS)
        if n in tracker.extrinsics:
            rvec, tvec, rms = tracker.extrinsics[n]
            R, _ = cv2.Rodrigues(rvec)
            p = -R.T @ tvec
            print(f"{n}: wall markers {seen}, reprojection {rms:.2f} px -> camera at x {p[0]:+.2f} m, "
                  f"{p[1]:.2f} m high, {p[2]:.2f} m from the wall")
        else:
            print(f"{n}: no wall marker seen")
    return len(tracker.extrinsics) == len(images)


args = argparse.ArgumentParser(description=__doc__.split("\n")[0])
args.add_argument("sources", nargs="+", help="camera index or image file, left camera first (1 or 2)")
args.add_argument("--cameras", help="camera file (camera editor's JSON) with each camera's calibration")
args.add_argument("--once", action="store_true", help="one image per camera, no window: print and save")
args = args.parse_args()

names = ["cam_left", "cam_right"][:len(args.sources)]
readers = {n: open_source(s) for n, s in zip(names, args.sources)}
cameras = {}
if args.cameras:
    cameras = {c["name"]: c for c in json.loads(pathlib.Path(args.cameras).read_text())["cameras"]}
images = {n: r() for n, r in readers.items()}
K, dist = {}, {}
for n in names:
    K[n], dist[n] = intrinsics(n, images[n], cameras)
tracker = Tracker(K, CODE, R_TRUNK_MARKER, P_TRUNK_MARKER, dist)
calibrated = calibrate(tracker, images)

while True:
    pose = tracker.update(images) if tracker.extrinsics else None
    seen_by = [n for n in names if ROBOT_ID in detect(images[n])]
    if not calibrated:
        status = "cameras not calibrated: show them the four wall markers, then press c"
    elif pose is None:
        status = "robot: marker not seen" if not seen_by else "robot: seen, but no calibrated camera sees it"
    else:
        p = pose[1] * 1000
        status = f"robot: x {p[0]:6.0f} mm, height {p[1]:6.0f} mm, {p[2]:5.0f} mm from the wall, seen by {seen_by}"
    shown = {n: annotate(images[n], tracker, n, pose) for n in names}
    if args.once:
        print(status)
        OUT.mkdir(exist_ok=True)
        for n, img in shown.items():
            cv2.imwrite(str(OUT / f"live_{n}.png"), img)
        print(f"annotated images: {OUT}")
        break
    view = np.hstack([cv2.resize(img, (640, 360)) for img in shown.values()])
    cv2.putText(view, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imshow("perception (c: recalibrate, s: save, q: quit)", view)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("c"):
        calibrated = calibrate(tracker, images)
    if key == ord("s"):
        OUT.mkdir(exist_ok=True)
        for n, img in shown.items():
            cv2.imwrite(str(OUT / f"live_{n}.png"), img)
    images = {n: r() for n, r in readers.items()}
    if any(img is None for img in images.values()):
        raise SystemExit("a camera stopped sending images")
cv2.destroyAllWindows()
