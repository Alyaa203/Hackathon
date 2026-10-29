"""Perception with ArUco markers on the simulator, checked against the ground truth, and shown.

    cd simulator && uv run --with opencv-contrib-python python ../perception/demo.py [N1|N2|N3] [options]

1. the two cameras' poses from the four wall markers (compared with the cameras the simulator was built with);
2. the holds, from their colour (compared with the wall's holds): written to perception/out/holds.json, a wall file
   of the simulator;
3. the robot's trunk tracked with its marker ID 5 while it sways on the wall, from the marker alone and with the
   IMU's tilt (compared with the simulated trunk); only when both cameras see the marker.

Options:
  --shuffle SEED   the two cameras somewhere else in front of the wall (each still seeing the four corner markers),
                   as the organisers may place them: the code only knows their intrinsics, never where they are
  --res WxH        the cameras' image size, a D435i colour mode (default 1920x1080; the simulator's own: 1280x720)
  --show           a window with the cameras' views while it runs (a key at the end closes it)

The video of the run goes to perception/out/demo.mp4: both cameras (top) and zoomed on the robot (bottom), with
the markers found (green), the holds found (magenta), the trunk as the cameras see it (its frame: x red, y green,
z blue) and where it truly is (white cross).  Annotated and raw camera images go to perception/out/ too.
"""
import argparse
import pathlib

import cv2
import mujoco
import numpy as np

from perception import Tracker, annotate, detect, level_with_gravity, to_world
from robot_marker import CODE, P_TRUNK_MARKER, R_TRUNK_MARKER, MarkedSim
from climbing_sim import LIMBS, Wall, world_to_wall
from climbing_sim.cameras import Camera, d435i_intrinsics, default_cameras, marker_visibility

OUT = pathlib.Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)


def angle_deg(Ra, Rb) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def quat_to_mat(q) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def random_cameras(seed: int, width: int, height: int) -> list[Camera]:
    """Two D435i at random in front of the wall (wall frame: x right, y up, z out), one on each side, 1.9-3 m away,
    0.5-1.4 m high, aimed near the wall's centre; drawn again until it sees the four corner markers."""
    rng = np.random.default_rng(seed)
    cams = []
    for name, side in (("cam_left", -1), ("cam_right", 1)):
        while True:
            pos = [0.4 + side * rng.uniform(0.2, 0.9), rng.uniform(0.5, 1.4), rng.uniform(1.9, 3.0)]
            target = [0.4 + rng.uniform(-0.1, 0.1), 0.7 + rng.uniform(-0.15, 0.15), 0.0]
            cam = Camera.looking_at(name, pos, target, **d435i_intrinsics(width, height))
            if all(marker_visibility(cam, margin_px=10).values()):
                cams.append(cam)
                break
    return cams


def frame(images, pose, p_true, text) -> np.ndarray:
    """One video frame, 1280 x 720: the two cameras (top) and zoomed on the robot (bottom)."""
    top, bottom = [], []
    for n in names:
        img = annotate(images[n], tracker, n, pose, holds)
        u, v = np.round(sim.cameras[n].project_wall(world_to_wall(p_true).reshape(1, 3))[0]).astype(int)
        cv2.drawMarker(img, (int(u), int(v)), (255, 255, 255), cv2.MARKER_CROSS, 14, 2)
        h, w = img.shape[:2]
        cw, ch = w // 6, h // 6                          # zoom: a sixth of the image, around the trunk
        x0 = int(np.clip(u - cw // 2, 0, w - cw))
        y0 = int(np.clip(v - ch * 2 // 3, 0, h - ch))
        top.append(cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA))
        bottom.append(cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (640, 360), interpolation=cv2.INTER_LINEAR))
    out = np.vstack([np.hstack(top), np.hstack(bottom)])
    cv2.rectangle(out, (0, 0), (1280, 26), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


args = argparse.ArgumentParser()
args.add_argument("wall", nargs="?", default="N1")
args.add_argument("--shuffle", type=int, metavar="SEED", help="cameras at random places (seed)")
args.add_argument("--res", default="1920x1080", help="cameras' image size (default 1920x1080)")
args.add_argument("--show", action="store_true", help="a window with the cameras' views while it runs")
args = args.parse_args()
wall = args.wall
width, height = (int(v) for v in args.res.lower().split("x"))
cams = default_cameras(width, height) if args.shuffle is None else random_cameras(args.shuffle, width, height)
sim = MarkedSim(wall, cameras=cams)
obs = sim.reset_hanging()
names = list(sim.cameras)
for n in names:
    p = sim.cameras[n].position_wall()
    print(f"{n} at x {p[0]:+.2f} m, {p[1]:.2f} m high, {p[2]:.2f} m from the wall, {width} x {height} pixels")
tracker = Tracker({n: sim.cameras[n].K() for n in names}, CODE, R_TRUNK_MARKER, P_TRUNK_MARKER)

# ---- 1. camera extrinsics from the wall markers ------------------------------------------------------------
images = {n: sim.render(n) for n in names}
tracker.calibrate(images)
print(f"wall {wall}: camera poses from the wall markers (ground truth: the simulator's cameras)")
for n in names:
    seen = sorted(detect(images[n]))
    if n not in tracker.extrinsics:
        print(f"  {n}: no wall marker seen")
        continue
    rvec, tvec, rms = tracker.extrinsics[n]
    R, _ = cv2.Rodrigues(rvec)
    cam = sim.cameras[n]
    pos_err = np.linalg.norm(-R.T @ tvec - cam.position_wall()) * 1000
    print(f"  {n}: markers {seen}, reprojection {rms:.2f} px, position error {pos_err:.1f} mm, "
          f"angle error {angle_deg(R, cam.R_wall_to_cam()):.3f} deg")

# ---- 2. the holds, from their colour --------------------------------------------------------------------------
holds = tracker.find_holds(images)
truth = {(h.col, h.row) for h in sim.wall.holds}
got = {(f.hold.col, f.hold.row) for f in holds}
print(f"holds from their colour (ground truth: the {len(truth)} holds of wall {wall}): found {len(holds)}, correct "
      f"{len(got & truth)}, false {len(got - truth)}, missed {len(truth - got)} (hidden behind the robot)")
for f in holds[:4] + [None] + holds[-2:]:
    if f is None:
        print("  ...")
        continue
    p, pw = np.round(f.hold.socket * 1000).astype(int), np.round(f.socket_wall * 1000).astype(int)
    print(f"  col {f.hold.col:2d} row {f.hold.row:2d} {f.color:7s} socket: world {p} mm = on the wall x {pw[0]} mm, "
          f"height {pw[1]} mm  ({'ok' if (f.hold.col, f.hold.row) in truth else 'FALSE'})")
Wall(name=f"found_{wall}", holds=[f.hold for f in holds],
     description="holds found by perception/demo.py").to_json(OUT / "holds.json")
print(f"  all of them: {OUT / 'holds.json'}")

# ---- 3. the robot tracked with its marker, while it sways -------------------------------------------------
print("trunk pose from marker ID 5, with the IMU's tilt (ground truth: the simulated trunk)")
video = cv2.VideoWriter(str(OUT / "demo.mp4"), cv2.VideoWriter_fourcc(*"avc1"), 1 / sim.control_dt, (1280, 720))
if not video.isOpened():
    video = cv2.VideoWriter(str(OUT / "demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 1 / sim.control_dt, (1280, 720))
tp0, tq0 = sim.trunk_pose()
balls = {l: sim.wall.start[l].socket for l in LIMBS}
q = sim.joint_positions()
errs = {"marker": [], "imu": []}
one_camera, steps = 0, 150
for k in range(steps + 1):
    s = np.sin(2 * np.pi * k / steps)
    q, _ = sim.joint_angles_for(balls, tp0 + [0.03 * s, 0.0, 0.015 * s], tq0, q_start=q)
    obs = sim.step(q)
    images = {n: sim.render(n) for n in names}
    pose_marker = tracker.update(images)                       # the marker alone
    p_true, q_true = sim.trunk_pose()
    if pose_marker is None:
        one_camera += 1
        pose = None
        text = f"{wall}  t={sim.data.time:4.2f} s   robot marker seen by {tracker.seen_by or 'no camera'}: not used"
    else:
        pose = level_with_gravity(pose_marker, obs["gravity"], P_TRUNK_MARKER)     # + the IMU's tilt
        for key, ps in (("marker", pose_marker), ("imu", pose)):
            R, p = to_world(*ps)
            errs[key].append((np.linalg.norm(p - p_true) * 1000, angle_deg(R, quat_to_mat(q_true))))
        e, e_m = errs["imu"][-1], errs["marker"][-1]
        text = (f"{wall}  t={sim.data.time:4.2f} s   error {e[0]:4.1f} mm {e[1]:3.1f} deg   "
                f"(marker alone: {e_m[0]:4.1f} mm {e_m[1]:3.1f} deg)   white cross: true trunk")
    shown = frame(images, pose, p_true, text)
    video.write(shown)
    if args.show:
        cv2.imshow("demo: cameras (top), zoomed on the robot (bottom)", shown)
        cv2.waitKey(1)
    if k % 15 == 0:
        if pose is None:
            print(f"  t={sim.data.time:5.2f} s: marker seen by {tracker.seen_by or 'no camera'}: not used "
                  f"(both cameras needed)")
        else:
            R, p = to_world(*pose)
            print(f"  t={sim.data.time:5.2f} s: trunk at {np.round(p * 1000).astype(int)} mm (true "
                  f"{np.round(p_true * 1000).astype(int)}), error {e[0]:4.1f} mm {e[1]:4.2f} deg "
                  f"(marker alone: {e_m[0]:4.1f} mm)")
    if k == steps // 2:
        for n in names:
            cv2.imwrite(str(OUT / f"{n}.png"), annotate(images[n], tracker, n, pose, holds))
            cv2.imwrite(str(OUT / f"{n}_raw.png"), cv2.cvtColor(images[n], cv2.COLOR_RGB2BGR))
video.release()
for key, name in (("marker", "marker alone  "), ("imu", "marker + IMU  ")):
    if errs[key]:
        e = np.array(errs[key])
        print(f"{name}: mean error {e[:, 0].mean():.1f} mm (max {e[:, 0].max():.1f}), {e[:, 1].mean():.2f} deg "
              f"(max {e[:, 1].max():.2f}) over {len(e)} images")
print(f"{one_camera} of {steps + 1} images not used: the robot's marker seen by one camera only")
print(f"video: {OUT / 'demo.mp4'}   annotated images: {OUT}")
if args.show:
    print("press a key in the window to close it")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
