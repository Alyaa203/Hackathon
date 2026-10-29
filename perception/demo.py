"""Perception with ArUco markers on the simulator, checked against the ground truth.

    cd simulator && uv run --with opencv-contrib-python python ../perception/demo.py [N1|N2|N3] [--shuffle SEED]

1. the two cameras' poses from the four wall markers (compared with the cameras the simulator was built with);
2. the robot's trunk tracked with its marker ID 5 while it sways on the wall (compared with the simulated trunk).
Annotated camera images go to perception/out/.

--shuffle SEED places the two cameras somewhere else in front of the wall (each still seeing the four corner
markers), as the organisers may place them: the code only knows their intrinsics, never where they are.
"""
import argparse
import pathlib

import cv2
import mujoco
import numpy as np

from perception import ROBOT_ID, Tracker, annotate, detect, to_world
from robot_marker import CODE, P_TRUNK_MARKER, R_TRUNK_MARKER, MarkedSim
from climbing_sim import LIMBS
from climbing_sim.cameras import Camera, d435i_intrinsics, marker_visibility

OUT = pathlib.Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)


def angle_deg(Ra, Rb) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def quat_to_mat(q) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def random_cameras(seed: int) -> list[Camera]:
    """Two D435i at random in front of the wall (wall frame: x right, y up, z out), one on each side, 1.9-3 m away,
    0.5-1.4 m high, aimed near the wall's centre; drawn again until it sees the four corner markers."""
    rng = np.random.default_rng(seed)
    cams = []
    for name, side in (("cam_left", -1), ("cam_right", 1)):
        while True:
            pos = [0.4 + side * rng.uniform(0.2, 0.9), rng.uniform(0.5, 1.4), rng.uniform(1.9, 3.0)]
            target = [0.4 + rng.uniform(-0.1, 0.1), 0.7 + rng.uniform(-0.15, 0.15), 0.0]
            cam = Camera.looking_at(name, pos, target, **d435i_intrinsics(1280, 720))
            if all(marker_visibility(cam, margin_px=10).values()):
                cams.append(cam)
                break
    return cams


args = argparse.ArgumentParser()
args.add_argument("wall", nargs="?", default="N1")
args.add_argument("--shuffle", type=int, metavar="SEED", help="cameras at random places (seed)")
args = args.parse_args()
wall = args.wall
sim = MarkedSim(wall, cameras=None if args.shuffle is None else random_cameras(args.shuffle))
sim.reset_hanging()
names = list(sim.cameras)
for n in names:
    p = sim.cameras[n].position_wall()
    print(f"{n} at x {p[0]:+.2f} m, {p[1]:.2f} m high, {p[2]:.2f} m from the wall")
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

# ---- 2. the robot tracked with its marker, while it sways -------------------------------------------------
print("trunk pose from marker ID 5 (ground truth: the simulated trunk)")
tp0, tq0 = sim.trunk_pose()
balls = {l: sim.wall.start[l].socket for l in LIMBS}
q = sim.joint_positions()
errs, steps = [], 150
for k in range(steps + 1):
    s = np.sin(2 * np.pi * k / steps)
    q, _ = sim.joint_angles_for(balls, tp0 + [0.03 * s, 0.0, 0.015 * s], tq0, q_start=q)
    sim.step(q)
    if k % 15:
        continue
    images = {n: sim.render(n) for n in names}
    pose = tracker.update(images)
    seen_by = [n for n in names if ROBOT_ID in detect(images[n])]
    p_true, q_true = sim.trunk_pose()
    if pose is None:
        print(f"  t={sim.data.time:5.2f} s: marker not seen")
        continue
    R, p = to_world(*pose)
    e_pos, e_ang = np.linalg.norm(p - p_true) * 1000, angle_deg(R, quat_to_mat(q_true))
    errs.append((e_pos, e_ang))
    print(f"  t={sim.data.time:5.2f} s: trunk at {np.round(p * 1000).astype(int)} mm (true "
          f"{np.round(p_true * 1000).astype(int)}), error {e_pos:5.1f} mm {e_ang:4.2f} deg, seen by {seen_by}")
    if k == steps // 2:
        for n in names:
            cv2.imwrite(str(OUT / f"{n}.png"), annotate(images[n], tracker, n, pose))
            cv2.imwrite(str(OUT / f"{n}_raw.png"), cv2.cvtColor(images[n], cv2.COLOR_RGB2BGR))
if errs:
    e = np.array(errs)
    print(f"mean error {e[:, 0].mean():.1f} mm (max {e[:, 0].max():.1f}), {e[:, 1].mean():.2f} deg "
          f"(max {e[:, 1].max():.2f}) over {len(e)} frames")
print(f"annotated images: {OUT}")
