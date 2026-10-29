"""The report's images, rendered by the simulator (perception/doc/img/).

    cd simulator && uv run --with opencv-contrib-python python ../perception/doc/make_figures.py
"""
import pathlib
import sys

import cv2
import mujoco
import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from perception import Tracker, annotate                     # noqa: E402
from robot_marker import CODE, P_TRUNK_MARKER, R_TRUNK_MARKER, MarkedSim, marker_image  # noqa: E402

IMG = HERE / "img"
IMG.mkdir(exist_ok=True)


def free_view(sim, azimuth, elevation, distance, lookat, w=900, h=900):
    r = mujoco.Renderer(sim.model, height=h, width=w)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance
    r.update_scene(sim.data, camera=cam)
    img = r.render()
    r.close()
    return img


sim = MarkedSim("N1")
sim.reset_hanging()
tp = sim.trunk_pose()[0]

Image.fromarray(sim.render("overview", 1280, 960)).crop((300, 0, 980, 960)).save(IMG / "scene.png")
Image.fromarray(free_view(sim, 90, -8, 0.55, tp + [0, 0, 0.06])).save(IMG / "robot_back.png")
Image.fromarray(free_view(sim, 0, 0, 0.45, tp + [0, 0.0, 0.06])).save(IMG / "robot_side.png")
Image.fromarray(marker_image(32)).save(IMG / "marker_id5.png")

names = list(sim.cameras)
tracker = Tracker({n: sim.cameras[n].K() for n in names}, CODE, R_TRUNK_MARKER, P_TRUNK_MARKER)
images = {n: sim.render(n) for n in names}
tracker.calibrate(images)
pose = tracker.update(images)
for n in names:
    out = annotate(images[n], tracker, n, pose)
    cv2.imwrite(str(IMG / f"{n}_annotated.png"), out)
    u = int(np.mean([c[0] for c in sim.cameras[n].project_wall([[tp[0], tp[2], 0.1]])]))
    v = int(np.mean([c[1] for c in sim.cameras[n].project_wall([[tp[0], tp[2] + 0.05, 0.1]])]))
    cv2.imwrite(str(IMG / f"{n}_zoom.png"), out[max(0, v - 120):v + 90, max(0, u - 110):u + 110])
print("images in", IMG)
