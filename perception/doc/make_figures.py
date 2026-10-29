"""The report's images, rendered by the simulator (perception/doc/img/).

    cd simulator && uv run --with opencv-contrib-python python ../perception/doc/make_figures.py

(real_holds.jpg and real_classes.jpg come from a photo of the real wall: not made here.)
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
from robot_marker import CODE, P_TRUNK_MARKER, R_TRUNK_MARKER, MarkedSim, sticker_image  # noqa: E402
from climbing_sim import default_cameras, world_to_wall      # noqa: E402

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


sim = MarkedSim("N2", cameras=default_cameras(1920, 1080))
obs = sim.reset_hanging()
tp = sim.trunk_pose()[0]

Image.fromarray(sim.render("overview", 1280, 960)).crop((300, 0, 980, 960)).save(IMG / "scene.png")
Image.fromarray(free_view(sim, 90, -8, 0.55, tp + [0, 0, 0.06])).save(IMG / "robot_back.png")
Image.fromarray(free_view(sim, 0, 0, 0.45, tp + [0, 0.0, 0.06])).save(IMG / "robot_side.png")
Image.fromarray(free_view(sim, 90, -5, 0.3, tp + [0, 0, 0.07])).save(IMG / "sticker_on_robot.png")
Image.fromarray(sticker_image()).save(IMG / "marker_id5.png")

names = list(sim.cameras)
tracker = Tracker({n: sim.cameras[n].K() for n in names}, CODE, R_TRUNK_MARKER, P_TRUNK_MARKER)
images = {n: sim.render(n) for n in names}
tracker.calibrate(images)
holds = tracker.find_holds(images)
pose = tracker.update(images, obs["gravity"])
for n in names:
    out = annotate(images[n], tracker, n, pose, holds)
    u, v = np.round(sim.cameras[n].project_wall(world_to_wall(tp).reshape(1, 3))[0]).astype(int)
    cv2.drawMarker(out, (int(u), int(v)), (255, 255, 255), cv2.MARKER_CROSS, 14, 2)
    cv2.imwrite(str(IMG / f"{n}_annotated.png"), cv2.resize(out, (1280, 720), interpolation=cv2.INTER_AREA))
    h, w = out.shape[:2]
    cw, ch = w // 6, h // 6
    x0, y0 = int(np.clip(u - cw // 2, 0, w - cw)), int(np.clip(v - ch * 2 // 3, 0, h - ch))
    cv2.imwrite(str(IMG / f"{n}_zoom.png"), cv2.resize(out[y0:y0 + ch, x0:x0 + cw], (2 * cw, 2 * ch)))
print("images in", IMG)
