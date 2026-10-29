"""The robot's ArUco marker ID 5 (rulebook 5.2), as the organisers' sticker sheet gives it, and in the simulator.

The organisers' sheet ("ArUco marker ID 5 (DICT_4X4_50) for the robot's back"): a 58 x 84.5 mm sticker stuck flat on
the trunk's back (chest + top cover), a 42 mm black code; its top slot (6.19 x 13.18 mm) goes around the handle's
foot, which centres it.  Print the sheet at 100 %, check its ruler, cut along the outline.  In the trunk frame
(x forward, y to the robot's left, z up) the code's centre is at (-33.0, 0, 54.75) mm, on the back's surface.

    sim = MarkedSim("N1")            # ClimbingSim's arguments, with the sticker on the robot (visual only)
"""
from __future__ import annotations

import pathlib
import tempfile

import climbing_sim.sim as _sim_module
import cv2
import mujoco
import numpy as np
from climbing_sim import ClimbingSim, build_spec
from PIL import Image

MARKER_ID = 5
CODE = 0.042                            # black code side (m): the size the tracker needs (organisers' sheet)
#: the marker's frame in the trunk frame (trunk: x forward to the wall, y left, z up; marker: x right and y up as
#: seen from behind the robot, z out of the back): the columns are the marker's axes (organisers' sheet)
R_TRUNK_MARKER = np.array([[0.0, 0.0, -1.0],
                           [-1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]])
P_TRUNK_MARKER = np.array([-0.033, 0.0, 0.05475])     # centre of the code, trunk frame (m) (organisers' sheet)

# the sticker, for the simulator's picture of it
STICKER_W, STICKER_H = 0.058, 0.0845    # (m), organisers' sheet
SLOT_W, SLOT_H = 0.00619, 0.01318       # the slot at the top, around the handle's foot (m), organisers' sheet
CODE_FROM_BOTTOM = 0.037                # code's centre above the sticker's bottom edge (m): measured on the sheet
PROUD = 0.0003                          # the simulated sticker's face stands this far off the back (m): visible


def sticker_image(px_per_mm: int = 10) -> np.ndarray:
    """The sticker as printed (white, the code, the slot left grey), uint8 grey image, top row first."""
    w, h = round(STICKER_W * 1000 * px_per_mm), round(STICKER_H * 1000 * px_per_mm)
    img = np.full((h, w), 255, np.uint8)
    side = round(CODE * 1000 * px_per_mm)
    code = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), MARKER_ID, side)
    top = h - round(CODE_FROM_BOTTOM * 1000 * px_per_mm) - side // 2
    left = (w - side) // 2
    img[top:top + side, left:left + side] = code
    sw, sh = round(SLOT_W * 1000 * px_per_mm), round(SLOT_H * 1000 * px_per_mm)
    img[:sh, (w - sw) // 2:(w + sw) // 2] = 110
    return img


def add_robot_marker(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Add the sticker to the trunk of a scene's spec (``climbing_sim.build_spec``): visual only, no mass."""
    path = pathlib.Path(tempfile.gettempdir()) / "climbing_sim_textures" / f"aruco_robot_sticker_{MARKER_ID}.png"
    path.parent.mkdir(exist_ok=True)
    Image.fromarray(sticker_image()).convert("RGB").save(path)
    spec.add_texture(name="aruco_robot", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(path))
    spec.add_material(name="aruco_robot").textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "aruco_robot"
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R_TRUNK_MARKER.flatten())
    # a thin box whose +z face (the marker's z: out of the back) carries the texture (stretched over it), x right,
    # y up; the face is the sticker, the box goes 1 mm into the trunk
    x_m, y_m, z_m = R_TRUNK_MARKER.T
    face = P_TRUNK_MARKER + y_m * (STICKER_H / 2 - CODE_FROM_BOTTOM) + z_m * PROUD
    g = spec.body("trunk").add_geom(name="aruco_robot", type=mujoco.mjtGeom.mjGEOM_BOX, material="aruco_robot",
                                    size=[STICKER_W / 2, STICKER_H / 2, 0.0005], pos=(face - z_m * 0.0005).tolist(),
                                    quat=quat.tolist(), contype=0, conaffinity=0, group=1)
    g.density = 0.0
    return spec


class MarkedSim(ClimbingSim):
    """ClimbingSim with the robot's marker ID 5 on its back (same arguments)."""

    def __init__(self, *args, **kwargs):
        _sim_module.build_spec = lambda **kw: add_robot_marker(build_spec(**kw))
        try:
            super().__init__(*args, **kwargs)
        finally:
            _sim_module.build_spec = build_spec
