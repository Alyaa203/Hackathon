"""The robot's ArUco marker ID 5 in the simulator (rulebook 5.2: the team picks where it goes and its size).

A 100 x 100 mm plate (80 mm black code, 10 mm white border) on the trunk's back, the side the cameras see while
the robot climbs.  The trunk's back is flat at 33 mm behind the trunk origin (a handle sticks out to 44 mm): the
plate stands 48 mm behind it.  Visual only: no collision, no mass.

    sim = MarkedSim("N1")            # ClimbingSim's arguments, with the marker on the robot
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
CODE = 0.080                            # black code side (m): the size the tracker needs
BORDER = 0.010                          # white border (m)
PLATE = CODE + 2 * BORDER
THICKNESS = 0.001
#: the marker's frame in the trunk frame (trunk: x forward to the wall, y left, z up; marker: x right and y up as
#: seen from behind the robot, z out of the plate): the columns are the marker's axes
R_TRUNK_MARKER = np.array([[0.0, 0.0, -1.0],
                           [-1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]])
P_TRUNK_MARKER = np.array([-0.048, 0.0, 0.050])       # centre of the code on the plate's face, trunk frame (m)


def marker_image(px_per_cell: int = 64) -> np.ndarray:
    """The marker with its white border, uint8 grey image, top row first (print it at PLATE x PLATE for the robot)."""
    code = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), MARKER_ID,
                                         6 * px_per_cell)
    return np.pad(code, int(round(6 * px_per_cell * BORDER / CODE)), constant_values=255)


def add_robot_marker(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Add the marker plate to the trunk of a scene's spec (``climbing_sim.build_spec``)."""
    path = pathlib.Path(tempfile.gettempdir()) / "climbing_sim_textures" / f"aruco_robot_{MARKER_ID}.png"
    path.parent.mkdir(exist_ok=True)
    Image.fromarray(marker_image()).convert("RGB").save(path)
    spec.add_texture(name="aruco_robot", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(path))
    spec.add_material(name="aruco_robot").textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "aruco_robot"
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R_TRUNK_MARKER.flatten())
    # a thin box whose +z face (the marker's z: out of the plate) carries the texture, x right, y up
    g = spec.body("trunk").add_geom(name="aruco_robot", type=mujoco.mjtGeom.mjGEOM_BOX, material="aruco_robot",
                                    size=[PLATE / 2, PLATE / 2, THICKNESS / 2],
                                    pos=(P_TRUNK_MARKER - R_TRUNK_MARKER[:, 2] * THICKNESS / 2).tolist(),
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
