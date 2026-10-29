"""Perception with ArUco markers (rulebook 6.1).

* the four wall markers (IDs 1-4) give each camera's pose with respect to the wall (its extrinsics);
* the robot's marker (ID 5) then gives the robot's pose in the wall frame.

Works on any RGB image: the simulator's renders or the real cameras' frames (pass their calibration: K and the
distortion coefficients).  Poses are in the wall frame (x right, y up, z out of the wall, origin at the bottom-left
corner); ``to_world`` converts them to the simulator's world frame.
"""
from __future__ import annotations

import cv2
import numpy as np
from climbing_sim import aruco
from climbing_sim.wall import WALL_HEIGHT, WALL_WIDTH, _R_WALL

ROBOT_ID = 5
WALL_IDS = (1, 2, 3, 4)
PLATE_Z = 0.001          # front face of the wall markers, out of the wall (m): the simulator's plates are 1 mm thick

_params = cv2.aruco.DetectorParameters()
_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
_detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), _params)


def detect(image_rgb) -> dict[int, np.ndarray]:
    """Marker ID -> its 4 corners in the image (4 x 2 pixels), in OpenCV's order: top-left, top-right,
    bottom-right, bottom-left of the code."""
    gray = cv2.cvtColor(np.ascontiguousarray(image_rgb), cv2.COLOR_RGB2GRAY)
    corners, ids, _ = _detector.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners)}


def wall_marker_corners(marker_id: int) -> np.ndarray:
    """The 4 corners of a wall marker's black code in the wall frame (4 x 3), in the order of :func:`detect`."""
    x0, y0 = aruco.corner_plate_origin(marker_id, WALL_WIDTH, WALL_HEIGHT) + aruco.BORDER
    c = aruco.CODE
    return np.array([[x0, y0 + c, PLATE_Z], [x0 + c, y0 + c, PLATE_Z], [x0 + c, y0, PLATE_Z], [x0, y0, PLATE_Z]])


def square_corners(size: float) -> np.ndarray:
    """The 4 corners of a square marker of side ``size`` in its own frame (x right, y up, z out of the marker)."""
    h = size / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])


def camera_pose(markers: dict, K, dist=None):
    """Camera extrinsics from the wall markers seen in its image: (rvec, tvec, rms) such that a wall point P is at
    R(rvec) P + tvec in the camera frame (OpenCV), rms = reprojection error (pixels).  None if no wall marker."""
    ids = [i for i in WALL_IDS if i in markers]
    if not ids:
        return None
    obj = np.concatenate([wall_marker_corners(i) for i in ids])
    img = np.concatenate([markers[i] for i in ids]).astype(np.float64)
    K = np.asarray(K, float)
    dist = np.zeros(5) if dist is None else np.asarray(dist, float)
    _, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
    return rvec.ravel(), tvec.ravel(), rms


def _rigid_fit(src, dst):
    """R, t minimising |R src + t - dst| (Kabsch)."""
    cs, cd = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - cs).T @ (dst - cd))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cd - R @ cs


def marker_pose(views: list, size: float):
    """Pose of a square marker of side ``size`` in the wall frame, from the cameras that see it.

    views: one dict per camera that sees it, {"corners": 4 x 2, "K", "dist", "rvec", "tvec"} (the camera's
    extrinsics from :func:`camera_pose`).  Two cameras or more: the corners are triangulated (accurate in depth);
    one camera: PnP on the marker alone.  Returns (R, t): the marker frame in the wall frame, or None."""
    obj = square_corners(size)
    if len(views) >= 2:
        P, pts = [], []
        for v in views[:2]:
            R, _ = cv2.Rodrigues(np.asarray(v["rvec"], float))
            P.append(np.asarray(v["K"], float) @ np.hstack([R, np.asarray(v["tvec"], float).reshape(3, 1)]))
            dist = np.zeros(5) if v.get("dist") is None else np.asarray(v["dist"], float)
            pts.append(cv2.undistortPoints(v["corners"].reshape(-1, 1, 2).astype(np.float64), v["K"], dist,
                                           P=v["K"]).reshape(-1, 2))
        X = cv2.triangulatePoints(P[0], P[1], pts[0].T, pts[1].T)
        return _rigid_fit(obj, (X[:3] / X[3]).T)
    if len(views) == 1:
        v = views[0]
        dist = np.zeros(5) if v.get("dist") is None else np.asarray(v["dist"], float)
        _, rvec, tvec = cv2.solvePnP(obj, v["corners"].astype(np.float64), v["K"], dist,
                                     flags=cv2.SOLVEPNP_IPPE_SQUARE)
        R_cm, _ = cv2.Rodrigues(rvec)
        R_cw, _ = cv2.Rodrigues(np.asarray(v["rvec"], float))
        t_cw = np.asarray(v["tvec"], float)
        return R_cw.T @ R_cm, R_cw.T @ (tvec.ravel() - t_cw)
    return None


def robot_pose(views: list, size: float, R_trunk_marker, p_trunk_marker):
    """Pose of the robot's trunk in the wall frame (R, p), from its marker ID 5 seen by the cameras (``views`` as
    in :func:`marker_pose`) and where the marker is on the trunk (its frame in the trunk frame).  None if unseen."""
    pose = marker_pose(views, size)
    if pose is None:
        return None
    R_wm, t_wm = pose
    R = R_wm @ np.asarray(R_trunk_marker, float).T
    return R, t_wm - R @ np.asarray(p_trunk_marker, float)


def to_world(R_wall, p_wall):
    """A pose in the wall frame -> the same pose in the simulator's world frame (x right, y into the wall, z up)."""
    return _R_WALL @ R_wall, _R_WALL @ np.asarray(p_wall, float)


def annotate(image_rgb, tracker: "Tracker", name: str, pose=None) -> np.ndarray:
    """BGR image (for cv2.imshow / imwrite) with the detected markers and the trunk pose (R, p, wall frame) drawn
    as its frame: x red (to the wall), y green (the robot's left), z blue (up)."""
    out = cv2.cvtColor(np.ascontiguousarray(image_rgb), cv2.COLOR_RGB2BGR)
    markers = detect(image_rgb)
    if markers:
        ids = list(markers)
        cv2.aruco.drawDetectedMarkers(out, [markers[i].reshape(1, 4, 2) for i in ids], np.array(ids).reshape(-1, 1))
    if pose is not None and name in tracker.extrinsics:
        rvec, tvec, _ = tracker.extrinsics[name]
        R_cw, _ = cv2.Rodrigues(rvec)
        R, p = pose
        dist = tracker.dist.get(name)
        cv2.drawFrameAxes(out, tracker.K[name], np.zeros(5) if dist is None else np.asarray(dist, float),
                          cv2.Rodrigues(R_cw @ R)[0], R_cw @ p + tvec, 0.08, 2)
    return out


class Tracker:
    """The whole pipeline on a set of cameras, image after image.

        tracker = Tracker({"cam_left": K_left, "cam_right": K_right}, marker_size, R_trunk_marker, p_trunk_marker)
        tracker.calibrate({"cam_left": img_left, "cam_right": img_right})   # camera extrinsics, once per attempt
        R, p = tracker.update({"cam_left": img_left, "cam_right": img_right})  # trunk pose, wall frame, or None
    """

    def __init__(self, intrinsics: dict, marker_size: float, R_trunk_marker, p_trunk_marker, dist: dict | None = None):
        self.K = {n: np.asarray(k, float) for n, k in intrinsics.items()}
        self.dist = dist or {}
        self.size, self.R_tm, self.p_tm = marker_size, R_trunk_marker, p_trunk_marker
        self.extrinsics = {}             # name -> (rvec, tvec, rms)

    def calibrate(self, images: dict) -> dict:
        """Each camera's pose from the wall markers; returns name -> (rvec, tvec, rms) for the cameras that saw some."""
        for name, img in images.items():
            pose = camera_pose(detect(img), self.K[name], self.dist.get(name))
            if pose is not None:
                self.extrinsics[name] = pose
        return self.extrinsics

    def update(self, images: dict):
        """The trunk's pose in the wall frame (R, p) from the robot's marker, or None when no camera sees it."""
        views = []
        for name, img in images.items():
            if name not in self.extrinsics:
                continue
            corners = detect(img).get(ROBOT_ID)
            if corners is not None:
                rvec, tvec, _ = self.extrinsics[name]
                views.append({"corners": corners, "K": self.K[name], "dist": self.dist.get(name),
                              "rvec": rvec, "tvec": tvec})
        return robot_pose(views, self.size, self.R_tm, self.p_tm)
