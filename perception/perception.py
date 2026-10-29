"""Perception with ArUco markers (rulebook 6.1).

* the four wall markers (IDs 1-4) give each camera's pose with respect to the wall (its extrinsics);
* the robot's marker (ID 5) then gives the robot's pose in the wall frame;
* the holds are found on the wall's grid of holes, from their colour in the images.

Works on any RGB image: the simulator's renders or the real cameras' frames (pass their calibration: K and the
distortion coefficients).  Poses are in the wall frame (x right, y up, z out of the wall, origin at the bottom-left
corner); ``to_world`` converts them to the simulator's world frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
from climbing_sim import aruco
from climbing_sim.scene import HOLD_VISUAL
from climbing_sim.wall import GRID_COLS, GRID_ROWS, OVERLAP, WALL_HEIGHT, WALL_WIDTH, Hold, _R_WALL, world_to_wall

ROBOT_ID = 5
WALL_IDS = (1, 2, 3, 4)
PLATE_Z = 0.001          # front face of the wall markers, out of the wall (m): the simulator's plates are 1 mm thick

_params = cv2.aruco.DetectorParameters()
_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
_params.perspectiveRemovePixelPerCell = 8      # read each cell on 8 x 8 pixels (default 4): the robot's 42 mm code,
                                               # about 20 px wide in the images, is then decoded every time
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


def level_with_gravity(pose, gravity, p_trunk_marker):
    """The trunk pose (R, p, wall frame) from the marker, its tilt corrected by the IMU: ``gravity`` is the direction
    of gravity in the trunk frame (the robot's observation "gravity").  The marker's orientation is turned the least
    that makes it agree with gravity: its heading (the turn about the vertical, which the IMU cannot know) stays.
    The marker's centre stays where it is (well measured); the trunk's origin, 6 cm from it, follows."""
    R, p = pose
    centre = p + R @ np.asarray(p_trunk_marker, float)
    v = R @ (np.asarray(gravity, float) / np.linalg.norm(gravity))     # gravity, as the marker's orientation puts it
    down = np.array([0.0, -1.0, 0.0])                                   # gravity in the wall frame
    axis = np.cross(v, down)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return R, p
    Q, _ = cv2.Rodrigues(axis / s * np.arctan2(s, float(v @ down)))
    R = Q @ R
    return R, centre - R @ np.asarray(p_trunk_marker, float)


def to_world(R_wall, p_wall):
    """A pose in the wall frame -> the same pose in the simulator's world frame (x right, y into the wall, z up)."""
    return _R_WALL @ R_wall, _R_WALL @ np.asarray(p_wall, float)


# ---- holds ------------------------------------------------------------------------------------------------
# Each pixel is a hold's colour (vivid, any hue: the holds are green, orange, blue...), the wall's (pale: plywood) or
# unknown (grey, black, white: the robot, the markers, deep shadows; it may hide a hold).  Each place a hold can be
# bolted (the grid of holes) is tried: the hold's 3D shape is projected there into each image, and its score is the
# fraction of its visible (not unknown) points that land on a hold's colour.
HOLD_SAT_MIN = 120          # saturation (0-255) from which a pixel is a hold's colour (plywood: 15-40 real, 67 sim)
GREY_SAT_MAX = 10           # saturation below which a pixel is grey or white (robot, markers): unknown
DARK_VAL_MAX = 50           # brightness (0-255) below which a pixel is too dark to tell: unknown
HOLD_SCORE_MIN = 0.8        # fraction of a hold's visible points that must be a hold's colour
HOLD_SEEN_MIN = 0.4         # fraction of a hold's points that must be visible (not hidden by the robot...)
PAIR_SCORE_MIN = 0.7        # two touching holds, seen as one blob: the score each of the two must reach
HOLD_POINTS = 200           # points sampled on the hold's surface
UNKNOWN, WALL, HOLD = 0, 1, 2


@dataclass
class FoundHold:
    """A hold found in the images: ``hold.socket`` is where a ball sits in it (world frame), ``socket_wall`` the same
    point in the wall frame."""
    hold: Hold
    score: float            # fraction of its visible points seen in a hold's colour, all cameras (0-1)
    seen: float             # fraction of its points visible (not hidden by the robot), all cameras (0-1)
    color: str              # its colour's name, from the mean hue

    @property
    def socket_wall(self) -> np.ndarray:
        return world_to_wall(self.hold.socket)


def pixel_classes(image_rgb) -> np.ndarray:
    """Image of UNKNOWN (grey, black, white, too dark), WALL (pale colour) or HOLD (vivid colour) per pixel."""
    hsv = cv2.cvtColor(np.ascontiguousarray(image_rgb), cv2.COLOR_RGB2HSV)
    sat, val = hsv[..., 1], hsv[..., 2]
    out = np.full(sat.shape, WALL, np.uint8)
    out[(sat < GREY_SAT_MAX) | (val < DARK_VAL_MAX)] = UNKNOWN
    out[(sat >= HOLD_SAT_MIN) & (val >= DARK_VAL_MAX)] = HOLD
    return out


@lru_cache
def hold_points() -> np.ndarray:
    """HOLD_POINTS points spread evenly over the hold's surface, around its socket (m, world axes), from the
    simulator's mesh of the real hold.  Hidden points (behind the cup) still land on the hold in an image."""
    raw = HOLD_VISUAL.read_bytes()                       # binary STL: 80-byte header, count, 50-byte triangles
    count = int(np.frombuffer(raw, "<u4", 1, 80)[0])
    tri = np.frombuffer(raw, np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count, 84)["v"]
    tri = tri.astype(float)
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    pick = np.random.default_rng(0).choice(count, HOLD_POINTS, replace=False, p=area / area.sum())
    return tri[pick].mean(axis=1)


@lru_cache
def hold_candidates() -> tuple[list[Hold], np.ndarray]:
    """Every place a hold can be bolted (left hole (col, row), off the corner markers) and its socket (world)."""
    holds = []
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS - 2):
            h = Hold(col, row)
            near_corner = (col < 12 or col > GRID_COLS - 14) and (row < 12 or row > GRID_ROWS - 13)
            if not (near_corner and h.on_markers()):
                holds.append(h)
    return holds, np.array([h.socket for h in holds])


def hold_pixels(K, dist, rvec, tvec, shape):
    """Where the points of a hold at each candidate place (:func:`hold_candidates`) land in this camera's images of
    size ``shape`` (h, w): (flat pixel indices, inside the image), both (places, HOLD_POINTS).  Only the places the
    camera sees whole count.  Depends on the camera's pose only: computed once per calibration."""
    _, sockets = hold_candidates()
    pts = world_to_wall(sockets[:, None, :] + hold_points()[None])
    uv, _ = cv2.projectPoints(pts.reshape(-1, 3), np.asarray(rvec, float), np.asarray(tvec, float),
                              np.asarray(K, float), np.zeros(5) if dist is None else np.asarray(dist, float))
    uv = np.round(uv.reshape(len(sockets), -1, 2)).astype(np.int64)
    h, w = shape
    inside = (uv[..., 0] >= 0) & (uv[..., 0] < w) & (uv[..., 1] >= 0) & (uv[..., 1] < h)
    inside &= inside.all(axis=1, keepdims=True)
    return np.where(inside, uv[..., 1] * w + uv[..., 0], 0), inside


def hold_votes(classes, pixels):
    """For each candidate place, in a :func:`pixel_classes` image: how many points of a hold there land on a hold's
    colour, on a known pixel (hold or wall), and in the image (``pixels`` from :func:`hold_pixels`)."""
    idx, inside = pixels
    cls = np.where(inside, classes.ravel()[idx], UNKNOWN)
    return (cls == HOLD).sum(axis=1), (cls != UNKNOWN).sum(axis=1), inside.sum(axis=1)


def select_holds(hit, known, seen):
    """The places that hold a hold, from the votes (``hit``, ``known``: added up over cameras and images; ``seen``:
    the largest fraction of each place visible in one set of images): (index, score, seen) for each.

    A place is a hold when enough of it was seen and most of what was seen is a hold's colour; the places are taken
    best first, skipping those whose hold would overlap one already taken (two holds cannot be bolted there).  Two
    holds side by side make one blob, whose best place is often its middle: each hold taken is then replaced by two
    that explain the blob better (both above PAIR_SCORE_MIN, a larger sum above it), when there are such two."""
    holds, _ = hold_candidates()
    with np.errstate(invalid="ignore", divide="ignore"):
        score = np.where(known > 0, hit / known, 0.0)
    score[seen < HOLD_SEEN_MIN] = 0.0
    at = {(h.col, h.row): i for i, h in enumerate(holds)}

    def clash(a, b):
        return (holds[a].col - holds[b].col, holds[a].row - holds[b].row) in OVERLAP

    taken = []
    for i in np.argsort(-score, kind="stable"):
        if score[i] < HOLD_SCORE_MIN:
            break
        if not any(clash(i, j) for j in taken):
            taken.append(int(i))
    for k, s in enumerate(list(taken)):
        others = [j for j in taken if j != s]
        near = [at[(holds[s].col + dc, holds[s].row + dr)] for dc, dr in OVERLAP
                if (holds[s].col + dc, holds[s].row + dr) in at]
        near = [j for j in near if score[j] >= PAIR_SCORE_MIN and not any(clash(j, o) for o in others)]
        gain = lambda *js: sum(score[j] - PAIR_SCORE_MIN for j in js)
        pairs = [(a, b) for n, a in enumerate(near) for b in near[n + 1:] if not clash(a, b)]
        if pairs:
            a, b = max(pairs, key=lambda p: gain(*p))
            if gain(a, b) > gain(s):
                taken[k] = a
                taken.append(b)
    return [(i, float(score[i]), float(seen[i])) for i in taken]


def hold_color(images: dict, pixels: dict, i: int) -> str:
    """The colour name of the hold at candidate place i: mean hue of its vivid pixels in the images."""
    px = []
    for name, img in images.items():
        idx, inside = pixels[name]
        hsv = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2HSV).reshape(-1, 3)[idx[i][inside[i]]]
        px.append(hsv[(hsv[:, 1] >= HOLD_SAT_MIN) & (hsv[:, 2] >= DARK_VAL_MAX)])
    px = np.concatenate(px)
    return _color_name(px[:, 0]) if len(px) else "?"


def _color_name(hues) -> str:
    """Name of the mean of OpenCV hues (0-180, a circle: red is at both ends)."""
    a = np.radians(np.asarray(hues, float) * 2)
    hue = np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360 / 2
    for top, name in ((5, "rouge"), (22, "orange"), (35, "jaune"), (85, "vert"), (130, "bleu"), (165, "violet")):
        if hue < top:
            return name
    return "rouge"


def annotate(image_rgb, tracker: "Tracker", name: str, pose=None, holds=()) -> np.ndarray:
    """BGR image (for cv2.imshow / imwrite) with the detected markers, the found holds (outlined in magenta) and
    the trunk pose (R, p, wall frame) drawn as its frame: x red (to the wall), y green (the robot's left), z blue
    (up)."""
    out = cv2.cvtColor(np.ascontiguousarray(image_rgb), cv2.COLOR_RGB2BGR)
    if holds and name in tracker.extrinsics:
        rvec, tvec, _ = tracker.extrinsics[name]
        dist = tracker.dist.get(name)
        for f in holds:
            uv, _ = cv2.projectPoints(world_to_wall(f.hold.socket + hold_points()), rvec, tvec, tracker.K[name],
                                      np.zeros(5) if dist is None else np.asarray(dist, float))
            cv2.polylines(out, [cv2.convexHull(uv.reshape(-1, 2).astype(np.int32))], True, (255, 0, 255), 1)
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
        holds = tracker.find_holds({"cam_left": img_left, "cam_right": img_right})   # the holds, now and then
        R, p = tracker.update({"cam_left": img_left, "cam_right": img_right}, obs["gravity"])  # trunk, or None
    """

    def __init__(self, intrinsics: dict, marker_size: float, R_trunk_marker, p_trunk_marker, dist: dict | None = None,
                 min_views: int = 2):
        self.K = {n: np.asarray(k, float) for n, k in intrinsics.items()}
        self.dist = dist or {}
        self.size, self.R_tm, self.p_tm = marker_size, R_trunk_marker, p_trunk_marker
        self.min_views = min_views       # cameras that must see the robot's marker (one camera: cm wrong in depth)
        self.extrinsics = {}             # name -> (rvec, tvec, rms)
        self.seen_by = []                # the calibrated cameras that saw the robot's marker in the last update
        self.reset_holds()

    def calibrate(self, images: dict) -> dict:
        """Each camera's pose from the wall markers; returns name -> (rvec, tvec, rms) for the cameras that saw some."""
        for name, img in images.items():
            pose = camera_pose(detect(img), self.K[name], self.dist.get(name))
            if pose is not None:
                self.extrinsics[name] = pose
                self._hold_px.pop(name, None)            # projected again with the new pose
        return self.extrinsics

    def reset_holds(self):
        """Forget what was seen of the holds (when they have been moved)."""
        self._hold_px = {}               # camera -> where each candidate place lands in its images (hold_pixels)
        self._votes = None               # (hit, known) added up over all the images given to find_holds, and seen:
                                         # the largest fraction of each place visible in one set of images
        self._colors = {}                # candidate place -> colour name

    def find_holds(self, images: dict) -> list[FoundHold]:
        """The holds on the wall, sorted from the bottom up.  The votes of these images are added to those of the
        previous calls, since the holds do not move: a hold hidden by the robot in one image counts once it shows in
        another.  Best: once before the robot is in front of the wall, then now and then while it climbs."""
        hit = known = total = 0
        used = {}
        for name, img in images.items():
            if name not in self.extrinsics:
                continue
            if name not in self._hold_px:
                rvec, tvec, _ = self.extrinsics[name]
                self._hold_px[name] = hold_pixels(self.K[name], self.dist.get(name), rvec, tvec, img.shape[:2])
            h, k, t = hold_votes(pixel_classes(img), self._hold_px[name])
            hit, known, total = hit + h, known + k, total + t
            used[name] = img
        if not used:
            return []
        with np.errstate(invalid="ignore", divide="ignore"):
            seen = np.where(total > 0, known / total, 0.0)
        if self._votes is not None:
            hit, known, seen = hit + self._votes[0], known + self._votes[1], np.maximum(seen, self._votes[2])
        self._votes = (hit, known, seen)
        holds, _ = hold_candidates()
        found = []
        for i, score, seen in select_holds(hit, known, seen):
            if self._colors.get(i, "?") == "?":
                self._colors[i] = hold_color(used, self._hold_px, i)
            found.append(FoundHold(holds[i], score, seen, self._colors[i]))
        return sorted(found, key=lambda f: (f.hold.row, f.hold.col))

    def update(self, images: dict, gravity=None):
        """The trunk's pose in the wall frame (R, p) from the robot's marker, or None when fewer than ``min_views``
        cameras see it.  ``gravity``: the IMU's gravity direction in the trunk frame (the observation's "gravity"),
        which then gives the trunk's tilt (:func:`level_with_gravity`): more accurate, the position too."""
        views, self.seen_by = [], []
        for name, img in images.items():
            if name not in self.extrinsics:
                continue
            corners = detect(img).get(ROBOT_ID)
            if corners is not None:
                rvec, tvec, _ = self.extrinsics[name]
                views.append({"corners": corners, "K": self.K[name], "dist": self.dist.get(name),
                              "rvec": rvec, "tvec": tvec})
                self.seen_by.append(name)
        if len(views) < self.min_views:
            return None
        pose = robot_pose(views, self.size, self.R_tm, self.p_tm)
        if pose is not None and gravity is not None:
            pose = level_with_gravity(pose, gravity, self.p_tm)
        return pose
