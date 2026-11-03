"""Survey the lunar base's 2D localization map with LunarBot's own lidar.

The prebuilt map AMCL localizes against (research repo F50 §4, user decision: wheel-inertial odometry +
localization against a prebuilt map). A base is surveyed when it is built; here the survey is exact by
construction: the rover is placed on every road, route and work/staging pose, the Mid-360 RayCaster
scans the same `/lunar_base/lidar` mesh it sees in every deploy run, and each return is kept only if
`pointcloud_to_laserscan` would keep it (chassis-frame height 0.05-0.6 m, planar range 0.1-40 m;
openral `nav2.launch.py`). Accumulating those returns in the world frame gives the occupancy grid the
live `/scan` is matched against. A cell is occupied only if returns land in it from at least
MIN_POSES survey poses: a vertical surface is hit in the same place from everywhere, while the slice's
ground returns (terrain rising into the band) fall on arcs that move with the rover -- first survey:
90 k occupied cells, most of them such arcs. Those stay unmodelled clutter for AMCL's z_rand.

    ~/Programs/isaac-sim/python.sh assets/lunar_base/scene/srb_nav_map_survey.py <out_dir> [min_poses=3]

Writes <out_dir>/map.yaml + map.pgm (nav2_map_server), survey.png (preview with the survey poses)
and survey.json (poses, hit counts).
"""

import json
import math
import sys
from pathlib import Path

from srb.core.app import AppLauncher
from srb.utils.path import SRB_APPS_DIR

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
launcher = AppLauncher(headless=True, enable_cameras=False, experience=SRB_APPS_DIR.joinpath("srb.headless.kit"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from isaaclab.utils.math import quat_apply_inverse, quat_from_euler_xyz  # noqa: E402

from srb.tasks.mobile_manipulation.lunar_base.task import Task, TaskCfg  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
L = yaml.safe_load(open(ROOT / "scene/layout.yaml"))
IX = json.loads((ROOT / "usd/scene/lunar_base/interaction.json").read_text())
H = np.load(ROOT / "scene/generated/terrain_near.npy").astype(np.float64)
HALF, RES_T = L["terrain"]["near_size_m"] / 2, L["terrain"]["near_res_m"]
BAND_Z, RANGE = (0.05, 0.6), (0.1, 40.0)  # pointcloud_to_laserscan's slice (openral nav2.launch.py)
GRID = 0.05  # map resolution, m
STEP_M = 1.0  # pose spacing along roads and routes
MIN_POSES = int(sys.argv[2]) if len(sys.argv) > 2 else 3
## ...and only as part of a lump of at least MIN_CELLS cells: MIN_POSES still left ~2,200 single cells where three or
## more poses' ground arcs happened to cross. AMCL treats a stray return as z_rand anyway, but Nav2's global costmap
## plans on this map, and each stray cell inflated to the rover's inscribed radius (0.26 m, ~85 cells) -- scattered
## ones joined into no-go areas metres across: the depot stand's approach pose sat inside one (research repo
## chain_l2b_pdu: "failed to create plan", 6 of 6). Every surveyed structure keeps its cells up to 12 (checked).
MIN_CELLS = 5


def terrain_z(x, y):
    i, j = (HALF - y) / RES_T, (x + HALF) / RES_T
    return float(cv2.remap(H.astype(np.float32), np.float32([[j]]), np.float32([[i]]), cv2.INTER_LINEAR)[0, 0])


def along(polyline):
    pts = np.asarray(polyline, float)
    for a, b in zip(pts[:-1], pts[1:]):
        d = b - a
        n = max(1, int(math.ceil(np.linalg.norm(d) / STEP_M)))
        yaw = math.atan2(d[1], d[0])
        for t in np.arange(n) / n:
            yield (a + t * d).tolist(), yaw
    yield pts[-1].tolist(), yaw


poses = []
for pl in [*L["roads"].values(), *L["missions"]["routes"].values()]:
    poses.extend(along(pl))
for wp in IX["work_poses"].values():
    yaw = math.radians(wp["yaw_deg"])
    poses.append((wp["xy"], yaw))
    poses.append(([wp["xy"][0] + 1.5 * math.cos(yaw), wp["xy"][1] + 1.5 * math.sin(yaw)], yaw))  # docking staging pose
# ...and every place the rover could drive through that none of those passes: a grid over the base's free ground. Nav2
# plans across any free cell, and a stretch no survey pose had seen is a stretch the map has nothing of: the habitat
# row's north face had not one cell, AMCL drove the corridor beside it on odometry alone and lost 7 m (research repo
# F141, ours_l2b_habitat_r2).
GRID_STEP_M = 2.5
STRUCT = json.loads((ROOT / "usd/scene/lunar_base/site_structures.json").read_text())["structures"]
FOOT = np.array([[0.6378, 0.57235], [-0.6678, 0.57235], [-0.6678, -0.57235], [0.6378, -0.57235]])  # openral_ext footprint.py


def clear_of_structures(xy, margin=0.3):
    """The footprint at xy (yaw 0) is at least `margin` from every surveyed structure's box (separating axes)."""
    hull = FOOT + np.asarray(xy)
    for s_ in STRUCT:
        c, sn = math.cos(s_["yaw"]), math.sin(s_["yaw"])
        R = np.array([[c, -sn], [sn, c]])
        box = (R @ (np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * np.array(s_["size_xy"]) / 2).T).T + s_["center"]
        axes = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), R[:, 0], R[:, 1]]
        if all(max(hull @ a) + margin > min(box @ a) and max(box @ a) + margin > min(hull @ a) for a in axes):
            return False
    return True


P0 = np.array([xy for xy, _ in poses])
lo_p, hi_p = P0.min(0) - 5.0, P0.max(0) + 5.0
for gx in np.arange(lo_p[0], hi_p[0] + 1e-6, GRID_STEP_M):
    for gy in np.arange(lo_p[1], hi_p[1] + 1e-6, GRID_STEP_M):
        if np.min(np.linalg.norm(P0 - [gx, gy], axis=1)) > 0.6 * GRID_STEP_M and clear_of_structures((gx, gy)):
            poses.append(([float(gx), float(gy)], 0.0))
print("POSES", len(poses), flush=True)

cfg = TaskCfg()
cfg.scene.num_envs = 1
cfg.__post_init__()
env = Task(cfg)
env.reset()
robot, lidar = env._robot, env._lidar_robot
dt = env.step_dt
arm_ids, _ = robot.find_joints([f"joint{i}" for i in range(1, 8)], preserve_order=True)
origin = env.scene.env_origins[0]
hold = torch.zeros(env.action_space.shape, device=env.device)
hold[:, 3] = 1.0  # arm: hold joint targets (srb_traction_check.py layout)
hold[:, 10:17] = robot.data.joint_pos[0, arm_ids]
hold[:, 17] = 1.0

hits_xy, rows = [], []
for k, (xy, yaw) in enumerate(poses):
    with torch.inference_mode():
        q = quat_from_euler_xyz(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([float(yaw)])).to(env.device)
        p = torch.tensor([[xy[0], xy[1], terrain_z(*xy) + 0.15]], device=env.device) + origin
        robot.write_root_pose_to_sim(torch.cat([p, q], 1))
        robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))
        for _ in range(int(1.5 / dt)):
            env.step(hold)
        w = lidar.data.ray_hits_w[0]
        w = w[torch.isfinite(w).all(1)]
        rel = quat_apply_inverse(robot.data.root_quat_w[0:1].expand(len(w), 4), w - robot.data.root_pos_w[0:1])
        rng = torch.linalg.norm(rel[:, :2], dim=1)
        keep = (rel[:, 2] >= BAND_Z[0]) & (rel[:, 2] <= BAND_Z[1]) & (rng >= RANGE[0]) & (rng <= RANGE[1])
        kept = (w[keep, :2] - origin[:2]).cpu().numpy()
    hits_xy.append(kept)
    at = (robot.data.root_pos_w[0] - origin).cpu().numpy()
    rows.append(dict(xy=[round(v, 2) for v in xy], yaw_deg=round(math.degrees(yaw)), settled=[round(float(v), 3) for v in at],
                     rays=int(len(w)), band_hits=int(len(kept))))
    if k % 20 == 0:
        print("POSE", k, json.dumps(rows[-1]), flush=True)

pts = np.concatenate(hits_xy)
lo = np.floor(pts.min(0) - 2.0)
hi = np.ceil(pts.max(0) + 2.0)
nx, ny = (int(round(v)) for v in (hi - lo) / GRID)
count = np.zeros((ny, nx), np.int32)
for h in hits_xy:  # count poses per cell, not returns
    cells = np.unique(((h[:, 1] - lo[1]) / GRID).astype(int) * nx + ((h[:, 0] - lo[0]) / GRID).astype(int))
    np.add.at(count.reshape(-1), cells, 1)
occ = count >= MIN_POSES
n_lumps, lumps, stats, _ = cv2.connectedComponentsWithStats(occ.astype(np.uint8), connectivity=8)
occ &= (stats[:, cv2.CC_STAT_AREA] >= MIN_CELLS)[lumps] & (lumps > 0)
np.savez_compressed(out / "survey_hits.npz", origin=lo, resolution=GRID, size=[nx, ny], pose_count=count,
                    hits=np.concatenate([np.c_[h, np.full(len(h), k)] for k, h in enumerate(hits_xy)]))
img = np.where(occ, 0, 254).astype(np.uint8)[::-1]  # PGM row 0 is the map's top (max y)
cv2.imwrite(str(out / "map.pgm"), img)
(out / "map.yaml").write_text(yaml.safe_dump(dict(image="map.pgm", mode="trinary", resolution=GRID,
                                                  origin=[float(lo[0]), float(lo[1]), 0.0], negate=0,
                                                  occupied_thresh=0.65, free_thresh=0.25), sort_keys=False))
prev = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for r in rows:
    c = (int((r["xy"][0] - lo[0]) / GRID), ny - 1 - int((r["xy"][1] - lo[1]) / GRID))
    cv2.circle(prev, c, 4, (0, 0, 255), -1)
cv2.imwrite(str(out / "survey.png"), prev)
(out / "survey.json").write_text(json.dumps(dict(origin=lo.tolist(), size=[nx, ny], resolution=GRID, band_z=BAND_Z, range=RANGE,
                                                 hits=int(len(pts)), min_poses=MIN_POSES, occupied_cells=int(occ.sum()), poses=rows), indent=1))
print("DONE", json.dumps(dict(poses=len(rows), hits=int(len(pts)), occupied=int(occ.sum()), size=[nx, ny], origin=lo.tolist())), flush=True)
env.close()
launcher.app.close()
