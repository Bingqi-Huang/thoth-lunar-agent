#!/usr/bin/env python3
"""Assemble the lunar base scenery USD from layout.yaml, the generated terrain and the processed assets.

    python3 scene/build_terrain.py && python3 scene/build_scene.py

Geometry is computed here (numpy); scene/usd_writer.py (Blender's bundled pxr) writes
usd/scene/lunar_base/{lunar_base.usda, terrain.usdc, textures/, rocks/}. Terrain meshes: near (collider),
mid ring and far ring (visual), see build_terrain.py.

Prim order matters: /lunar_base/lidar is the first Mesh under the scenery prim, because the
LunarBot RayCaster only ray-casts the first Mesh it finds. It holds the near-field terrain plus a
box per static asset and has no collider; /lunar_base/terrain carries the physics collider.
"""

import json
import math
import shutil
import subprocess
from pathlib import Path

import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "scene/generated"
OUT = ROOT / "usd/scene/lunar_base"
BLENDER = os.environ.get("BLENDER", "blender")
ROCKS = ROOT / "raw/terrain/omnilrs_regolith_rocks/lunar_rocks/rocks_s4_r512"

L = yaml.safe_load(open(ROOT / "scene/layout.yaml"))
# Scene variants (layout.yaml `variants:`) add static assets -- baked into the lidar mesh like any structure, which a
# runtime-spawned obstacle would not be (the RayCaster only sees `/lunar_base/lidar`, F41). A variant writes
# lunar_base_<variant>.usda + terrain_<variant>.usdc next to the base scene and shares its textures, rocks and
# interaction.json.
VARIANT = sys.argv[sys.argv.index("--variant") + 1] if "--variant" in sys.argv else ""
if VARIANT:
    L["assets"] = {**L["assets"], **L["variants"][VARIANT]["assets"]}
T = L["terrain"]
rng = np.random.default_rng(T["seed"] + 1)
h = np.load(GEN / "terrain_near.npy").astype(np.float64)
mid = np.load(GEN / "terrain_mid.npy").astype(np.float64)
far = np.load(GEN / "terrain_far.npy").astype(np.float64)
platform = np.load(GEN / "platform_mask.npy")
half, res = T["near_size_m"] / 2, T["near_res_m"]


def ij(x, y):
    return (half - y) / res, (x + half) / res


def height(x, y):
    i, j = ij(np.asarray(x, float), np.asarray(y, float))
    return cv2.remap(h.astype(np.float32), j.astype(np.float32).reshape(-1, 1), i.astype(np.float32).reshape(-1, 1),
                     cv2.INTER_LINEAR).ravel()


def grid_mesh(z, size, keep=None):
    """Regular grid → (points, triangle indices); keep = per-cell boolean mask to emit only some cells."""
    n = z.shape[0]
    v = np.linspace(-size / 2, size / 2, n)
    xs, ys = np.meshgrid(v, v[::-1])
    pts = np.stack([xs, ys, z], -1).reshape(-1, 3)
    a = np.arange(n * n).reshape(n, n)
    q00, q01, q10, q11 = a[:-1, :-1], a[:-1, 1:], a[1:, :-1], a[1:, 1:]
    tris = np.stack([np.stack([q10, q11, q01], -1), np.stack([q10, q01, q00], -1)], 2)  # CCW seen from +z
    if keep is not None:
        tris = tris[keep]
    return pts, tris.reshape(-1, 3)


def vertex_normals(pts, tris):
    fn = np.cross(pts[tris[:, 1]] - pts[tris[:, 0]], pts[tris[:, 2]] - pts[tris[:, 0]])
    vn = np.zeros_like(pts)
    for k in range(3):
        np.add.at(vn, tris[:, k], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-9)


# ---- asset placements --------------------------------------------------------------------------
placements, boxes, problems = [], [], []
for name, a in L["assets"].items():
    usd = ROOT / "usd" / a["usd"] / f"{Path(a['usd']).name}.usdc"
    meta = json.loads((usd.parent / f"{Path(a['usd']).name}.json").read_text())
    sx, sy, sz = meta["size_m"]
    x, y = a["pos"]
    yaw = math.radians(a.get("yaw_deg", 0.0))
    c, s = math.cos(yaw), math.sin(yaw)
    corners = np.array([[x + c * u - s * v, y + s * u + c * v] for u in (-sx / 2, sx / 2) for v in (-sy / 2, sy / 2)])
    fx = np.linspace(-sx / 2, sx / 2, 9)
    fy = np.linspace(-sy / 2, sy / 2, 9)
    fu, fv = np.meshgrid(fx, fy)
    foot = height(x + c * fu - s * fv, y + s * fu + c * fv)
    z = float(foot.max())
    if foot.max() - foot.min() > 0.05:
        problems.append(f"{name}: footprint not level ({foot.max() - foot.min():.2f} m relief)")
    if meta["physics"] != "static":
        problems.append(f"{name}: dynamic asset in the static layout")
    placements.append(dict(name=name, usd=str(usd), pos=[x, y, z], yaw_deg=a.get("yaw_deg", 0.0), zone=a["zone"]))
    boxes.append(dict(name=name, center=[x, y], half=[sx / 2, sy / 2], yaw=yaw, z0=z, z1=z + min(sz, 4.0), corners=corners))

# ---- interaction layer (Phase C3): props, dynamic objects, interaction frames, robot work/view poses ----
from interaction import build as build_interaction  # noqa: E402

prop_placements, prop_boxes, interaction, ix_problems = build_interaction(L, ROOT, placements, boxes, lambda x, y: float(height(x, y)[0]))
problems += ix_problems
placements += prop_placements
boxes += prop_boxes
interaction["objects_usd_root"] = "assets/lunar_base"
if not VARIANT:
    (GEN / "interaction.json").write_text(json.dumps(interaction, indent=1))
    (OUT / "interaction.json").write_text(json.dumps(interaction, indent=1))
# a change to the dynamic objects only (layout `dynamic_objects`): the static scenery, its lidar mesh and the nav map
# built from it stay as they are
if "--interaction-only" in sys.argv:
    print(json.dumps(dict(problems=problems), indent=1))
    sys.exit(0)

# Clearance report: LunarBot needs >= 2 m aisles (footprint r 0.82 m + inflation).
def poly_distance(p, q):
    """Distance between two convex polygons (0 if they overlap)."""
    def seg(pt, a, b):
        t = np.clip(np.dot(pt - a, b - a) / max(np.dot(b - a, b - a), 1e-12), 0, 1)
        return np.linalg.norm(pt - (a + t * (b - a)))

    if any(cv2.pointPolygonTest(q.astype(np.float32), tuple(map(float, v)), False) >= 0 for v in p) or any(
        cv2.pointPolygonTest(p.astype(np.float32), tuple(map(float, v)), False) >= 0 for v in q
    ):
        return 0.0
    return min(min(seg(v, e[k], e[(k + 1) % 4]) for v in o for k in range(4)) for o, e in ((p, q), (q, p)))


clearances = []
for i, b1 in enumerate(boxes):
    for b2 in boxes[i + 1 :]:
        if b1["name"] in (b2.get("attached_to"),) or b2["name"] in (b1.get("attached_to"),) or (
            b1.get("attached_to") and b1.get("attached_to") == b2.get("attached_to")
        ):
            continue  # props mounted on the same structure
        d = poly_distance(b1["corners"][[0, 1, 3, 2]], b2["corners"][[0, 1, 3, 2]])
        if d < 2.0:
            clearances.append(f"{b1['name']} <-> {b2['name']}: {d:.2f} m")

# Roads and reference routes vs asset footprints: the robot centre line needs >= 1 m (half width 0.57 m
# + margin) from every structure.
def box_distance(p, b):
    c, s_ = math.cos(b["yaw"]), math.sin(b["yaw"])
    q = np.array([[c, s_], [-s_, c]]) @ (np.asarray(p) - np.asarray(b["center"]))
    d = np.maximum(np.abs(q) - np.asarray(b["half"]), 0.0)
    return float(np.linalg.norm(d)) if (np.abs(q) > b["half"]).any() else -float(np.min(np.asarray(b["half"]) - np.abs(q)))


def path_clearance(polyline, need=1.0):
    hits = {}
    pts = np.asarray(polyline, float)
    for a_, b_ in zip(pts[:-1], pts[1:]):
        for t in np.linspace(0, 1, max(2, int(np.linalg.norm(b_ - a_) / 0.25))):
            p = a_ + t * (b_ - a_)
            for b in boxes:
                d = box_distance(p, b)
                if d < need and d < hits.get(b["name"], (np.inf,))[0]:
                    hits[b["name"]] = (round(d, 2), np.round(p, 1).tolist())
    return hits


for kind, paths in (("road", L["roads"]), ("route", L.get("missions", {}).get("routes", {}))):
    for pname, pl in paths.items():
        for asset, (d, at) in path_clearance(pl).items():
            clearances.append(f"{kind} {pname} <-> {asset}: {d:.2f} m at {at}")

# ---- lidar mesh: terrain + static asset boxes ---------------------------------------------------
t_pts, t_tris = grid_mesh(h, T["near_size_m"])
lid_pts, lid_tris = [t_pts], [t_tris]
off = len(t_pts)
box_faces = np.array([[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 4], [1, 5, 4],
                      [2, 6, 3], [3, 6, 7], [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5]])
for b in boxes:
    cz = []
    for zc in (b["z0"] - 0.2, b["z1"]):
        for k in (0, 1, 2, 3):
            cz.append([*b["corners"][k], zc])
    lid_pts.append(np.array(cz))
    lid_tris.append(box_faces + off)
    off += 8
lidar_pts, lidar_tris = np.concatenate(lid_pts), np.concatenate(lid_tris)

# ---- rocks: scattered outside the graded platform and across the mid ring (visual only) ----------
MID, mid_res = T["mid_size_m"] / 2, T["mid_res_m"]


def mid_height(x, y):
    i, j = (MID - y) / mid_res, (x + MID) / mid_res
    return float(cv2.remap(mid.astype(np.float32), np.float32([[j]]), np.float32([[i]]), cv2.INTER_LINEAR)[0, 0])


rock_files = sorted(ROCKS.glob("*.usdz"))[:16]
rocks = []


def add_rock(x, y, z, scale):
    rocks.append(dict(file=rock_files[rng.integers(len(rock_files))].name, pos=[x, y, z - 0.35 * scale],
                      yaw_deg=float(rng.uniform(0, 360)), tilt_deg=[float(v) for v in rng.uniform(-15, 15, 2)], scale=scale))


while len(rocks) < T["rocks"]["near"]:
    x, y = rng.uniform(-half + 1, half - 1, 2)
    i, j = (int(round(v)) for v in ij(x, y))
    if platform[i, j]:
        continue
    add_rock(x, y, float(height(x, y)[0]), float(np.clip(rng.lognormal(-1.2, 0.7), 0.08, 1.6)))
n_near = len(rocks)
while len(rocks) < n_near + T["rocks"]["mid"]:
    x, y = rng.uniform(-MID + 20, MID - 20, 2)
    ring = max(abs(x), abs(y))
    if ring <= half or rng.random() > (half / ring) ** 1.5:  # thinning with distance, like the site's scatter
        continue
    add_rock(x, y, mid_height(x, y), float(np.clip(rng.lognormal(-0.9, 0.7), 0.1, 2.5)))


def ring_mesh(z, size, res, inner_half):
    """Grid mesh without the cells that lie entirely inside the finer field."""
    n = z.shape[0]
    v = np.linspace(-size / 2, size / 2, n)
    gx, gy = np.meshgrid(v, v[::-1])
    inner = (np.abs(gx) < inner_half - res + 1e-6) & (np.abs(gy) < inner_half - res + 1e-6)
    keep = ~(inner[:-1, :-1] & inner[1:, 1:] & inner[:-1, 1:] & inner[1:, :-1])
    return grid_mesh(z, size, keep=np.repeat(keep[:, :, None], 2, 2))


m_pts, m_tris = ring_mesh(mid, T["mid_size_m"], mid_res, half)
f_pts, f_tris = ring_mesh(far, T["far_size_m"], T["far_res_m"], MID)

# ---- textures (baked near-field albedo + tiling 2K regolith maps, bake_albedo.py) ------------------
if not VARIANT:
    subprocess.run(["python3", str(ROOT / "scene/bake_albedo.py")], check=True)
(OUT / "rocks").mkdir(exist_ok=True)
for f in rock_files:
    if not (OUT / "rocks" / f.name).exists():
        shutil.copy2(f, OUT / "rocks" / f.name)

# ---- hand off to the USD writer ------------------------------------------------------------------
job = GEN / f"scene_job{'_' + VARIANT if VARIANT else ''}.npz"
np.savez_compressed(
    job,
    terrain_pts=t_pts, terrain_tris=t_tris, terrain_nrm=vertex_normals(t_pts, t_tris),
    lidar_pts=lidar_pts, lidar_tris=lidar_tris,
    mid_pts=m_pts, mid_tris=m_tris, mid_nrm=np.tile([0.0, 0.0, 1.0], (len(m_pts), 1)),  # relief lives in the baked normal map
    far_pts=f_pts, far_tris=f_tris, far_nrm=vertex_normals(f_pts, f_tris),
)
meta = dict(out=str(OUT), suffix=f"_{VARIANT}" if VARIANT else "", job=str(job), placements=placements, rocks=rocks, tile_m=L["albedo"]["tile_m"], interaction=interaction,
            near_size_m=T["near_size_m"], mid_size_m=T["mid_size_m"])
# the surveyed static structures as oriented boxes, for planners that need the worksite (MoveIt's planning scene)
if not VARIANT:
    (OUT / "site_structures.json").write_text(json.dumps(dict(
        frame="site (= map)", source="build_scene.py static asset boxes (the /lunar_base/lidar structures)",
        structures=[dict(name=b["name"], center=[round(float(v), 4) for v in np.mean(np.asarray(b["corners"], float), 0)],
                         size_xy=[round(2 * b["half"][0], 4), round(2 * b["half"][1], 4)], yaw=round(float(b["yaw"]), 5),
                         z0=round(float(b["z0"]), 4), z1=round(float(b["z1"]), 4)) for b in boxes]), indent=1))
job_json = GEN / f"scene_job{'_' + VARIANT if VARIANT else ''}.json"
job_json.write_text(json.dumps(meta, indent=1))
r = subprocess.run([BLENDER, "-b", "--factory-startup", "-P", str(ROOT / "scene/usd_writer.py"), "--", str(job_json)],
                   capture_output=True, text=True)
print("\n".join(l for l in r.stdout.splitlines() if l.startswith(("WROTE", "Error", "Traceback", "  "))), r.stderr[-2000:])

report = dict(
    assets=len(placements), rocks=len(rocks),
    faces=dict(terrain=len(t_tris), lidar=len(lidar_tris), mid=len(m_tris), far=len(f_tris)),
    problems=problems, clearance_below_2m=clearances,
    work_poses={k: (v["clearance_m"], v["nearest"]) for k, v in interaction["work_poses"].items()},
    view_poses={k: (v["clearance_m"], v["nearest"]) for k, v in interaction["view_poses"].items()},
)
(GEN / f"scene_report{'_' + VARIANT if VARIANT else ''}.json").write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
