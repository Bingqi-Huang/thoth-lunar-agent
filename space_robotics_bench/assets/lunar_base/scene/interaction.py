"""Interaction layer of the lunar base, used by build_scene.py.

From layout.yaml `props`, `dynamic_objects` and `inspection_points` it derives, in the scenery frame:
- world poses of the authored props (attached to an asset face, or free-standing on the platform);
- interaction frames (press / indicator / place / grasp) from each prop's authored frames.json;
- spawn poses of the dynamic objects on their supports;
- LunarBot base poses: a work pose per frame (arm end -X toward the frame, the frame `reach` past the
  -X hull edge) and a view pose per inspection point (front camera +X facing it);
and the offline checks: supports under every object, no spawn overlaps, grasp
features inside the reach table, work/view poses clear of every structure.
"""

import json
import math
from pathlib import Path

import numpy as np

# LunarBot envelope (openral robots/lunar_bot/robot.yaml)
HULL_X_BACK = 0.6678  # -X hull edge of the footprint polygon (arm end)
FOOTPRINT = np.array([[0.6378, 0.57235], [-0.6678, 0.57235], [-0.6678, -0.57235], [0.6378, -0.57235]])
# Tool point distance past the -X hull edge, all side-on (tool horizontal into the frame's surface). 0.45 m keeps the
# RM-75 on its shoulder-back branch with >=0.2 joint-range margin for z 0.78-1.10 m, which is what a press needs.
# Top-down poses at these heights are out of reach; spares carry a top bail handle grasped side-on instead.
# how far past the hull the work pose puts a site point, by what the arm does there. A press only needs the tool tip
# there; a grasp needs the last centimetres past the part and room for the arm's elbow to stay above the item, which
# the RM-75 does not have at its full 0.61 m radius; at 0.35 m it folds the elbow into its stop instead.
REACH = {"press": 0.45, "place": 0.40, "pick": 0.40}
GRASP_Z = (0.45, 1.10)


def rot(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def rect_corners(center, half, yaw):
    return np.array([center + rot(yaw) @ np.array([u, v]) for u, v in ((-half[0], -half[1]), (half[0], -half[1]), (half[0], half[1]), (-half[0], half[1]))])


def polys_overlap_depth(p, q):
    """Separating-axis test for convex polygons: >0 = penetration depth, <=0 = minus the gap."""
    best = -np.inf
    for poly in (p, q):
        for i in range(len(poly)):
            e = poly[(i + 1) % len(poly)] - poly[i]
            n = np.array([-e[1], e[0]]) / (np.linalg.norm(e) + 1e-12)
            a, b = p @ n, q @ n
            sep = max(b.min() - a.max(), a.min() - b.max())
            best = max(best, sep)
    return -best


def build(L, root, asset_placements, asset_boxes, height):
    """Return (prop_placements, prop_boxes, interaction, problems)."""
    root = Path(root)
    by_name = {p["name"]: p for p in asset_placements}
    box_by_name = {b["name"]: b for b in asset_boxes}
    problems = []

    def meta(usd_rel):
        name = Path(usd_rel).name
        m = json.loads((root / "usd" / usd_rel / f"{name}.json").read_text())
        f = root / "authored" / name / "frames.json"
        m["frames"] = json.loads(f.read_text()) if f.exists() else {}
        return m

    # ---- props -----------------------------------------------------------------------------------
    props, prop_boxes, prop_pose = [], [], {}
    for name, p in L.get("props", {}).items():
        m = meta(p["usd"])
        sx, sy, sz = m["size_m"]
        if "attach" in p:
            parent = by_name[p["attach"]]
            pb = box_by_name[p["attach"]]
            pyaw = math.radians(parent["yaw_deg"])
            lx = pb["half"][0] if p["local"][0] == "front" else float(p["local"][0])
            xy = np.array(parent["pos"][:2]) + rot(pyaw) @ np.array([lx, p["local"][1]])
            z = parent["pos"][2] + float(p["local"][2])
            yaw = pyaw + math.radians(p.get("yaw_deg", 0.0))
        else:
            xy, yaw = np.array(p["pos"], float), math.radians(p.get("yaw_deg", 0.0))
            sx_, sy_ = meta(p["usd"])["size_m"][:2]
            gu, gv = np.meshgrid(np.linspace(-sx_ / 2, sx_ / 2, 5), np.linspace(-sy_ / 2, sy_ / 2, 5))
            foot = [height(*(xy + rot(yaw) @ np.array([u, v]))) for u, v in zip(gu.ravel(), gv.ravel())]
            z = float(max(foot))
            if max(foot) - min(foot) > 0.02:
                problems.append(f"{name}: footprint not level ({max(foot) - min(foot):.3f} m)")
        # keep_origin props: extent is [0, sx] along x from the mount point; others are centred
        cx = sx / 2 if "attach" in p else 0.0
        center = xy + rot(yaw) @ np.array([cx, 0.0])
        prop_pose[name] = dict(xy=xy, z=z, yaw=yaw, frames=m["frames"], parent=p.get("attach"))
        props.append(dict(name=name, usd=str(root / "usd" / p["usd"] / f"{Path(p['usd']).name}.usdc"), pos=[float(xy[0]), float(xy[1]), float(z)],
                          yaw_deg=math.degrees(yaw), zone=by_name[p["attach"]]["zone"] if "attach" in p else p.get("zone", "")))
        # A prop that declares a working surface is an obstacle up to that surface, not up to its own bounding
        # height: staging_stand measures 0.61 because of the lip along its back, while its table -- the slots the
        # arm works over, and the front edge the asset leaves free for it -- is at 0.55. Taking the bounding height
        # put every item standing on a slot 6 cm inside the obstacle before the arm had touched it, so a carried
        # item had to rise 0.115 m before the arm's own collision checks would let it move at all, and bring_in
        # stalled with the ORU still over the stand.
        # `sz` is how *tall* the prop is, not where its top is: a prop that keeps its modelled origin need not
        # start at z = 0. The service shelf spans 0.16..0.53 in its own frame, so a box built to sz put a
        # phantom obstacle in the 0.16 m of air beneath it and left the tray itself uncovered. The asset
        # now reports where it is; `top_z` still overrides, for a prop whose working surface is below its top.
        top = float(m["frames"].get("top_z", m.get("z_range_m", [0.0, sz])[1]))
        prop_boxes.append(dict(name=name, center=center.tolist(), half=[sx / 2, sy / 2], yaw=yaw, z0=0.0 if "attach" in p else z,
                               z1=z + top if "attach" not in p else z + float(p["local"][2]) + top,
                               corners=rect_corners(center, [sx / 2, sy / 2], yaw), attached_to=p.get("attach")))

    def frame_world(prop, fname):
        pp = prop_pose[prop]
        f = pp["frames"][fname]
        xy = pp["xy"] + rot(pp["yaw"]) @ np.array(f["xyz"][:2])
        n = rot(pp["yaw"]) @ np.array(f["normal"][:2])
        return dict(xyz=[float(xy[0]), float(xy[1]), float(pp["z"] + f["xyz"][2])], normal=[float(n[0]), float(n[1]), 0.0], type=f["type"])

    points = {}
    for prop, pp in prop_pose.items():
        for fname, f in pp["frames"].items():
            if isinstance(f, dict) and "type" in f:
                points[f"{prop}/{fname}"] = dict(frame_world(prop, fname), prop=prop, asset=pp["parent"] or prop)
                for extra in ("material", "interface", "role"):  # what the prop's author says about the feature
                    if extra in f:
                        points[f"{prop}/{fname}"][extra] = f[extra]

    # ---- dynamic objects -----------------------------------------------------------------------------
    objects = {}
    for name, o in L.get("dynamic_objects", {}).items():
        m = meta(o["usd"])
        sx, sy, sz = m["size_m"]
        support = points[o["support"]]
        yaw = math.atan2(support["normal"][1], support["normal"][0]) + math.radians(o.get("yaw_deg", 0.0))
        pitch = math.radians(o.get("pitch_deg", 0.0))
        xy = np.array(support["xyz"][:2]) + np.array(support["normal"][:2]) * float(o.get("forward_m", 0.0))
        z = support["xyz"][2] + float(o.get("lift_m", 0.005))
        keep_origin = (root / "authored" / Path(o["usd"]).name).exists()
        center = xy + rot(yaw) @ np.array([sx / 2 if keep_origin else 0.0, 0.0])
        objects[name] = dict(usd=str(root / "usd" / o["usd"] / f"{Path(o['usd']).name}.usdc"), pos=[float(xy[0]), float(xy[1]), float(z)],
                             yaw_deg=math.degrees(yaw), pitch_deg=math.degrees(pitch), mass_kg=m.get("mass_kg"), size_m=m["size_m"],
                             support=o["support"], item=o["item"], serial=o["serial"], top_z=float(z + sz * math.cos(pitch)),
                             footprint=rect_corners(center, [sx / 2, sy / 2], yaw), **({"status": o["status"]} if "status" in o else {}))
        if m["physics"] != "dynamic":
            problems.append(f"{name}: {o['usd']} is not a dynamic asset")
        objects[name]["handle"] = m["handle"]  # lifting handle on the lid (asset json, bottom-centred local frame)
        feat_z = z + m["handle"]["xyz"][2]
        if not GRASP_Z[0] <= feat_z <= GRASP_Z[1]:
            problems.append(f"{name}: grasp feature z {feat_z:.2f} outside {GRASP_Z}")
        points[f"{name}/pick"] = dict(xyz=[float(center[0]), float(center[1]), float(feat_z)], normal=support["normal"], type="pick",
                                        asset=support["asset"], prop=points[o["support"]]["prop"], object=name)
    names = list(objects)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if objects[a]["support"] == objects[b]["support"] or points[objects[a]["support"]]["prop"] == points[objects[b]["support"]]["prop"]:
                d = polys_overlap_depth(objects[a]["footprint"], objects[b]["footprint"])
                if d > -0.01:
                    problems.append(f"spawn overlap {a} <-> {b}: {d:+.3f} m")

    # ---- robot work / view poses -------------------------------------------------------------------
    all_boxes = list(asset_boxes) + prop_boxes

    def pose_clear(xy, yaw, ignore):
        fp = np.array([xy + rot(yaw) @ c for c in FOOTPRINT])
        worst = (np.inf, None)
        for b in all_boxes:
            if b["name"] in ignore or b.get("attached_to") in ignore:
                continue
            gap = -polys_overlap_depth(fp, np.asarray(b["corners"]))
            if gap < worst[0]:
                worst = (gap, b["name"])
        return worst

    work_poses = {}
    for key, pt in points.items():
        if pt["type"] not in REACH:
            continue
        n = np.array(pt["normal"][:2])
        dist = HULL_X_BACK + REACH[pt["type"]]
        xy = np.array(pt["xyz"][:2]) + n * dist
        yaw = math.atan2(n[1], n[0])
        # nothing is ignored: the hull must clear the target structure, its props (e.g. the PDU shelf beside
        # the reset panel) and the support the arm reaches over
        gap, near = pose_clear(xy, yaw, set())
        work_poses[key] = dict(xy=[round(float(xy[0]), 3), round(float(xy[1]), 3)], yaw_deg=round(math.degrees(yaw), 1), reach_m=REACH[pt["type"]],
                               clearance_m=round(float(gap), 3), nearest=near)
        if gap < 0.03:
            problems.append(f"work pose {key}: footprint {gap:+.2f} m from {near}")
    view_poses = {}
    for name, v in L.get("inspection_points", {}).items():
        a = by_name[v["asset"]]
        ab = box_by_name[v["asset"]]
        yaw_a = math.radians(a["yaw_deg"])
        front = rot(yaw_a) @ np.array([1.0, 0.0]) if v.get("face", "front") == "front" else rot(yaw_a) @ np.array([-1.0, 0.0])
        if "normal" in v:
            front = np.array(v["normal"], float)
        lx = ab["half"][0] if v["local"][0] == "front" else float(v["local"][0])
        target = np.array(a["pos"][:2]) + rot(yaw_a) @ np.array([lx, v["local"][1]])
        xy = target + front * float(v.get("distance_m", 3.0))
        yaw = math.atan2(-front[1], -front[0])
        gap, near = pose_clear(xy, yaw, set())
        view_poses[name] = dict(xy=[round(float(xy[0]), 3), round(float(xy[1]), 3)], yaw_deg=round(math.degrees(yaw), 1),
                                target=[round(float(target[0]), 3), round(float(target[1]), 3), round(a["pos"][2] + float(v["local"][2]), 3)],
                                clearance_m=round(float(gap), 3), nearest=near)
        if gap < 0.05:
            problems.append(f"view pose {name}: footprint {gap:+.2f} m from {near}")

    # serviced units: every point on a unit's control face, thermal cover or module cradles carries the unit's id -- the
    # name its telemetry, the authority rule and mission control use (the prop's parent asset is not always the unit)
    units = L.get("units", {})
    for u, spec in units.items():
        for prop in [spec["panel"], spec["cover"], *spec["bays"]]:
            for key, pt in points.items():
                if pt.get("prop") == prop:
                    pt["unit"] = u
    for o in objects.values():
        o["footprint"] = np.round(o["footprint"], 3).tolist()
    interaction = dict(frame="scenery (lunar_base prim); +x east, +y north, dock at origin", points=points, objects=objects, units=units,
                       work_poses=work_poses, view_poses=view_poses, roads={k: [[round(float(x), 2), round(float(y), 2)] for x, y in v]
                                                                            for k, v in (L.get("roads") or {}).items()},
                       robot=dict(hull_x_back_m=HULL_X_BACK, reach_m=REACH, grasp_z_m=GRASP_Z))
    for b in prop_boxes:
        b["corners"] = np.asarray(b["corners"])
    return props, prop_boxes, interaction, problems
