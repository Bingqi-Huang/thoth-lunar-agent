"""High-resolution stills of a finished run, from the same kinematic replay as `replay_render.py` (scene, robot and items
posed from the bag's /tf, lamps and radiator shutters from the evaluator's mission_state), but one Kit start for a whole
list of shots, each with its own sim time, camera and resolution.

    isaac-sim/python.sh still_render.py <run dir> <shots.json> <out dir> [--fill 150]

shots.json is a list of {"name", "t", "res": [w, h], "focal": mm, "cam": {...}}; the camera is one of
    {"type": "follow", "off": [x, y, z]}                      offset in the rover's frame from the point between body and gripper
    {"type": "orbit", "az": deg, "el": deg, "dist": m, "target": "robot" | "grip" | "mid" | [x, y, z]}
                                                               az 0 = rover's front, positive to its left
    {"type": "world", "eye": [x, y, z], "look": [x, y, z]}    scenery frame (+x east, +y north, dock at the origin)
and an optional "roll_free": true is not needed (z is up). A shot is written to <out dir>/<name>.png.
"""
import argparse
import glob
import json
import math
import sqlite3
import time
from pathlib import Path

import numpy as np

from replay_render import (LAMP_GREEN, LAMP_RED, LB, ROBOT_USD, SCENE, ALT_EYES, occluded, poses_at, quat_mat)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("shots", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--fill", type=float, default=150.0)
    a = ap.parse_args()
    shots = json.loads(a.shots.read_text())

    db = sqlite3.connect(glob.glob(str(a.run / "bag" / "*.db3"))[0])
    tid = db.execute("select id from topics where name='/tf'").fetchone()[0]
    t0, t1 = (v / 1e9 for v in db.execute("select min(timestamp), max(timestamp) from messages where topic_id=? and length(data)>2000", (tid,)).fetchone())
    db.close()
    for s in shots:
        s["t"] = min(max(s["t"], t0 + 0.05), t1)
    frames = poses_at(a.run / "bag", [s["t"] for s in shots])
    states = [(e["sim_t"], e["state"]["units"]) for e in map(json.loads, (a.run / "evaluator/events.jsonl").read_text().splitlines())
              if e["kind"] == "mission_state"]

    from isaacsim import SimulationApp

    w0, h0 = shots[0]["res"]
    app = SimulationApp({"headless": True, "width": w0, "height": h0})
    import carb  # noqa: E402
    import omni.replicator.core as rep  # noqa: E402
    import omni.usd  # noqa: E402
    from PIL import Image  # noqa: E402
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

    carb.settings.get_settings().set("/app/player/playSimulations", False)
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.DefinePrim("/World/lunar_base", "Xform").GetReferences().AddReference(str(SCENE / "lunar_base.usda"))
    stage.DefinePrim("/World/lunarbot", "Xform").GetReferences().AddReference(str(ROBOT_USD))
    ix = json.loads((SCENE / "interaction.json").read_text())
    items = {}
    for name, o in ix["objects"].items():
        prim = stage.DefinePrim(f"/World/objects/lb_{name}", "Xform")
        prim.GetReferences().AddReference(str(LB / o["usd"][o["usd"].index("usd/"):]))
        UsdGeom.Xformable(prim).ClearXformOpOrder()
        items[f"srb/env0/lb_{name}"] = UsdGeom.Xformable(prim).AddTransformOp()
    sun = UsdLux.DistantLight.Define(stage, "/World/sun")
    sun.CreateIntensityAttr(3000)
    sun.CreateAngleAttr(0.53)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(90 - 12, 0, 180 - 200))
    UsdLux.DomeLight.Define(stage, "/World/fill").CreateIntensityAttr(a.fill)

    cache = UsdGeom.XformCache()
    bodies = {}
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/lunarbot")):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            xf = UsdGeom.Xformable(prim)
            scale = Gf.Transform(xf.GetLocalTransformation()).GetScale()
            xf.ClearXformOpOrder()
            bodies[f"srb/env0/robot/{prim.GetName()}"] = (xf.AddTransformOp(), Gf.Matrix4d().SetScale(scale),
                                                          cache.GetLocalToWorldTransform(prim.GetParent()).GetInverse())
    lamps = {u: stage.GetPrimAtPath(f"/World/lunar_base/assets/{s['panel']}/_materials/StatusLamp/Principled_BSDF")
             for u, s in ix["units"].items()}
    covers = {u: [c for c in stage.GetPrimAtPath(f"/World/lunar_base/assets/{s['cover']}").GetChildren()
                  if c.GetName().startswith("cover_closed")] for u, s in ix["units"].items() if s["cover"]}

    def mat(p, q) -> Gf.Matrix4d:
        return Gf.Matrix4d().SetTransform(Gf.Rotation(Gf.Quatd(q[3], q[0], q[1], q[2])), Gf.Vec3d(*p))

    cam = UsdGeom.Camera.Define(stage, "/World/still_cam")
    cam.CreateHorizontalApertureAttr(20.955)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 8000.0))
    focal_attr, vap_attr = cam.CreateFocalLengthAttr(18.0), cam.CreateVerticalApertureAttr(11.8)
    cam_op = UsdGeom.Xformable(cam).AddTransformOp()
    rp = rep.create.render_product("/World/still_cam", (w0, h0))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach(rp)
    a.out.mkdir(parents=True, exist_ok=True)

    def where(shot: dict, f: dict) -> tuple[np.ndarray, np.ndarray]:
        p, q = f["srb/env0/robot"]
        r = quat_mat(q)
        grip = p + r @ f["srb/env0/robot/eg2_base"][0]
        yaw = math.atan2(r[1, 0], r[0, 0])
        mid = 0.5 * (p + np.array([0.0, 0.0, 0.8])) + 0.5 * grip
        c = shot["cam"]
        if c["type"] == "world":
            return np.array(c["eye"], float), np.array(c["look"], float)
        tgt = c.get("target", "mid")
        look = {"robot": p + np.array([0.0, 0.0, 0.55]), "grip": grip, "mid": mid}[tgt] if isinstance(tgt, str) else np.array(tgt, float)
        if c["type"] == "follow":
            ca, sa = math.cos(yaw), math.sin(yaw)
            o = c["off"]
            eye = look + np.array([ca * o[0] - sa * o[1], sa * o[0] + ca * o[1], o[2]])
            if occluded(look, eye):
                eye = min([eye] + [look + np.array([ca * v[0] - sa * v[1], sa * v[0] + ca * v[1], v[2]]) for v in ALT_EYES],
                          key=lambda e: occluded(look, e))
            return eye, look
        az, el, d = math.radians(c["az"]) + yaw, math.radians(c["el"]), c["dist"]
        return look + d * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]), look

    shown = {}
    start = time.time()
    for n, (shot, f) in enumerate(zip(shots, frames)):
        t, (w, h) = shot["t"], shot["res"]
        root = mat(*f["srb/env0/robot"])
        for name, (op, scale, parent_inv) in bodies.items():
            op.Set(scale * mat(*f[name]) * root * parent_inv)
        for name, op in items.items():
            op.Set(mat(*f[name]))
        st_t = [s[0] for s in states]
        units = states[max(sum(1 for x in st_t if x <= t) - 1, 0)][1]
        for u, s in units.items():
            key = (s["tripped"], s["cover_closed"])
            if shown.get(u) != key:
                shown[u] = key
                colour = Gf.Vec3f(*(LAMP_RED if s["tripped"] else LAMP_GREEN))
                lamps[u].GetAttribute("inputs:diffuseColor").Set(colour)
                lamps[u].GetAttribute("inputs:emissiveColor").Set(colour * 1.5)
                for c in covers.get(u, []):
                    (UsdGeom.Imageable(c).MakeVisible if s["cover_closed"] else UsdGeom.Imageable(c).MakeInvisible)()
        eye, look = where(shot, f)
        focal_attr.Set(float(shot.get("focal", 18.0)))
        vap_attr.Set(20.955 * h / w)
        cam_op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*look), Gf.Vec3d(0, 0, 1)).GetInverse())
        for _ in range(80 if n == 0 else 6):
            rep.orchestrator.step(rt_subframes=6)
        img = np.asarray(rgb.get_data())[..., :3]
        if img.shape[:2] != (h, w):
            print(f"resolution stuck at {img.shape[1]}x{img.shape[0]} for {shot['name']}", flush=True)
        Image.fromarray(img).save(a.out / f"{shot['name']}.png")
        print(f"{n + 1}/{len(shots)} {shot['name']} {(time.time() - start) / (n + 1):.1f} s/shot", flush=True)
    print(f"RENDERED {len(shots)} stills in {time.time() - start:.0f} s -> {a.out}", flush=True)
    app.close()


if __name__ == "__main__":
    main()
