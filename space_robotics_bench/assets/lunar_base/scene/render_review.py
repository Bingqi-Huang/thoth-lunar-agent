"""RTX review renders of the assembled lunar base with LunarBot parked on the dock (no physics).

    ~/Programs/isaac-sim/python.sh scene/render_review.py <out_dir> [view ...]

A view is a name below or `name=ex,ey,ez,tx,ty,tz` (eye and target, scenery frame). The interaction layer's movable
objects (ORUs, modules, containers) are placed at their authored spawn poses, as a task resets them.
"""

import sys
from pathlib import Path

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "width": 1600, "height": 900})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SRB = ROOT.parent.parent
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)

omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)

base = stage.DefinePrim("/World/lunar_base", "Xform")
base.GetReferences().AddReference(str(ROOT / "usd/scene/lunar_base/lunar_base.usda"))
bot = stage.DefinePrim("/World/lunarbot", "Xform")
bot.GetReferences().AddReference(str(SRB / "assets/srb_assets/robot/mobile_manipulation/lunarbot/lunarbot.usdc"))
UsdGeom.XformCommonAPI(bot).SetTranslate(Gf.Vec3d(0.0, 0.0, 0.0))

import json  # noqa: E402

for name, o in json.loads((ROOT / "usd/scene/lunar_base/interaction.json").read_text())["objects"].items():
    prim = stage.DefinePrim(f"/World/objects/{name}", "Xform")
    prim.GetReferences().AddReference(str(ROOT / o["usd"][o["usd"].index("usd/"):]))
    api = UsdGeom.XformCommonAPI(prim)
    api.SetTranslate(Gf.Vec3d(*o["pos"]))
    api.SetRotate(Gf.Vec3f(0.0, o["pitch_deg"], o["yaw_deg"]))

sun = UsdLux.DistantLight.Define(stage, "/World/sun")
sun.CreateIntensityAttr(3000)
sun.CreateAngleAttr(0.53)
UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(90 - 12, 0, 180 - 200))
dome = UsdLux.DomeLight.Define(stage, "/World/fill")
dome.CreateIntensityAttr(40)

views = {
    "aerial": ((62, -72, 48), (2, 4, 0)),
    "top": ((0, -0.5, 150), (0, 0, 0)),
    "top_high": ((0, -0.5, 420), (0, 0, 0)),
    "oblique_high": ((230, -260, 170), (0, 0, 0)),
    "seam_grazing": ((130, -20, 25), (80, 30, 0)),
    "ship_pad_north": ((-40, 70, 3.0), (-30, 140, 0)),
    "dock": ((-7, -9, 2.4), (6, 6, 0.8)),
    "power": ((6, -13, 4.5), (27, 8, 1.2)),
    "habitat_comms": ((-40, 2, 7), (-8, 25, 2)),
    "depot_isru": ((22, -38, 7), (-18, -12, 2)),
    "starship": ((-18, 22, 6), (-46, 56, 16)),
    "robot_eye": ((0.9, 0.0, 1.25), (20, 4, 0.3)),
    "pdu_north": ((24, 10, 2.0), (24, 5, 1.1)),
    "pdu_south": ((24, 0, 2.0), (24, 5, 1.1)),
    "tracker_south": ((22, 9, 1.8), (22, 14, 1.2)),
    "tracker_north": ((22, 19, 1.8), (22, 14, 1.2)),
}
for arg in sys.argv[2:]:
    if "=" in arg:
        name, nums = arg.split("=")
        v = [float(x) for x in nums.split(",")]
        views[name] = (tuple(v[:3]), tuple(v[3:]))
wanted = [a.split("=")[0] for a in sys.argv[2:]] or list(views)
cam = rep.create.camera(focal_length=18.0)
rp = rep.create.render_product(cam, (1600, 900))
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach(rp)
for name in wanted:
    eye, target = views[name]
    with cam:
        rep.modify.pose(position=eye, look_at=target)
    for _ in range(40):
        rep.orchestrator.step(rt_subframes=4)
    Image.fromarray(np.asarray(rgb.get_data())[..., :3]).save(out / f"{name}.png")
    print(f"RENDERED {name}", flush=True)
app.close()
