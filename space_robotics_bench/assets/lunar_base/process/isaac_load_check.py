"""Load every processed USD in Isaac Sim: physics settle trace + RTX overview renders.

    ~/Programs/isaac-sim/python.sh process/isaac_load_check.py <out_dir>
"""

import json
import sys
from pathlib import Path

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "width": 1600, "height": 900})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import SimulationContext  # noqa: E402
from isaacsim.core.prims import RigidPrim  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)

sim = SimulationContext(physics_dt=1 / 120, rendering_dt=1 / 30, stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
scene.CreateGravityMagnitudeAttr(1.62)

ground = UsdGeom.Mesh.Define(stage, "/World/ground")
s = 200.0
ground.CreatePointsAttr([(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)])
ground.CreateFaceVertexCountsAttr([4])
ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
ground.CreateDisplayColorAttr([(0.33, 0.32, 0.31)])
UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

sun = UsdLux.DistantLight.Define(stage, "/World/sun")
sun.CreateIntensityAttr(3000)
sun.CreateAngleAttr(0.53)
UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(78, 0, 135))
dome = UsdLux.DomeLight.Define(stage, "/World/fill")
dome.CreateIntensityAttr(60)

usds = sorted(p for p in (ROOT / "usd").glob("*/*/*.usdc") if p.stem != "starship_hls")
placed = {}
cols = 5
for i, p in enumerate(usds):
    x, y = (i % cols) * 12.0, (i // cols) * 12.0
    prim = stage.DefinePrim(f"/World/assets/{p.stem}", "Xform")
    prim.GetReferences().AddReference(str(p))
    dynamic = prim.HasAPI(UsdPhysics.RigidBodyAPI)
    UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(x, y, 0.02 if dynamic else 0.0))
    placed[p.stem] = dict(xy=[x, y], dynamic=dynamic)
ship = stage.DefinePrim("/World/assets/starship_hls", "Xform")
ship.GetReferences().AddReference(str(ROOT / "usd/landers/starship_hls/starship_hls.usdc"))
UsdGeom.XformCommonAPI(ship).SetTranslate(Gf.Vec3d(80.0, 18.0, 0.0))

sim.initialize_physics()
dyn_names = [n for n, v in placed.items() if v["dynamic"]]
bodies = RigidPrim([f"/World/assets/{n}" for n in dyn_names], name="dynamic_bodies")
sim.play()
trace = []
for step in range(600):
    sim.step(render=False)
    if step % 20 == 0:
        pos, _ = bodies.get_world_poses()
        vel = bodies.get_linear_velocities()
        trace.append(dict(step=step, z=np.round(pos[:, 2].cpu().numpy() if hasattr(pos, "cpu") else pos[:, 2], 4).tolist(),
                          speed=np.round(np.linalg.norm(vel.cpu().numpy() if hasattr(vel, "cpu") else vel, axis=1), 4).tolist()))
(out / "settle_trace.json").write_text(json.dumps(dict(bodies=dyn_names, trace=trace), indent=1))

views = {
    "overview": ((-18, -22, 22), (24, 18, 1)),
    "power_row": ((10, -10, 5), (24, 12, 1.5)),
    "vehicles_logistics": ((40, 20, 6), (24, 36, 1)),
    "starship": ((40, -30, 18), (80, 18, 20)),
}
rp_cam = rep.create.camera()
rp = rep.create.render_product(rp_cam, (1600, 900))
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach(rp)
for name, (eye, target) in views.items():
    with rp_cam:
        rep.modify.pose(position=eye, look_at=target)
    for _ in range(40):  # RTX accumulation / first-run shader warmup
        rep.orchestrator.step(rt_subframes=4)
    Image.fromarray(np.asarray(rgb.get_data())[..., :3]).save(out / f"{name}.png")
    print(f"RENDERED {name}", flush=True)

print("LOADCHECK dynamic bodies:", dyn_names, flush=True)
print("LOADCHECK final z:", trace[-1]["z"], "speed:", trace[-1]["speed"], flush=True)
app.close()
