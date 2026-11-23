"""Checks on the lunar_base task (observable-state and reach renders).

    ~/Programs/isaac-sim/python.sh assets/lunar_base/scene/srb_settle_check.py <out_dir> [--no-render]

1. Settle traces: every dynamic object, 3 resets, per-tick pose/velocity for 3 s after each reset.
2. Support check: spare crate moved onto pdu_1's shelf and the spare ORU onto pdu_2's shelf, settled.
3. Status lamps: pdu_2 red, pdu_3 amber (runtime recolour).
4. Renders: panel with cover + lamps, depot stand with spares, shelves loaded, robot at work poses.
Pass: after 0.2 s no object faster than 0.05 m/s, settled drift <= 2 cm, tilt change <= 5 deg.
"""

import json
import math
import sys
from pathlib import Path

from srb.core.app import AppLauncher
from srb.utils.path import SRB_APPS_DIR

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
RENDER = "--no-render" not in sys.argv
launcher = AppLauncher(headless=True, enable_cameras=RENDER,
                       experience=SRB_APPS_DIR.joinpath("srb.headless.rendering.kit" if RENDER else "srb.headless.kit"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz  # noqa: E402

from srb.assets.scenery.lunar_base import OBJECT_PREFIX, load_interaction, set_status_lamp  # noqa: E402
from srb.tasks.mobile_manipulation.lunar_base.task import Task, TaskCfg  # noqa: E402

cfg = TaskCfg()
cfg.scene.num_envs = 1
cfg.__post_init__()
env = Task(cfg)
ix = load_interaction()
names = list(ix["objects"])
bodies = {n: env.scene[OBJECT_PREFIX + n] for n in names}
origin = env.scene.env_origins[0]
dt = env.step_dt
idle = torch.zeros(env.action_space.shape, device=env.device)
report = dict(resets=[], shelves=None, lamps=None, verdict=None)


def tilt_deg(q):
    r, p, _ = euler_xyz_from_quat(q)
    return float(math.degrees(math.acos(max(-1.0, min(1.0, math.cos(float(r[0])) * math.cos(float(p[0])))))))


def settle(label, seconds=3.0, watch=None):
    watch = watch or names
    start = {n: (bodies[n].data.root_pos_w[0].clone(), tilt_deg(bodies[n].data.root_quat_w[0:1])) for n in watch}
    peak = {n: 0.0 for n in watch}
    first_still = {n: None for n in watch}
    with torch.inference_mode():
        for k in range(int(seconds / dt)):
            env.step(idle)
            t = (k + 1) * dt
            for n in watch:
                v = float(torch.linalg.norm(bodies[n].data.root_lin_vel_w[0]))
                if t > 0.2:
                    peak[n] = max(peak[n], v)
                if v < 0.01 and first_still[n] is None:
                    first_still[n] = round(t, 2)
                elif v >= 0.01:
                    first_still[n] = None
    res = {}
    for n in watch:
        p0, t0 = start[n]
        p1 = bodies[n].data.root_pos_w[0]
        res[n] = dict(peak_speed_after_0p2s=round(peak[n], 4), still_from_s=first_still[n],
                      drift_m=round(float(torch.linalg.norm(p1 - p0)), 4), dz_m=round(float(p1[2] - p0[2]), 4),
                      tilt_change_deg=round(abs(tilt_deg(bodies[n].data.root_quat_w[0:1]) - t0), 2),
                      final_pos=[round(float(c), 3) for c in (p1 - origin)])
    print("SETTLE", label, json.dumps(res), flush=True)
    return res


def ok(res):
    return all(r["peak_speed_after_0p2s"] <= 0.05 and r["drift_m"] <= 0.02 + 0.006 and r["tilt_change_deg"] <= 5.0 for r in res.values())


# 1. resets
for i in range(3):
    with torch.inference_mode():
        env.reset()
    report["resets"].append(settle(f"reset_{i}"))

# 2. shelves: move the spares onto pdu_1 / pdu_2 service shelves (yaw so the shorter side spans the 0.42 m depth)
def place_on(obj, point, yaw_rel_deg, lift=0.005):
    pt = ix["points"][point]
    yaw = math.atan2(pt["normal"][1], pt["normal"][0]) + math.radians(yaw_rel_deg)
    q = quat_from_euler_xyz(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([yaw])).to(env.device)
    pos = torch.tensor([[pt["xyz"][0], pt["xyz"][1], pt["xyz"][2] + lift]], device=env.device) + origin
    bodies[obj].write_root_pose_to_sim(torch.cat([pos, q], 1))
    bodies[obj].write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))


with torch.inference_mode():
    place_on("spare_crate", "pdu_1_shelf/shelf", 0.0)
    place_on("spare_oru", "pdu_2_shelf/shelf", 90.0)
report["shelves"] = settle("on_pdu_shelves", watch=["spare_crate", "spare_oru"])

# 3. lamps
set_status_lamp([0], "pdu_2_panel", (1.0, 0.0, 0.0))
set_status_lamp([0], "pdu_3_panel", (1.0, 0.5, 0.0))
report["lamps"] = dict(pdu_1_panel="green (authored)", pdu_2_panel="red", pdu_3_panel="amber")

passed = all(ok(r) for r in report["resets"]) and ok(report["shelves"])
report["verdict"] = "PASS" if passed else "FAIL"
print("C3_SETTLE", report["verdict"], flush=True)

# 4. renders
if RENDER:
    import omni.replicator.core as rep
    import omni.usd
    from PIL import Image
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    cam = UsdGeom.Camera.Define(stage, "/World/review_camera")
    cam.CreateFocalLengthAttr(18.0)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.02, 20000.0))
    rp = rep.create.render_product("/World/review_camera", (1280, 800))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach(rp)
    robot = env._robot
    o = origin.cpu().numpy()

    def shoot(name, eye, target):
        m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, o + np.array(eye))), Gf.Vec3d(*map(float, o + np.array(target))), Gf.Vec3d(0, 0, 1)).GetInverse()
        xf = UsdGeom.Xformable(cam)
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(m)
        for _ in range(40):
            env.sim.render()
        Image.fromarray(np.asarray(rgb.get_data())[..., :3]).save(out / f"{name}.png")
        print("RENDERED", name, flush=True)

    def park(pose_key, table):
        wp = ix[table][pose_key]
        yaw = math.radians(wp["yaw_deg"])
        with torch.inference_mode():
            q = quat_from_euler_xyz(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([yaw])).to(env.device)
            robot.write_root_pose_to_sim(torch.cat([torch.tensor([[wp["xy"][0], wp["xy"][1], 0.05]], device=env.device) + origin, q], 1))
            robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))
            for _ in range(int(1.0 / dt)):
                env.step(idle)

    press = ix["points"]["pdu_1_panel/press"]["xyz"]
    press2 = ix["points"]["pdu_2_panel/press"]["xyz"]
    press3 = ix["points"]["pdu_3_panel/press"]["xyz"]
    shelf1 = ix["points"]["pdu_1_shelf/shelf"]["xyz"]
    slot0 = ix["points"]["depot_stand/slot_0"]["xyz"]
    shoot("pdu_panels_lamps", (press[0] - 1.2, press[1] - 3.2, 1.6), (press[0] + 4.0, press[1], 0.9))
    shoot("pdu_1_panel_cover_closeup", (press[0] - 0.7, press[1] - 1.1, 1.2), (press[0], press[1], 0.8))
    shoot("pdu_2_lamp_red_closeup", (press2[0] - 0.6, press2[1] - 1.0, 1.1), (press2[0], press2[1], 0.9))
    shoot("pdu_3_lamp_amber_closeup", (press3[0] - 0.6, press3[1] - 1.0, 1.1), (press3[0], press3[1], 0.9))
    shoot("pdu_1_shelf_loaded", (shelf1[0] - 0.4, shelf1[1] - 1.4, 1.2), (shelf1[0], shelf1[1], 0.55))
    with torch.inference_mode():
        env.reset()
        for _ in range(int(1.0 / dt)):
            env.step(idle)
    shoot("depot_stand_spares", (slot0[0] - 1.8, slot0[1] + 1.8, 1.5), (slot0[0] - 0.2, slot0[1], 0.6))
    park("pdu_1_panel/press", "work_poses")
    shoot("work_pose_press_pdu_1", (press[0] + 2.2, press[1] - 2.0, 1.8), (press[0], press[1] - 0.8, 0.7))
    park("depot_stand/slot_0", "work_poses")
    shoot("work_pose_pick_depot", (slot0[0] + 2.5, slot0[1] + 1.2, 1.8), (slot0[0], slot0[1] + 0.8, 0.6))
    park("pdu_1", "view_poses")
    vp = ix["view_poses"]["pdu_1"]
    yaw = math.radians(vp["yaw_deg"])
    cam_xy = np.array(vp["xy"]) + 0.75 * np.array([math.cos(yaw), math.sin(yaw)])  # ahead of the front mast
    shoot("view_pose_pdu_1_robot_eye", (cam_xy[0], cam_xy[1], 0.95), vp["target"])

(out / "c3_settle.json").write_text(json.dumps(report, indent=1))
env.close()
launcher.app.close()
