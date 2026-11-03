"""Bring up the `lunar_base` SRB task with LunarBot and check it: robot settles on the dock,
drives on the terrain, the lidar sees the structures; then RTX review renders from the env.

    ~/Programs/isaac-sim/python.sh assets/lunar_base/scene/srb_scene_check.py <out_dir>
"""

import json
import sys
from pathlib import Path

from srb.core.app import AppLauncher
from srb.utils.path import SRB_APPS_DIR

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
launcher = AppLauncher(headless=True, enable_cameras=True, experience=SRB_APPS_DIR.joinpath("srb.headless.rendering.kit"))

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import torch  # noqa: E402
from omni.physx import acquire_physx_interface  # noqa: E402
from PIL import Image  # noqa: E402

acquire_physx_interface().overwrite_gpu_setting(1)

from srb.tasks.mobile_manipulation.lunar_base.task import Task, TaskCfg  # noqa: E402

cfg = TaskCfg()
cfg.scene.num_envs = 1
cfg.__post_init__()
env = Task(cfg)
env.reset()
robot, lidar = env._robot, env._lidar_robot
terms = {n: env.action_manager.get_term(n).action_dim for n in env.action_manager.active_terms}
print("ACTION TERMS", terms, flush=True)


def snapshot(step):
    pos = robot.data.root_pos_w[0].cpu().numpy()
    vel = robot.data.root_lin_vel_w[0].cpu().numpy()
    rec = dict(step=step, pos=np.round(pos, 3).tolist(), speed=round(float(np.linalg.norm(vel)), 3))
    if lidar is not None:
        hits = lidar.data.ray_hits_w[0].cpu().numpy()
        ok = np.isfinite(hits).all(1)
        rel = hits[ok] - lidar.data.pos_w[0].cpu().numpy()
        dist = np.linalg.norm(rel, axis=1)
        above = hits[ok][:, 2] > 0.3  # higher than the graded platform + wheel bumps: structures
        rec.update(lidar_rays=int(len(hits)), lidar_hits=int(ok.sum()), lidar_hits_above_0p3m=int(above.sum()),
                   lidar_structure_range_m=[round(float(dist[above].min()), 2), round(float(dist[above].max()), 2)] if above.any() else None)
    return rec


def lidar_diagnostics():
    """Scan geometry in the world, returns near the dock, and the sensor-frame cloud's round trip."""
    from isaaclab.utils.math import quat_apply

    n = lidar.num_rays
    q = lidar.data.quat_w[0:1].repeat(n, 1)
    dirs = quat_apply(q, lidar.ray_directions[0]).cpu().numpy()
    elev = np.degrees(np.arcsin(np.clip(dirs[:, 2], -1, 1)))
    hits_w = lidar.data.ray_hits_w[0]
    src = lidar.data.pos_w[0].cpu().numpy()
    hits_s = lidar.ray_hits_sensor_frame()[0]
    ok = torch.isfinite(hits_w).all(1)
    back = quat_apply(q[ok], hits_s[ok]) + lidar.data.pos_w[0]
    roundtrip = float((back - hits_w[ok]).abs().max())
    hw = hits_w.cpu().numpy()
    okn = ok.cpu().numpy()
    ground = float(robot.data.root_pos_w[0, 2])
    band = okn & (hw[:, 2] > ground + 0.05) & (hw[:, 2] < ground + 0.6)  # Nav2 pointcloud_to_laserscan band
    horiz = np.linalg.norm(hw[band, :2] - src[:2], axis=1)
    south = okn & (dirs[:, 1] < -0.9) & (np.abs(elev) < 7.5)
    rng = np.linalg.norm(hw[south] - src, axis=1)
    return dict(
        lidar_pos=np.round(src, 3).tolist(), lidar_quat=np.round(lidar.data.quat_w[0].cpu().numpy(), 3).tolist(),
        elevation_deg=[round(float(elev.min()), 1), round(float(elev.max()), 1)],
        blind_cone_ground_radius_m=round(float((src[2] - ground) / np.tan(np.radians(-elev.min()))), 2),
        sensor_frame_roundtrip_err_m=roundtrip,
        nav2_band_points=int(band.sum()), nav2_band_nearest_m=round(float(horiz.min()), 2) if band.any() else None,
        south_near_horizontal_hits=int(south.sum()), south_nearest_m=np.round(np.sort(rng)[:5], 2).tolist(),
    )


trace = []
zero = torch.zeros(env.action_space.shape, device=env.device)
with torch.inference_mode():
    for step in range(240):  # settle on the dock
        env.step(zero)
        if step % 20 == 0 or step == 239:
            trace.append(snapshot(step))
    diag = lidar_diagnostics() if lidar is not None else None
    print("LIDAR", json.dumps(diag), flush=True)
    drive = zero.clone()
    drive[:, 0] = 0.6  # first cmd_vel channel: forward
    for step in range(240, 480):  # drive
        env.step(drive)
        if step % 40 == 0 or step == 479:
            trace.append(snapshot(step))
    for step in range(480, 560):  # stop
        env.step(zero)
    trace.append(snapshot(559))
(out / "trace.json").write_text(json.dumps(dict(action_terms=terms, lidar=diag, trace=trace), indent=1))
for r in trace:
    print("TRACE", json.dumps(r), flush=True)

origin = env.scene.env_origins[0].cpu().numpy()
views = {
    "srb_aerial": ((62, -72, 48), (2, 4, 0)),
    "srb_dock": ((-6, -7, 2.6), (2, 1, 0.6)),
    "srb_robot_follow": ("robot", (-5.0, -3.0, 2.5)),
    "srb_lidar_mount": ("lidar", (1.1, -0.9, 0.35)),
}
# Replicator's orchestrator.step() hangs inside a running IsaacLab env; move a plain USD camera and
# pump env.sim.render() instead, reading the annotator after the RTX accumulation settles.
import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

stage = omni.usd.get_context().get_stage()
cam = UsdGeom.Camera.Define(stage, "/World/review_camera")
cam.CreateFocalLengthAttr(18.0)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 20000.0))
rp = rep.create.render_product("/World/review_camera", (1600, 900))
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach(rp)
rpos = robot.data.root_pos_w[0].cpu().numpy()
for name, (eye, target) in views.items():
    if eye == "lidar":
        lpos = lidar.data.pos_w[0].cpu().numpy()
        eye, target = lpos + np.array(target), lpos
    elif isinstance(eye, str):
        eye, target = rpos + np.array(target), rpos + np.array((0, 0, 0.6))
    else:
        eye, target = origin + np.array(eye), origin + np.array(target)
    m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, target)), Gf.Vec3d(0, 0, 1)).GetInverse()
    xf = UsdGeom.Xformable(cam)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(m)
    for _ in range(60):
        env.sim.render()
    Image.fromarray(np.asarray(rgb.get_data())[..., :3]).save(out / f"{name}.png")
    print(f"RENDERED {name}", flush=True)
env.close()
launcher.app.close()
