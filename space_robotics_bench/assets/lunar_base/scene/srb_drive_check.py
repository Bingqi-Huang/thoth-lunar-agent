"""LunarBot base at the kernel's speed caps on the lunar base terrain: straight, stop, spin, arc, and
the access road into the landing pad; checks the BODY_TWIST sign convention (+v toward chassis +X,
+omega about +Z).

    ~/Programs/isaac-sim/python.sh assets/lunar_base/scene/srb_drive_check.py <out_dir> [v_m_s] [w_rad_s] [routes|all]

By default only the first reference route runs (a quick check); pass comma-separated route names or "all".

Commands go straight to the env (raw cmd_vel = m/s, rad/s for four_wheel_steer); the kernel bounds are
openral robots/lunar_bot/robot.yaml safety.max_base_*.
"""

import json
import math
import sys
from pathlib import Path

from srb.core.app import AppLauncher
from srb.utils.path import SRB_APPS_DIR

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
V = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
W = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
launcher = AppLauncher(headless=True, enable_cameras=False, experience=SRB_APPS_DIR.joinpath("srb.headless.kit"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omni.physx import acquire_physx_interface  # noqa: E402

acquire_physx_interface().overwrite_gpu_setting(1)

from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz  # noqa: E402

from srb.tasks.mobile_manipulation.lunar_base.task import Task, TaskCfg  # noqa: E402

cfg = TaskCfg()
cfg.scene.num_envs = 1
cfg.__post_init__()
env = Task(cfg)
env.reset()
robot = env._robot
dt = env.step_dt
zero = torch.zeros(env.action_space.shape, device=env.device)


def state():
    p = robot.data.root_pos_w[0].cpu().numpy()
    r, pi, y = (float(a[0]) for a in euler_xyz_from_quat(robot.data.root_quat_w[0:1]))
    v = robot.data.root_lin_vel_w[0].cpu().numpy()
    w = float(robot.data.root_ang_vel_w[0, 2])
    wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi  # noqa: E731
    return dict(pos=p, roll=math.degrees(wrap(r)), pitch=math.degrees(wrap(pi)), yaw=y, speed=float(np.linalg.norm(v[:2])), wz=w)


def run(name, v, w, seconds, record):
    act = zero.clone()
    act[:, 0], act[:, 2] = v, w
    s0 = state()
    rows, max_tilt = [], 0.0
    yaw_prev, yaw_total = s0["yaw"], 0.0
    with torch.inference_mode():
        for k in range(int(round(seconds / dt))):
            env.step(act)
            s = state()
            d = (s["yaw"] - yaw_prev + math.pi) % (2 * math.pi) - math.pi
            yaw_total, yaw_prev = yaw_total + d, s["yaw"]
            max_tilt = max(max_tilt, abs(s["roll"]), abs(s["pitch"]))
            if k % int(round(0.5 / dt)) == 0:
                rows.append(dict(t=round((k + 1) * dt, 2), pos=np.round(s["pos"], 3).tolist(), speed=round(s["speed"], 3),
                                 wz=round(s["wz"], 3), roll=round(s["roll"], 2), pitch=round(s["pitch"], 2)))
    s1 = state()
    tail = rows[len(rows) // 2 :]
    res = dict(
        segment=name, cmd=[v, w], seconds=seconds,
        displacement_m=round(float(np.linalg.norm(s1["pos"][:2] - s0["pos"][:2])), 3),
        dz_m=round(float(s1["pos"][2] - s0["pos"][2]), 3),
        yaw_change_deg=round(math.degrees(yaw_total), 1),
        steady_speed_m_s=round(float(np.mean([r["speed"] for r in tail])), 3) if tail else None,
        steady_wz_rad_s=round(float(np.mean([r["wz"] for r in tail])), 3) if tail else None,
        final_speed_m_s=round(s1["speed"], 3), max_tilt_deg=round(max_tilt, 2),
        final_pos=np.round(s1["pos"], 3).tolist(),
        # direction of travel relative to the chassis +X axis at the segment start (0 = forward, 180 = backward)
        travel_vs_chassis_x_deg=round(math.degrees(abs((math.atan2(*(s1["pos"][:2] - s0["pos"][:2])[::-1]) - s0["yaw"] + math.pi)
                                                         % (2 * math.pi) - math.pi)), 1)
        if np.linalg.norm(s1["pos"][:2] - s0["pos"][:2]) > 0.3 else None,
    )
    record.append(dict(result=res, trace=rows))
    print("SEGMENT", json.dumps(res), flush=True)
    return res


def teleport(x, y, heading_rad):
    """Place the robot with its chassis +X (driving direction) along heading_rad."""
    yaw = heading_rad
    q = quat_from_euler_xyz(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([yaw])).to(env.device)
    pose = torch.cat([torch.tensor([[x, y, 0.08]], device=env.device) + env.scene.env_origins[0:1], q], dim=1)
    robot.write_root_pose_to_sim(pose)
    robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))


record = []
with torch.inference_mode():
    for _ in range(100):
        env.step(zero)
run("settle", 0.0, 0.0, 1.0, record)
run("straight", V, 0.0, 8.0, record)            # west from the dock across the platform
run("stop", 0.0, 0.0, 3.0, record)
run("spin", 0.0, W, math.pi / W, record)        # half turn in place
run("stop_after_spin", 0.0, 0.0, 2.0, record)
run("arc", V, 0.6 * W, 6.0, record)
run("stop_after_arc", 0.0, 0.0, 3.0, record)


def follow(name, waypoints, record, lookahead=1.2, timeout_s=240.0):
    """Closed-loop waypoint following (heading P-control, like a minimal Nav2 DiffDrive controller)."""
    import time

    pts = [np.array(p, float) for p in waypoints]
    i, t, dist, max_tilt, slow = 1, 0.0, 0.0, 0.0, 0.0
    prev = state()["pos"][:2].copy()
    trace = []
    wall0 = time.time()
    with torch.inference_mode():
        while i < len(pts) and t < timeout_s:
            s = state()
            pos = s["pos"][:2]
            if np.linalg.norm(pts[i] - pos) < (0.4 if i == len(pts) - 1 else lookahead):
                i += 1
                continue
            err = (math.atan2(*(pts[i] - pos)[::-1]) - s["yaw"] + math.pi) % (2 * math.pi) - math.pi
            act = zero.clone()
            act[:, 1] = float(np.clip(1.5 * err, -W, W))
            act[:, 0] = V * max(0.0, math.cos(err)) ** 2
            env.step(act)
            t += dt
            s = state()
            dist += float(np.linalg.norm(s["pos"][:2] - prev))
            prev = s["pos"][:2].copy()
            max_tilt = max(max_tilt, abs(s["roll"]), abs(s["pitch"]))
            slow += dt if s["speed"] < 0.2 else 0.0
            if int(round(t / dt)) % int(round(0.25 / dt)) == 0:
                trace.append([round(t, 2), *np.round(s["pos"][:2], 2).tolist(), round(math.degrees(s["yaw"]), 1),
                              round(math.degrees(err), 1), round(float(act[0, 0]), 2), round(float(act[0, 1]), 2),
                              round(s["speed"], 2), round(s["wz"], 2)])
    for _ in range(int(2.0 / dt)):
        env.step(zero)
    s = state()
    res = dict(segment=name, arrived=i >= len(pts), sim_time_s=round(t, 1), wall_time_s=round(time.time() - wall0, 1),
               path_m=round(dist, 1), mean_speed_m_s=round(dist / max(t, 1e-6), 3), time_below_0p2_m_s=round(slow, 1),
               max_tilt_deg=round(max_tilt, 2), final_pos=np.round(s["pos"], 3).tolist(), goal=pts[-1].tolist())
    record.append(dict(result=res, trace_cols="t x y yaw_deg err_deg cmd_v cmd_w speed wz", trace=trace))
    print("ROUTE", json.dumps(res), flush=True)


# Reference routes along the road network (layout.yaml missions.routes).
import yaml  # noqa: E402

routes = yaml.safe_load(open(Path(__file__).resolve().parent / "layout.yaml"))["missions"]["routes"]
with torch.inference_mode():
    teleport(0.0, 0.0, 0.0)
    for _ in range(150):
        env.step(zero)
wanted = sys.argv[4].split(",") if len(sys.argv) > 4 and sys.argv[4] != "all" else None
if len(sys.argv) <= 4:
    wanted = [next(iter(routes))]
for rname, wps in routes.items():
    if wanted is None or rname in wanted:
        follow(rname, wps, record)

(out / "drive_check.json").write_text(json.dumps(dict(v=V, w=W, dt=dt, segments=record), indent=1))
env.close()
launcher.app.close()
