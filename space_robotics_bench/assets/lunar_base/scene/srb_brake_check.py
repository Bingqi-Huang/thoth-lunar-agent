"""LunarBot base braking response on the lunar base terrain: step the velocity command up, hold, step it to
zero, and log body speed against wheel rim speed every control step. Separates drive lag (rim speed decays
slowly with the body) from wheel slip (rim speed drops while the body keeps going) -- the along-track Nav2
overshoot (research repo F47) needs the controller's acceleration model to match whichever it is.

    ~/Programs/isaac-sim/python.sh assets/lunar_base/scene/srb_brake_check.py <out_dir> [v_m_s]
"""

import json
import sys
from pathlib import Path

from srb.core.app import AppLauncher
from srb.utils.path import SRB_APPS_DIR

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
V = float(sys.argv[2]) if len(sys.argv) > 2 else 0.55
launcher = AppLauncher(headless=True, enable_cameras=False, experience=SRB_APPS_DIR.joinpath("srb.headless.kit"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from srb.tasks.mobile_manipulation.lunar_base.task import Task, TaskCfg  # noqa: E402

cfg = TaskCfg()
cfg.scene.num_envs = 1
cfg.__post_init__()
env = Task(cfg)
env.reset()
robot = env._robot
dt = env.step_dt
drive_ids, drive_names = robot.find_joints([".*_steering_to_wheel_joint"])
R_WHEEL = 0.1453
masses = robot.root_physx_view.get_masses()[0].cpu().numpy()
print("MASS total", round(float(masses.sum()), 1), "kg; step_dt", dt, flush=True)
zero = torch.zeros(env.action_space.shape, device=env.device)
zero[:, 3] = 1.0  # arm: hold the joint-position targets that follow
zero[:, 10:17] = robot.data.joint_pos[0, robot.find_joints([f"joint{i}" for i in range(1, 8)], preserve_order=True)[0]]
zero[:, 17] = 1.0


def row(t, v_cmd):
    v = float(np.linalg.norm(robot.data.root_lin_vel_w[0, :2].cpu().numpy()))
    rim = float(robot.data.joint_vel[0, drive_ids].abs().mean().cpu()) * R_WHEEL
    return dict(t=round(t, 3), v_cmd=v_cmd, v_body=round(v, 4), v_rim=round(rim, 4))


rows = []
t = 0.0


def run(v_cmd, seconds):
    global t
    act = zero.clone()
    act[:, 0] = v_cmd
    for _ in range(int(seconds / dt)):
        with torch.inference_mode():
            env.step(act)
        t += dt
        rows.append(row(t, v_cmd))


run(0.0, 1.0)
run(V, 6.0)
p0 = robot.data.root_pos_w[0, :2].cpu().numpy().copy()
run(0.0, 5.0)
p1 = robot.data.root_pos_w[0, :2].cpu().numpy().copy()
stop_dist = float(np.linalg.norm(p1 - p0))
# ramp down at 0.3 m/s^2 for comparison
run(V, 6.0)
p2 = robot.data.root_pos_w[0, :2].cpu().numpy().copy()
v = V
while v > 0:
    v = max(0.0, v - 0.3 * dt)
    run(v, dt)
run(0.0, 3.0)
p3 = robot.data.root_pos_w[0, :2].cpu().numpy().copy()
ramp_dist = float(np.linalg.norm(p3 - p2))
(out / "brake.json").write_text(json.dumps(dict(v=V, step_stop_dist_m=stop_dist, ramp03_stop_dist_m=ramp_dist, rows=rows), indent=0))
# time constant of the step-down: first time body speed falls under V/e after the step
step_rows = [r for r in rows if 7.0 <= r["t"] <= 12.0]
tau = next((r["t"] - 7.0 for r in step_rows if r["v_body"] < V / np.e), None)
peak = max(r["v_body"] for r in rows if 3.0 <= r["t"] <= 7.0)
print("RESULT", json.dumps(dict(v_cmd=V, v_body_steady=round(peak, 3), step_stop_dist_m=round(stop_dist, 3), tau_s=tau,
                                 ramp03_stop_dist_m=round(ramp_dist, 3), ideal_ramp_dist_m=round(V * V / 0.6, 3))), flush=True)
for r in step_rows[::5][:16]:
    print("TRACE", r, flush=True)
env.close()
launcher.app.close()
