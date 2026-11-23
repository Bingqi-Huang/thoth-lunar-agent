"""LunarBot in-place turn response on the lunar base terrain: command (v=0, w) steps and log yaw rate and the
steering angles every control step. The docking servo turned only ~0.6 deg/s at a commanded 0.3 rad/s; this
separates the base's own turning ability from anything upstream of SRB's cmd_vel.

    ~/Programs/isaac-sim/python.sh assets/lunar_base/scene/srb_turn_check.py <out_dir>
"""
import json
import math
import sys
from pathlib import Path

from srb.core.app import AppLauncher
from srb.utils.path import SRB_APPS_DIR

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
launcher = AppLauncher(headless=True, enable_cameras=False, experience=SRB_APPS_DIR.joinpath("srb.headless.kit"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import euler_xyz_from_quat  # noqa: E402

from srb.tasks.mobile_manipulation.lunar_base.task import Task, TaskCfg  # noqa: E402

cfg = TaskCfg(); cfg.scene.num_envs = 1; cfg.__post_init__()
env = Task(cfg); env.reset()
robot = env._robot; dt = env.step_dt
steer_ids, _ = robot.find_joints(["chassis_to_.*_steering_joint"])
zero = torch.zeros(env.action_space.shape, device=env.device)
zero[:, 3] = 1.0
zero[:, 10:17] = robot.data.joint_pos[0, robot.find_joints([f"joint{i}" for i in range(1, 8)], preserve_order=True)[0]]
zero[:, 17] = 1.0
rows = []; t = 0.0


def yaw():
    return float(euler_xyz_from_quat(robot.data.root_quat_w[0:1])[2][0])


def run(v, w, seconds):
    global t
    a = zero.clone(); a[:, 0], a[:, 2] = v, w
    y0 = yaw(); t0 = t
    for _ in range(int(seconds / dt)):
        with torch.inference_mode():
            env.step(a)
        t += dt
        rows.append(dict(t=round(t, 2), w_cmd=w, wz=round(float(robot.data.root_ang_vel_w[0, 2]), 3),
                         steer_deg=[round(math.degrees(float(s)), 1) for s in robot.data.joint_pos[0, steer_ids]]))
    dy = (yaw() - y0 + math.pi) % (2 * math.pi) - math.pi
    print("RESULT", json.dumps(dict(v=v, w_cmd=w, seconds=seconds, yaw_change_deg=round(math.degrees(dy), 1), mean_wz=round(dy / seconds, 3))), flush=True)


run(0.0, 0.0, 1.0)
run(0.0, 0.3, 8.0)
run(0.0, 0.0, 2.0)
run(0.0, -0.5, 8.0)
run(0.0, 0.0, 2.0)
run(0.15, 0.3, 6.0)
(out / "turn.json").write_text(json.dumps(rows))
for r in rows[50:450:25]:
    print("TRACE", r, flush=True)
env.close(); launcher.app.close()
