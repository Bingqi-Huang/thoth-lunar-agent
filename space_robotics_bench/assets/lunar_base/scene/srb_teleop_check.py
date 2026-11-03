"""Check LunarBot's base command conventions end to end on the lunar_base task, without a GUI.

    ~/Programs/isaac-sim/python.sh assets/lunar_base/scene/srb_teleop_check.py

Keyboard path: the twist IsaacLab's Se3Keyboard produces per key -> srb.__main__.teleop_twist_for_env
(the same call `srb agent teleop` makes) -> action group map_cmd_to_action -> FourWheelSteerAction.
ROS path: a cmd_vel twist -> map_cmd_to_action (as srb/interfaces/interface/ros.py does, no inversion).
Expected (REP-103 in chassis_base_link): W/S drive +X/-X, A/C yaw left (+Z), D/V yaw right.
"""

import json
import math

from srb.core.app import AppLauncher
from srb.utils.path import SRB_APPS_DIR

launcher = AppLauncher(headless=True, enable_cameras=False, experience=SRB_APPS_DIR.joinpath("srb.headless.kit"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz  # noqa: E402

from srb.__main__ import teleop_twist_for_env  # noqa: E402
from srb.core.env import ManipulationEnv  # noqa: E402
from srb.tasks.mobile_manipulation.lunar_base.task import Task, TaskCfg  # noqa: E402

cfg = TaskCfg()
cfg.scene.num_envs = 1
cfg.__post_init__()
env = Task(cfg)
env.reset()
robot = env._robot

# The live action group behind map_cmd_to_action (same closure walk as srb.__main__.teleop_agent).
map_fn = env.cfg.actions.map_cmd_to_action
targets = []
for cell in getattr(map_fn, "__closure__", None) or ():
    val = cell.cell_contents
    for fn in val if isinstance(val, list) else ():
        obj = getattr(fn, "__self__", None)
        if obj is not None and hasattr(obj, "toggle_mode") and obj not in targets:
            targets.append(obj)
print("TELEOP TARGETS", [type(t).__name__ for t in targets], [t.teleop_mode for t in targets], flush=True)
invert_controls = isinstance(env, ManipulationEnv)  # srb agent teleop: --invert_controls defaults to True

POS, ROT = 1.0, 3.1415927  # srb agent teleop defaults (--pos_sensitivity, --rot_sensitivity), gain 1.0
KEYS = {
    "W": [POS, 0, 0, 0, 0, 0], "S": [-POS, 0, 0, 0, 0, 0],
    "A": [0, POS, 0, 0, 0, 0], "D": [0, -POS, 0, 0, 0, 0],
    "C": [0, 0, 0, 0, 0, ROT], "V": [0, 0, 0, 0, 0, -ROT],
}


def pose():
    p = robot.data.root_pos_w[0].cpu().numpy()
    yaw = float(euler_xyz_from_quat(robot.data.root_quat_w[0:1])[2][0])
    return p, yaw


def reset_pose():
    q = quat_from_euler_xyz(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0])).to(env.device)
    robot.write_root_pose_to_sim(torch.cat([env.scene.env_origins[0:1] + torch.tensor([[0.0, 0.0, 0.06]], device=env.device), q], 1))
    robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))
    idle = env.cfg.actions.map_cmd_to_action(torch.zeros(6, device=env.device), False).unsqueeze(0)
    for _ in range(int(1.5 / env.step_dt)):
        env.step(idle)


def hold(twist_np, seconds=2.5):
    reset_pose()
    p0, y0 = pose()
    act = env.cfg.actions.map_cmd_to_action(torch.tensor(twist_np, dtype=torch.float32, device=env.device), False).unsqueeze(0)
    yaw_total, yaw_prev = 0.0, y0
    for _ in range(int(seconds / env.step_dt)):
        env.step(act)
        _, y = pose()
        yaw_total += (y - yaw_prev + math.pi) % (2 * math.pi) - math.pi
        yaw_prev = y
    p1, _ = pose()
    d = p1[:2] - p0[:2]
    body_x = float(np.cos(y0) * d[0] + np.sin(y0) * d[1])  # displacement along the chassis +X at the start
    return dict(base_cmd=[round(float(act[0, 0]), 3), round(float(act[0, 1]), 3)], body_dx_m=round(body_x, 3),
                yaw_change_deg=round(math.degrees(yaw_total), 1))


results = {}
with torch.inference_mode():
    for key, raw in KEYS.items():
        twist = teleop_twist_for_env(np.array(raw, dtype=np.float32), invert_controls, targets)
        results[f"keyboard_{key}"] = hold(twist)
    results["ros_cmd_vel_forward_left"] = hold(np.array([0.5, 0, 0, 0, 0, 0.3], dtype=np.float32))
    results["ros_cmd_vel_reverse_right"] = hold(np.array([-0.5, 0, 0, 0, 0, -0.3], dtype=np.float32))

expect = {"keyboard_W": (1, 0), "keyboard_S": (-1, 0), "keyboard_A": (0, 1), "keyboard_D": (0, -1),
          "keyboard_C": (0, 1), "keyboard_V": (0, -1), "ros_cmd_vel_forward_left": (1, 1), "ros_cmd_vel_reverse_right": (-1, 1)}
# reverse with -omega: yaw rate is negative, but travelling backwards; check yaw sign against the command
expect["ros_cmd_vel_reverse_right"] = (-1, -1)
ok = True
for name, r in results.items():
    ex, ey = expect[name]
    good = (ex == 0 or np.sign(r["body_dx_m"]) == ex and abs(r["body_dx_m"]) > 0.2) and (ey == 0 or np.sign(r["yaw_change_deg"]) == ey and abs(r["yaw_change_deg"]) > 10)
    if ex == 0:
        good = good and abs(r["body_dx_m"]) < 0.3
    ok &= bool(good)
    print("CHECK", name, "OK" if good else "WRONG", json.dumps(r), flush=True)
print("TELEOP_CHECK", "PASS" if ok else "FAIL", flush=True)
env.close()
launcher.app.close()
