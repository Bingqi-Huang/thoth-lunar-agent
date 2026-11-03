"""Does SRB's switchable arm term rotate the tool on a pure angular IK twist? (D0-3 diagnosis)

Drives LunarBot's arm to READY in joint mode, then sends IK-mode (mode flag 0) commands: 2 s of pure
angular twist about the chassis -X axis (the tool axis at READY), then 2 s of pure linear +Z, then 3 s
of zero twist (hold). Logs Link7 rotation / TCP position every 0.5 s.

    CUDA_VISIBLE_DEVICES=1 ~/Programs/isaac-sim/python.sh assets/lunar_base/scene/srb_arm_twist_check.py <out_dir>
"""
import json
import math
import sys
from pathlib import Path

from srb.core.app import AppLauncher
from srb.utils.path import SRB_APPS_DIR

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
launcher = AppLauncher(headless=True, enable_cameras=False, experience=SRB_APPS_DIR.joinpath("srb.headless.kit"),
                       kit_args="--/renderer/activeGpu=1 --/physics/cudaDevice=1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import matrix_from_quat  # noqa: E402

from srb.tasks.mobile_manipulation.lunar_base.task import Task, TaskCfg  # noqa: E402

cfg = TaskCfg()
cfg.scene.num_envs = 1
cfg.__post_init__()
env = Task(cfg)
env.reset()
robot = env._robot
dt = env.step_dt
ARM, _ = robot.find_joints([f"joint{i}" for i in range(1, 8)], preserve_order=True)
L7 = robot.find_bodies("Link7")[0][0]
READY = torch.tensor([0.0, -1.0, 0.0, -1.4, 0.0, 1.2, 0.0], device=env.device)


def pose():
    Rb = matrix_from_quat(robot.data.root_quat_w[0:1])[0]
    R7 = matrix_from_quat(robot.data.body_quat_w[0:1, L7])[0]
    p7 = robot.data.body_pos_w[0, L7] - robot.data.root_pos_w[0]
    Rc = Rb.T @ R7
    pc = Rb.T @ p7 + Rc @ torch.tensor([0.0, 0.0, 0.14], device=env.device)
    return pc.cpu().numpy(), Rc.cpu().numpy()


def step(mode, ik=(0,) * 6, q=None, n=1):
    a = torch.zeros(env.action_space.shape, device=env.device)
    a[0, 3] = mode
    a[0, 4:10] = torch.tensor(ik, device=env.device, dtype=torch.float32)
    a[0, 10:17] = READY if q is None else q
    a[0, 17] = 1.0
    for _ in range(n):
        with torch.inference_mode():
            env.step(a)


def report(tag):
    p, R = pose()
    q7 = robot.data.joint_pos[0, ARM].cpu().numpy()
    print("POSE", tag, "tcp", np.round(p, 4).tolist(), "tool_z", np.round(R[:, 2], 3).tolist(), "jaw_x", np.round(R[:, 0], 3).tolist(),
          "q", np.round(q7, 3).tolist(), flush=True)


step(1.0, n=int(6 / dt))
report("READY (joint mode)")
R0 = pose()[1]
# pure angular twist about chassis -X (the tool axis): raw 0.33 -> 0.033 rad per step with ik_scale 0.1
for k in range(int(2 / dt)):
    step(0.0, ik=(0, 0, 0, -0.33, 0, 0))
    if k % int(0.5 / dt) == 0:
        report(f"angular -X t={k * dt:.1f}")
R1 = pose()[1]
ang = math.degrees(math.acos(max(-1, min(1, (np.trace(R1 @ R0.T) - 1) / 2))))
print("RESULT angular: tool rotated", round(ang, 1), "deg in 2 s (commanded ~", round(math.degrees(0.033 * 2 / dt)), "deg)", flush=True)
p0 = pose()[0]
for k in range(int(2 / dt)):
    step(0.0, ik=(0, 0, 0.2, 0, 0, 0))
    if k % int(0.5 / dt) == 0:
        report(f"linear +Z t={k * dt:.1f}")
p1 = pose()[0]
print("RESULT linear: tcp moved", np.round(p1 - p0, 4).tolist(), "in 2 s (commanded ~", round(0.02 * 2 / dt, 2), "m)", flush=True)
step(0.0, n=int(3 / dt))
p2 = pose()[0]
print("RESULT hold: drift", round(float(np.linalg.norm(p2 - p1)), 4), "m over 3 s of zero twist", flush=True)
report("end")
env.close()
launcher.app.close()
