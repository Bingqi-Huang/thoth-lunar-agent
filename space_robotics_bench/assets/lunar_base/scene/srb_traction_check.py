"""What fraction of a commanded base velocity does LunarBot actually achieve, on the road and at a work pose?

The Nav2 leg realises ~86 % of a 0.48 m/s command on the road network, while the docking servo realised
15 % of 0.15 m/s and 9 % of 0.45 rad/s at PDU-2's work pose. Two candidates: the
command magnitude (the wheel drives are velocity-controlled with damping 40 N·m·s/rad, so a small speed
error buys little torque) or the terrain off the prepared road. This drives both, at both places.

    ~/Programs/isaac-sim/python.sh assets/lunar_base/scene/srb_traction_check.py <out_dir>
"""

import json
import math
import sys
from pathlib import Path

from srb.core.app import AppLauncher
from srb.utils.path import SRB_APPS_DIR

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
launcher = AppLauncher(headless=True, enable_cameras=False, experience=SRB_APPS_DIR.joinpath("srb.headless.kit"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz  # noqa: E402

from srb.tasks.mobile_manipulation.lunar_base.task import Task, TaskCfg  # noqa: E402

cfg = TaskCfg()
cfg.scene.num_envs = 1
cfg.__post_init__()
env = Task(cfg)
env.reset()
robot = env._robot
dt = env.step_dt
steer_ids, _ = robot.find_joints(["chassis_to_.*_steering_joint"])
drive_ids, _ = robot.find_joints([".*_steering_to_wheel_joint"])
arm_ids, _ = robot.find_joints([f"joint{i}" for i in range(1, 8)], preserve_order=True)
origin = env.scene.env_origins[0].cpu().numpy()
zero = torch.zeros(env.action_space.shape, device=env.device)
zero[:, 3] = 1.0  # arm: hold joint targets
zero[:, 10:17] = robot.data.joint_pos[0, arm_ids]
zero[:, 17] = 1.0


def teleport(xy, yaw):
    with torch.inference_mode():
        q = quat_from_euler_xyz(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([float(yaw)])).to(env.device)
        pose = torch.cat([torch.tensor([[xy[0], xy[1], 0.12]], device=env.device) + env.scene.env_origins[0:1], q], 1)
        robot.write_root_pose_to_sim(pose)
        robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))
    for _ in range(int(2.0 / dt)):
        with torch.inference_mode():
            env.step(zero)


def pose():
    p = robot.data.root_pos_w[0].cpu().numpy() - origin
    yaw = float(euler_xyz_from_quat(robot.data.root_quat_w[0:1])[2][0])
    return p, (yaw + math.pi) % (2 * math.pi) - math.pi


def drive(v, w, seconds):
    a = zero.clone()
    a[:, 0], a[:, 2] = v, w
    p0, y0 = pose()
    peak_v, peak_w, wheel = 0.0, 0.0, 0.0
    for _ in range(int(seconds / dt)):
        with torch.inference_mode():
            env.step(a)
        peak_v = max(peak_v, float(np.linalg.norm(robot.data.root_lin_vel_w[0, :2].cpu().numpy())))
        peak_w = max(peak_w, abs(float(robot.data.root_ang_vel_w[0, 2])))
        wheel = max(wheel, float(robot.data.joint_vel[0, drive_ids].abs().max().cpu()))
    p1, y1 = pose()
    dy = (y1 - y0 + math.pi) % (2 * math.pi) - math.pi
    mean_v = float(np.linalg.norm((p1 - p0)[:2])) / seconds
    mean_w = dy / seconds
    torque = robot.data.applied_torque[0, drive_ids].abs().max().item()
    return dict(v_cmd=v, w_cmd=w, mean_v=round(mean_v, 3), peak_v=round(peak_v, 3), frac_v=round(mean_v / v, 2) if v else None,
                mean_w=round(mean_w, 3), peak_w=round(peak_w, 3), frac_w=round(mean_w / w, 2) if w else None,
                wheel_rad_s=round(wheel, 2), wheel_torque_Nm=round(torque, 1),
                steer_deg=[round(math.degrees(float(s)), 1) for s in robot.data.joint_pos[0, steer_ids]],
                pitch_deg=round(math.degrees(float(euler_xyz_from_quat(robot.data.root_quat_w[0:1])[1][0])), 1))


SITES = {
    "road_aisle": ((10.0, 2.0), 0.0),
    "pdu2_work": ((17.2, 2.4), math.radians(-90)),  # on the approach axis into PDU-2's work pose
    "depot_axis": ((5.82, -12.5), math.radians(90)),  # Mission B's pick approach axis
}
rows = []
for name, (xy, yaw) in SITES.items():
    teleport(xy, yaw)
    for v, w, s in ((0.15, 0.0, 6.0), (0.3, 0.0, 6.0), (0.5, 0.0, 6.0), (-0.15, 0.0, 6.0), (-0.3, 0.0, 6.0),
                    (0.0, 0.2, 6.0), (0.0, 0.45, 6.0)):
        teleport(xy, yaw)
        r = drive(v, w, s)
        r["site"] = name
        rows.append(r)
        print("RESULT", json.dumps(r), flush=True)
(out / "traction.json").write_text(json.dumps(rows, indent=1))
env.close()
launcher.app.close()
