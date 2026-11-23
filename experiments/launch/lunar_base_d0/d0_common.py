"""Helpers shared by the live mission checks (the evaluator side of a run).

Importing this module connects to the graph and waits until Nav2, the rskill runner and TF are ready, then logs the
mission the task publishes. The simulator pose (`/openral/pose_truth`) is read for evaluation only (errors at the dock,
the tool point, object poses); it never reaches a method, which sees the robot only through the capability boundary.
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from lifecycle_msgs.srv import GetState
from openral_msgs.action import ExecuteRskill
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

HERE = Path(__file__).resolve().parent
IX = json.loads((HERE / "../../../space_robotics_bench/assets/lunar_base/usd/scene/lunar_base/interaction.json").resolve().read_text())
SKILL = dict(release="OpenRAL/rskill-procedural-lunar_bot-release")
TRACE_OBJ = os.environ.get("D0_TRACE_OBJECT")  # e.g. spare_oru: sample its handle against the TCP during every step


def mission_b_goal() -> str:
    """Mission B's formal goal for the baselines: the
    brief's item by its equipment-catalogue name, where the manifest has it staged and the destination, as site place
    frames -- from the same task overrides the brief is made from (D0_ENV env.cargo / env.destination, the task's
    defaults otherwise; lunar_base_mission_b/task.py). Both baselines read it from MISSION_GOAL. When the brief names
    the cargo by its serial (env.name_serial=true), so does the goal."""
    cfg = dict(kv.split("=", 1) for kv in os.environ.get("D0_ENV", "").split())
    cargo = IX["objects"][cfg.get("env.cargo", "spare_oru")]
    destination = {**{u: f"{u}_shelf/shelf" for u in ("pdu_1", "pdu_2", "pdu_3")},
                   "bess": "bess_stand/slot_0", "habitat": "habitat_stand/slot_0"}[cfg.get("env.destination", "pdu_1")]
    serial = {"serial": cargo["serial"]} if cfg.get("env.name_serial", "false").lower() == "true" else {}
    return json.dumps(dict(item=cargo["item"], source=cargo["support"], destination=destination, **serial))

rclpy.init()
n = Node("d0_mission_b_check", parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)])
buf = Buffer()
TransformListener(buf, n)
mission = {}
n.create_subscription(String, "/srb/env0/mission_state", lambda m: mission.update(json.loads(m.data)), 10)
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
vel = dict(cmd=[], srb=[], odom=[])  # Nav2's /cmd_vel, what the HAL hands SRB, and measured /odom base speeds during a base step
n.create_subscription(Twist, "/cmd_vel", lambda m: vel["cmd"].append((round(m.linear.x, 3), round(m.angular.z, 3))), 50)
n.create_subscription(Twist, "/srb/env0/action/cmd_vel", lambda m: vel["srb"].append((round(m.linear.x, 3), round(m.angular.z, 3))), 50)
n.create_subscription(Odometry, "/odom", lambda m: vel["odom"].append((round(m.twist.twist.linear.x, 3), round(m.twist.twist.angular.z, 3))), 50)
truth = {}  # latest /openral/pose_truth: evaluator-only
from sensor_msgs.msg import JointState
jaw = {}  # EG2 jaw joint (rad): 0.10 closed on nothing, 0.82 open -- "clamped, not hanging" is measured, not inferred
arm = {}  # the seven arm joints (rad), by name: a snapshot restore is checked against these, never trusted


def _on_joint_states(m):
    if "eg2_joint1" in m.name:
        jaw.update(q=m.position[m.name.index("eg2_joint1")])
    arm.update({k: float(v) for k, v in zip(m.name, m.position) if k in ARM_JOINTS})


ARM_JOINTS = frozenset(f"joint{i}" for i in range(1, 8))
n.create_subscription(JointState, "/srb/env0/robot/joint_states", _on_joint_states, 10)
n.create_subscription(Odometry, "/openral/pose_truth", lambda m: truth.update(p=m.pose.pose.position, q=m.pose.pose.orientation), 10)
T0 = time.time()
sim = dict(t=0.0)  # SRB's /clock: a rendering graph runs at a fraction of real time, and every skill
n.create_subscription(Clock, "/clock", lambda m: sim.update(t=m.clock.sec + m.clock.nanosec * 1e-9), 10)  # timeout is measured on it


def log(kind, **kw):
    print(kind, json.dumps(dict(t=round(time.time() - T0, 1), **kw), default=lambda o: round(float(o), 4) if isinstance(o, (np.floating, float)) else str(o)), flush=True)


def latched_string(topic: str) -> str:
    """The value of a latched std_msgs/String topic (published once, transient-local), e.g. the mission brief."""
    got = []
    sub = n.create_subscription(String, topic, lambda m: got.append(m.data),
                                QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    while not got:
        rclpy.spin_once(n, timeout_sec=0.1)
    n.destroy_subscription(sub)
    return got[0]


def spin(seconds):
    t = time.time()
    while time.time() - t < seconds:
        rclpy.spin_once(n, timeout_sec=0.05)


# ---- geometry ---------------------------------------------------------------------------------------------------
def quat_to_mat(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def tf_buffered(target, source):
    """The transform already in the buffer, without spinning: safe from a second thread while the main thread spins
    (rclpy's executor is not re-entrant -- two threads spinning one node corrupt its wait set). None when not there."""
    if not buf.can_transform(target, source, Time()):
        return None
    tr = buf.lookup_transform(target, source, Time()).transform
    return (np.array([tr.translation.x, tr.translation.y, tr.translation.z]),
            quat_to_mat(tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w))


def tf(target, source, timeout=10.0):
    t = time.time()
    while time.time() - t < timeout:
        rclpy.spin_once(n, timeout_sec=0.05)
        got = tf_buffered(target, source)
        if got is not None:
            return got
    raise RuntimeError(f"no TF {target} <- {source}")


def truth_pose():
    """The simulator's base pose (world = map by construction). Evaluation only; never reaches a method."""
    rclpy.spin_once(n, timeout_sec=0.05)
    p, q = truth["p"], truth["q"]
    R = quat_to_mat(q.x, q.y, q.z, q.w)
    return np.array([p.x, p.y, p.z]), math.atan2(R[1, 0], R[0, 0]), R


def est_pose(frame="map"):
    """The robot's own estimate of its base pose: `map` (AMCL) or `odom` (wheel-inertial)."""
    p, R = tf(frame, "chassis_base_link")
    return p, math.atan2(R[1, 0], R[0, 0]), R


def tcp_pose():
    """TCP in the world, through the true base pose (evaluation)."""
    pb, _, Rb = truth_pose()
    pc, Rc = tf("chassis_base_link", "tcp_frame")
    return pb + Rb @ pc, Rb @ Rc


def handle_pos(name, spin=True):
    got = tf("srb/world", f"srb/env0/lb_{name}") if spin else tf_buffered("srb/world", f"srb/env0/lb_{name}")
    if got is None:
        return None
    p, R = got
    return p + R @ np.array(IX["objects"][name]["handle"]["xyz"])


# ---- dispatch -----------------------------------------------------------------------------------------------------
ac = ActionClient(n, ExecuteRskill, "/openral/execute_rskill")


def dispatch(key, params, deadline=120.0, prompt=""):
    goal = ExecuteRskill.Goal()
    goal.rskill_id = SKILL[key]
    goal.prompt = prompt or f"d0 {key} {int(time.time())}"
    goal.goal_params_json = json.dumps(params)
    goal.deadline_s = float(deadline)
    t, sim0 = time.time(), sim["t"]
    h = None
    for attempt in range(3):  # the runner's first goal response after boot has taken >30 s (run 7)
        f = ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(n, f, timeout_sec=90)
        h = f.result()
        if h is not None:
            break
        log("RETRY", key=key, attempt=attempt + 1, waited_s=round(time.time() - t, 1))
    if h is None or not h.accepted:
        return dict(success=False, failure_reason="goal rejected" if h is not None else "no goal response", wall_s=round(time.time() - t, 1))
    rf = h.get_result_async()
    track, last, obj, last_obj = [], 0.0, [], 0.0
    # the hard deadline is on the sim clock like the skill's own budget: a rendering graph at 0.06x gave a 400 s
    # Nav2 goal 28 s of sim time on the wall and abandoned a rover still driving correctly
    while not rf.done() and sim["t"] - sim0 < deadline + 60:
        rclpy.spin_once(n, timeout_sec=0.1)
        if time.time - last > 3.0 and truth and buf.can_transform("map", "chassis_base_link", Time):  # sparse track: truth | map estimate
            tr = buf.lookup_transform("map", "chassis_base_link", Time()).transform
            pt, yt, _ = truth_pose()
            track.append((round(float(pt[0]), 2), round(float(pt[1]), 2), round(math.degrees(yt)),
                          round(tr.translation.x, 2), round(tr.translation.y, 2), round(math.degrees(2 * math.atan2(tr.rotation.z, tr.rotation.w)))))
            last = time.time()
        if TRACE_OBJ and time.time() - last_obj > 0.5 and truth and buf.can_transform("srb/world", f"srb/env0/lb_{TRACE_OBJ}", Time()) \
                and buf.can_transform("chassis_base_link", "tcp_frame", Time()):
            tr = buf.lookup_transform("srb/world", f"srb/env0/lb_{TRACE_OBJ}", Time()).transform
            Ro = quat_to_mat(tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w)
            hp = np.array([tr.translation.x, tr.translation.y, tr.translation.z]) + Ro @ np.array(IX["objects"][TRACE_OBJ]["handle"]["xyz"])
            tc = buf.lookup_transform("chassis_base_link", "tcp_frame", Time()).transform
            pt, _, Rt = truth_pose()
            tw = pt + Rt @ np.array([tc.translation.x, tc.translation.y, tc.translation.z])
            obj.append((round(sim["t"] - sim0, 1), round(float(hp[2]), 3), round(float(np.linalg.norm(hp - tw)), 3), round(float(pt[0]), 2), round(float(pt[1]), 2),
                        round(float(jaw.get("q", -1.0)), 3)))
            last_obj = time.time()
    out = dict(wall_s=round(time.time() - t, 1), sim_s=round(sim["t"] - sim0, 1),
               sim_rate=round((sim["t"] - sim0) / max(time.time() - t, 1e-6), 2))
    if obj:
        out["obj_track"] = obj  # (sim s, handle z, handle-TCP distance, base x, base y, jaw rad)
    out["jaw_rad"] = round(float(jaw.get("q", -1.0)), 3)
    if key in ("nav", "dock"):
        out["track"] = track
        for src in ("cmd", "srb", "odom"):
            s = vel[src]
            if s:
                v = [abs(a) for a, _ in s]; w = [abs(b) for _, b in s]
                out[f"{src}_v_max"], out[f"{src}_v_mean"], out[f"{src}_w_max"], out[f"{src}_n"] = round(max(v), 3), round(sum(v) / len(v), 3), round(max(w), 3), len(s)
    vel["cmd"].clear(); vel["srb"].clear(); vel["odom"].clear()
    if not rf.done():
        return dict(success=False, failure_reason="no result before hard deadline", **out)
    r = rf.result().result
    return dict(success=bool(r.success), failure_reason=r.failure_reason, failure_kind=int(r.failure_kind), **out)


def _yaw_deg(a, b):
    return round(math.degrees((a - b + math.pi) % (2 * math.pi) - math.pi), 1)


def base_err(xy, yaw):
    """Where the base is against (xy, yaw): physically (pos_err, truth), in the robot's belief (map_pos_err, AMCL),
    and how wrong each estimate is (loc_err: AMCL vs truth; odom_err: wheel-inertial odometry vs truth -- odom
    starts at the dock, the map origin, so the two frames coincide at t=0)."""
    pt, yt, _ = truth_pose()
    pm, ym, _ = est_pose("map")
    po, yo, _ = est_pose("odom")
    goal = np.asarray(xy, float)
    return dict(pos_err_m=round(float(np.linalg.norm(pt[:2] - goal)), 3), yaw_err_deg=_yaw_deg(yt, yaw),
                map_pos_err_m=round(float(np.linalg.norm(pm[:2] - goal)), 3), map_yaw_err_deg=_yaw_deg(ym, yaw),
                loc_err_m=round(float(np.linalg.norm(pm[:2] - pt[:2])), 3), loc_yaw_err_deg=_yaw_deg(ym, yt),
                odom_err_m=round(float(np.linalg.norm(po[:2] - pt[:2])), 3), odom_yaw_err_deg=_yaw_deg(yo, yt),
                at=[round(float(pt[0]), 3), round(float(pt[1]), 3)], map_at=[round(float(pm[0]), 3), round(float(pm[1]), 3)])


# ---- readiness ---------------------------------------------------------------------------------------------------
def lifecycle_state(node):
    c = n.create_client(GetState, f"{node}/get_state")
    if not c.wait_for_service(timeout_sec=1.0):
        return None
    f = c.call_async(GetState.Request())
    rclpy.spin_until_future_complete(n, f, timeout_sec=3.0)
    return f.result().current_state.label if f.result() else None


t = time.time()
while time.time() - t < 1500:
    s = lifecycle_state("/bt_navigator")
    if s == "active" and ac.server_is_ready():
        break
    spin(3.0)
else:
    log("RESULT", error="graph never ready"); sys.exit(2)
log("READY", after_s=round(time.time() - t), bt_navigator="active")
# Localization is up when AMCL is matching scans, not when `map` first exists: AMCL publishes the initial pose (and
# /amcl_pose, stamped ~0 s) at activation, and `map -> odom` keeps coming only once it processes scans, re-sent with
# every scan. A goal sent before that is judged against a stale transform, and Nav2's controller, failing to
# transform the goal, compares the robot at the map origin (the dock) with a zero pose and reports "Reached the
# goal!" at once (site_check_11: three navigations "succeeded" in 0.1 s without moving, map -> odom 5 s old).
while True:
    spin(0.5)
    if buf.can_transform("map", "odom", Time()):
        st = buf.lookup_transform("map", "odom", Time()).header.stamp
        if st.sec + st.nanosec * 1e-9 > sim["t"] - 0.5:
            break
log("LOCALIZED", sim_t=round(sim["t"], 1), map_odom_stamp=round(st.sec + st.nanosec * 1e-9, 2))
tf("map", "chassis_base_link", timeout=600)
tf("odom", "tcp_frame", timeout=60)
while "p" not in truth:
    spin(0.5)
try:
    tf("chassis_base_link", "base_link", timeout=20)
    log("TF", arm_root="ok")
except RuntimeError as e:
    log("TF", arm_root=str(e))
spin(20.0)  # costmaps
# Functional probe, not a presence check: `server_is_ready` can be True against a rskill runner left
# over from an earlier run's incomplete teardown. Opening the (empty) jaws is the harmless dispatch that proves the
# runner in THIS graph answers.
probe = dispatch("release", {"mode": "open", "timeout_s": 8.0}, deadline=15.0, prompt="d0 readiness probe")
log("PROBE", **probe)
if "no goal response" in probe.get("failure_reason", ""):
    log("RESULT", error="rskill runner does not answer — check for nodes left over from an earlier run on this ROS_DOMAIN_ID")
    sys.exit(2)
log("MISSION", **mission)


# ---- snapshot restore -----------------------------------------------------------------------------------------------
# The simulator's full state is saved before every agent step (<log>/snapshots/before_step_<n>.pt, with the rover's own
# localization estimate beside it).
snap_dir = Path(os.environ["D0_LOG"]) / "snapshots"
snap_dir.mkdir(exist_ok=True)
snap_pub = n.create_publisher(String, "/srb/envs/snapshot", 10)
snap_done = []
n.create_subscription(String, "/srb/envs/snapshot/done", lambda m: snap_done.append(json.loads(m.data)), 10)
while snap_pub.get_subscription_count() == 0:
    spin(0.5)


def snapshot(op, path):
    snap_done.clear()
    snap_pub.publish(String(data=json.dumps(dict(op=op, path=str(path)))))
    while not snap_done:
        spin(0.1)


