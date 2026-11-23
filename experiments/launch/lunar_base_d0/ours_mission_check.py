"""The agent on the live graph: the space-native mission agent (space_runtime) runs one Mission A/B case.

The agent receives the operator's mission brief (the same wording every method gets) and reaches the robot only
through the common capability boundary. This script is the evaluator: it starts the agent, waits for the autonomous
run to end, and reports the task verifier's state and the rover's truth error at the dock -- neither ever reaches
the agent. The agent's own record is <log>/ours/{events.jsonl,report.md,agent.log}.
Run with D0_CAPABILITY_SERVER=1 D0_CHECK=ours_mission_check.py (the scene and D0_SCENARIO pick the mission).
"""
import subprocess

from d0_common import *  # noqa:

REPO = (HERE / "../../..").resolve()
# the operator's brief, exactly as the SRB task publishes it for every method (latched /srb/env0/mission_brief)
BRIEF = latched_string("/srb/env0/mission_brief")
run_dir = Path(os.environ["D0_LOG"]) / "ours"
run_dir.mkdir(parents=True, exist_ok=True)
# the run tag puts the agent under the graph's teardown: an agent blocked in rclpy ignores SIGTERM
env = dict(os.environ, PYTHONPATH=f"{REPO}:{os.environ.get('PYTHONPATH', '')}", D0_RUN_TAG=os.environ["D0_LOG"])

log("PLAN", method="ours", brief=BRIEF, run_dir=str(run_dir))
t0, sim0 = time.time(), sim["t"]


def steps():
    p = run_dir / "events.jsonl"
    return sum('"kind": "tool_proposed"' in line or '"kind": "reasoner_step_failed"' in line for line in p.read_text().splitlines()) if p.exists() else 0


# OURS_CONFIG=<path>: the harness configuration
CONFIG = ["--config", os.environ["OURS_CONFIG"]] if os.environ.get("OURS_CONFIG") else []


def start(extra):
    return subprocess.Popen([sys.executable, "-m", "space_runtime.agent", "--run-dir", str(run_dir), *CONFIG, *extra], cwd=REPO, env=env,
                            stdout=(run_dir / "agent.log").open("a"), stderr=subprocess.STDOUT)


agent = start(["--mission", BRIEF])
last = time.time()
saved = -1


def save_before_next_step(evs):
    """Once the last recorded step has ended (or none started), save the state the next step will start from."""
    global saved
    k = steps()
    last = max((i for i, x in enumerate(evs) if '"kind": "tool_proposed"' in x or '"kind": "reasoner_step_failed"' in x), default=-1)
    ended = last < 0 or '"reasoner_step_failed"' in evs[last] or any(
        m in x for x in evs[last + 1:] for m in ('"kind": "verification_finished"', '"kind": "query_answered"', '"kind": "local_tool"', '"outcome": "deny', '"outcome": "approval'))
    if ended and k != saved:
        path = snap_dir / f"before_step_{k + 1}.pt"
        p, yaw, _ = est_pose()
        path.with_suffix(".json").write_text(json.dumps(dict(x=float(p[0]), y=float(p[1]), yaw=float(yaw),
                                                            sim_t=sim["t"], arm=dict(arm))))
        snapshot("save", path)
        saved = k


# D0_TRACE_OBJECT=<name>: log the object's truth pose whenever it moves > 2 cm, with the agent's last recorded step
trace_prev = handle_pos(TRACE_OBJ) if TRACE_OBJ else None
while agent.poll() is None:
    rclpy.spin_once(n, timeout_sec=0.2)
    evs = (run_dir / "events.jsonl").read_text().splitlines() if (run_dir / "events.jsonl").exists() else []
    save_before_next_step(evs)
    if TRACE_OBJ:
        hp = handle_pos(TRACE_OBJ)
        if np.linalg.norm(hp - trace_prev) > 0.02:
            evs = (run_dir / "events.jsonl").read_text().splitlines()
            lastev = next((json.loads(x) for x in reversed(evs) if '"tool_proposed"' in x), {})
            log("OBJECT", name=TRACE_OBJ, sim_t=round(sim["t"], 2), handle=[round(float(v), 3) for v in hp],
                moved_m=round(float(np.linalg.norm(hp - trace_prev)), 3), last_tool=lastev.get("tool"), last_args=lastev.get("args"),
                tcp=[round(float(v), 3) for v in tcp_pose()[0]], jaw=round(float(jaw.get("q", -1)), 3))
            trace_prev = hp
    if time.time() - last > 120:
        log("PROGRESS", wall_s=round(time.time() - t0), sim_s=round(sim["t"] - sim0, 1), mission=dict(mission))
        last = time.time()
spin(3.0)
log("AGENT", exit=agent.returncode, wall_s=round(time.time() - t0, 1), sim_s=round(sim["t"] - sim0, 1))
log("DOCK", **base_err([0.0, 0.0], 0.0))
log("RESULT", mission=dict(mission))
