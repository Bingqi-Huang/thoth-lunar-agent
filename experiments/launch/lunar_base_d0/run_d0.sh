#!/bin/bash
# Phase D0 live check: boots `openral deploy sim` on a Mission A/B scene (reasoner off) and runs
# the capability-path check against it, then tears the whole graph down.
#
#   [D0_SCENE=scenes/deploy/srb_lunar_base_mission_b.yaml] [D0_CHECK=d0_mission_b_check.py] [D0_STEPS=1,2] \
#   experiments/launch/lunar_base_d0/run_d0.sh <log_dir>
#
# Launch detached (setsid nohup ... &): this tool's background tasks get killed as "low memory"
# around Isaac Sim. (no `set -u`: ROS setup.bash reads unset variables)
LOG=${1:?log dir}
mkdir -p "$LOG"
# bash reads a script as it runs: editing this file under a live run made three runners die mid-run on a syntax
# error and skip their teardown. Every run executes its own frozen copy.
if [ -z "$D0_HERE" ]; then
  D0_HERE=$(cd "$(dirname "$0")" && pwd) && export D0_HERE
  cp "$0" "$LOG/run_d0.sh" && exec bash "$LOG/run_d0.sh" "$@"
fi
HERE=$D0_HERE
cd "$HERE/../../../openral"
# openral is a link to the main checkout in a worktree, so `..` from here is the main checkout: this's own
# components are named from $HERE (the console ran from the main checkout until 2026-11-10 and ignored AKR_GROUND_DECISION)
source /opt/ros/humble/setup.bash
source install/setup.bash
# D0_DOMAIN: one per graph on a host (default 77). Humble does not read ROS_AUTOMATIC_DISCOVERY_RANGE
# (Iron and later); the domain is what keeps two graphs apart
export ROS_DOMAIN_ID=${D0_DOMAIN:-77}
# camera images need large shared-memory segments, or most frames are lost in transport (fastdds_large_shm.xml)
export FASTRTPS_DEFAULT_PROFILES_FILE="$HERE/fastdds_large_shm.xml"
# Two graphs share this host (GPU 0 / GPU 1): the deploy CLI's argv-signature reaps at start and exit, and its
# Fast-DDS SHM purge, would kill or silence the sibling. This script's own teardown is by run tag.
export OPENRAL_SKIP_ORPHAN_REAP=1
# Both the reasoner and the scene/grasp VLM run on one OpenAI-compatible gateway (model DeepSeek-V4.1-Flash in the
# paper's runs). One 640x480 view per request answers in seconds.
# The key is read from a file outside the repository (AKR_KEY_FILE) or taken from SPACE_LLM_API_KEY; never echoed.
if [ -n "$AKR_KEY_FILE" ] && [ -r "$AKR_KEY_FILE" ]; then
  SPACE_LLM_API_KEY=$(tr -d '[:space:]' < "$AKR_KEY_FILE") && export SPACE_LLM_API_KEY
fi
: "${SPACE_LLM_API_KEY:?set SPACE_LLM_API_KEY (or AKR_KEY_FILE) for your OpenAI-compatible provider}"; export SPACE_LLM_API_KEY
SPACE_REASONER_API_KEY=$SPACE_LLM_API_KEY && export SPACE_REASONER_API_KEY
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
# Which checkout every component of this run is loaded from, with its commit (<run>/components.json); this research
# repo's own components must resolve inside it and the simulator inside D0_SRB, or the run stops here (2026-11-10: the
# console had been loaded from the main checkout through the openral link)
.venv/bin/python - "$LOG/components.json" "$HERE" <<'PY' || exit 1
import json, os, subprocess, sys
from pathlib import Path
out, here = Path(sys.argv[1]), Path(sys.argv[2])
repo = (here / "../../..").resolve()
def git(p):
    d = p if p.is_dir() else p.parent
    run = lambda *a: subprocess.run(["git", "-C", str(d), *a], capture_output=True, text=True).stdout.strip()
    return {"toplevel": run("rev-parse", "--show-toplevel"), "head": run("rev-parse", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "--", str(p)))}
own = {"console": here / "../../operator_console/console.py", "evaluator": here / "../../evaluator_recorder/recorder.py",
       "capability_server": here / "../../../openral_ext/capability_server/node.py",
       "planning_scene": here / "../../../openral_ext/planning_scene/node.py",
       "space_runtime": here / "../../../space_runtime", "check": here / os.environ.get("D0_CHECK", "d0_mission_b_check.py")}
if os.environ.get("OURS_CONFIG"):
    own["ours_config"] = Path(os.environ["OURS_CONFIG"])
srb = Path(os.environ.get("D0_SRB") or repo / "space_robotics_bench")
shared = {"openral": Path("."), "ros_install": Path("../ros/install"), "simulator": srb}
comp, bad = {}, []
for name, p in {**own, **shared}.items():
    rp = p.resolve()
    comp[name] = {"path": str(rp), "exists": rp.exists(), **(git(rp) if rp.exists() else {})}
    if name in own and (not rp.exists() or not str(rp).startswith(str(repo) + "/")):
        bad.append(f"{name}: {rp} is not inside {repo}")
if not comp["simulator"]["exists"] or not (srb / "srb").is_dir():
    bad.append(f"simulator: {srb} is not an SRB checkout")
comp["research_repo"] = {"path": str(repo), **git(repo)}
if os.environ.get("OURS_CONFIG"):    # the models a run talks to (never the key)
    import yaml
    rc = yaml.safe_load(Path(os.environ["OURS_CONFIG"]).read_text()).get("reasoner", {})
    comp["models"] = {"planner": rc.get("model"), "planner_endpoint": rc.get("endpoint"), "thinking": rc.get("thinking"),
                      "camera_questions": os.environ.get("SPACE_VLM_MODEL", "DeepSeek-V4.1-Flash-1017 (capability server default)"),
                      "key_from_file": bool(os.environ.get("AKR_KEY_FILE"))}
out.write_text(json.dumps(comp, indent=1))
if bad:
    print("COMPONENTS WRONG:\n  " + "\n  ".join(bad), file=sys.stderr)
    sys.exit(1)
print(f"components: research {comp['research_repo']['head'][:9]}, simulator {comp['simulator'].get('head', '?')[:9]} ({srb})")
PY
SCENE=${D0_SCENE:-scenes/deploy/srb_lunar_base_mission_b.yaml}
if [ -n "$D0_VIDEO$D0_GPU$D0_SCENERY$D0_SCENARIO$D0_PROFILE$D0_GPU_PHYSICS$D0_ENV$D0_SRB" ]; then
  # A copy of the scene, adjusted for this run:
  #   D0_VIDEO=<file.mp4>  SRB streams a robot-following viewer camera to that file (srb/utils/video_stream.py);
  #                        rendering costs wall time, so only for runs already known to pass. D0_VIDEO_RES ("W,H",
  #                        default 960,540), D0_VIDEO_STRIDE (env steps per frame, 5 = 10 fps of sim time) and
  #                        D0_VIDEO_SPEEDUP (1 = played at sim time) go with it. Raise the resolution before the
  #                        frame rate: bv1 rendered enough to drag the sim to 0.06x real time and the arm's hold
  #                        sagged 15 cm until `servo over shelf` stalled -- the render broke the run, not the skill.
  #   D0_GPU=1             SRB on the second GPU (a parallel graph; pair with D0_DOMAIN=78).
  #   D0_SCENERY=<name>    another SRB scenery for the same task (e.g. lunar_base_route_block).
  #   D0_SCENARIO=<name>   the task's env.scenario (e.g. heldout_module_failure).
  #   D0_OPERATOR=0        no scripted mission-control console (default: on, the protocol of heldout_design §8).
  #   D0_EVALUATOR=0       no external evaluator recorder (default: on; evaluation/violations.py reads it).
  #   D0_ENV="k=v ..."     further task overrides, e.g. "env.faulted_unit=bess" or "env.cargo=consumables_container
  #                        env.destination=habitat".
  #   D0_GPU_PHYSICS=1     GPU PhysX (SRB_SIM_DEVICE=cuda, CCD off) instead of CPU PhysX.
  #   D0_PROFILE=1         SRB's per-tick bucket timing.
  #   D0_SRB=<dir>         run the simulator from another SRB checkout (a worktree of a simulator branch): the scene's
  #                        paths into the main checkout's space_robotics_bench point there instead.
  .venv/bin/python - "$SCENE" "$LOG/scene_run.yaml" "$D0_VIDEO" "$D0_GPU" "$D0_SCENERY" "$D0_SCENARIO" "$D0_PROFILE" "$D0_GPU_PHYSICS" "$D0_ENV" <<'PY'
import os, sys, yaml
src, dst, video, gpu, scenery, scenario, profile, gpu_physics, env_args = sys.argv[1:]
d = yaml.safe_load(open(src))
if os.environ.get("D0_SRB"):
    main_srb = "${THOTH_ROOT}/space_robotics_bench"
    d = yaml.safe_load(yaml.safe_dump(d).replace(main_srb, os.environ["D0_SRB"]))
argv = d["simulator"]["argv"]
if video:
    argv.insert(argv.index("--env"), "--video")
    d["simulator"].setdefault("env_set", {}).update(SRB_VIDEO_PATH=video, SRB_VIDEO_FOLLOW="robot",
                                                   SRB_VIDEO_STRIDE=os.environ.get("D0_VIDEO_STRIDE", "5"),
                                                   SRB_VIDEO_SPEEDUP=os.environ.get("D0_VIDEO_SPEEDUP", "1"),
                                                   SRB_VIDEO_RES=os.environ.get("D0_VIDEO_RES", "960,540"))
    d["simulator"]["boot_timeout_s"] = 1800.0
if gpu_physics:
    d["simulator"].setdefault("env_set", {})["SRB_SIM_DEVICE"] = "cuda"
if profile:
    d["simulator"].setdefault("env_set", {})["SRB_ROS_PROFILE"] = "1"
if gpu:
    argv[argv.index("--env"):argv.index("--env")] = ["--kit_args", f"--/renderer/activeGpu={gpu} --/physics/cudaDevice={gpu}"]
if scenario:
    argv[:] = [f"env.scenario={scenario}" if a.startswith("env.scenario=") else a for a in argv]
if scenery:
    argv.insert(argv.index("--headless"), f"env.scenery={scenery}")
for kv in env_args.split():
    argv.insert(argv.index("--headless"), kv)
yaml.safe_dump(d, open(dst, "w"), sort_keys=False)
PY
  SCENE="$LOG/scene_run.yaml"
fi
D0_RUN_TAG="$LOG" setsid .venv/bin/openral deploy sim --config "$SCENE" --no-enable-octomap --no-object-detector > "$LOG/deploy.log" 2>&1 &
DEPLOY=$!
echo "DEPLOY_PID $DEPLOY sid $(ps -o sid= -p $DEPLOY)" > "$LOG/deploy.pid"
sleep 30
# the CLI setsid()s its `ros2 launch` child: a second session that neither the group kill nor a
# session kill on $DEPLOY reaches (stale launch trees survived runs 3-7 that way)
LAUNCH=$(pgrep -P "$DEPLOY" | head -1)
echo "LAUNCH_PID $LAUNCH sid $(ps -o sid= -p "$LAUNCH")" >> "$LOG/deploy.pid"
# The raw record every run keeps (docs/preflight_checklist.md R1: a metric defined later is recomputed from the run dir,
# never re-simulated): simulator truth (/tf carries every srb/env0/* frame), joint states, unit telemetry, the latched
# brief/inventory/space context (transient_local in record_qos.yaml), raw contact forces, navigation inputs and
# outputs (scans and the static map, not the 1 Hz full global costmap: ~20 GB/h and recomputable from them), safety
# edges and the operator channel. D0_BAG adds topics on top.
D0_BAG_DEFAULT="/tf /tf_static /clock /srb/env0/robot/joint_states /srb/env0/power_telemetry /srb/env0/mission_brief \
/srb/env0/inventory_record /srb/env0/space_context /srb/env0/mission_state \
/srb/env0/reward/penalty_undesired_robot_contacts /srb/env0/contacts_robot/force_norm /srb/env0/contacts_end_effector/force_norm /openral/safety_status /openral/estop /openral/estop_cleared \
/amcl_pose /plan /cmd_vel /cmd_vel_smoothed /odom /map /scan /openral/nav2/scan /local_costmap/costmap \
/local_costmap/published_footprint /global_costmap/published_footprint \
/space/operator/report /space/operator/request /space/operator/reply"
D0_BAG_TOPICS=$(echo "$D0_BAG_DEFAULT $D0_BAG" | tr -s ' ' '\n' | sort -u | tr '\n' ' ')
D0_RUN_TAG="$LOG" setsid ros2 bag record -o "$LOG/bag" --use-sim-time --qos-profile-overrides-path "$HERE/record_qos.yaml" \
  $D0_BAG_TOPICS > "$LOG/bag.log" 2>&1 &
if [ -n "$D0_CAPABILITY_SERVER" ]; then
  # the common capability boundary (§21.4) for boundary-driven checks; tagged so the teardown takes it down too
  source ../ros/install/setup.bash
  export D0_LOG="$LOG"
  D0_RUN_TAG="$LOG" setsid .venv/bin/python "$HERE/../../../openral_ext/capability_server/node.py" > "$LOG/capability_server.log" 2>&1 &
fi
# wall-clock cap only against a hung check: every skill and check deadline is on the sim clock, and a rendering
# Mission B at 0.06x needs more than 50 min of wall time (run bv1 was cut at 3000 s)
if [ -n "$D0_PERCEPTION" ]; then
  # relative perception (openral_ext/perception): fiducial targets in the camera images -> carrier poses; needs a scene
  # with cameras; the boundary and the planning scene take their targets from it (no simulator truth); tagged
  D0_RUN_TAG="$LOG" setsid .venv/bin/python "$HERE/../../../openral_ext/perception/fiducial_node.py" > "$LOG/perception.log" 2>&1 &
fi
if [ "${D0_OPERATOR:-1}" != 0 ]; then
  # mission control behind the SPICE link, the same scripted protocol for every method
  # (docs/heldout_design.md §8); tagged. A person at the console (demonstration runs): D0_OPERATOR=0 and run
  # console.py --manual in a terminal on the same domain.
  source ../ros/install/setup.bash
  D0_RUN_TAG="$LOG" setsid .venv/bin/python -u "$HERE/../../operator_console/console.py" --run-dir "$LOG" \
    > "$LOG/operator_console.log" 2>&1 &
fi
if [ "${D0_EVALUATOR:-1}" != 0 ]; then
  # the external evaluator's recorder (simulator truth for the violation classes; never read by a method), on for every
  # method's run; D0_EVALUATOR=0 turns it off; tagged
  D0_RUN_TAG="$LOG" setsid .venv/bin/python -u "$HERE/../../evaluator_recorder/recorder.py" --run-dir "$LOG" \
    > "$LOG/evaluator_recorder.log" 2>&1 &
fi
if [ "${D0_PLANNING_SCENE:-1}" != 0 ]; then
  # the worksite in MoveIt's planning scene (openral_ext/planning_scene) -- part of the shared stack every method
  # plans in, on by default (D0_PLANNING_SCENE=0 turns it off); tagged for the teardown
  D0_RUN_TAG="$LOG" setsid .venv/bin/python "$HERE/../../../openral_ext/planning_scene/node.py" > "$LOG/planning_scene.log" 2>&1 &
fi
# The check is an actuator client too. A failed boot must not leave it waiting
# to command a later graph in the same ROS domain after teardown.
D0_RUN_TAG="$LOG" timeout 43200 .venv/bin/python "$HERE/${D0_CHECK:-d0_mission_b_check.py}" > "$LOG/check.log" 2>&1
echo "CHECK_EXIT $?" >> "$LOG/check.log"
if [ -n "$D0_HOLD" ]; then
  # step-wise development: the graph stays up after the check for attach_d0.sh (continue from the failed step);
  # teardown_d0.sh <log> takes it down
  echo held > "$LOG/held"
  exit 0
fi
bash "$HERE/teardown_d0.sh" "$LOG"
# §19 run manifest (openral_ext/run_manifest): this script launches `openral deploy sim` itself rather than through
# RunManifestWriter, so it backfills post-hoc from what the finished run dir already has on disk -- read-only,
# starts nothing (openral_ext/run_manifest/backfill.py's own hard rule)
.venv/bin/python - "$LOG" "$HERE/../../.." <<'PY'
import json
import sys
from pathlib import Path

import yaml

run_dir, repo_root = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repo_root))
from openral_ext.run_manifest.backfill import backfill_run_manifest
from openral_ext.run_manifest.capability_registry import resolve_capability_registry

manifest = backfill_run_manifest(run_dir, repo_root=repo_root)
(run_dir / "run_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
robot_id = manifest.get("robot_id")
if robot_id:
    registry = resolve_capability_registry(robot_yaml=repo_root / "openral/robots" / robot_id / "robot.yaml",
                                            rskills_dir=repo_root / "openral/rskills")
    (run_dir / "capability_registry.json").write_text(json.dumps(registry, indent=2))
PY
echo done > "$LOG/done"
