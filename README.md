# thoth-lunar-agent

Code for the paper **Trustworthy agentic lunar robotics: safety-constrained contingent planning with a
foundation-model planner**.

A lunar-base rover repairs equipment and moves cargo while its records may be wrong and its observations
uncertain. The paper models this as safety-constrained contingent planning and gives **asymmetric knowledge-level
replanning (AKR)**: a foundation model proposes plans, and the executor runs a reversible action on assumptions but an
irreversible action (removing a module from a unit, handing over cargo) only once every precondition is known, sensing
on demand to decide the ones that are not. Three theorems state soundness for any planner, the cost of that safety,
and completion. **Thoth** is the executor that implements AKR on the rover.

This repository contains the executor, the comparison rules (K-replanner rule, FF-Replan rule), the two authored
baselines (PLEXIL and EUROPA to PLEXIL, original and revised task models), the simulated lunar base with the LunarBot
rover and arm, the full-stack experiment launchers, the per-run measures and the Monte Carlo study.

## Layout

| Path | Contents |
|---|---|
| `space_runtime/akr/` | AKR executor (Algorithm 1): knowledge state, record, failure set, sensing cover, episodes, end rules; one strategy switch for AKR, K-replanner rule and FF-Replan rule |
| `configs/akr_vocabulary.yaml` | The declared vocabulary: propositions, actions with preconditions and effects, sensing actions with costs and windows, evidence rules, missions, initial knowledge and record |
| `configs/space_runtime_akr.yaml`, `prompts/akr/` | Executor configuration of the full-stack runs (planner model, episode cap L = 16, skill budget 80) and the planner prompts |
| `space_runtime/`, `space_context/` | Live agent around the executor, planner client; lunar clock, Earth visibility (SPICE), link state |
| `openral_ext/`, `ros/` | The common capability boundary every method calls (skills, evidence, telemetry), ROS 2 packages |
| `experiments/akr/` | Run list, per-run measures, Monte Carlo trials, plan-length check, dry runs |
| `experiments/launch/lunar_base_d0/` | One full-stack run in simulation (`run_d0.sh`) and the per-method mission checks |
| `experiments/operator_console/` | Mission control behind the Earth link (ground requests and authorizations) |
| `baselines/plexil/`, `baselines/europa/` | Authored baselines, original task models |
| `baselines/revised/<baseline>-<case>/` | Revised task models written for the four contingency classes |
| `experiments/reauthoring/` | Validation and engineering-time accounting of the revised task models |
| `experiments/scenarios/` | Scenario definitions of the missions |
| `openral/` | Modified OpenRAL (Apache-2.0), see `NOTICE` and `openral/MODIFICATIONS.md` |
| `space_robotics_bench/` | Modified Space Robotics Bench with the lunar-base scene and its equipment dynamics (Apache-2.0 OR MIT) |
| `docs/` | Design notes: baselines, authoring rules, scene, robot |

## Where each part of the paper is

| Paper | Code |
|---|---|
| Vocabulary, knowledge state and its update | `configs/akr_vocabulary.yaml`, `space_runtime/akr/vocabulary.py`, `space_runtime/akr/knowledge.py` |
| Admissibility and the three acting rules | `space_runtime/akr/executor.py` (`AKR_STRATEGY=akr`, `kreplanner`, `ffreplan`) |
| Sensing cover, its lower bound and the greedy cover | `space_runtime/akr/cover.py` |
| Algorithm 1: episodes, subgoals, certified holds, end rules | `space_runtime/akr/executor.py` |
| Execution-rule checks on every run log | `space_runtime/akr/accounting.py` |
| Inserted proposals, Earth-set time, ground decision | `space_runtime/akr/injection.py` |
| Evidence rules (pick, place, press, docking) and telemetry | `openral_ext/capability_server/node.py` |
| Earth link and ground authority | `space_context/`, `experiments/operator_console/console.py` |
| Equipment dynamics (trips, shutters, connector damage, handover) | `space_robotics_bench/srb/tasks/mobile_manipulation/lunar_base*`, `space_robotics_bench/assets/lunar_base/` |
| Missions and conditions | `configs/akr_vocabulary.yaml` (goals), `experiments/akr/campaign_jobs.py` (run list), `experiments/scenarios/` |
| Episode cap L (Condition 3) | `experiments/akr/plan_length.py`, `experiments/akr/initial_states.py` |
| Per-run measures (hazardous actions, covers, mission time, evidence statistics) | `experiments/akr/run_records.py`, `experiments/akr/truth_join.py` |
| Monte Carlo trials | `experiments/akr/montecarlo.py`, `experiments/akr/site_world.py`, `experiments/akr/bfs_planner.py`, `experiments/akr/mc_summary.py` |
| Revision effort of the authored baselines | `experiments/reauthoring/cost_table.py`, `experiments/reauthoring/worklog/` |

## Requirements

Ubuntu 22.04, ROS 2 Humble, Python 3.10 (OpenRAL side), Isaac Sim 5.1 with Isaac Lab (Python 3.11, separate process),
an NVIDIA GPU, and an OpenAI-compatible model endpoint. The two sides meet only on the ROS 2 (DDS) bus. The executor
core (`space_runtime/akr/`) and the Monte Carlo study need only Python 3.10, PyYAML and the site survey from the asset
archive (`interaction.json`).

## Quick start

All commands run from the repository root.

```bash
export THOTH_ROOT=$(pwd)

# 1. Simulation assets (not stored in git)
ASSETS_URL=<asset archive url> scripts/fetch_assets.sh        # or: scripts/fetch_assets.sh path/to/archive.tar.gz
# Upstream Space Robotics Bench assets: install with the bench's own tooling (space_robotics_bench/README.md).
space_context/fetch_kernels.sh                                 # NAIF generic kernels for the Earth-visibility profile

# 2. OpenRAL (Python 3.10) and the ROS 2 packages
(cd openral && ./scripts/install.sh)                           # see openral/README.md
source /opt/ros/humble/setup.bash && source openral/install/setup.bash
(cd ros && colcon build --merge-install)

# 3. Model access (planner and camera questions; DeepSeek-V4.1-Flash in the paper)
export SPACE_LLM_API_KEY=<key>                                 # or AKR_KEY_FILE=<file holding the key>
export SPACE_VLM_ENDPOINT=<https://your-endpoint/v1>          # and set reasoner.endpoint in configs/space_runtime_akr.yaml

# 4. One AKR mission end to end: restore PDU-2 after an overcurrent trip
D0_CAPABILITY_SERVER=1 D0_CHECK=ours_mission_check.py D0_SCENE=scenes/deploy/srb_lunar_base_mission_a.yaml \
D0_SCENARIO=covered D0_ENV=env.faulted_unit=pdu_2 OURS_CONFIG=$THOTH_ROOT/configs/space_runtime_akr.yaml \
AKR_STRATEGY=akr AKR_MISSION=a_l0_pdu2_reset \
  experiments/launch/lunar_base_d0/run_d0.sh /tmp/thoth_run
```

`python3 -m experiments.akr.campaign_jobs` prints the campaign's run list, one line per run (run
name, then the variables to pass to `run_d0.sh`). The `POOL_MACHINE` field only records which machine ran the run.
The authored baselines additionally need NASA PLEXIL and EUROPA (`baselines/europa/build_europa.sh`, `PLEXIL_HOME`);
`BASELINE_REVISION=reauth/<baseline>-<case>` selects a revised task model.

The simulation clock is held while the planner model answers, so mission time is simulated time without model
latency. Every run writes its raw data (events, model input and output, bag, evaluator record) to its run directory;
`python3 -m experiments.akr.run_records <run dir>` derives the measures from it.

## Monte Carlo trials

The rover and simulator are replaced by a sampled site world with the simulator's equipment rules, and a
breadth-first planner stands in for the foundation model; the executor is unchanged. Each trial's seed is derived from
its job specification, so any trial is reproduced from its specification alone. The static site (map and equipment
catalogue) is read from `experiments/akr/data/site_static.json`, as the full-stack runs recorded it.

```bash
python3 -m experiments.akr.montecarlo --grid v2 --rho 0.847 --q 0.962 --trials 1000 --jobs 16 --out mc_main.jsonl
python3 -m experiments.akr.montecarlo --grid episodes --rho 0.847 --q 0.962 --trials 1000 --jobs 16 --out mc_episodes.jsonl
python3 -m experiments.akr.montecarlo --grid ground --p-g 0.5 --rho 0.847 --q 0.962 --trials 1000 --jobs 16 --out mc_ground.jsonl
python3 -m experiments.akr.mc_summary mc_main.jsonl
```

ρ and q are the effect-realization and conclusive-judgment rates measured in the full-stack runs. The main grid has
84,000 trials, the episode grid 2,000 and the ground-refusal grid 168,000.

## Tests

`python3 -m experiments.akr.dryrun --out /tmp/akr_dry` runs scripted planner outputs through the executor without a
model or simulator. Unit tests live beside the code (`space_runtime/akr/test_*.py`, `experiments/akr/test_montecarlo.py`).

## Scope of this release

Included: everything the runs of record execute, the measures and the Monte Carlo study. Not included: the scheduler
and watchers of the multi-machine run farm, job lists with machine paths, and internal notes. The repository's first
commit holds the earlier release (harness experiments with held-out cases, ablations and a new-capability study) that
the present paper no longer reports.

## License

Apache-2.0 (see `LICENSE`). Bundled third-party code and simulation asset licenses are listed in `NOTICE` and
`space_robotics_bench/assets/lunar_base/ATTRIBUTION.md`.
