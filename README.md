# thoth-lunar-agent

**Thoth** (THinking Onboard, Trusted by Humans) is an executor for a lunar-base rover that repairs equipment and moves
cargo under a foundation-model planner. The planner proposes the actions; Thoth runs a reversible action on
assumptions, but an irreversible action (removing a module from an energized unit, handing cargo over) only once every
precondition is known, and senses on demand to decide the ones that are not. This is asymmetric knowledge-level
replanning (AKR).

The repository contains the executor, two comparison acting rules, two authored baselines (PLEXIL, and EUROPA
dispatching through PLEXIL), the simulated lunar base with the LunarBot rover and arm, the launcher for one mission in
simulation and a Monte Carlo study of the executor.

## Layout

| Path | Contents |
|---|---|
| `space_runtime/akr/` | The executor: knowledge state, record, sensing cover, episodes, end rules; one switch selects the acting rule |
| `configs/akr_vocabulary.yaml` | The declared vocabulary: propositions, actions with preconditions and effects, sensing actions with costs, evidence bands, mission goals |
| `configs/space_runtime_akr.yaml`, `prompts/` | Executor and planner configuration, and the texts the planner model reads |
| `space_runtime/`, `space_context/` | The live agent around the executor; lunar clock, Earth visibility (SPICE) and link state |
| `openral_ext/`, `ros/` | The common capability boundary every method calls (skills, evidence, telemetry) and its ROS 2 packages |
| `experiments/akr/` | Mission list, per-run records, Monte Carlo trials, plan-length check, dry runs |
| `experiments/launch/lunar_base_d0/` | One mission in simulation (`run_d0.sh`) and the per-method mission checks |
| `experiments/operator_console/` | Mission control behind the Earth link (ground requests and authorizations) |
| `experiments/evaluator_recorder/` | The external evaluator's record of simulator truth |
| `baselines/` | The authored baselines, original and revised task models |
| `openral/` | Modified OpenRAL (Apache-2.0), see `NOTICE` and `openral/MODIFICATIONS.md` |
| `space_robotics_bench/` | Modified Space Robotics Bench with the lunar-base scene and its equipment dynamics (Apache-2.0 OR MIT) |

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

# 3. Model access (planner and camera questions)
export SPACE_LLM_API_KEY=<key>                                 # or AKR_KEY_FILE=<file holding the key>
export SPACE_VLM_ENDPOINT=<https://your-endpoint/v1>          # and set reasoner.endpoint in configs/space_runtime_akr.yaml

# 4. One mission end to end: restore PDU-2 after an overcurrent trip
python3 -m experiments.akr.missions                            # the missions
eval "$(python3 -m experiments.akr.missions restore_pdu2 --rule akr) experiments/launch/lunar_base_d0/run_d0.sh /tmp/thoth_run"
```

`--rule` selects the acting rule (`akr`, `kreplanner`, `ffreplan`); `--baseline plexil|europa` launches an authored
baseline instead, and `--revised` its revised task model where one exists. The baselines need NASA PLEXIL and EUROPA
(`baselines/europa/build_europa.sh`, `PLEXIL_HOME`).

The simulation clock is held while the planner model answers, so mission time is simulated time. Every run writes its
raw data (events, model input and output, bag, evaluator record) to its run directory;
`python3 -m experiments.akr.run_records <run dir>` derives the per-run records from it.

## Monte Carlo trials

The rover and simulator are replaced by a sampled site world with the simulator's equipment rules, and a
breadth-first planner stands in for the foundation model; the executor is unchanged. Each trial's seed is derived from
its job specification, so any trial is reproduced from its specification alone.

```bash
python3 -m experiments.akr.montecarlo --grid main --trials 1000 --jobs 16 --out mc_main.jsonl
python3 -m experiments.akr.montecarlo --grid episodes --trials 1000 --jobs 16 --out mc_episodes.jsonl
python3 -m experiments.akr.montecarlo --grid ground --p-g 0.5 --trials 1000 --jobs 16 --out mc_ground.jsonl
python3 -m experiments.akr.mc_summary mc_main.jsonl
```

## Tests

`python3 -m experiments.akr.dryrun --out /tmp/akr_dry` runs scripted planner outputs through the executor without a
model or simulator. Unit tests live beside the code (`space_runtime/akr/test_*.py`, `experiments/akr/test_montecarlo.py`).

## License

Apache-2.0 (see `LICENSE`). Bundled third-party code and simulation asset licenses are listed in `NOTICE` and
`space_robotics_bench/assets/lunar_base/ATTRIBUTION.md`.
