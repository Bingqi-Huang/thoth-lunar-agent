# thoth-lunar-agent

Thoth is a trustworthy agent harness for runtime mission adaptation in lunar robotics. A large language model adapts
the mission at run time; the harness keeps the mission state authoritative and verifies every action against it.
The repository contains the harness, two classical baselines (PLEXIL and EUROPA to PLEXIL), a simulated lunar base
with the LunarBot rover and arm, and the experiment and evaluation tooling.

## Layout

| Path | Contents |
|---|---|
| `space_runtime/`, `space_context/` | Mission state, decision layer, verification; lunar clock, geometry, illumination, link state |
| `openral_ext/`, `ros/` | Adapters and ROS 2 packages that connect the harness to OpenRAL |
| `baselines/` | PLEXIL and EUROPA to PLEXIL baselines (build scripts, plans, domain model; planners are not bundled) |
| `experiments/`, `evaluation/` | Launchers, scenarios, perturbations, metrics and tables |
| `configs/`, `prompts/`, `missions/` | Harness configuration, reasoner prompts, mission specifications |
| `openral/` | Modified OpenRAL (Apache-2.0), see `NOTICE` and `openral/MODIFICATIONS.md` |
| `space_robotics_bench/` | Modified Space Robotics Bench with the lunar-base scene (Apache-2.0 OR MIT) |
| `docs/` | Design notes for baselines, held-out cases and evaluation |

## Requirements

Ubuntu 22.04, ROS 2 Humble, Python 3.10 (OpenRAL side), Isaac Sim 5.1 with Isaac Lab (Python 3.11, separate process),
an NVIDIA GPU, and an OpenAI-compatible model endpoint. The two sides meet only on the ROS 2 (DDS) bus.

## Quick start

All commands run from the repository root.

```bash
export THOTH_ROOT=$(pwd)

# 1. Simulation assets (not stored in git)
ASSETS_URL=<asset archive url> scripts/fetch_assets.sh        # or: scripts/fetch_assets.sh path/to/archive.tar.gz
# Upstream Space Robotics Bench assets: install with the bench's own tooling (space_robotics_bench/README.md).

# 2. OpenRAL (Python 3.10) and the ROS 2 packages
(cd openral && ./scripts/install.sh)                           # see openral/README.md
source /opt/ros/humble/setup.bash && source openral/install/setup.bash
(cd ros && colcon build --merge-install)

# 3. Model access
export SPACE_LLM_API_KEY=<key>
export OPENRAL_REASONER_ENDPOINT=<https://your-endpoint/v1>   # also set in configs/space_runtime*.yaml
source experiments/launch/reasoner_env.sh

# 4. One mission end to end in simulation
D0_SCENE=scenes/deploy/srb_lunar_base_mission_b.yaml experiments/launch/lunar_base_d0/run_d0.sh /tmp/thoth_run
```

The baselines additionally need NASA PLEXIL and EUROPA (`baselines/europa/build_europa.sh`, `PLEXIL_HOME`).
Metrics and tables are described in `evaluation/README.md`.

Status: paths were generalised for this release and the full pipeline has not yet been re-run from a clean checkout.

## License

Apache-2.0 (see `LICENSE`). Bundled third-party code and simulation asset licenses are listed in `NOTICE` and
`space_robotics_bench/assets/lunar_base/ATTRIBUTION.md`.
