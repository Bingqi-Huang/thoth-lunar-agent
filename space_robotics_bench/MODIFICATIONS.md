# Modifications

This directory is Space Robotics Bench (Apache-2.0 OR MIT) at DsslRobot/space_robotics_bench commit 94696f61,
modified for the Thoth lunar-robotics work: the `lunar_base` scene and task set (`assets/lunar_base`), the LunarBot
robot with four-wheel steering and RM-75 arm, a Mid-360 lidar model, and ROS adapters.

Added for the lunar-base missions:

- equipment telemetry of every serviced unit: breaker, lamp, heat-sink temperature, trip cause, module serial and
  connector state per bay;
- unit dynamics: re-trip with a closed heat-rejection shutter or an unserviceable module, permanent connector damage
  when a module is removed from an energized unit;
- handover points: an item that comes to rest on a delivery's receiving stand is latched and cannot be taken back;
- record faults (false locations, exchanged identities) and the sequential-fault scenario;
- a pause service (`envs/pause`) that holds physics and the clock while the planner model answers;
- battery modules with readable data plates (`power/bat_010x_v2`, the default).

The upstream `assets/srb_assets` payload and the large processed lunar-base assets are not stored in git: install
the upstream assets with the bench's own tooling and the lunar-base archive with `scripts/fetch_assets.sh`.
