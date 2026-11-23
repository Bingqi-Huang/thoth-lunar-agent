# experiments/evaluator_recorder

`recorder.py` is the external evaluator's recorder: it logs the simulator-truth signals the physical violation
classes are counted from (undesired robot contact, mission state, safety-kernel state, object poses) to
`<run_dir>/evaluator/events.jsonl`, the same topics whichever method is driving the run.

```bash
source /opt/ros/humble/setup.bash && source <openral>/install/setup.bash && source <ros>/install/setup.bash
ROS_DOMAIN_ID=96 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
    python3 experiments/evaluator_recorder/recorder.py --run-dir <run_dir> --selftest   # proves the mechanism, no simulation

# real use (against a live graph a session is actually running):
ROS_DOMAIN_ID=77 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
    python3 experiments/evaluator_recorder/recorder.py --run-dir <run_dir>
```

Never run with `ROS_DOMAIN_ID=77` except against a live graph a session is actually running -- the
`--selftest` path always uses 96.
