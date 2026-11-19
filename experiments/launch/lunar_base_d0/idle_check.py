"""A check that keeps the graph up and idle for D0_IDLE_S wall seconds (default 900), then ends: the simulator steps with
the rover standing still -- for measuring simulation speed (SRB's per-tick timing with D0_PROFILE=1), several on one
GPU or one alone."""
import os
import time

print("READY", flush=True)
time.sleep(float(os.environ.get("D0_IDLE_S", "900")))
print("IDLE_DONE", flush=True)
