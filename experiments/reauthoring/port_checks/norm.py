"""norm.py <run_dir>...: each run's events without wall clock, and its problem files, as one text for diffing."""
import json, sys
from pathlib import Path
for d in map(Path, sys.argv[1:]):
    print("##", d.name)
    for l in (d / "events.jsonl").read_text().splitlines():
        e = json.loads(l); e.pop("wall_t", None)
        if e["kind"] == "operator_answered": e.pop("interaction_id", None)
        print(json.dumps(e, sort_keys=True))
    for p in sorted(d.glob("problem_*.nddl"), key=lambda p: int(p.stem.split("_")[1])):
        print("#", p.name); print(p.read_text())
