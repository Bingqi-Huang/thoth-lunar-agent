"""Ground truth at every dispatch of a live AKR run, for the accounting (work package D): the evaluator's 5 Hz
`mission_state` record joined to the executor's `dispatch` / `outcome` events by mission time, read through the
vocabulary's `truth`. The grant propositions' truth is the ground's decision fixed for the run (AKR_GROUND_DECISION),
which is what the request reveals. Writes `<run>/ours/truth.jsonl` in the row format accounting.check_run reads; a
literal whose proposition has no truth source is left out of `pre_truth` (and of the realization check).

    python3 -m experiments.akr.truth_join <run dir> [<run dir> ...]
"""
import bisect
import json
import sys
from pathlib import Path

import yaml

from space_runtime.akr.types import parse_lit
from space_runtime.akr.vocabulary import Vocabulary

REPO = Path(__file__).resolve().parents[2]


def join(run: Path) -> int:
    ev = [json.loads(l) for l in (run / "ours/events.jsonl").read_text().splitlines()]
    inp = next(e for e in ev if e["kind"] == "akr_inputs")
    site = next(e["site"] for e in ev if e["kind"] == "site_knowledge")
    cfg = yaml.safe_load((REPO / "configs/akr_vocabulary.yaml").read_text())
    inj = inp["injection"]
    vocab = Vocabulary(cfg, site, inp["telemetry"], inp["manifest"], inp["mission_key"], t_vis=inj["earth_set_s"],
                       telemetry_removed=tuple(inj["telemetry_removed"]))
    decision = inj["ground_decision"] if inj["ground_decision"] != "rule" else None
    if decision is None and (run / "operator/events.jsonl").exists():      # runs that set it only for the console
        ends = [e for e in map(json.loads, (run / "operator/events.jsonl").open())
                if e["kind"] == "interaction_end" and "decided before the mission" in (e.get("text") or "")]
        decision = ("grant" if ends[0]["outcome"] == "grant" else "refuse") if ends else None
    states = [(e["sim_t"], e["state"]) for e in map(json.loads, (run / "evaluator/events.jsonl").open()) if e["kind"] == "mission_state"]
    times = [t for t, _ in states]

    def truth_at(t: float) -> dict:
        s = states[max(0, bisect.bisect_right(times, t) - 1)][1]
        tr = vocab.truth(s)
        if decision:
            tr.update({g: decision == "grant" for g in vocab.grants})
        return tr

    rows, pending = [], None
    for e in ev:
        if e["kind"] == "dispatch":
            tr = truth_at(e["sim_t"])
            pre = {s: tr[parse_lit(s)[0]] == parse_lit(s)[1] for s in e["pre"] if parse_lit(s)[0] in tr}
            pending = {"t": e["t"], "action": e["action"], "kind": e["action_kind"], "pre_truth": pre, "pre_holds": all(pre.values())}
            rows.append(pending)
        elif e["kind"] == "outcome" and pending is not None:
            tr = truth_at(e["sim_t"] + 1.0)
            lits = [parse_lit(s) for s in e["judgments"]]
            known = [tr[p] == v for p, v in lits if p in tr]
            pending["effects_realized"] = all(known) if known else None
            pending = None
    (run / "ours/truth.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return len(rows)


if __name__ == "__main__":
    for r in sys.argv[1:]:
        print(r, join(Path(r)), "dispatches")
