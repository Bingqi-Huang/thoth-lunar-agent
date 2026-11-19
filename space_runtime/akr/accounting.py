"""Work package D: the per-run summary line and the three verification checks, computed from the event log alone.

Nothing here is consulted by the executor. Executor rules and episode bounds are bugs if violated; assumption failures
(the map held a soundly written value that the simulator contradicts) are results. `truth.jsonl` holds the world's ground
truth at every actuation dispatch, one row per dispatch in dispatch order:
{"t", "action", "kind", "pre_truth": {literal: bool}, "pre_holds"} and optionally "effects_realized" (live evaluator).

An episode is the span from its first planner call to the first planner call of the next episode (Section 3.3): the
telemetry update and the cover processing that follow its last element belong to it. Events before the first planner
call (K0, the first telemetry update) belong to no episode.

    python3 -m space_runtime.akr.accounting <root>      # one summary line per run under root, root/summary.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

from .types import IRREVERSIBLE, REPORT, REVERSIBLE, SENSING, Proposal, parse_lit

SOURCES = {"k0", "sensing", "telemetry", "judgment"}
QUIET = {"kappa_write", "record_pruned", "failure_set", "cover_cleared", "subgoal", "episode_end", "telemetry"}
BRANCH_CAP = 2 ** 8


def read_events(run_dir: Path, name: str = "events.jsonl") -> list[dict]:
    """The executor's events from its `run_started` on (a live run's record holds the agent's own events before it)."""
    ev = [json.loads(l) for l in (Path(run_dir) / name).read_text().splitlines() if l.strip()]
    for e in ev:                                # the agent's own events carry the mission time as sim_t
        e.setdefault("t", e.get("sim_t"))
    return ev[next(i for i, e in enumerate(ev) if e["kind"] == "run_started"):]


def read_truth(run_dir: Path) -> list[dict] | None:
    p = Path(run_dir) / "truth.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else None


def _props(judgments: dict) -> set[str]:
    return {parse_lit(s)[0] for s in judgments}


def _episode_sets(ev: list[dict]) -> tuple[set, set]:
    """Episodes with a refuted record literal pruned in their span, and with a telemetry overturn in their span."""
    cur, refuted, exo = 0, set(), set()
    for e in ev:
        if e["kind"] == "planner_call":
            cur = e["episode"]
        elif cur and e["kind"] == "record_pruned" and e["refuted"]:
            refuted.add(cur)          # every pruned literal was in R_cur at the episode start: literals never return
        elif cur and e["kind"] == "telemetry" and e["changed_decided"]:
            exo.add(cur)
    return refuted, exo


# ---- the three checks -------------------------------------------------------------------------------------------------
def check_run(run_dir: Path, events_name: str = "events.jsonl") -> dict:
    ev, truth = read_events(run_dir, events_name), read_truth(run_dir)
    rs = ev[0]
    strategy, grants, exogenous = rs["strategy"], set(rs["grants"]), set(rs["exogenous"])
    rules: list[dict] = []

    def flag(e, rule, detail):
        rules.append({"t": e["t"], "rule": rule, "detail": detail})

    F, calls, last_out, obs_of, refused = set(), 0, None, {}, None
    for e in ev:
        k = e["kind"]
        if refused and k not in QUIET and k != "terminal":
            flag(e, "d_refusal", f"{k} after the refusal of {refused} and before the terminal")
        if k == "failure_set":
            F |= set(e["added"])
            F -= set(e["removed"])
        elif k == "planner_call":
            calls += 1
        elif k == "sense_started":
            obs_of[e["action"]] = e["obs"]
        elif k == "outcome":
            last_out = e
        elif k == "kappa_write":
            src, p, by = e["source"], e["prop"], e["by"]
            if src not in SOURCES:
                flag(e, "c_kappa_write", f"{p} written from source {src!r}")
            elif src == "k0" and calls:
                flag(e, "c_kappa_write", f"{p} written from k0 after the first planner call")
            elif src == "judgment" and (last_out is None or by != last_out["action"] or p not in _props(last_out["judgments"])):
                flag(e, "c_kappa_write", f"{p} judged by {by}, not an effect of the preceding outcome")
            elif src == "sensing" and p not in obs_of.get(by, ()):
                flag(e, "c_kappa_write", f"{p} written by sensing {by}, which does not observe it")
            elif src == "telemetry" and p not in exogenous:
                flag(e, "c_kappa_write", f"{p} written by telemetry but is not exogenous")
            if e["new"] == "F" and p in grants and refused is None:
                refused = p
        elif k == "dispatch":
            kp, pre = e["kappa_pre"], [parse_lit(s) for s in e["pre"]]
            if strategy != "ffreplan":
                if e["action_kind"] == IRREVERSIBLE:
                    for p, v in pre:
                        if kp[p] == "U" or kp[p] == ("F" if v else "T"):
                            flag(e, "a_irreversible_pre", f"{e['action']}: {p} is {kp[p]} in kappa")
                for p, v in pre:
                    if p in grants and kp[p] != "T":
                        flag(e, "b_restricted_grant", f"{e['action']}: grant {p} is {kp[p]} in kappa")
            if strategy == "akr" and e["action_kind"] == REVERSIBLE:
                for p, v in pre:
                    if kp[p] == "U" and p in F:
                        flag(e, "e_one_shot", f"{e['action']}: undecided {p} is in the failure set")
            if strategy == "kreplanner":
                for p, v in pre:
                    if kp[p] != ("T" if v else "F"):
                        flag(e, "f_kreplanner", f"{e['action']}: {p} is {kp[p]} in kappa")
        elif k == "terminal" and refused and e["outcome"] != "certified_hold":
            flag(e, "d_refusal", f"refusal of {refused} but the run ended with {e['outcome']}")

    fails = []
    if truth is not None:
        rows, seen = defaultdict(list), defaultdict(int)
        for r in truth:
            rows[r["action"]].append(r)
        for e in ev:
            if e["kind"] != "dispatch":
                continue
            r = rows[e["action"]][seen[e["action"]]]
            seen[e["action"]] += 1
            for s in e["pre"]:
                if s not in r["pre_truth"]:             # no truth source for this proposition (live evaluator)
                    continue
                p, v = parse_lit(s)
                k = e["kappa_pre"][p]
                world = r["pre_truth"][s] == v          # the value of the proposition in the world
                if k != "U" and (k == "T") != world:
                    fails.append({"t": e["t"], "action": e["action"], "literal": s, "kappa": k,
                                  "truth": "T" if world else "F"})

    bounds = []
    refuted, exo = _episode_sets(ev)
    n_exo_events = sum(1 for e in ev if e["kind"] == "telemetry" and e["changed_decided"])
    pruned = [e["literal"] for e in ev if e["kind"] == "record_pruned"]
    end = ev[-1]
    if len(refuted) > len(rs["record"]):
        bounds.append({"t": end["t"], "rule": "refutation_episodes", "detail": f"{len(refuted)} > |R| = {len(rs['record'])}"})
    if len(exo) > n_exo_events:
        bounds.append({"t": end["t"], "rule": "exogenous_episodes", "detail": f"{len(exo)} > {n_exo_events} telemetry overturns"})
    if len(set(pruned)) != len(pruned) or not set(pruned) <= set(rs["record"]):
        bounds.append({"t": end["t"], "rule": "record_literal_returned", "detail": f"pruned {pruned} from {rs['record']}"})
    return {"executor_rule_violations": rules, "assumption_failures": fails, "episode_bound_violations": bounds,
            "truth": "absent" if truth is None else "present"}


# ---- offline plan validity (Definition 4) -----------------------------------------------------------------------------
class _K:
    """The slice of the knowledge state that vocab.ground reads, rebuilt from a planner_call event."""

    def __init__(self, call: dict):
        self.kappa, self.record = call["kappa"], [parse_lit(s) for s in call["record"]]

    def value(self, p: str) -> str:
        return self.kappa.get(p, "U")


def _lit(x) -> tuple:
    return parse_lit(x) if isinstance(x, str) else (x[0], bool(x[1]))


def _decide(K: set, A: dict, p: str, v: bool) -> None:
    """Kl in, K-not-l and A-not-l out."""
    K.discard((p, not v))
    K.add((p, v))
    if A.get(p) == (not v):
        del A[p]


def plan_length(proposals: list[dict]) -> int:
    """Actuation and sensing elements; assumptions are not elements and the planner's report is not an action."""
    return sum(1 for p in proposals if p["name"] != REPORT)


def plan_valid(proposals: list[dict], call: dict, goal: list, vocab, kind: str) -> bool:
    """Whether some choice of assumptions and sensing values makes the sequence solve the compiled problem of Definition 4
    at the state of `call` (a planner_call event). `kind` "route" admits reversible actuation only; the goal is then the
    subgoal. A proposal that does not ground makes the plan invalid; the planner's report is ignored.

    Assumptions are made when first needed (equivalent to any earlier time: an effect on the proposition deletes both),
    for propositions undecided at the call and outside F, only against no K value of the opposite sign, and one value
    per proposition at a time. A sensed proposition that no later precondition or goal literal reads is not branched on;
    the branches over the others, each tried first with the value that the first later reader needs, are capped
    at 2^8, beyond which the plan is reported invalid."""
    view, steps = _K(call), []
    for p in proposals:
        if p["name"] == REPORT:
            continue
        a = vocab.ground(Proposal.of(p["name"], p["args"]), view)
        if a is None or (kind == "route" and a.kind == IRREVERSIBLE):
            return False
        steps.append(a)
    kappa, F, open_sensing = call["kappa"], set(call["failure_set"]), set(call["open_sensing"])
    goal = [_lit(l) for l in goal]
    read = [{p for p, _ in a.pre} for a in steps]
    read_after = [set().union({p for p, _ in goal}, *read[i + 1:]) for i in range(len(steps))]
    budget = [BRANCH_CAP]

    def first_read(i: int, p: str) -> bool:
        """The value of p that the first later precondition (else the goal) reads: tried first."""
        lits = [l for a in steps[i + 1:] for l in a.pre] + goal
        return next((v for q, v in lits if q == p), True)

    def go(i: int, K: frozenset, A: dict) -> bool:
        if i == len(steps):
            return all(l in K for l in goal)
        a = steps[i]
        if a.kind == SENSING:
            if a.name not in open_sensing or not all(l in K for l in a.pre):
                return False
            free = [p for p in a.obs if p not in kappa]
            branch = [p for p in free if p in read_after[i]]
            first = [first_read(i, p) for p in branch]
            for vals in itertools.product(*((f, not f) for f in first)):
                budget[0] -= 1
                if budget[0] < 0:
                    return False
                K2, A2 = set(K), dict(A)
                for p in free:
                    _decide(K2, A2, p, dict(zip(branch, vals)).get(p, True))
                if go(i + 1, frozenset(K2), A2):
                    return True
            return False
        A2 = dict(A)
        for p, v in a.pre:
            if (p, v) in K:
                continue
            if p in kappa or p in F or (p, not v) in K or A2.get(p, v) != v:
                return False
            A2[p] = v
        K2 = set(K)
        for p, v in a.eff:
            _decide(K2, A2, p, v)
        return go(i + 1, frozenset(K2), A2)

    return go(0, frozenset((p, v == "T") for p, v in kappa.items()), {})


def _liveness(ev: list[dict], vocab) -> dict:
    """Condition 2: per subsequence of planner calls (no subgoal; the same subgoal), the longest run of consecutive calls
    whose returned sequence is not a valid plan of at most L actions for the call's own problem. Every problem is taken
    to be solvable (Condition 3), so this is an upper bound on what the condition counts. The injected proposals are
    not the planner's and are dropped."""
    rs = ev[0]
    cur, best, call, invalid, n = defaultdict(int), defaultdict(int), None, 0, 0
    for e in ev:
        if e["kind"] == "planner_call":
            call = e
        elif e["kind"] == "planner_returned":
            own = [p for p in e["proposals"] if not p["injected"]]
            sub = call["subgoal"]
            ok = plan_length(own) <= rs["cap_L"] and plan_valid(own, call, sub or rs["goal"], vocab, "route" if sub else "main")
            key = " & ".join(sorted(sub)) if sub else None
            cur[key] = 0 if ok else cur[key] + 1
            best[key] = max(best[key], cur[key])
            invalid, n = invalid + (not ok), n + 1
    return {"longest_invalid_run_main": best[None],
            "longest_invalid_run_by_subgoal": {k: v for k, v in best.items() if k is not None},
            "b": 1 + max(best.values(), default=0), "invalid_calls": invalid, "calls": n}


# ---- the summary line -------------------------------------------------------------------------------------------------
def _is_request(name: str) -> bool:
    return name.startswith("ground_request")


def _decision_points(ev: list[dict]) -> list[dict]:
    covers = {}
    for e in ev:
        if e["kind"] == "cover_set":
            covers.setdefault(e["decision_point"], e)
    out = []
    for i, d in enumerate(ev):
        if d["kind"] != "decision_point":
            continue
        U = set(d["undecided"])
        j = next((j for j in range(i + 1, len(ev)) if ev[j]["kind"] == "dispatch" and ev[j]["action"] == d["action"]), None)
        res = "executed"
        if j is None:
            j = next((j for j in range(i + 1, len(ev)) if ev[j]["kind"] == "terminal"), None)
            res = "hold" if j is not None and ev[j]["outcome"] == "certified_hold" and ev[j]["pending_action"] == d["action"] else "open"
        stop = len(ev) if res == "open" else j
        paid = 0.0
        for e in ev[i + 1:stop]:
            if e["kind"] == "sense_started":
                paid += e["cost"]
            elif e["kind"] == "outcome" and U & _props(e["judgments"]):
                paid += e["duration"]
        c = covers.get(d["id"])
        out.append({"id": d["id"], "action": d["action"], "undecided": d["undecided"], "proposer": d["proposer"],
                    "t": d["t"], "resolved_t": None if res == "open" else ev[j]["t"], "resolution": res, "paid": paid,
                    "c_star_ge_t": c and c["c_star_ge_t"], "c_star_d": c and c["c_star_d"],
                    "greedy_cost": c and c["greedy_cost"]})
    return out


def _skill_rates(ev: list[dict], truth: list[dict] | None) -> dict:
    s = defaultdict(lambda: {"executions": 0, "conclusive": 0, "verified_all": 0})
    for e in ev:
        if e["kind"] == "outcome":
            r, js = s[e["action"].split("(")[0]], list(e["judgments"].values())
            r["executions"] += 1
            r["conclusive"] += all(j in "+-" for j in js)
            r["verified_all"] += all(j == "+" for j in js)
    for name, r in s.items():
        r["q"] = r["conclusive"] / r["executions"]
        rows = [x for x in truth or [] if x["action"].split("(")[0] == name and x["pre_holds"] and x.get("effects_realized") is not None]
        r["rho"] = sum(x["effects_realized"] for x in rows) / len(rows) if rows else None
    return dict(s)


def _ground_hold(ev: list[dict], rs: dict) -> str | None:
    """A run with no terminal event whose last proposal is a wait for mission control that the planner asked to last
    past the closure of the communication window (Earth set): no answer can arrive within the mission, so the mission
    is not completed -- the robot holds for the ground, no action pending. Classified by this rule, not by the stop
    that ends the run. Returns the wait's description, or None."""
    earth_set = ((rs.get("injection") or {}).get("earth_set_s"))
    last = next((e for e in reversed(ev) if e["kind"] == "tool_proposed"), None)
    if earth_set is None or last is None or "wait_for_mission_control" not in str(last.get("tool")):
        return None
    end = last["sim_t"] + float((last.get("args") or {}).get("max_wait_s") or 0)
    if end <= earth_set:
        return None
    return f"wait_for_mission_control until t={end:.0f} s (window closes at {earth_set:.0f} s)"


def summary(run_dir: Path, events_name: str = "events.jsonl", evaluator: dict | None = None, vocab=None) -> dict:
    """The per-run line. `evaluator` is the external verdict {"success": bool, ...}; `vocab` enables the liveness check."""
    run_dir = Path(run_dir)
    ev, truth = read_events(run_dir, events_name), read_truth(run_dir)
    rs = ev[0]
    term = next((e for e in reversed(ev) if e["kind"] == "terminal"), None)
    calls = [e for e in ev if e["kind"] == "planner_call"]
    tokens = [e["prompt_tokens"] for e in ev if e["kind"] == "planner_returned" and e["prompt_tokens"] is not None]
    starts = [e for e in ev if e["kind"] == "sense_started"]
    requests = [e for e in starts if _is_request(e["action"])]
    pruned = [e for e in ev if e["kind"] == "record_pruned"]
    refuted, exo = _episode_sets(ev)
    episodes = {e["episode"] for e in calls}
    props = [e for e in ev if e["kind"] == "proposal"]
    own = [e for e in props if not e["injected"]]
    rejected = [e for e in own if e["decision"] not in ("executed", "sensed", "report")]
    inj = [e for e in props if e["injected"]]
    kappa, repeated, uninformative = {}, 0, []
    for e in ev:
        if e["kind"] == "kappa_write":
            kappa[e["prop"]] = e["new"]
        elif e["kind"] == "sense_started" and all(kappa.get(p, "U") != "U" for p in e["obs"]):
            repeated += 1
            if _is_request(e["action"]):
                uninformative.append(e)
    skills = _skill_rates(ev, truth)
    outcome = term and term["outcome"]
    reason = term and term["reason"]
    hold = None if term else _ground_hold(ev, rs)
    if hold:                                    # protocol rule: a wait the planner asked for past the link's closure
        outcome, reason = "ground_hold", "wait_beyond_window"
    success = evaluator["success"] if evaluator else None
    n_exec = sum(r["executions"] for r in skills.values())
    return {
        "run": run_dir.name, "strategy": rs["strategy"], "mission": rs["mission"], "injection": rs["injection"],
        "cap_L": rs["cap_L"], "outcome": outcome, "reason": reason,
        "pending_action": term["pending_action"] if term else hold,
        "evaluator_success": success, "false_completion": None if success is None else outcome == "success" and not success,
        "planner_reported_complete": any(e["kind"] == "planner_report" for e in ev),
        "unsafe_executions": None if truth is None else sum(1 for r in truth if r["kind"] == IRREVERSIBLE and not r["pre_holds"]),
        "mission_time": ev[-1]["t"] - ev[0]["t"], "planner_calls": len(calls),
        "prompt_tokens": sum(tokens) if tokens else None,
        "ground_requests": len(requests),
        "first_request_t": requests[0]["t"] if requests else None,
        "request_bound": requests[0]["window"] if requests else None,
        "episodes": len(episodes), "record_size": len(rs["record"]),
        "injected_false_records": (rs["injection"] or {}).get("record_false_count"),
        "refutations": sum(e["refuted"] for e in pruned),
        "refutations_robot_changed": sum(e["refuted"] and e["robot_changed"] for e in pruned),
        "episodes_ended_by_refutation": len(refuted), "episodes_ended_by_exogenous": len(exo),
        "decision_points": _decision_points(ev),
        "actions_executed": n_exec,
        "q_overall": sum(r["conclusive"] for r in skills.values()) / n_exec if n_exec else None,
        "skill_rates": skills,
        "proposals_rejected": len(rejected),
        "proposals_rejected_per_episode": len(rejected) / len(episodes) if episodes else None,
        "rejected_by_decision": {d: sum(1 for e in rejected if e["decision"] == d) for d in sorted({e["decision"] for e in rejected})},
        "injected_proposals": {"total": len(inj), "executed": sum(e["decision"] == "executed" for e in inj)},
        "repeated_observations": repeated,
        "uninformative_requests": len(uninformative), "uninformative_request_cost": sum(e["cost"] for e in uninformative),
        "liveness": None if vocab is None else _liveness(ev, vocab),
    }


def run_dirs(root: Path) -> list[Path]:
    """Directories under root whose events.jsonl holds an executor run (a `run_started` event)."""
    return [p.parent for p in sorted(Path(root).rglob("events.jsonl")) if '"kind": "run_started"' in p.read_text()]


def summarize_dir(root: Path, vocab=None) -> list[dict]:
    """One summary per run directory, with the counts of the three checks; `<run>/evaluator.json` is read when present."""
    out = []
    for d in run_dirs(root):
        ev_file = d / "evaluator.json"
        row = summary(d, evaluator=json.loads(ev_file.read_text()) if ev_file.exists() else None, vocab=vocab)
        c = check_run(d)
        out.append({**row, **{k: len(c[k]) for k in ("executor_rule_violations", "assumption_failures", "episode_bound_violations")}})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    a = ap.parse_args()
    rows = summarize_dir(a.root)
    (a.root / "summary.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for r in rows:
        print(json.dumps(r))
