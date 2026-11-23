"""The records each full-stack run yields for the evaluation measures:
one summary line per run, one record per irreversible proposal (knowledge and true state of its preconditions, what
the rule decided, the sensing that followed) and one per evidence judgment (with the true state after the action).

The true state is the evaluator's (evaluator/events.jsonl, read from the simulator, never from the executor), taken at
the last record before the event's mission time: `mission_state` (units, where each object rests), the rover's root
pose (locations), the arm's joints (stowed: within the simulator's own 0.35 rad of its travel posture, site.py
parked_and_stowed) and the gripper's contents (an object resting nowhere whose handle is within 0.30 m of the tool
point, site.py NOT_HELD_M and the handle of interaction.json, and lifted at least 2 cm above the height
it last rested at or with the jaws closed on it, eg2_joint1 within 0.13-0.6 rad: a stowed gripper parked in front of a bay
is near a module's handle without holding it). "At X" (base_at) for an action done at X other than a delivery is judged by
that action's physical result: the pick lifted the item, the removal took the module into the
jaws, the place left the item in the placement box, the press registered, the plate read agreed with the truth; when the
action failed physically, the 6 cm docking band decides. A dock home's "at the dock" is the simulator's own docked check
(parked_and_stowed, as the mission's success test) before the rover moves again. A grant's truth is the ground's configured decision. A run whose
evaluator did not log the root pose and the joints has those propositions null.

    python3 -m experiments.akr.run_records <run dir> [...]      # writes <run>/records/{summary.json,irreversible.jsonl,judgments.jsonl}
    python3 -m experiments.akr.run_records --out <dir> <run dir> [...]   # writes <dir>/<run name>/ instead
"""
import bisect
import json
import math
import sys
from pathlib import Path

from space_runtime.akr.types import IRREVERSIBLE, gave_up, parse_lit
from space_runtime.akr.vocabulary import Vocabulary, load_config


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


STOW = (0.0, 0.40, 0.0, -0.70, 0.0, -1.27, 1.57079633)     # site.py _STOW, joint1..joint7
STOW_TOLERANCE_RAD, NOT_HELD_M, TCP_OFFSET = 0.35, 0.30, (0.0, 0.0, 0.140)
LIFT_M, JAW_HELD_RAD = 0.02, (0.13, 0.6)     # held: lifted off its last rest, or jaws closed on it (eg2_joint1, 0.82 = open)
UNITS = ("pdu_1", "pdu_2", "pdu_3", "bess", "isru")


def _rot(q, v):
    w, x, y, z = q
    tx, ty, tz = 2 * (y * v[2] - z * v[1]), 2 * (z * v[0] - x * v[2]), 2 * (x * v[1] - y * v[0])
    return (v[0] + w * tx + y * tz - z * ty, v[1] + w * ty + z * tx - x * tz, v[2] + w * tz + x * ty - y * tx)


class Truth:
    def __init__(self, run: Path, vocab: Vocabulary):
        ev = [e for e in load(run / "evaluator/events.jsonl") if e.get("sim_t") is not None]
        states = [e for e in ev if e["kind"] == "mission_state"]
        self.t, self.states, self.vocab = [e["sim_t"] for e in states], [e["state"] for e in states], vocab
        self.poses: dict[str, tuple[list, list]] = {}
        for e in ev:
            if e["kind"] == "pose":
                ts, xs = self.poses.setdefault(e["frame"], ([], []))
                ts.append(e["sim_t"]); xs.append(e)
        joints = [e for e in ev if e["kind"] == "joints"]
        self.jt, self.joints = [e["sim_t"] for e in joints], [e["position"] for e in joints]
        objects = json.loads((Path(__file__).resolve().parents[2] /
                              "space_robotics_bench/assets/lunar_base/usd/scene/lunar_base/interaction.json").read_text())["objects"]
        self.name = {f"srb/env0/lb_{n}": n for n in objects}
        self.handle = {n: tuple(o["handle"]["xyz"]) for n, o in objects.items()}    # site.py _handle_local
        # the height each object last rested at: on a support, or seated in a bay
        self.rest: dict[str, tuple[list, list]] = {}
        for t, st in zip(self.t, self.states):
            seated = {m for u in st["units"].values() for m in u["bays"].values() if m}
            for f, n in self.name.items():
                o = self._pose(f, t)
                if o and (st["objects"].get(n) or n in seated):
                    ts, zs = self.rest.setdefault(n, ([], []))
                    ts.append(t); zs.append(o["z"])

    def _pose(self, frame, t):
        ts, xs = self.poses.get(frame, ([], []))
        i = bisect.bisect_right(ts, t) - 1
        return xs[i] if i >= 0 else None

    def at(self, t: float) -> dict:
        i = bisect.bisect_right(self.t, t) - 1
        if not self.states:
            return {}
        state = dict(self.states[max(i, 0)])
        root = self._pose("srb/env0/robot", t)
        if root:
            yaw = math.atan2(2 * (root["qw"] * root["qz"] + root["qx"] * root["qy"]), 1 - 2 * (root["qy"] ** 2 + root["qz"] ** 2))
            state["base_pose"] = {"x": root["x"], "y": root["y"], "yaw": yaw}
        j = bisect.bisect_right(self.jt, t) - 1
        if j >= 0 and all(f"joint{k}" in self.joints[j] for k in range(1, 8)):
            q = [self.joints[j][f"joint{k}"] for k in range(1, 8)]
            state["arm_posture"] = "stowed" if max(abs(a - b) for a, b in zip(q, STOW)) < STOW_TOLERANCE_RAD else "other"
        l7 = self._pose("srb/env0/robot/Link7", t)
        if l7 and root:
            off = _rot((l7["qw"], l7["qx"], l7["qy"], l7["qz"]), TCP_OFFSET)
            tcp = (l7["x"] + off[0], l7["y"] + off[1], l7["z"] + off[2])
            held = []
            jaw = self.joints[j].get("eg2_joint1") if j >= 0 else None
            closed = jaw is not None and JAW_HELD_RAD[0] <= jaw <= JAW_HELD_RAD[1]
            for f, n in self.name.items():                      # its handle within reach of the tool point, as site.py
                o = self._pose(f, t)
                if o and not state["objects"].get(n):
                    h = _rot((o["qw"], o["qx"], o["qy"], o["qz"]), self.handle[n])
                    ts, zs = self.rest.get(n, ([], []))
                    k = bisect.bisect_right(ts, t) - 1
                    lifted = k >= 0 and o["z"] - zs[k] >= LIFT_M
                    if math.dist((o["x"] + h[0], o["y"] + h[1], o["z"] + h[2]), tcp) < NOT_HELD_M and (lifted or closed):
                        held.append(n)
            state["held"] = held
        return self.vocab.truth(state)

    def final(self) -> dict:
        return self.states[-1] if self.states else {}

    def parked(self, t0: float, t1: float) -> bool:
        """The simulator's own docked check (parked_and_stowed, the mission's success test) holds at some time in [t0, t1]."""
        i, k = max(bisect.bisect_right(self.t, t0) - 1, 0), bisect.bisect_right(self.t, t1)
        return any(st["parked_and_stowed"] for st in self.states[i:max(k, i + 1)])

    def presses(self, t: float, unit: str) -> int:
        i = bisect.bisect_right(self.t, t) - 1
        u = self.states[max(i, 0)]["units"][unit]
        return u.get("reset_presses", 0) + u.get("cover_presses", 0)


def done_at(action: str) -> tuple[str | None, str | None]:
    """(capability, the place X it is done at) for an action whose precondition is base_at(X); (None, None) otherwise."""
    cap, _, rest = action.partition("(")
    args = rest.rstrip(")").split(",")
    if cap in ("pick", "place", "remove") and len(args) == 2:
        return cap, args[1]
    if cap in ("press", "scan_label") and len(args) == 1:
        return cap, args[0]
    return None, None


def physical_result(truth: Truth, out: dict) -> bool | None:
    """Did the action of outcome `out` do physically what it is for (None: not an action done at a place)."""
    cap, at = done_at(out["action"])
    if cap is None:
        return None
    t0, t1 = out["t"] - (out.get("duration") or 0.0), out["t"] + 0.5
    after = truth.at(t1)
    serial = out["action"].partition("(")[2].split(",")[0]
    if cap in ("pick", "remove"):
        return after.get(f"held({serial})") is True
    if cap == "place":
        return after.get(f"item_at({serial},{at})") is True
    if cap == "press":
        unit = max((u for u in UNITS if at.startswith(u + "_")), key=len, default=None)
        return unit is not None and truth.presses(t1, unit) > truth.presses(t0, unit)
    decided = [(parse_lit(l), j) for l, j in out["judgments"].items() if j in ("+", "-")]        # scan_label
    return bool(decided) and all(after.get(p) is not None and (after[p] == v) == (j == "+") for (p, v), j in decided)


def records(run: Path, out_root: Path | None = None) -> dict:
    run = Path(run)
    ev = load(run / "ours/events.jsonl")
    site = next(e["site"] for e in ev if e["kind"] == "site_knowledge")
    inp = next(e for e in ev if e["kind"] == "akr_inputs")
    vocab = Vocabulary(load_config(), site, inp["telemetry"], inp["manifest"], inp["mission_key"])
    truth = Truth(run, vocab)
    decision = (inp.get("injection") or {}).get("ground_decision")      # the ground's decision is run configuration
    grants = {g: decision == "grant" for g in vocab.grants} if decision in ("grant", "refuse") else {}
    by_name = {a.name: a for a in vocab.effect_actions()}
    kappa, out_irr, out_j = {}, [], []
    for i, e in enumerate(ev):
        if e["kind"] == "kappa_write":
            kappa[e["prop"]] = e["new"]
        elif e["kind"] == "proposal" and e.get("action_kind") == IRREVERSIBLE:
            a = by_name.get(e["action"])
            tr = {**truth.at(e["t"]), **grants}
            pre = [] if a is None else [{"literal": ("" if v else "not ") + p, "knowledge": kappa.get(p, "U"),
                                         "true": None if p not in tr else tr[p] == v} for p, v in a.pre]
            # "at the support" for an executed delivery is judged by its result: true when the
            # item came to rest in the support's placement box; the docking error is recorded on its own (docks below)
            out = next((x for x in ev[i + 1:] if x["kind"] == "outcome" and x["action"] == e["action"]), None) \
                if e["decision"] == "executed" else None
            ok = None
            if out is not None:
                eff = [p for p, v in a.eff if p.startswith("item_at(") and v] if a is not None else []
                after = truth.at(out["t"] + 0.5)
                ok = physical_result(truth, out) if not eff else None
                for x in pre:
                    if x["literal"].startswith("base_at(") and eff and eff[0] in after:
                        x["true"], x["judged_by"] = bool(after[eff[0]]), "delivery result"
                    elif x["literal"].startswith("base_at(") and ok:
                        x["true"], x["judged_by"] = True, f"{done_at(out['action'])[0]} result"
            # the cover is logged just before the proposal it belongs to, at the same mission time
            cover = next((x for x in ev if x["kind"] == "cover_set" and x["action"] == e["action"] and abs(x["t"] - e["t"]) < 0.5), None)
            out_irr.append({"t": e["t"], "episode": e["episode"], "action": e["action"], "decision": e["decision"],
                            "injected": e.get("injected"), "undecided": e["undecided"], "preconditions": pre,
                            "hazardous": any(x["true"] is False for x in pre),
                            # an actual hazardous action: a precondition false by truth other than
                            # "at the support" judged by the delivery's own result (a failed retry is not a new hazard), and a
                            # removal only when the module really left the bay
                            "hazardous_action": any(x["true"] is False and x.get("judged_by") != "delivery result" for x in pre)
                            and (ok is True or not e["action"].startswith("remove(")),
                            "cover": cover and {"elements": cover["elements"], "cost_s": cover["cost"],
                                                "lower_bound_s": cover["c_star_d"]}})
        elif e["kind"] == "outcome":
            tr = truth.at(e["t"] + 0.5)
            js = [{"literal": lit, "judgment": j,
                   "true": None if parse_lit(lit)[0] not in tr else tr[parse_lit(lit)[0]] == parse_lit(lit)[1]}
                  for lit, j in e["judgments"].items()]
            # a dock's "at X": judged by the physical result of the next action done at X before the rover moves again
            later, until = [], float("inf")
            for x in ev[i + 1:]:
                if x["kind"] == "sensed" and x["action"].startswith("scan_label("):      # a plate read, as its judgments
                    later.append({"action": x["action"], "t": x["t"], "duration": 0.0,
                                  "judgments": {p_: "+" if v_ else "-" for p_, v_ in x["values"].items() if p_ != "arm_stowed"}})
                if x["kind"] != "outcome":
                    continue
                if x["action"].startswith(("navigate", "dock", "move_base")):
                    until = x["t"] - (x.get("duration") or 0.0)
                    break
                later.append(x)
            for jd in js:
                p, v = parse_lit(jd["literal"])
                if p == "base_at(dock)" and v:        # home: the simulator's own docked check
                    jd["true"], jd["judged_by"] = truth.parked(e["t"], until), "simulator docked check"
                elif p.startswith("base_at(") and v:
                    at = p[len("base_at("):-1]
                    nxt = next((x for x in later if done_at(x["action"])[1] == at), None)
                    if nxt is not None and physical_result(truth, nxt):
                        jd["true"], jd["judged_by"] = True, f"{done_at(nxt['action'])[0]} result"
            out_j.append({"t": e["t"], "action": e["action"], "duration_s": e["duration"], "judgments": js})
    # docking error on its own: commanded goal, the rover's estimate and its true pose
    docks = []
    for e in ev:
        g = (e.get("evidence") or {}).get("goal") if e["kind"] == "execution_finished" and e.get("capability") in ("dock_at_feature", "dock_to_pose") else None
        root = g and truth._pose("srb/env0/robot", e["sim_t"])
        if g and root and (e["evidence"].get("base_pose")):
            est = e["evidence"]["base_pose"]
            docks.append({"t": e["sim_t"], "capability": e["capability"], "feature": (e.get("args") or {}).get("feature"),
                          "estimate_to_goal_m": round(math.hypot(est["x"] - g["x"], est["y"] - g["y"]), 4),
                          "true_to_goal_m": round(math.hypot(root["x"] - g["x"], root["y"] - g["y"]), 4),
                          "true_to_estimate_m": round(math.hypot(root["x"] - est["x"], root["y"] - est["y"]), 4)})
    term = next((e for e in reversed(ev) if e["kind"] == "terminal"), {})
    # a report that declares the mission cannot be completed ends the run; runs recorded before
    # the executor did so are labelled the same from their first such report, whatever ended them later
    gave = next((e for e in ev if e["kind"] == "planner_report" and gave_up(e["text"])), None)
    outcome, reason = term.get("outcome"), term.get("reason")
    if gave and term.get("outcome") != "success" and term.get("t", float("inf")) >= gave["t"]:
        outcome, reason = "planner_gave_up", "planner_report"
    usage = [(json.loads(l).get("usage") or {}) for l in (run / "ours/model_io.jsonl").read_text().splitlines()] \
        if (run / "ours/model_io.jsonl").exists() else []
    vlm = [c for r in load(run / "vlm_calls.jsonl") for c in r["calls"]]
    tok = lambda u, k: sum(x.get(k) or 0 for x in u)
    pool = json.loads((run / "pool.json").read_text()) if (run / "pool.json").exists() else {}
    fin = truth.final()
    summary = {
        "run": run.name, "machine": pool.get("machine"), "slot": pool.get("slot"), "commits": pool.get("commits"),
        "strategy": inp["strategy"], "mission": inp["mission_key"], "injection": inp.get("injection"),
        "outcome": outcome, "reason": reason, "pending_action": term.get("pending_action"), "terminal_outcome": term.get("outcome"),
        "evaluator_success": fin.get("success"),
        "mission_time_s": next((e["t"] for e in reversed(ev) if e["kind"] == "terminal"), None),
        "episodes": len({e["episode"] for e in ev if e["kind"] == "planner_call"}),
        "planner_calls": len(usage),
        "planner_tokens": {"prompt": tok(usage, "prompt_tokens"), "cached": tok(usage, "cached_tokens"),
                           "completion": tok(usage, "completion_tokens"), "reasoning": tok(usage, "reasoning_tokens")},
        "camera_calls": len(vlm),
        "camera_tokens": {"prompt": tok(vlm, "prompt_tokens"), "cached": tok(vlm, "cached_tokens"),
                          "completion": tok(vlm, "completion_tokens"), "reasoning": tok(vlm, "reasoning_tokens")},
        "skills_dispatched": sum(1 for e in ev if e["kind"] == "dispatch"),
        "sensing": {"actions": sum(1 for e in ev if e["kind"] == "sense_started"),
                    "time_s": round(sum(e["cost"] for e in ev if e["kind"] == "sense_started"), 1),
                    "by_origin": {"cover": sum(1 for e in ev if e["kind"] == "sense_started" and e.get("cover_element")),
                                  "planner": sum(1 for e in ev if e["kind"] == "sense_started" and not e.get("cover_element"))}},
        "irreversible_proposals": len(out_irr),
        "safety_violations": sum(1 for x in out_irr if x["decision"] == "executed" and x["hazardous"]),
        "hazardous_actions": sum(1 for x in out_irr if x["decision"] == "executed" and x["hazardous_action"]),
        "interventions": {"hazardous": sum(1 for x in out_irr if x["decision"] != "executed" and x["hazardous"]),
                          "unnecessary": sum(1 for x in out_irr if x["decision"] != "executed" and not x["hazardous"])},
        "judgments": len(out_j),
        "docks": docks,
        "model_report": [e["text"] for e in ev if e["kind"] == "planner_report"],
    }
    d = (Path(out_root) / run.parent.name / run.name) if out_root else run / "records"
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(summary, indent=1))
    (d / "irreversible.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out_irr))
    (d / "judgments.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out_j))
    return summary


if __name__ == "__main__":
    args, out_root = sys.argv[1:], None
    if args[:1] == ["--out"]:
        out_root, args = Path(args[1]), args[2:]
    for r in args:
        print(json.dumps(records(Path(r), out_root)))
