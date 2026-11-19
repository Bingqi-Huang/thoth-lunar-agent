"""Condition 3 check: the length of the shortest valid plan for each mission's goal from the initial knowledge state.

A plan is valid in the sense of the manuscript's compiled problem under the AKR admissibility rule: a reversible action
applies unless one of its preconditions is known false (and then writes its effects), an irreversible action only when
every precondition is known true, a sensing action when its preconditions are known true, and it returns the value the
plan chooses for each proposition it observes. Breadth-first over the propositions relevant to the goal (the backward
closure of preconditions over the actions that can change them), from k0 plus the telemetry snapshot of a recorded run.

    python3 -m experiments.akr.plan_length [--runs artifacts/runs/akr]

`bfs_plan` is the planner the Monte Carlo study reuses.
"""
import os
import argparse
import glob
import json
import re
from collections import deque
from pathlib import Path

from space_runtime.akr.types import IRREVERSIBLE, SENSING
from space_runtime.akr.vocabulary import Vocabulary, load_config

RUNS = os.path.expandvars("${THOTH_ROOT}/artifacts/runs/akr")


def snapshots(root: str) -> dict[str, dict]:
    """One recorded snapshot (site, telemetry, manifest) per mission key, from the AKR runs' event logs."""
    out = {}
    for f in sorted(glob.glob(f"{root}/*/*/ours/events.jsonl")):
        site = inputs = None
        for line in open(f):
            if '"site_knowledge"' in line and site is None:
                site = json.loads(line)["site"]
            elif '"akr_inputs"' in line:
                inputs = json.loads(line)
                break
        if site and inputs and inputs["mission_key"] not in out:
            out[inputs["mission_key"]] = dict(site=site, telemetry=inputs["telemetry"], manifest=inputs["manifest"], run=f)
    return out


def relevant(vocab: Vocabulary, actions: list, goal: list | None = None) -> tuple[set, list]:
    """The propositions the goal depends on and the actions that change or observe them."""
    R = {p for p, _ in (goal or vocab.goal())}
    while True:
        acts = [a for a in actions if R & {p for p, _ in a.eff} or (a.kind == SENSING and R & set(a.obs))]
        R2 = R | {p for a in acts for p, _ in a.pre}
        if R2 == R:
            return R, acts
        R = R2


def _args(name: str) -> list[str]:
    inner = name[name.index("(") + 1:-1] if "(" in name else ""
    return [x for x in re.split(r"[(),]", inner) if x]


def focus(vocab: Vocabulary, start: dict, record: list, goal: list, failed: set = frozenset(), narrow: bool = False) -> tuple[set, set]:
    """The serials and site points a shortest plan can need. Plan length does not depend on which free staging supports
    items are set down on, so all free supports but one per item are left out (symmetric choices); the rest is what the goal
    names: the goal units' bays, panels, shutters and views, the cargo and its destination, whatever may be in the jaws,
    where knowledge or the record has those items (anywhere knowledge does not rule out, for an item neither places),
    and whatever stands on those supports."""
    units = {p[p.index("(") + 1:-1] for p, _ in goal if p.startswith("breaker_closed(")}
    bays = [b for b in vocab.bays if vocab.pts[b].get("unit") in units]
    serials = {x for b in bays for x in vocab.serials if start.get(f"item_at({x},{b})") is True}
    serials |= {vocab.cargo} if vocab.cargo else set()
    serials |= {_args(p)[0] for p, _ in goal if p.startswith("item_at(")}      # every item the goal names
    serials |= {x for x in vocab.serials if start.get(f"held({x})") is not False}   # what may be in the jaws
    where = lambda x: [k for k in vocab.supports if start.get(f"item_at({x},{k})") is True or (f"item_at({x},{k})", True) in record]
    points = set(bays) | {k for x in serials for k in where(x)} | {a for p, _ in goal if p.startswith("item_at(") for a in _args(p)[1:]}
    # an item that neither knowledge nor the record places can be wherever knowledge does not rule it out
    # (narrow: only the first such support in name order -- one hypothesis at a time, for when the full search exceeds
    # its limit; a refuted hypothesis rules that support out and the next call takes the next one)
    for x in [x for x in serials if not where(x) and start.get(f"held({x})") is not True]:
        cands = sorted(k for k in vocab.supports if start.get(f"item_at({x},{k})") is not False)
        if narrow:                              # a hypothesis that may be assumed first (not in the failure set F)
            cands = sorted(cands, key=lambda k: f"item_at({x},{k})" in failed)[:1]
        points |= set(cands)
    points |= {k for k in vocab.dockable if vocab.pts[k].get("unit") in units and vocab.pts[k]["type"] == "press"}
    named = {_args(p_)[0] for p_, _ in goal if p_.startswith(("base_at(", "at_view("))}   # a route problem's targets
    points |= named
    serials |= {x for x in vocab.serials for k in points if f"item_at({x},{k})" in start and start[f"item_at({x},{k})"]}
    may_be_on = lambda x, k: (start.get(f"item_at({x},{k})") is not False and start.get(f"held({x})") is not True
                              and not any(start.get(f"item_at({x},{k2})") is True for k2 in vocab.supports if k2 != k))
    # a support known occupied with no item known on it (e.g. after a failed pick of its occupier left the occupier's
    # location undecided): every item that knowledge does not rule out there may be the occupier to move
    for k in [k for k in points if k in vocab.supports and start.get(vocab._occ(k)) is True]:
        if not any(start.get(f"item_at({x},{k})") is True for x in vocab.serials):
            # only items that may be there: not ruled out on it, not known elsewhere or in the jaws
            serials |= {x for x in vocab.serials if may_be_on(x, k)}
    # a support the goal needs free whose occupancy is undecided (e.g. a place on it judged inconclusive): the item that
    # may still be on it, unless the record places that item elsewhere
    need_free = {_args(p)[0] for p, val in goal if p.startswith("occupied(") and val is False}
    for k in [k for k in points & need_free if k in vocab.supports and start.get(vocab._occ(k)) is None]:
        if not any(start.get(f"item_at({x},{k})") is True for x in vocab.serials):
            serials |= {x for x in vocab.serials if may_be_on(x, k) and not any(
                (f"item_at({x},{k2})", True) in record for k2 in vocab.supports if k2 != k)}
    free = [k for k in vocab.staging() if k not in points and vocab._occ(k) not in failed
            and start.get(vocab._occ(k)) is not True and not any(start.get(f"item_at({x},{k})") for x in vocab.serials)]
    # supports known free first, then those no record entry claims, then those only the record claims
    free.sort(key=lambda k: (start.get(vocab._occ(k)) is not False,
                             any((f"item_at({x},{k})", True) in record for x in vocab.serials)))
    points |= set(free[:len(serials)])          # one place to set each item down: the cargo and an occupier both
    views = {v for u in units for v in (vocab.cfg["camera"]["lamp"].get(u), vocab.cfg["camera"]["shutter"].get(u)) if v}
    return serials, points | views | {"dock"}


def _in_focus(a, vocab: Vocabulary, serials: set, points: set) -> bool:
    if a.capability in ("navigate_to_feature",):
        return False                            # no location claim: never part of a shortest plan
    if a.capability in ("navigate_to_pose", "dock_to_pose"):
        at = [p for p, v in a.eff if v]
        name = at[0][at[0].index("(") + 1:-1] if at else None
        return name in points if at else False
    toks = set(_args(a.name)) | {p[p.index("(") + 1:-1] for p in a.obs if p.startswith(("at_view(",))}
    toks |= {t for p, _ in a.pre for t in _args(p)}
    if a.capability == "observe":
        return any(_args(p)[0] in {vocab.pts[k].get("unit") for k in points if k in vocab.pts} for p in a.obs)
    return all(t in serials or t in points or t not in set(vocab.serials) | set(vocab.pts) | set(vocab.views) for t in toks)


def exclusive_consequences(vocab: Vocabulary, eff) -> list:
    """What an effect implies through the site's exclusivity -- an item rests on at most one support and is not both held
    and resting, a support carries at most one item: an item put on a support is on no other and not held, the support
    carries no other item; an item taken into the jaws rests nowhere. Planner-side only (the vocabulary's effects are
    unchanged); without it the search could assume an item on a support it has just filled."""
    own = {p for p, _ in eff}
    out = []
    for p, v in eff:
        if v and p.startswith("item_at("):
            x, k = p[len("item_at("):-1].split(",")
            out += [(f"item_at({x},{k2})", False) for k2 in vocab.supports if k2 != k]
            out += [(f"item_at({y},{k})", False) for y in vocab.serials if y != x] + [(f"held({x})", False)]
        elif v and p.startswith("held("):
            x = p[len("held("):-1]
            out += [(f"item_at({x},{k2})", False) for k2 in vocab.supports]
    seen, res = set(own), []
    for q, w in out:
        if q not in seen:
            seen.add(q); res.append((q, w))
    return res


def exclusive_closure(vocab: Vocabulary, start: dict) -> dict:
    """The same exclusivity applied to a planning start state: what is placed (or held) rules out the other places."""
    out = dict(start)
    if start.get("gripper_empty") is True:      # empty jaws hold nothing
        for x in vocab.serials:
            out.setdefault(f"held({x})", False)
    if any(start.get(f"held({x})") is True for x in vocab.serials):
        out.setdefault("gripper_empty", False)
    for x in vocab.serials:
        at = [k for k in vocab.supports if start.get(f"item_at({x},{k})") is True]
        if at or start.get(f"held({x})") is True:
            for k in vocab.supports:
                out.setdefault(f"item_at({x},{k})", False)
    for k in vocab.supports:
        if any(start.get(f"item_at({x},{k})") is True for x in vocab.serials):
            for x in vocab.serials:
                out.setdefault(f"item_at({x},{k})", False)
    return out


def bfs_plan(vocab: Vocabulary, telemetry: dict | None = None, limit: int = 2_000_000, k0: dict | None = None,
             start: dict | None = None, record: list | None = None, goal: list | None = None,
             reversible_only: bool = False, irreversible_on_knowledge: bool = False,
             known_only: bool = False, failed: set | None = None, narrow: bool = False) -> list[str] | None:
    """The shortest valid plan (action names) for the goal, or None when none exists within `limit` expanded states.
    `start` (proposition -> bool, the decided part of knowledge) replaces k0 plus the telemetry reading; `goal` and
    `reversible_only` give the route problem of a sensing action (its preconditions, reversible actions only). The
    compiled problem does not represent irreversibility (an assumption satisfies any actuation precondition);
    `irreversible_on_knowledge` makes it a planner that takes irreversible steps on knowledge only; `known_only` admits
    an actuation action only when every precondition is known true (no assumptions: the K-replanner rule's own).
    `failed` is the failure set F: an undecided proposition in F has no assumption action (Sec. 3.6), so a precondition
    on it must be decided first."""
    goal = list(goal or vocab.goal())
    if start is None:
        start = {**(k0 or vocab.k0()), **vocab.telemetry_values(telemetry)}
    record = vocab.record() if record is None else record
    serials, points = focus(vocab, start, record, goal, set(failed or ()), narrow)
    actions = [a for a in vocab.effect_actions() + vocab.sensors(None) if _in_focus(a, vocab, serials, points)
               and not (reversible_only and a.kind == IRREVERSIBLE)]
    R, acts = relevant(vocab, actions, goal)
    want = {p: v for p, v in goal}
    for a in acts:
        for p, v in a.pre:
            want.setdefault(p, v)
    # bit sets over R: K the decided propositions, V their values (V is a subset of K)
    ix = {p: i for i, p in enumerate(sorted(R))}
    bits = lambda lits, val=None: sum(1 << ix[p] for p, v in lits if p in ix and (val is None or v == val))
    comp = []                                   # (name, sensing, knowledge only, pre mask, pre true, changed, set true)
    for a in acts:
        pos, neg = bits(a.pre, True), bits(a.pre, False)
        if a.kind == SENSING:
            obs = [p for p in a.obs if p in ix]
            comp.append((a.name, True, True, pos | neg, pos, bits((p, True) for p in obs),
                         bits((p, True) for p in obs if want.get(p, True))))
        else:
            eff = list(a.eff) + exclusive_consequences(vocab, a.eff)
            comp.append((a.name, False, known_only or (a.kind == IRREVERSIBLE and irreversible_on_knowledge), pos | neg, pos,
                         bits(eff), bits(eff, True)))
    K0 = sum(1 << ix[p] for p in start if p in ix)
    V0 = sum(1 << ix[p] for p, v in start.items() if p in ix and v)
    gm, gv = bits(goal), bits(goal, True)
    fm = sum(1 << ix[p] for p in (failed or ()) if p in ix)
    done = lambda K, V: K & gm == gm and V & gm == gv
    if done(K0, V0):
        return []
    seen, q = {(K0, V0): None}, deque([(K0, V0)])
    while q:
        if len(seen) > limit:
            bfs_plan.exhausted = False          # stopped at the limit: no conclusion
            return None
        K, V = q.popleft()
        nK, KVt, KVf = ~K, K & V, K & ~V
        # among plans of the same length, the one resting on fewer assumptions is found first: actions whose
        # preconditions are all decided are expanded before those that need an assumption (applicable actions in
        # their declared order, stably sorted by that count -- the same order as sorting all of them first)
        succ = []
        for name, sensing, known, pre, pos, m, mv in comp:
            if known:                           # every precondition known true (sensing; irreversible if asked)
                if K & pre != pre or V & pre != pos:
                    continue
            elif KVt & (pre & ~pos) or KVf & pos or pre & fm & nK:   # no precondition known false, none
                continue                                         # undecided in F
            if sensing:                         # decides its undecided propositions, to the value the plan needs
                m &= nK
                if not m:
                    continue
            succ.append(((pre & nK).bit_count(), name, m, mv))
        succ.sort(key=lambda t: t[0])
        for _, name, m, mv in succ:
            n = (K | m, (V & ~m) | (mv & m))
            if n in seen:
                continue
            seen[n] = ((K, V), name)
            if done(*n):
                plan, cur = [], n
                while seen[cur] is not None:
                    cur, nm = seen[cur]
                    plan.append(nm)
                return plan[::-1]
            q.append(n)
    bfs_plan.exhausted = True                   # every reachable state expanded: no plan exists in this model
    return None


def record_known(vocab: Vocabulary, start: dict) -> dict:
    """A later knowledge state: every recorded location read (the record's items where it has them, nowhere else), the
    supports' occupancy following, and the units' heat-rejection shutters read as they are drawn in the contingency
    missions (shutter_open false for the goal units with a shutter): the state after the assumptions a shortest plan
    from k0 rests on have been refuted."""
    st = dict(start)
    for p_, v in vocab.record():
        if p_.startswith("item_at("):
            x = _args(p_)[0]
            st.update({f"item_at({x},{k})": False for k in vocab.supports if f"item_at({x},{k})" not in st})
            st[p_] = v
    for k in vocab.supports:
        if k not in vocab.bays:
            st.setdefault(f"occupied({k})", any(st.get(f"item_at({x},{k})") for x in vocab.serials))
    units = {_args(p_)[0] for p_, _ in vocab.goal() if p_.startswith("breaker_closed(")}
    st.update({f"shutter_open({u})": False for u in units if u in vocab.shutter_units})
    return st


def initial(mission: str) -> dict:
    """The mission's initial state (site, telemetry, manifest), generated from its scenario (initial_states.py)."""
    from experiments.akr.initial_states import MISSIONS, generate, parse_env
    if not hasattr(initial, "static"):
        d = json.loads((Path(__file__).parent / "data" / "site_static.json").read_text())   # as the campaign runs recorded it
        initial.static = d["site"], d["equipment_catalogue"]
    short, scene, scenario, env, key = next(m for m in MISSIONS.values() if m[4] == mission)
    e = {**parse_env(env), "scenario": scenario.replace("covered_misreport", "covered")}
    return generate("A" if "mission_a" in scene else "B", e, 0.0, *initial.static)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--later", action="store_true", help="also from the later state of record_known")
    a = ap.parse_args()
    cfg = load_config()
    for m in cfg["missions"]:
        s = initial(m)
        vocab = Vocabulary(cfg, s["site"], s["telemetry"], s["manifest"], m)
        start = {**vocab.k0(), **vocab.telemetry_values(s["telemetry"])}
        plan = bfs_plan(vocab, start=start)
        print(f"{m:36s} k0     {('none' if plan is None else len(plan)):>4}  {' > '.join(plan or [])}", flush=True)
        if a.later:
            plan = bfs_plan(vocab, start=record_known(vocab, start))
            print(f"{m:36s} later  {('none' if plan is None else len(plan)):>4}  {' > '.join(plan or [])}", flush=True)


if __name__ == "__main__":
    main()
