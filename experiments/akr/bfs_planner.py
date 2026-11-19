"""The breadth-first planner of plan_length as a Planner for the executor (Monte Carlo v1, dry cases).

Each call plans from the decided part of knowledge, with the record's literals tried first as values for the
propositions knowledge leaves undecided (the manuscript passes R_cur as preferred values), then without them. Plans are
valid in the optimistic relaxation of manuscript Sec. 3.6 (an assumption satisfies any actuation precondition,
irreversible ones included), so the planner is complete in the sense of that section. With an executor subgoal it solves
the route problem (the subgoal, reversible actions only). Under the K-replanner rule every plan, main plan and route,
is the original K-replanner's: no actuation precondition is assumed, so an action with an
undecided precondition is preceded by a sensing action that decides it, whose outcome is taken as the plan needs.
AKR and the FF-Replan rule plan in the relaxation.
The plan's actions are returned as the capability calls the vocabulary grounds back to the same actions.
"""
from space_runtime.akr.types import IRREVERSIBLE, Proposal

from experiments.akr.plan_length import _args, bfs_plan, exclusive_closure


def to_proposal(vocab, a) -> Proposal:
    """The capability call that `vocab.ground` maps to action `a`."""
    c, args = a.capability, _args(a.name)
    if c in ("navigate_to_feature", "dock_at_feature", "press"):
        return Proposal.of(c, {"feature": args[0]})
    if c in ("navigate_to_pose", "dock_to_pose"):
        x, y, yaw = map(float, args)
        return Proposal.of(c, {"x": x, "y": y, "yaw": yaw})
    if c in ("pick", "place"):
        return Proposal.of(c, {"serial": args[0], "support": args[1]})
    if c in ("stow", "arm_ready"):
        return Proposal.of(c)
    if c == "scan_label":
        return Proposal.of(c, {"support": args[0]})
    if c == "request_operator":
        return Proposal.of(c, {"kind": "authorization", "grant": a.obs[0]})
    return Proposal.of("check_proposition", {"proposition": a.obs[0]})


def consistent(start: dict, v) -> dict:
    """The decided part of knowledge with every group of mutually exclusive propositions that knowledge holds
    contradictorily (wrong observations can leave such groups) left undecided, so that the plan decides them again
    by sensing: an item in at most one place (held, or on one support), a support with at most one item and an
    occupancy that agrees with the items known on it, and empty jaws holding nothing. Knowledge itself is not changed."""
    out, drop = dict(start), set()
    for x in v.serials:
        g = [f"held({x})"] + [f"item_at({x},{k})" for k in v.supports]
        if sum(start.get(p) is True for p in g) > 1:
            drop.update(g)
    for k in v.supports:
        items, occ = [f"item_at({x},{k})" for x in v.serials], v._occ(k)
        n_true = sum(start.get(p) is True for p in items)
        if n_true > 1 or (start.get(occ) is False and n_true) or (start.get(occ) is True and all(start.get(p) is False for p in items)):
            drop.update(items + [occ])
    held = [f"held({x})" for x in v.serials]
    if start.get("gripper_empty") is True and any(start.get(p) is True for p in held):
        drop.update(held + ["gripper_empty"])
    for p in drop:
        out.pop(p, None)
    return out


class BfsPlanner:
    def __init__(self, vocab, limit: int = 2_000_000, rule: str = "akr"):
        self.vocab, self.limit, self.calls, self.rule = vocab, limit, 0, rule
        self.last_plan: list = []
        self.memo: dict = {}
        self.extra_goal: list = []              # planner-side oracle remedy (hidden-cause repair missions only)
        self.tried_failed: set = set()          # planner-side memory: undecided preconditions of executed actions that failed
        self.failed = lambda: set()             # the executor's failure set F (Sec. 3.6: no assumption on it)
        self.lengths: list[int] = []            # length of the plan returned at each call (0: none found)
        self.last_prompt_tokens = self.last_planner_s = None

    def plan(self, view) -> list[Proposal]:
        self.calls += 1
        v = self.vocab
        raw = {p: x == "T" for p, x in view.kappa.items()}
        start = consistent(raw, v)
        goal, rev = (list(view.subgoal), True) if view.subgoal else (list(view.goal) + list(self.extra_goal), False)
        plan = None
        # planner-side memory, the same for every rule: an executed irreversible action judged other than + leaves its
        # undecided preconditions unassumable for the planner (the executor's F records this for reversible actions
        # only), so a failed irreversible attempt is not repeated on the same assumption
        acts = {a.name: a for a in v.effect_actions()}
        for h in view.history:
            if h.get("kind") == "executed" and any(j != "+" for j in h.get("judgments", {}).values()) and h["action"] in acts \
                    and acts[h["action"]].kind == IRREVERSIBLE:
                self.tried_failed |= {p for p, _ in acts[h["action"]].pre if view.kappa.get(p) is None}
        # no assumption on these (Sec. 3.6), the record's preferred value neither; contradictory knowledge left undecided
        # by `consistent` is re-observed before it is relied on (in F for planning: it can only be decided by sensing)
        failed = set(self.failed()) | {p for p in self.tried_failed if p not in start} | (set(raw) - set(start))
        prefer = {**{p: val for p, val in view.record if p not in start and p not in failed}, **start}
        start, prefer = exclusive_closure(v, start), exclusive_closure(v, prefer)
        # the full search first; only when it finds nothing within its limit, the search with one location hypothesis
        # per unplaced item (plan_length.focus, narrow)
        # last, when neither finds a plan within the limit, the full search again with a ten times larger limit
        big = max(10 * self.limit, 10_000_000)      # last resort: plans of 10-12 steps need up to 10M states
        for st, narrow, lim in ((prefer, False, self.limit), (start, False, self.limit), (prefer, True, self.limit),
                                (start, True, self.limit), (prefer, False, big), (start, False, big)):
            # bfs_plan is a function of these inputs alone: a repeated call (knowledge unchanged after a call that
            # returned no plan) reuses the earlier result instead of searching again
            key = (frozenset(st.items()), tuple(view.record), tuple(goal), rev, frozenset(failed), narrow, lim)
            if key not in self.memo:
                self.memo[key] = bfs_plan(v, start=st, record=list(view.record), goal=goal, reversible_only=rev,
                                          known_only=self.rule == "kreplanner", failed=failed, limit=lim, narrow=narrow)
            plan = self.memo[key]
            # an empty plan under the record's defaults means only that the record makes the goal look met; the goal
            # is not known, so plan again without the defaults (the plan then decides the goal by sensing or acting)
            if plan is not None and (plan or st is start):
                break
        self.lengths.append(len(plan or []))
        if not plan:
            self.last_plan = []
            return []
        by = {a.name: a for a in v.effect_actions() + v.sensors(None)}
        self.last_plan = [by[n] for n in plan]
        return [to_proposal(v, a) for a in self.last_plan]
