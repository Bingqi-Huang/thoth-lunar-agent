"""The executor loop of asymmetric knowledge-level replanning, with a strategy switch between three acting rules.

Soundness holds by construction: kappa is written only by `Knowledge` (K0, sensing, telemetry, judgments); an irreversible
action under akr / kreplanner reaches `_execute` only through the branch where U(a, kappa) is empty and no
precondition is known false, immediately after the telemetry update of `_tick`; a refusal ends the run in `_refused`;
under akr a reversible action with an undecided precondition in F goes to a cover, never to `_execute`.
Methods return True when the run has ended (`self.out` set) wherever the run can end inside them.
"""
from __future__ import annotations

from .cover import exact_cover, greedy_cover, lower_bound
from .knowledge import Knowledge
from .types import IRREVERSIBLE, REPORT, SENSING, PlannerView, Proposal, gave_up, lit_str


def _strs(lits) -> list[str]:
    return [lit_str(l) for l in lits]



#: sensing services that are not skills on the robot and are not charged to the skill budget: the ground request, the
#: equipment telemetry and the free localization and arm-state reads
NON_SKILL = frozenset({"request_operator", "power_status", "base_pose", "arm_state"})

class Executor:
    def __init__(self, vocab, world, planner, strategy, cap_L, max_planner_calls, goal, k0, record, log,
                 injection_meta=None, mission=None, skill_budget=None):
        """`skill_budget`: the run's budget of skill calls -- every skill dispatched to the
        robot, actuation and sensing, the executor's own cover elements included; not planner calls, not ground requests,
        telemetry or the free localization / arm-state reads. `max_planner_calls` is only an infrastructure guard."""
        self.vocab, self.world, self.planner, self.strategy = vocab, world, planner, strategy
        self.cap_L, self.max_calls, self.goal, self.k0, self.record = cap_L, max_planner_calls, goal, k0, record
        self.skill_budget, self.skills = skill_budget, 0
        self.log, self.meta, self.mission = log, injection_meta, mission
        self.K: Knowledge
        self.cover: list = []                   # pending cover: sensing actions in order
        self.cover_ran: set[str] = set()        # elements of the pending cover already executed (each runs once)
        self.cover_for: set[str] = set()        # the undecided preconditions the pending cover was set for
        self.cover_action: str | None = None    # the action they are preconditions of
        self.cover_excluded: set[str] = set()   # sensors left out of it: a precondition known false, not reachable
        self.subgoal: list | None = None
        self.inflight: dict = {}                # sensing action name -> action, readings that have not arrived
        self.history: list[dict] = []           # executor feedback for the next planner call
        self.dp_of: dict[str, int] = {}         # irreversible action name -> id of its latest decision point
        self.calls = self.dp_n = self.episode = self.executed = 0
        self.in_ep = False
        self.deferred = None                    # an episode whose sequence ran out, kept open for a continuing planner
        self.out: dict | None = None
        self.proposed: list = []                # actuation actions proposed so far (a hold names the one it is for)

    def _log(self, event: str, /, **fields) -> None:
        self.log(event, t=self.world.now(), **fields)

    def run(self) -> dict:
        self._log("run_started", strategy=self.strategy, mission=self.mission, cap_L=self.cap_L, injection=self.meta,
                  goal=_strs(self.goal), k0=_strs(self.k0),
                  record=_strs(self.record), exogenous=sorted(self.vocab.exogenous),
                  grants=sorted(self.vocab.grants))
        self.K = Knowledge(self.k0, self.record, self._log)
        while self.out is None:
            self._iteration()
        return self.out

    def _finish(self, outcome: str, reason: str, pending: str | None = None) -> bool:
        if self.in_ep or self.deferred:
            self._end("hold" if outcome == "certified_hold" else outcome, final=True)
        self._log("terminal", outcome=outcome, reason=reason, pending_action=pending)
        self.out = dict(outcome=outcome, reason=reason, pending_action=pending)
        return True

    def _end(self, reason: str, final: bool = False) -> None:
        self.in_ep = False
        if reason == "sequence_exhausted" and not final and hasattr(self.planner, "continues"):
            self.deferred = reason              # may go on with the next call; logged once it really ends
            return
        self.deferred = None
        self._log("episode_end", episode=self.episode, reason=reason, executed=self.executed)

    # ---- the outer loop -----------------------------------------------------------------------------------------
    def _iteration(self) -> None:
        if self._tick():
            return
        if self.cover:
            if self._process_cover() or not self.cover:     # ended, or cleared: back to the telemetry update
                return
        if self.calls == self.max_calls or getattr(self.planner, "exhausted", False):
            self._finish("budget_exhausted", "planner_call_guard")      # infrastructure guard, not the mission budget
            return
        self._episode()

    def _refresh(self) -> bool:
        """Telemetry and poll, the update that precedes every action start."""
        values = self.world.telemetry()
        changed = self.K.telemetry(values)
        self._log("telemetry", values=values, changed_decided=changed)
        if changed:
            self.history.append(dict(kind="exogenous_change", props=changed))
        for s, vals in self.world.poll():
            self.inflight.pop(s.name, None)
            self._sensed(s, vals)
        return self._refused()

    def _restricted_for(self, grant: str) -> str | None:
        """The restricted action this grant is for: the latest one proposed, else the vocabulary's grounding of it for
        the item knowledge has where it acts (the module installed in the bay), else its first grounding."""
        for e in reversed(self.proposed):
            if e.grant == grant:
                return e.name
        cands = [a for a in self.vocab.effect_actions() if a.grant == grant]
        known = [a for a in cands if all(self.K.models(l) for l in a.pre if l[0].startswith("item_at("))]
        return (known or cands or [None])[0] and (known or cands)[0].name

    def _refused(self) -> bool:
        """A grant proposition written F ends the run now: a refusal is permanent."""
        for g in sorted(self.vocab.grants):
            if self.K.value(g) == "F":
                return self._finish("certified_hold", "refused", self._restricted_for(g))
        return False

    def _tick(self) -> bool:
        """Telemetry update, then the success check on knowledge alone."""
        if self._refresh():
            return True
        if all(self.K.models(l) for l in self.goal):
            return self._finish("success", "goal_known")
        return False

    # ---- sensing ----------------------------------------------------------------------------------------------
    def _skill(self, capability: str) -> bool:
        """Charges one skill call against the run's budget; False (and the run ends) when the budget is spent."""
        if capability in NON_SKILL:
            return True
        if self.skill_budget is not None and self.skills >= self.skill_budget:
            self._finish("budget_exhausted", "skill_budget_exhausted")
            return False
        self.skills += 1
        return True

    def _do_sense(self, s, cover_element: bool) -> bool:
        if not self._skill(s.capability):
            return True
        self._log("sense_started", action=s.name, obs=list(s.obs), cost=s.cost, window=s.window,
                  cover_element=cover_element, skills_used=self.skills)
        vals = self.world.sense(s)
        if vals is None:                        # a ground request: the reply comes through poll
            self.inflight[s.name] = s
        else:
            self._sensed(s, vals)
        return self._refused()

    def _sensed(self, s, vals: dict) -> None:
        self._log("sensed", action=s.name, values=vals)
        self.history.append(dict(kind="sensed", action=s.name, values=vals))
        self.K.sense(s, vals)

    def _set_subgoal(self, lits: list | None) -> None:
        if lits == self.subgoal:
            return
        self.subgoal = lits
        self._log("subgoal", posted=lits is not None, literals=_strs(lits or []))
        if lits is not None:
            self.history.append(dict(kind="subgoal", literals=_strs(lits)))

    def _clear_cover(self, reason: str) -> None:
        self._log("cover_cleared", reason=reason)
        self.cover, self.cover_ran, self.cover_for, self.cover_excluded = [], set(), set(), set()
        self._set_subgoal(None)

    def _unreachable(self, s) -> bool:
        """A precondition of sensor s is known false and no reversible action makes it true. A location is always
        reached by moving: it stays a route goal."""
        from .types import REVERSIBLE
        for l in s.pre:
            if self.K.known_false(l) and not l[0].startswith(("base_at(", "at_view(")) and not any(
                    a.kind == REVERSIBLE and l in a.eff for a in self.vocab.effect_actions()):
                return True
        return False

    def _process_cover(self) -> bool:
        """The pending cover, with the in-flight extension: an element whose reading is in flight counts as started."""
        K, fresh = self.K, True                 # `_tick` has just run
        while True:
            if not fresh and self._refresh():
                return True
            fresh = False
            # an element leaves the cover once executed, whatever its reading decided: a reading that leaves its
            # propositions undecided (a plate that cannot be read) is not repeated; the planner is asked again. It also
            # leaves once the propositions it was in the cover for are decided, however they were
            self.cover = [s for s in self.cover if s.name not in self.cover_ran
                          and any(p in self.cover_for and K.value(p) == "U" for p in s.obs)]
            todo = [s for s in self.cover if s.name not in self.inflight]
            if todo and self._unreachable(todo[0]):
                # its precondition can no longer be brought about: the cover is computed again without it
                self.cover_excluded.add(todo[0].name)
                left = {p for p in self.cover_for if K.value(p) == "U"}
                t = self.world.now()
                sensors = [s for s in self.vocab.sensors(K) if s.name not in self.cover_excluded | self.cover_ran]
                seq, cost = exact_cover(left, t, sensors)
                self._log("cover_set", action=self.cover_action, undecided=sorted(left),
                          elements=None if seq is None else [s.name for s in seq], cost=cost,
                          greedy_cost=greedy_cover(left, t, sensors)[1], c_star_ge_t=cost,
                          c_star_d=lower_bound(left, t, sensors, self.vocab.effect_actions(K)),
                          decision_point=self.dp_of.get(self.cover_action), excluded=sorted(self.cover_excluded))
                if seq is None:                 # no feasible cover: certified hold
                    self._finish("certified_hold", "no_feasible_cover", self.cover_action)
                    return True
                self.cover = seq
                continue
            if not todo or not (all(K.models(l) for l in todo[0].pre) and self.world.now() <= todo[0].window):
                break
            self.cover_ran.add(todo[0].name)
            if self._do_sense(todo[0], True):
                return True
        if not self.cover:
            self._clear_cover("exhausted")
        elif not todo:                          # only readings in flight remain: no subgoal, the robot keeps working
            self._set_subgoal(None)
        elif self.world.now() > todo[0].window:
            self._clear_cover("window_closed")
        else:
            self._set_subgoal(list(todo[0].pre))
        return False

    # ---- one episode: a planner call and up to L of its elements -----------------------------------------------------
    def _episode(self) -> None:
        K, t = self.K, self.world.now()
        self.calls += 1
        call = dict(subgoal=self.subgoal and _strs(self.subgoal), kappa=K.decided_view(), record=_strs(K.record),
                    failure_set=sorted(K.F), open_sensing=[s.name for s in self.vocab.sensors(K) if s.window >= t])
        view = PlannerView(kappa=K.decided_view(), record=list(K.record), goal=self.goal,
                           subgoal=self.subgoal and list(self.subgoal), history=self.history, t=t, episode=self.episode)
        self.history = []
        seq = self.planner.plan(view)
        # A planner consulted at every decision (the live model) keeps its episode while its working plan
        # is unchanged and the previous call's sequence ran out without a break; the cap L counts across those calls.
        if not (self.deferred and self.planner.continues and self.executed < self.cap_L):
            if self.deferred:
                self._end(self.deferred, final=True)
            self.episode, self.executed = self.episode + 1, 0
        self.deferred, self.in_ep = None, True
        self._log("planner_call", episode=self.episode, **call)
        self._log("planner_returned", episode=self.episode,
                  proposals=[dict(name=p.name, args=p.kwargs, injected=p.injected) for p in seq],
                  prompt_tokens=getattr(self.planner, "last_prompt_tokens", None),
                  planner_s=getattr(self.planner, "last_planner_s", None))
        L = self.cap_L - self.executed
        for i, p in enumerate(seq[:L]):
            reason = self._element(i, p, seq)
            if self.out:
                return
            if reason:
                return self._end(reason)
        for i, p in enumerate(seq[L:], L):
            self._propose(i, p, None, set(), "skipped_by_cap")
        self._end("cap" if len(seq) > L else "sequence_exhausted")

    def _propose(self, i: int, p: Proposal, a, U: set, decision: str) -> str:
        self._log("proposal", episode=self.episode, index=i, name=p.name, args=p.kwargs, action=a and a.name,
                  action_kind=a and a.kind, undecided=sorted(U), decision=decision, injected=p.injected)
        if decision not in ("executed", "sensed"):
            self.history.append(dict(kind="refused", action=a.name if a else p.name, decision=decision,
                                     undecided=sorted(U)))
        return decision

    def _element(self, i: int, p: Proposal, seq: list[Proposal]) -> str | None:
        """One element of the planner's sequence. Returns the reason the episode ends, or None to go on."""
        K, now = self.K, self.world.now
        if self._tick():
            return None
        if self.subgoal is not None and all(K.models(l) for l in self.subgoal):
            self._set_subgoal(None)
            return "subgoal_reached"
        if p.name == REPORT:
            text = " ".join(str(v) for _, v in p.args)
            self._log("planner_report", episode=self.episode, text=text)
            self._propose(i, p, None, set(), "report")
            if gave_up(text):           # a declared inability ends the run at once
                self._finish("planner_gave_up", "planner_report")
                return None
            return "report"
        a = self.vocab.ground(p, K)
        if a is None:
            return self._propose(i, p, None, set(), "ungroundable")
        if a.kind != SENSING:
            self.proposed.append(a)
        if a.kind == SENSING:                   # a reading already in flight is not sent twice
            grant = next((q for q in a.obs if q in self.vocab.grants), None)
            if grant is not None and now() > a.window and K.value(grant) == "U" and self.strategy != "ffreplan":
                # a request for a grant proposed after its latest request time w = T_vis - c_g can never
                # be answered in this mission, so the restricted action it is for can never become admissible (under
                # ffreplan an undecided grant does not block the action, so nothing is certified: the request is refused)
                self._propose(i, p, a, set(), "late_request")
                self._finish("certified_hold", "late_request", self._restricted_for(grant))
                return None
            if not (all(K.models(l) for l in a.pre) and now() <= a.window and a.name not in self.inflight):
                return self._propose(i, p, a, set(), "sensing_not_executable")
            self._propose(i, p, a, set(), "sensed")
            self.executed += 1
            self._do_sense(a, False)
            return None
        U = K.undecided(a)
        if any(K.known_false(l) for l in a.pre):
            return self._propose(i, p, a, U, "invalid_known_false")
        irr = a.kind == IRREVERSIBLE
        if self.strategy == "ffreplan":       # admissible unless a precondition is known false
            cov, dec = set(), ""
        elif self.strategy == "akr" and not irr:    # one-shot rule: undecided and already failed once
            cov, dec = U & K.F, "one_shot_cover"
        else:                                   # akr irreversible, kreplanner: every precondition known true
            cov, dec = U, "irreversible_cover" if irr else "not_admissible"
        if cov:
            if self.subgoal is not None:
                return self._propose(i, p, a, U, "subgoal_pending")
            return self._cover(i, p, a, U, cov, dec)
        if irr and U:                           # ffreplan only: counted as a decision point, then executed
            self._decision_point(a, U, p)
        if not self._skill(a.capability):
            return None
        self._propose(i, p, a, U, "executed")
        self.executed += 1
        judgments = self._execute(a, self.dp_of.pop(a.name, None) if irr else None)
        bad = {e[0] for e, j in judgments.items() if j != "+"}
        if irr or not bad:
            return None
        K.add_failures(U)
        later = set()
        for r in seq[i + 1:]:
            b = r.name != REPORT and self.vocab.ground(r, K)
            if b:
                later |= {q for q, _ in b.pre}
        return "revfail" if bad & later else None

    def _decision_point(self, a, U: set, p: Proposal) -> None:
        self.dp_n += 1
        self.dp_of[a.name] = self.dp_n
        self._log("decision_point", id=self.dp_n, action=a.name, undecided=sorted(U),
                  proposer="injection" if p.injected else "planner")

    def _cover(self, i: int, p: Proposal, a, U: set, cov: set, dec: str) -> str | None:
        """Set the sensing cover for an irreversible action. Propositions whose reading is in flight count as covered."""
        irr = a.kind == IRREVERSIBLE
        new = cov - {q for s in self.inflight.values() for q in s.obs}
        if irr and (new or a.name not in self.dp_of):
            self._decision_point(a, U, p)
        dp = self.dp_of.get(a.name) if irr else None
        if new:
            t, sensors = self.world.now(), self.vocab.sensors(self.K)
            seq, cost = exact_cover(new, t, sensors)
            self._log("cover_set", action=a.name, undecided=sorted(new),
                      elements=None if seq is None else [s.name for s in seq], cost=cost,
                      greedy_cost=greedy_cover(new, t, sensors)[1], c_star_ge_t=cost,
                      c_star_d=lower_bound(new, t, sensors, self.vocab.effect_actions(self.K)), decision_point=dp)
            if seq is None:                     # no feasible cover: certified hold
                self._propose(i, p, a, U, "hold")
                self._finish("certified_hold", "no_feasible_cover", a.name)
                return None
            if self.cover:
                self._log("cover_cleared", reason="replaced")
            self.cover, self.cover_ran = seq, set()
            self.cover_for, self.cover_action, self.cover_excluded = set(new), a.name, set()
            self.history.append(dict(kind="cover", action=a.name, undecided=sorted(new),
                                     elements=[s.name for s in seq], cost=cost))
        return self._propose(i, p, a, U, dec)

    def _execute(self, a, dp: int | None) -> dict:
        K = self.K
        self._log("dispatch", action=a.name, action_kind=a.kind, pre=_strs(a.pre),
                  kappa_pre={q: K.value(q) for q, _ in a.pre}, episode=self.episode, decision_point=dp)
        out = self.world.execute(a)
        judged = {lit_str(l): j for l, j in out.judgments.items()}
        self._log("outcome", action=a.name, judgments=judged, duration=out.duration, evidence=out.evidence)
        self.history.append(dict(kind="executed", action=a.name, judgments=judged))
        K.judge(a, out.judgments)
        return out.judgments
