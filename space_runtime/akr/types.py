"""Shared types and the event schema of the AKR executor (the contract between executor, accounting and dry runs)."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

Literal = tuple[str, bool]          # (proposition, value): ("breaker_closed(pdu_1)", True) is the positive literal
STRATEGIES = ("akr", "ffreplan", "kreplanner")
REVERSIBLE, IRREVERSIBLE, SENSING = "reversible", "irreversible", "sensing"
REPORT = "report_to_operator"       # the planner's terminal report (advisory)
INF = math.inf


def lit_str(l: Literal) -> str:
    return l[0] if l[1] else f"not {l[0]}"


def parse_lit(s: str) -> Literal:
    return (s[4:], False) if s.startswith("not ") else (s, True)


@dataclass(frozen=True)
class Proposal:
    """One element of a planner's sequence, as the model's tool call names it."""
    name: str
    args: tuple = ()                 # sorted (key, value) pairs; use Proposal.of(name, {...})
    injected: str | None = None      # proposal fault label when inserted by injection.py, else None

    @staticmethod
    def of(name: str, args: dict | None = None, injected: str | None = None) -> "Proposal":
        return Proposal(name, tuple(sorted((args or {}).items())), injected)

    @property
    def kwargs(self) -> dict:
        return dict(self.args)


@dataclass(frozen=True)
class Action:
    """A grounded action of Definition 2."""
    name: str                        # unique ground name, e.g. "pick(pdu_3_bay/slot_1)" or "ground_request(granted_remove(pdu_3_bay/slot_1))"
    capability: str                  # the boundary capability or sensing service that realizes it
    args: tuple                      # sorted (key, value) pairs passed to the capability
    kind: str                        # REVERSIBLE | IRREVERSIBLE | SENSING
    pre: tuple[Literal, ...] = ()
    eff: tuple[Literal, ...] = ()    # actuation only; a consistent set of literals
    obs: tuple[str, ...] = ()        # sensing only
    cost: float = 0.0                # sensing: c(sigma), mission time
    window: float = INF              # sensing: latest admissible start w_sigma
    duration: float = 0.0            # actuation: d(a), mission time (for c*_D of Theorem 2)
    grant: str | None = None         # restricted irreversible action: its grant proposition (also in pre)


@dataclass
class Outcome:
    """What world.execute returns: a judgment per effect literal, and the mission time the action took."""
    judgments: dict[Literal, str]    # literal -> "+" | "-" | "?"
    duration: float
    evidence: dict = field(default_factory=dict)   # the measurements the evidence rule read (logged, never consulted)


@dataclass
class PlannerView:
    """The planner's input (Definition 4): knowledge state, current record, goal, executor subgoal, history."""
    kappa: dict[str, str]            # decided propositions only ("T"/"F"); absent = U
    record: list[Literal]            # R_cur, preferred defaults for undecided propositions
    goal: list[Literal]
    subgoal: list[Literal] | None
    history: list[dict]              # executor feedback since the last call: what was executed / refused and why
    t: float
    episode: int


# ---- event schema: kind -> fields (besides `t`). accounting.py and the checks read only these. ---------------------
EVENT_FIELDS = {
    "run_started": ("strategy", "mission", "cap_L", "injection", "goal", "k0", "record", "exogenous", "grants"),
    "kappa_write": ("prop", "old", "new", "source", "by"),      # source: k0 | sensing | telemetry | judgment; by: action name or "telemetry"
    "telemetry": ("values", "changed_decided"),                 # changed_decided: P_x props whose decided value telemetry overturned
    "record_pruned": ("literal", "refuted", "robot_changed"),   # refuted: decided against the literal; robot_changed: prop was in an executed effect before
    "failure_set": ("added", "removed"),
    "planner_call": ("episode", "subgoal", "kappa", "record", "failure_set", "open_sensing"),
    "planner_returned": ("episode", "proposals", "prompt_tokens", "planner_s"),
    "proposal": ("episode", "index", "name", "args", "action", "action_kind", "undecided", "decision", "injected"),
    "decision_point": ("id", "action", "undecided", "proposer"),   # proposer: planner | injection
    "dispatch": ("action", "action_kind", "pre", "kappa_pre", "episode", "decision_point"),
    "outcome": ("action", "judgments", "duration"),
    "sense_started": ("action", "obs", "cost", "window", "cover_element"),
    "sensed": ("action", "values"),
    "cover_set": ("action", "undecided", "elements", "cost", "greedy_cost", "c_star_ge_t", "c_star_d", "decision_point"),
    "cover_cleared": ("reason",),
    "subgoal": ("posted", "literals"),
    "episode_end": ("episode", "reason", "executed"),
    "planner_report": ("episode", "text"),
    "terminal": ("outcome", "reason", "pending_action"),       # outcome: success | certified_hold | budget_exhausted
}
# decision values on a `proposal` event
DECISIONS = ("executed", "sensed", "invalid_known_false", "not_admissible", "one_shot_cover", "irreversible_cover",
             "subgoal_pending", "sensing_not_executable", "hold", "ungroundable", "report", "skipped_by_cap")

# A report that declares the mission cannot be completed ends the run for every method: the
# planner's own status line, read from the head of its report (every give-up report of the campaign states it there,
# e.g. "MISSION STATUS: NOT COMPLETE", "FINAL REPORT -- ...: NOT ACHIEVED", "MISSION CLOSED AS IMPOSSIBLE").
GAVE_UP_HEAD = 300
GAVE_UP = re.compile(r"\b(NOT (COMPLETE|COMPLETED|COMPLETABLE|ACHIEVED|ACHIEVABLE|MET|RESTORED|DELIVERED)|INCOMPLETE|IMPOSSIBLE|"
                     r"CANNOT BE (ACHIEVED|COMPLETED)|COULD NOT BE (ACHIEVED|COMPLETED|RESTORED|DELIVERED))\b", re.I)


def gave_up(report: str) -> bool:
    """True when the head of a planner report declares that the mission cannot be completed."""
    return bool(GAVE_UP.search(" ".join(report.split())[:GAVE_UP_HEAD]))
