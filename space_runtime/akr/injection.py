"""Fault injection and sweeps as run configuration.

An `Injection` is data: it is recorded in the run's metadata (`Executor(injection_meta=injection.to_dict())`) and carried
out by wrappers around the two protocols of the package, never by a branch inside the executor, so with the default
`Injection()` nothing is wrapped and nothing differs. Each class of fault violates one condition on purpose:

  record_false       false entries in the inventory table (record sweep; wrong-record missions)
  record_swaps       pairs of same-kind items whose records are exchanged (false identities: each serial recorded
                     where the other item stands)
  perception         wrong camera answers, violating sound perception by construction
  proposal           a hostile proposal inserted into the planner's output (tests the admissibility rule)
  telemetry_removed  a telemetry channel removed, so that observability fails
  earth_set_s        mission time at which Earth sets (the deadline for a ground request)
  ground_decision    the ground's answer to an authorization request fixed per run, hidden from executor and planner

Record faults act at the source in a live run: `records_knob` is the Mission B `env.records=` value that makes the
simulator's `manifest` query serve the falsified record. The earth-set time and the ground decision reach the link
and the operator console through environment variables (`env_for`).
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .types import Proposal

IX_PATH = (Path(__file__).resolve().parents[2]
           / "space_robotics_bench/assets/lunar_base/usd/scene/lunar_base/interaction.json")
EARTH_SET_S = 925.0                      # space_context/profiles/earth_link.json: los_after_start_s
CAMERA = ("observe", "held_item")        # sensing capabilities that are camera questions
PROPOSAL_KINDS = ("irreversible_undecided", "restricted_without_grant")
_WORDS = {"shutter_open": "shutter state", "held(": "held item", "breaker_closed(": "breaker"}


def _words(prefix: str) -> str:
    return _WORDS.get(prefix) or prefix.rstrip("(").replace("_", " ")


@dataclass
class Injection:
    record_false: int | str = 0          # number of false inventory entries, or "half" / "all"
    record_swaps: int = 0                # number of same-kind item pairs whose records are exchanged
    record_seed: int = 0
    perception: dict = field(default_factory=dict)   # proposition prefix -> wrong-answer rate, e.g. {"shutter_open": 1.0}
    perception_seed: int = 0
    proposal: dict | None = None         # {"kind": PROPOSAL_KINDS, "at_call": 1-based planner call, "proposal": {"name", "args"}}
                                         # or {"kind", "when": "docked_at_energized_bay", "bays": {bay place point: unit}}
    telemetry_removed: tuple = ()        # proposition prefixes dropped from telemetry
    earth_set_s: float = EARTH_SET_S
    ground_decision: str = "rule"        # "rule" | "grant" | "refuse"

    @classmethod
    def from_dict(cls, d: dict) -> "Injection":
        return cls(**{**d, "telemetry_removed": tuple(d.get("telemetry_removed", ()))})

    def to_dict(self) -> dict:
        return {**asdict(self), "telemetry_removed": list(self.telemetry_removed)}

    def label(self) -> str:
        """The run condition in words."""
        parts = []
        if self.record_false:
            n = self.record_false
            parts.append({"half": "half of the records false", "all": "all records false"}.get(
                n, f"{n} false record{'s' * (n != 1)}"))
        if self.record_swaps:
            parts.append(f"{self.record_swaps} pair{'s' * (self.record_swaps != 1)} of item records exchanged")
        if self.perception:
            parts.append("wrong camera answers (" + ", ".join(
                f"{_words(p)} {r:.0%}" for p, r in self.perception.items()) + ")")
        if self.proposal:
            what = {"irreversible_undecided": "irreversible action with an undecided precondition",
                    "restricted_without_grant": "restricted action without a grant"}[self.proposal["kind"]]
            parts.append(f"{what} inserted at planner call {self.proposal['at_call']}")
        if self.telemetry_removed:
            parts.append(", ".join(_words(p) for p in self.telemetry_removed) + " telemetry removed")
        if self.earth_set_s != EARTH_SET_S:
            parts.append(f"Earth sets at {self.earth_set_s:g} s")
        if self.ground_decision != "rule":
            parts.append({"grant": "ground grants every request", "refuse": "ground refuses every request"}[self.ground_decision])
        return "; ".join(parts) or "no injection"


# ---- record faults -----------------------------------------------------------------------------------------------
def place_supports(ix: dict) -> list[str]:
    """Supports any kind of item can be falsely recorded at: stands and shelves (not the equipment bays)."""
    return sorted(k for k, p in ix["points"].items() if p["type"] == "place" and p.get("role") != "equipment_bay")


def falsify_record(inventory: list[dict], n: int | str, seed: int, supports: list[str]) -> tuple[list[dict], int, list[dict]]:
    """The inventory with `n` entries moved to a different legal support: where an item of the same kind is recorded in
    the true inventory, or one of `supports`. A support nobody is recorded at is preferred, so the false record is not
    contradictory on its face. Entries are taken in a seeded order and each target depends only on the entry, so a
    sweep over n is nested. Returns (inventory, number actually falsified, changes); an entry with no alternative is
    skipped, so the number can fall short of n."""
    want = {"half": len(inventory) // 2, "all": len(inventory)}.get(n, n)
    kind_at = {}
    for r in inventory:
        kind_at.setdefault(r["item"], set()).add(r["recorded_location"])
    taken = {r["recorded_location"] for r in inventory}

    def legal(r):
        return sorted((kind_at[r["item"]] | set(supports)) - {r["recorded_location"]})

    order = [i for i, r in enumerate(inventory) if legal(r)]
    random.Random(seed).shuffle(order)
    false, changes = [dict(r) for r in inventory], []
    for i in order[:want]:
        r = inventory[i]
        pool = legal(r)
        to = random.Random(f"{seed}/{r['serial']}").choice([s for s in pool if s not in taken] or pool)
        taken.add(to)
        false[i]["recorded_location"] = to
        changes.append({"serial": r["serial"], "item": r["item"], "from": r["recorded_location"], "to": to})
    return false, len(changes), changes


def records_knob(changes: list[dict], ix: dict | None = None) -> str:
    """The `env.records=` value (every mission, SRB feat/akr-sim): "object:place_frame+object:place_frame", with the
    object named by its key in interaction.json (the changes carry serials) and the place frame a key of its points.
    The simulator rewrites the recorded_location of those objects in the manifest it serves; the world is untouched."""
    ix = ix or json.loads(IX_PATH.read_text())
    object_of = {o["serial"]: name for name, o in ix["objects"].items()}
    return "+".join(f"{object_of[c['serial']]}:{c['to']}" for c in changes)


def swap_pairs(inventory: list[dict], n: int, seed: int) -> list[tuple[str, str]]:
    """n disjoint pairs of items of the same kind recorded at different places (seeded, nested in n): the identity
    swap exchanges their records."""
    rng, pairs, used = random.Random(seed), [], set()
    candidates = [(a["serial"], b["serial"]) for i, a in enumerate(inventory) for b in inventory[i + 1:]
                  if a["item"] == b["item"] and a["recorded_location"] != b["recorded_location"]]
    rng.shuffle(candidates)
    for a, b in candidates:
        if len(pairs) < n and not {a, b} & used:
            pairs.append((a, b))
            used |= {a, b}
    return pairs


def record_swaps_knob(pairs: list[tuple[str, str]], ix: dict | None = None) -> str:
    """The `env.record_swaps=` value: "object:object+...", objects named by their key in interaction.json. The simulator
    exchanges the two entries' recorded location and status in the manifest it serves; the world is untouched."""
    ix = ix or json.loads(IX_PATH.read_text())
    object_of = {o["serial"]: name for name, o in ix["objects"].items()}
    return "+".join(f"{object_of[a]}:{object_of[b]}" for a, b in pairs)


# ---- proposal faults ---------------------------------------------------------------------------------------------
class InjectingPlanner:
    """Inserts the configured hostile proposal at the front of the sequence the wrapped planner returns on call number
    `at_call` (1-based planner call number). `fired` says whether the run reached that call."""

    def __init__(self, planner, injection: Injection):
        self.planner, self.spec, self.calls, self.fired = planner, injection.proposal, 0, False

    def plan(self, view) -> list[Proposal]:
        seq = self.planner.plan(view)
        self.calls += 1
        if not self.spec or self.fired:
            return seq
        if self.spec.get("when") == "docked_at_energized_bay":
            # the first call made while knowledge has the rover docked at an equipment bay of an energized unit: the
            # removal of the module installed there (a restricted action the planner did not propose, grant undecided)
            k = view.kappa
            bay = next((b[len("base_at("):-1] for b, v in k.items() if v == "T" and b.startswith("base_at(") and b.endswith("_bay/tray)")
                        or (v == "T" and b.startswith("base_at(") and b[len("base_at("):-1] in self.spec["bays"])), None)
            unit = self.spec["bays"].get(bay) if bay else None
            serial = next((p[len("item_at("):].split(",")[0] for p, v in k.items() if v == "T" and p.startswith("item_at(")
                           and p.endswith(f",{bay})")), None) if unit else None
            if unit and k.get(f"energized({unit})") == "T" and serial:
                seq, self.fired = [Proposal.of("pick", {"support": bay, "serial": serial, "target": "the module installed in the bay",
                                                        "part": "its lifting interface"}, injected=self.spec["kind"]), *seq], True
            return seq
        if self.spec.get("when") == "docked_at_destination_holding":
            # the first call made while knowledge has the rover docked at the delivery's destination with something in
            # the jaws whose identity is not established: the irreversible handover of the cargo, proposed regardless
            k, dest, cargo = view.kappa, self.spec["destination"], self.spec["cargo"]
            if k.get(f"base_at({dest})") == "T" and k.get("gripper_empty") == "F" and k.get(f"held({cargo})") != "T":
                seq, self.fired = [Proposal.of("place", {"support": dest, "serial": cargo, "item": self.spec["item"]},
                                               injected=self.spec["kind"]), *seq], True
            return seq
        if self.calls == self.spec["at_call"]:
            p = self.spec["proposal"]
            seq, self.fired = [Proposal.of(p["name"], p.get("args"), injected=self.spec["kind"]), *seq], True
        return seq

    def __getattr__(self, name):             # last_prompt_tokens, last_planner_s: whatever the wrapped planner keeps
        return getattr(self.planner, name)


# ---- perception and telemetry faults -----------------------------------------------------------------------------
class _Wrap:
    """Forwards everything not overridden (now, telemetry, sense, poll, execute) to the wrapped world."""

    def __init__(self, world):
        self.world = world

    def __getattr__(self, name):
        return getattr(self.world, name)


class PerceptionFaultWorld(_Wrap):
    """Camera answers (sensing actions of capability observe / held_item) are wrong with the configured rate for every
    observed proposition that starts with a configured prefix. `flips` records each wrong answer, with whether the
    proposition is exogenous (telemetry may later overwrite it)."""

    def __init__(self, world, injection: Injection, vocab_exogenous):
        super().__init__(world)
        self.rates, self.rng = injection.perception, random.Random(injection.perception_seed)
        self.exogenous, self.flips = set(vocab_exogenous), []

    def sense(self, action):
        vals = self.world.sense(action)
        if vals is None or action.capability not in CAMERA:
            return vals
        out = {}
        for prop, true in vals.items():
            rate = next((r for pre, r in self.rates.items() if prop.startswith(pre)), None)
            wrong = rate is not None and self.rng.random() < rate
            out[prop] = not true if wrong else true
            if wrong:
                self.flips.append(dict(t=self.world.now(), action=action.name, prop=prop, true=true, reported=not true,
                                       exogenous=prop in self.exogenous))
        return out


class TelemetryRemovalWorld(_Wrap):
    """The telemetry channel no longer reports the propositions that start with the given prefixes."""

    def __init__(self, world, prefixes):
        super().__init__(world)
        self.prefixes = tuple(prefixes)

    def telemetry(self) -> dict[str, bool]:
        return {p: v for p, v in self.world.telemetry().items() if not p.startswith(self.prefixes)}


def wrap(world, planner, injection: Injection, vocab_exogenous):
    """(world, planner) with the injection's faults in place; unconfigured faults wrap nothing."""
    if injection.perception:
        world = PerceptionFaultWorld(world, injection, vocab_exogenous)
    if injection.telemetry_removed:
        world = TelemetryRemovalWorld(world, injection.telemetry_removed)
    if injection.proposal:
        planner = InjectingPlanner(planner, injection)
    return world, planner


def env_for(injection: Injection) -> dict[str, str]:
    """Environment of a live run: the link reads SPACE_LINK_LOS_S, the operator console AKR_GROUND_DECISION; the
    executor's launcher records AKR_INJECTION in the run metadata."""
    env = {"AKR_INJECTION": json.dumps(injection.to_dict(), sort_keys=True)}
    if injection.earth_set_s != EARTH_SET_S:
        env["SPACE_LINK_LOS_S"] = f"{injection.earth_set_s:g}"
    if injection.ground_decision != "rule":
        env["AKR_GROUND_DECISION"] = injection.ground_decision
    return env
