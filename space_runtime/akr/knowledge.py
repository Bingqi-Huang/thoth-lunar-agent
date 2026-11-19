"""The knowledge state kappa, the current record R_cur and the failure set F of Definition 3, with the update operator of
eq. (update).

A value enters kappa only through K0, a sensing result (telemetry included) or a judgment on an executed effect -- the
only writers below -- which is what Theorem 1(i) needs. The record is never written into kappa; it only shrinks.
Every update ends with pruning, and every change is logged (`kappa_write`, `record_pruned`, `failure_set`).
"""
from __future__ import annotations

from .types import Action, Literal, lit_str


class Knowledge:
    def __init__(self, k0: list[Literal], record: list[Literal], log):
        self.log = log                          # log(event, /, **fields)
        self.kappa: dict[str, str] = {}         # prop -> "T" | "F"; absent = "U"
        self.record: list[Literal] = list(record)
        self.F: set[str] = set()
        self.executed: set[str] = set()         # propositions in the effect of an action executed in this run
        for p, v in k0:
            self._write(p, "T" if v else "F", "k0", "k0")
        self._prune()

    # ---- reads -----------------------------------------------------------------------------------------------
    def value(self, p: str) -> str:
        return self.kappa.get(p, "U")

    def models(self, lit: Literal) -> bool:
        return self.value(lit[0]) == ("T" if lit[1] else "F")

    def known_false(self, lit: Literal) -> bool:
        return self.value(lit[0]) == ("F" if lit[1] else "T")

    def undecided(self, action: Action) -> set[str]:
        """U(a, kappa): the propositions of pre(a) on which kappa is undecided."""
        return {p for p, _ in action.pre if self.value(p) == "U"}

    def decided_view(self) -> dict[str, str]:
        return dict(self.kappa)

    # ---- the three update operations of eq. (update) -------------------------------------------------------------
    def sense(self, action: Action, values: dict[str, bool]) -> None:
        for p in action.obs:                    # a reading that decided only part of obs (an unclear answer) writes that part
            if p in values:
                self._write(p, "T" if values[p] else "F", "sensing", action.name)
        if values.get("gripper_empty") is True:  # a gripper seen empty holds nothing
            for p in [q for q in self.kappa if q.startswith("held(")]:
                self._write(p, "F", "sensing", action.name)
        self._prune()

    def telemetry(self, values: dict[str, bool]) -> list[str]:
        """Sensing update for sigma_x; returns the props whose decided value it overturned."""
        changed = [p for p, v in values.items() if self.value(p) not in ("U", "T" if v else "F")]
        for p, v in values.items():
            self._write(p, "T" if v else "F", "telemetry", "telemetry")
        self._prune()
        return changed

    def judge(self, action: Action, judgments: dict[Literal, str]) -> None:
        """+ writes the value of the effect literal, - the opposite, ? returns the proposition to U."""
        self.executed.update(p for p, _ in action.eff)
        for e in action.eff:
            j = judgments[e]
            self._write(e[0], "U" if j == "?" else "T" if e[1] == (j == "+") else "F", "judgment", action.name)
        self._prune()

    def add_failures(self, props: set[str]) -> None:
        """F := F u props, pruned by kappa (a proposition decided meanwhile never enters)."""
        new = {p for p in props if self.value(p) == "U"} - self.F
        if new:
            self.F |= new
            self.log("failure_set", added=sorted(new), removed=[])

    # ---- internals -------------------------------------------------------------------------------------------
    def _write(self, p: str, new: str, source: str, by: str) -> None:
        old = self.value(p)
        if old == new:
            return
        if new == "U":
            del self.kappa[p]
        else:
            self.kappa[p] = new
        self.log("kappa_write", prop=p, old=old, new=new, source=source, by=by)

    def _prune(self) -> None:
        """R_cur := literals whose proposition is undecided; F := propositions that are undecided."""
        for lit in [l for l in self.record if self.value(l[0]) != "U"]:
            self.record.remove(lit)
            self.log("record_pruned", literal=lit_str(lit), refuted=not self.models(lit),
                     robot_changed=lit[0] in self.executed)
        gone = sorted(p for p in self.F if self.value(p) != "U")
        if gone:
            self.F -= set(gone)
            self.log("failure_set", added=[], removed=gone)
