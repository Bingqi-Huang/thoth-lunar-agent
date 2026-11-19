"""The sensing cover of eq. (cover) and the relaxed problem c*_D of Theorem 2(i).

Given the undecided propositions U of a proposed irreversible action, the start time t and the sensing actions, choose
an ordered sequence that observes every proposition of U at minimal total cost while each element starts within its
window: t + sum of the costs before it <= w. Infeasible is cost INF.

The exact solver enumerates minimal covers (every element contributes a proposition no other element observes; deleting
an element lowers every prefix sum, so a feasible cover stays feasible and never gets dearer when reduced to a minimal
one) and orders each by earliest completion deadline w + c. The greedy solver is the classical weighted set cover of
the greedy bound in Theorem 2(ii), which holds for unbounded windows only.
"""
from __future__ import annotations

import math
from dataclasses import replace

from .types import INF, SENSING, Action


def sequence_cost(seq: list[Action]) -> float:
    """Total cost of a sequence, summed exactly so that it does not depend on the order of the elements."""
    return math.fsum(a.cost for a in seq)


def _edf(seq) -> list[Action]:
    """Earliest-deadline-first on the completion deadline w + c, ties by name."""
    # Exchange: "i then j" fits and w_j + c_j <= w_i + c_i imply "j then i" fits (s + c_j <= w_i); w alone is not enough.
    return sorted(seq, key=lambda a: (a.window + a.cost, a.name))


def _feasible(seq: list[Action], t: float) -> bool:
    for a in seq:
        if t > a.window:
            return False
        t += a.cost
    return True


def _candidates(U: set[str], t: float, sensors) -> list[Action]:
    """Actions that observe something of U and can still start at t, in name order."""
    return sorted((a for a in sensors if a.window >= t and U & set(a.obs)), key=lambda a: a.name)


def _solve(U: set[str], t: float, cands: list[Action]) -> tuple[list[Action] | None, float]:
    """Optimal cover over `cands` (already filtered by _candidates). Ties: lower cost, fewer elements, names."""
    props = sorted(U)
    obs = [U & set(a.obs) for a in cands]
    by_prop = {p: [i for i, o in enumerate(obs) if p in o] for p in props}
    if not all(by_prop.values()):
        return None, INF
    best: list = [None, None]               # [(cost, size, names), sequence]
    seen: set[frozenset] = set()

    def leaf(chosen: tuple[int, ...]):
        key = frozenset(chosen)
        if key in seen:
            return
        seen.add(key)
        for i in chosen:                    # minimal: i observes something the others do not
            if not obs[i] - set().union(*(obs[j] for j in chosen if j != i)):
                return
        seq = _edf([cands[i] for i in chosen])
        if not _feasible(seq, t):
            return
        rank = (sequence_cost(seq), len(seq), tuple(sorted(a.name for a in seq)))
        if best[0] is None or rank < best[0]:
            best[0], best[1] = rank, seq

    def extend(chosen: tuple[int, ...], covered: set[str]):
        if best[0] is not None and math.fsum(cands[i].cost for i in chosen) > best[0][0]:
            return
        rest = [p for p in props if p not in covered]
        if not rest:
            leaf(chosen)
            return
        for i in by_prop[rest[0]]:          # every cover contains an element observing the first undecided proposition
            extend(chosen + (i,), covered | obs[i])

    extend((), set())
    return (None, INF) if best[1] is None else (best[1], best[0][0])


def exact_cover(U: set[str], t: float, sensors: list[Action]) -> tuple[list[Action] | None, float]:
    """The optimal sequence of eq. (cover) over the sensing actions, and its cost; (None, INF) when infeasible."""
    if not U:
        return [], 0.0
    return _solve(U, t, _candidates(U, t, (a for a in sensors if a.kind == SENSING)))


def greedy_cover(U: set[str], t: float, sensors: list[Action]) -> tuple[list[Action] | None, float]:
    """Greedy weighted set cover (maximal new propositions per unit cost, free actions first, ties by name), then
    ordered by completion deadline; (None, INF) if the windows rule that order out."""
    if not U:
        return [], 0.0
    cands = _candidates(U, t, (a for a in sensors if a.kind == SENSING))
    uncovered, picked = set(U), []
    while uncovered:
        scored = [(a, len(uncovered & set(a.obs))) for a in cands]
        scored = [(a, n) for a, n in scored if n]
        if not scored:
            return None, INF
        a, _ = min(scored, key=lambda an: (an[0].cost > 0, -an[1] if an[0].cost == 0 else -an[1] / an[0].cost,
                                           an[0].name))
        picked.append(a)
        uncovered -= set(a.obs)
    seq = _edf(picked)
    return (seq, sequence_cost(seq)) if _feasible(seq, t) else (None, INF)


def lower_bound(U: set[str], t: float, sensors: list[Action], effect_actions: list[Action]) -> float:
    """c*_D of Theorem 2(i): the cover over the sensing actions with w >= t plus every actuation action with a
    proposition of U in its effect, taken as a sensor of those propositions at cost d(a) and unbounded window."""
    if not U:
        return 0.0
    standins = [replace(a, kind=SENSING, obs=tuple(sorted(p for p, _ in a.eff if p in U)), cost=a.duration,
                        window=INF) for a in effect_actions if a.kind != SENSING]
    cands = _candidates(U, t, [a for a in sensors if a.kind == SENSING] + standins)
    return _solve(U, t, cands)[1]
