"""Offline cases of the Monte Carlo trials: with the complete planner and no faults (eps = p_R = eta = 0, rho = q = 1),
every acting rule completes the ten missions the declared model can solve, none completes the six repair missions, and
no trial executes an irreversible action on a false precondition. The occupied destination completes under the
K-replanner rule (its routes use no assumption).

    python3 -m pytest experiments/akr/test_montecarlo.py
"""
import pytest

from experiments.akr.montecarlo import MISSIONS, REPAIR, RULES, trial


def run(mission, rule):
    return trial(dict(mission=mission, rule=rule, eps=0.0, p_R=0.0, eta=0.0, seed=1, rho=1.0, q=1.0))


def test_kreplanner_occupied_destination():
    r = run("b_l2_destination_occupied_habitat", "kreplanner")
    assert r["outcome"] == "success" and r["violations"] == 0, r


@pytest.mark.parametrize("rule", RULES)
def test_no_faults(rule):
    rows = [run(m, rule) for m in MISSIONS]
    assert all(r["violations"] == 0 for r in rows)
    expected = set(MISSIONS) - REPAIR
    assert {r["mission"] for r in rows if r["outcome"] == "success"} == expected, \
        [(r["mission"], r["outcome"], r["reason"]) for r in rows]


def test_akr_no_hold_without_restricted_action():
    """With sampled skill outcomes (rho, q of the trials) and no faults, AKR never holds on a mission the declared
    model can solve: a held-item or occupancy cover is always reachable by a route (Algorithm 1), and only a ground
    request past its deadline can make a cover infeasible."""
    for m in sorted(set(MISSIONS) - REPAIR):
        for seed in range(8):
            r = trial(dict(mission=m, rule="akr", eps=0.0, p_R=0.0, eta=0.0, seed=seed))
            assert r["outcome"] != "certified_hold" and r["violations"] == 0, (m, seed, r["outcome"], r["reason"], r["pending"])


@pytest.mark.parametrize("mission,seed", [("b_l1_bat_spare_bess", 4100985876), ("b_l2_destination_occupied_habitat", 751472108)])
def test_cover_follows_knowledge(mission, seed):
    """Two trials whose cover once outlived its purpose: a held-item check kept after the item's identity was decided
    otherwise (the pick refuted it), its precondition then known false (the support found occupied), so the subgoal could
    not be reached and the planner had nothing to offer. The cover now drops an element whose propositions are decided
    and is computed again without an element whose precondition is known false and not reachable."""
    r = trial(dict(mission=mission, rule="akr", eps=0.0, p_R=0.0, eta=0.0, seed=seed))
    assert r["outcome"] == "success" and r["reason"] != "planner_call_guard", r
