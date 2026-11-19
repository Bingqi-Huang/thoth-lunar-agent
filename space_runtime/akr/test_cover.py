"""Tests of the sensing cover (eq. cover) and its relaxation c*_D (Theorem 2): optimality against brute force, window
ordering, the greedy gap and bound, and the deterministic tie-break."""
import itertools
import random

from pytest import approx

from space_runtime.akr.cover import exact_cover, greedy_cover, lower_bound, sequence_cost
from space_runtime.akr.types import INF, IRREVERSIBLE, REVERSIBLE, SENSING, Action

GRANT, BREAKER, PLATE = "granted_remove(pdu_3)", "breaker_closed(pdu_3)", "serial(slot_1)=A"


def sensor(name, obs, cost, window=INF):
    return Action(name, "sense", (), SENSING, obs=tuple(obs), cost=cost, window=window)


def actuator(name, eff, duration, kind=REVERSIBLE):
    return Action(name, "act", (), kind, eff=tuple((p, True) for p in eff), duration=duration)


def names(seq):
    return [a.name for a in seq]


def fits(seq, t):
    """The window constraint of eq. (cover), written out independently of the module."""
    return all(t + sum(a.cost for a in seq[:i]) <= a.window for i, a in enumerate(seq))


def test_empty_u_costs_nothing():
    assert exact_cover(set(), 0.0, [sensor("cam", [BREAKER], 5.0)]) == ([], 0.0)
    assert greedy_cover(set(), 0.0, []) == ([], 0.0)
    assert lower_bound(set(), 0.0, [], []) == 0.0


def test_single_sensor_and_missing_proposition():
    cam = sensor("cam", [BREAKER], 5.0)
    assert exact_cover({BREAKER}, 0.0, [cam]) == ([cam], 5.0)
    assert exact_cover({BREAKER, PLATE}, 0.0, [cam]) == (None, INF)      # nothing observes PLATE


def test_one_sensor_for_both_versus_two_partial_ones():
    a, b = sensor("a", [BREAKER], 3.0), sensor("b", [PLATE], 3.0)
    seq, cost = exact_cover({BREAKER, PLATE}, 0.0, [a, b, sensor("both", [BREAKER, PLATE], 5.0)])
    assert names(seq) == ["both"] and cost == 5.0
    seq, cost = exact_cover({BREAKER, PLATE}, 0.0, [a, b, sensor("both", [BREAKER, PLATE], 7.0)])
    assert sorted(names(seq)) == ["a", "b"] and cost == 6.0


def test_request_runs_before_the_camera_question_that_would_close_its_window():
    request = sensor("ground_request", [GRANT], 308.5, 616.0)
    camera = sensor("camera", [BREAKER], 200.0)
    seq, cost = exact_cover({GRANT, BREAKER}, 500.0, [camera, request])
    assert names(seq) == ["ground_request", "camera"] and cost == 508.5 and fits(seq, 500.0)
    assert not fits([camera, request], 500.0)                                   # the reverse order is infeasible


def test_order_is_by_completion_deadline_not_by_window():
    long = sensor("long", [BREAKER], 100.0, 10.0)
    short = sensor("short", [PLATE], 1.0, 11.0)
    seq, cost = exact_cover({BREAKER, PLATE}, 0.0, [long, short])
    assert names(seq) == ["short", "long"] and cost == 101.0 and fits(seq, 0.0)
    assert not fits([long, short], 0.0)                                         # window order would fail here


def test_closed_window_is_infeasible_and_excluded_candidates_are_ignored():
    request = sensor("ground_request", [GRANT], 308.5, 616.0)
    camera = sensor("camera", [BREAKER], 5.0)
    assert exact_cover({GRANT, BREAKER}, 616.0, [request, camera])[1] == 313.5          # w == t still admissible
    assert exact_cover({GRANT, BREAKER}, 700.0, [request, camera]) == (None, INF)
    assert greedy_cover({GRANT, BREAKER}, 700.0, [request, camera]) == (None, INF)
    cheap_late = sensor("cheap_late", [BREAKER], 1.0, 499.0)
    seq, cost = exact_cover({BREAKER}, 500.0, [cheap_late, camera])
    assert names(seq) == ["camera"] and cost == 5.0


def test_greedy_is_worse_than_exact_on_the_classic_instance_within_the_harmonic_bound():
    u = {f"p{i}" for i in range(1, 7)}
    t1 = sensor("t1", ["p1", "p3", "p5"], 1.0)
    t2 = sensor("t2", ["p2", "p4", "p6"], 1.0)
    g1 = sensor("g1", ["p1", "p2", "p3", "p4"], 1.2)
    g2 = sensor("g2", ["p5", "p6"], 1.0)
    sensors = [g1, g2, t1, t2]
    opt_seq, opt = exact_cover(u, 0.0, sensors)
    gr_seq, gr = greedy_cover(u, 0.0, sensors)
    assert sorted(names(opt_seq)) == ["t1", "t2"] and opt == 2.0
    assert sorted(names(gr_seq)) == ["g1", "g2"] and gr == sequence_cost(gr_seq) == approx(2.2)
    m = 4                                                                       # max |obs & U|
    assert opt < gr <= sum(1 / k for k in range(1, m + 1)) * opt


def test_greedy_gives_up_when_its_cover_has_no_feasible_order_but_exact_finds_one():
    x, y = sensor("x", ["a"], 5.0, 3.0), sensor("y", ["b"], 5.0, 3.0)
    z = sensor("z", ["a", "b"], 11.0)
    assert greedy_cover({"a", "b"}, 0.0, [x, y, z]) == (None, INF)
    seq, cost = exact_cover({"a", "b"}, 0.0, [x, y, z])
    assert names(seq) == ["z"] and cost == 11.0


def test_lower_bound_is_below_the_cover_when_an_effect_action_is_cheaper_than_the_sensor():
    request = sensor("ground_request", [GRANT], 308.5, 616.0)
    camera = sensor("camera", [BREAKER], 20.0)
    press = actuator("press(breaker_pdu_3)", [BREAKER, "lamp_on"], 4.0)
    unrelated = actuator("dock", ["docked"], 1.0)
    u = {GRANT, BREAKER}
    cover = exact_cover(u, 0.0, [request, camera])[1]
    assert cover == 328.5
    assert lower_bound(u, 0.0, [request, camera], [press, unrelated]) == 312.5
    assert lower_bound(u, 0.0, [request, camera], [unrelated]) == cover


def test_lower_bound_is_infinite_when_a_grant_is_observable_only_by_a_closed_request():
    request = sensor("ground_request", [GRANT], 308.5, 616.0)
    camera = sensor("camera", [BREAKER], 20.0)
    press = actuator("press(breaker_pdu_3)", [BREAKER], 4.0)
    remove = actuator("remove(module)", ["module_removed"], 30.0, IRREVERSIBLE)
    assert lower_bound({GRANT, BREAKER}, 700.0, [request, camera], [press, remove]) == INF
    assert lower_bound({GRANT, BREAKER}, 600.0, [request, camera], [press, remove]) == 308.5 + 4.0


def test_ties_break_by_cost_then_size_then_name_whatever_the_input_order():
    a, b = sensor("a_cam", [BREAKER], 3.0), sensor("b_cam", [BREAKER], 3.0)
    for pool in ([a, b], [b, a]):
        assert names(exact_cover({BREAKER}, 0.0, pool)[0]) == ["a_cam"]
    one = sensor("z_one", [BREAKER, PLATE], 2.0)
    two = [sensor("a1", [BREAKER], 1.0), sensor("a2", [PLATE], 1.0)]
    assert names(exact_cover({BREAKER, PLATE}, 0.0, two + [one])[0]) == ["z_one"]       # same cost, fewer elements
    assert names(exact_cover({BREAKER, PLATE}, 0.0, [one] + two[::-1])[0]) == ["z_one"]


def test_exact_cover_matches_brute_force_over_all_subsets_and_orders():
    rng = random.Random(7)
    props = ["p0", "p1", "p2"]
    for _ in range(200):
        n = rng.randint(1, 6)
        sensors = [sensor(f"s{i}", rng.sample(props, rng.randint(1, 3)), float(rng.randint(0, 9)),
                          rng.choice([INF, float(rng.randint(0, 25))])) for i in range(n)]
        t = float(rng.randint(0, 6))
        pool = [s for s in sensors if s.window >= t]
        best = INF
        for k in range(1, len(pool) + 1):
            for perm in itertools.permutations(pool, k):
                if set().union(*(p.obs for p in perm)) >= set(props) and fits(list(perm), t):
                    best = min(best, sum(p.cost for p in perm))
        seq, cost = exact_cover(set(props), t, sensors)
        assert cost == best
        assert (seq is None) == (best == INF) and (seq is None or fits(seq, t))
