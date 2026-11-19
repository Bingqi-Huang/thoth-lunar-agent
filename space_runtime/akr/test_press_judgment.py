"""The judgment on a press's equipment effects is the telemetry reading after the press (manuscript Sec. 4.2): verified
when the reading shows the breaker closed, refuted when it shows it tripped; what telemetry does not report (the
shutter, the folded arm) stays unverified. Uses the site, telemetry and manifest snapshot of a recorded run.

    python3 -m pytest space_runtime/akr/test_press_judgment.py
"""
import os
import copy
import glob
import json
import types

import pytest

from space_runtime.akr.types import Proposal
from space_runtime.akr.vocabulary import Vocabulary, load_config

RUNS = os.path.expandvars("${THOTH_ROOT}/artifacts/runs/akr")


def snapshot(mission: str):
    for f in sorted(glob.glob(f"{RUNS}/*/*/ours/events.jsonl")):
        site = inputs = None
        for line in open(f):
            if '"site_knowledge"' in line and site is None:
                site = json.loads(line)["site"]
            elif '"akr_inputs"' in line:
                inputs = json.loads(line)
                break
        if site and inputs and inputs["mission_key"] == mission:
            return site, inputs["telemetry"], inputs["manifest"]
    pytest.skip(f"no recorded snapshot of {mission}")


@pytest.fixture(scope="module")
def l2():
    site, tele, man = snapshot("a_l2_radiator_cover_pdu1")
    return Vocabulary(load_config(), site, tele, man, "a_l2_radiator_cover_pdu1"), tele


def reading(tele, breaker: str, cause=None):
    t = copy.deepcopy(tele)
    u = t["units"]["pdu_1"]
    u.update(breaker=breaker, energized=breaker == "closed", lamp="green" if breaker == "closed" else "red", trip_cause=cause)
    return t


def reset_action(v):
    return v.ground(Proposal.of("press", {"feature": "pdu_1_panel/press"}), types.SimpleNamespace(value=lambda p: "U", record=[]))


def test_reset_verified_by_telemetry(l2):
    v, tele = l2
    a = reset_action(v)
    j = {f"{'' if val else 'not '}{p}": x for (p, val), x in v.judge(a, [], {}, reading(tele, "closed")).items()}
    assert j["breaker_closed(pdu_1)"] == j["energized(pdu_1)"] == "+"
    assert j["arm_stowed"] == "?"                       # the folded arm is not measured by the press's evidence


def test_reset_refuted_by_telemetry(l2):
    v, tele = l2
    a = reset_action(v)
    j = {p: x for (p, _), x in v.judge(a, [], {}, reading(tele, "tripped", "ORU_SELFTEST_FAIL")).items()}
    assert j["breaker_closed(pdu_1)"] == j["energized(pdu_1)"] == "-"


def test_without_reading_unverified(l2):
    v, _ = l2
    assert set(v.judge(reset_action(v), [], {}).values()) == {"?"}


def test_shutter_press_unverified(l2):
    v, tele = l2
    a = v.ground(Proposal.of("press", {"feature": "pdu_1_radiator/cover_button"}), types.SimpleNamespace(value=lambda p: "U", record=[]))
    assert set(v.judge(a, [], {}, reading(tele, "closed")).values()) == {"?"}   # the shutter is not in telemetry


def test_only_press_reads_telemetry(l2):
    v, tele = l2
    a = v.ground(Proposal.of("dock_at_feature", {"feature": "pdu_1_panel/press"}), types.SimpleNamespace(value=lambda p: "U", record=[]))
    assert all(x == "?" for x in v.judge(a, [], {}, reading(tele, "closed")).values())   # no base-pose evidence given


def test_live_world_reads_telemetry_after_press(l2):
    """RosWorld.execute queries power_status after a press, and only after a press."""
    live = pytest.importorskip("space_runtime.akr.live")
    v, tele = l2
    queries, events = [], []
    agent = types.SimpleNamespace(sim_t=0.0, event=lambda kind, **f: events.append(kind),
                                  query=lambda name, args: queries.append(name) or {"success": True, "value": reading(tele, "closed")})
    w = live.RosWorld(agent, v, lambda: types.SimpleNamespace(value=lambda p: "U"))
    w._run = lambda a: []
    out = w.execute(reset_action(v))
    assert queries == ["power_status"] and out.judgments[("breaker_closed(pdu_1)", True)] == "+"
    assert "telemetry_after" in out.evidence
    queries.clear()
    w.execute(v.ground(Proposal.of("dock_at_feature", {"feature": "pdu_1_panel/press"}), types.SimpleNamespace(value=lambda p: "U", record=[])))
    assert queries == []
