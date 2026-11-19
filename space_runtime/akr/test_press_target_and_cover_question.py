"""The press names only its feature: the planner's palette offers no `target` for press, and a
target the planner writes anyway is not passed on, so the boundary looks the button up in the site survey. The dust-cover
question is a two-way choice, (B) the cover open. Uses the site, telemetry and manifest snapshot of a recorded run.

    python3 -m pytest space_runtime/akr/test_press_target_and_cover_question.py
"""
import types

import pytest

from space_runtime.akr.test_press_judgment import snapshot
from space_runtime.akr.types import Proposal
from space_runtime.akr.vocabulary import Vocabulary, load_config


@pytest.fixture(scope="module")
def v():
    site, tele, man = snapshot("a_l2_radiator_cover_pdu1")
    return Vocabulary(load_config(), site, tele, man, "a_l2_radiator_cover_pdu1")


def test_press_drops_the_planners_target(v):
    a = v.ground(Proposal.of("press", {"feature": "pdu_1_radiator/cover_button", "target": "the round red push button"}),
                 types.SimpleNamespace(value=lambda p: "U", record=[]))
    assert a.name == "press(pdu_1_radiator/cover_button)" and "target" not in dict(a.args)


def test_palette_press_has_no_target(monkeypatch):
    monkeypatch.setenv("AKR_STRATEGY", "akr")
    import yaml

    from space_runtime import ablation, tools
    press = next(t for t in tools.tool_schemas(yaml.safe_load(ablation.prompt("tools.yaml").read_text()))
                 if t["function"]["name"] == "press")["function"]["parameters"]
    assert "target" not in press["properties"] and press["required"] == ["feature"]


@pytest.mark.parametrize("answer,value", [("B. Two large round fans.", True), ("(A) a single flat plain grey panel", False),
                                          ("a", False), ("It cannot be seen clearly.", None)])
def test_cover_question_is_read_as_a_choice(v, answer, value):
    a = next(s for s in v.sensors(types.SimpleNamespace(value=lambda p: "U", models=lambda l: False)) if s.name == "observe(shutter_open(pdu_1))")
    assert "Answer A or B first" in dict(a.args)["question"]
    assert v.read(a, {"answer": answer}).get("shutter_open(pdu_1)") is value
