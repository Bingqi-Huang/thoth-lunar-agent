"""A report that declares the mission cannot be completed ends the run as "planner_gave_up":
the classifier on the report heads of the campaign, and the executor ending on such a report.

    python3 -m pytest space_runtime/akr/test_gave_up.py
"""
import random

import pytest

from experiments.akr.montecarlo import CFG
from experiments.akr.plan_length import initial
from experiments.akr.site_world import SiteWorld
from space_runtime.akr.executor import Executor
from space_runtime.akr.types import REPORT, Proposal, gave_up
from space_runtime.akr.vocabulary import Vocabulary

GAVE_UP = ["MISSION STATUS: NOT COMPLETE — payload lost; rover safe at the dock",
           "FINAL REPORT — consumables container delivery to the habitat logistics stand: NOT ACHIEVED (mission impossible)",
           "MISSION CLOSED AS IMPOSSIBLE — no capability left can establish the third goal.",
           "STATUS: MISSION NOT ACHIEVED. CNS-0301 is NOT at habitat_stand/slot_0",
           "MISSION OUTCOME: BESS identified and worked extensively, but NOT restored"]
GOES_ON = ["MISSION REPORT — ISRU fault inspection and restoration attempt. WHAT I OBSERVED - Telemetry (power_status, all five "
           "serviced units): PDU-1, PDU-2, PDU-3 and BESS are breaker-closed, energized, lamp green, heat sink 25 C, 0 resets"]


@pytest.mark.parametrize("text,value", [(t, True) for t in GAVE_UP] + [(t, False) for t in GOES_ON])
def test_classifier(text, value):
    assert gave_up(text) is value


class Reporter:
    """A planner that answers every call with one report."""
    last_prompt_tokens = last_planner_s = None

    def __init__(self, text):
        self.text = text

    def plan(self, view):
        return [Proposal.of(REPORT, {"text": self.text})]


@pytest.mark.parametrize("text,outcome", [(GAVE_UP[0], "planner_gave_up"), (GOES_ON[0], "budget_exhausted")])
def test_executor_ends_on_a_give_up_report(text, outcome):
    mission = "b_l0_spare_oru_pdu3"
    s = initial(mission)
    vocab = Vocabulary(CFG, s["site"], s["telemetry"], s["manifest"], mission)
    world = SiteWorld(vocab, s["telemetry"], s["truth"], rho=1.0, q=1.0, eta=0.0, seed=1)
    ex = Executor(vocab, world, Reporter(text), "akr", 16, 5, vocab.goal(), list(vocab.k0().items()), vocab.record(),
                  lambda kind, **f: None, mission=mission, skill_budget=80)
    assert ex.run()["outcome"] == outcome
