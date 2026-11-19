"""The flight-rule table of `decide` (authorization requests) and the per-run fixed ground decision."""
from experiments.operator_console.console import IX, decide


def _units(**over):
    base = {"pdu_1": ("crew-critical", "closed"), "pdu_2": ("crew-critical", "closed"), "pdu_3": ("non-critical", "closed"),
            "bess": ("mission-critical", "closed"), "isru": ("mission-critical", "closed")}
    base.update(over)
    return {"units": {u: {"criticality": c, "breaker": b, "load": u} for u, (c, b) in base.items()}}


BAY = {"bess": "bess_cradle/tray", "pdu_3": "pdu_3_bay/tray"}


def _target(unit):
    assert IX["points"][BAY[unit]]["unit"] == unit
    return BAY[unit]


def _ask(unit, tel, ground="rule"):
    return decide("authorization", [{"capability": "pick", "target": _target(unit)}], tel, ground)[3]


def test_own_module_of_a_unit_that_is_down_is_granted():
    # the telemetry-removal runs: BESS tripped, its own failed module -- the donor rule must not apply
    assert _ask("bess", _units(bess=("mission-critical", "tripped"))) == "grant"


def test_donor_rule_unchanged():
    # a running non-critical donor while a crew-critical unit is down: granted (the L4 PDU-1 runs)
    assert _ask("pdu_3", _units(pdu_1=("crew-critical", "tripped"))) == "grant"
    # a running mission-critical donor: denied
    assert _ask("bess", _units(pdu_1=("crew-critical", "tripped"))) == "deny"
    # a running non-critical unit with no crew-critical need: denied
    assert _ask("pdu_3", _units()) == "deny"


def test_fixed_ground_decision_overrides_the_table():
    tel = _units(bess=("mission-critical", "tripped"))
    assert _ask("bess", tel, "refuse") == "deny"
    assert _ask("pdu_3", _units(), "grant") == "grant"
