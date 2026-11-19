"""A gripper check that sees the jaws empty writes every held(X) false: a load lost on the
way leaves no "held" behind for a later action to rely on.

    python3 -m pytest space_runtime/akr/test_empty_gripper.py
"""
from space_runtime.akr.knowledge import Knowledge
from space_runtime.akr.types import SENSING, Action


def test_empty_gripper_clears_held():
    K = Knowledge([("held(CNS-0301)", True), ("held(RTN-0007)", False), ("gripper_empty", False)], [], lambda *a, **k: None)
    check = Action("held_item", "held_item", (), SENSING, obs=("gripper_empty",))
    K.sense(check, {"gripper_empty": True})
    assert K.value("gripper_empty") == "T" and K.value("held(CNS-0301)") == "F" and K.value("held(RTN-0007)") == "F"


def test_holding_reading_leaves_held_alone():
    K = Knowledge([("held(CNS-0301)", True), ("gripper_empty", False)], [], lambda *a, **k: None)
    K.sense(Action("held_item", "held_item", (), SENSING, obs=("gripper_empty",)), {"gripper_empty": False})
    assert K.value("held(CNS-0301)") == "T"
