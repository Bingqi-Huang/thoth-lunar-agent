"""Buttons and dust-cover camera question on the simulator, no model planning.

Every push button of the units in CHECK_UNITS is pressed PRESSES times from its standard dock pose (the feature's work
pose: navigate to it, dock with standoff 0), one press call per try with the button's surveyed look as the target
(BUTTON_APPEARANCE, the text a caller that names only the feature now gets). Per press: whether the skill found the
button (got past its locate stage), whether the simulator counted a press, and whether it failed for reach (approach:
no arm pose from the dock; locate: no camera view of it). For each unit with a dust cover, FRAMES camera questions
(SHUTTER_QUESTIONS, the candidate wording) from the unit's view pose with the cover closed and FRAMES with it open,
each against the simulator's cover state; the cover button's presses toggle it, and when they leave it as it was it is
pressed again (logged apart from the PRESSES tries). The rover's true and estimated pose are logged at every dock and
view pose (step 2a). The calls never see the simulator's state; only this log compares against it.

    D0_CAPABILITY_SERVER=1 D0_CHECK=button_camera_check.py D0_SCENE=scenes/deploy/srb_lunar_base_mission_a.yaml \
      D0_SCENARIO=heldout_radiator_cover D0_ENV=env.faulted_unit=bess CHECK_UNITS=bess,isru \
      BUTTON_APPEARANCE=<configs/button_appearance.yaml> SHUTTER_QUESTIONS=<yaml {pdu: ..., bess: ...}> run_d0.sh <run>
"""
import re

import yaml

from d0_boundary import cap, query
from d0_common import *  # noqa: F401,F403

UNITS = os.environ["CHECK_UNITS"].split(",")
PRESSES, FRAMES = int(os.environ.get("PRESSES", 5)), int(os.environ.get("FRAMES", 10))
APPEARANCE = yaml.safe_load(Path(os.environ["BUTTON_APPEARANCE"]).read_text())
_Q = yaml.safe_load(Path(os.environ["SHUTTER_QUESTIONS"]).read_text())   # {pdu, bess}, or the vocabulary file itself
QUESTIONS = _Q["camera"]["shutter_question"] if "camera" in _Q else _Q
IXJ = json.loads((Path(__file__).resolve().parents[3] / "space_robotics_bench/assets/lunar_base/usd/scene/lunar_base/interaction.json").read_text())
NAMES = {"pdu_1": "PDU-1", "pdu_2": "PDU-2", "pdu_3": "PDU-3", "bess": "BESS", "isru": "ISRU plant"}
COVER = {"pdu_1": "pdu_1_radiator", "pdu_2": "pdu_2_radiator", "pdu_3": "pdu_3_radiator", "bess": "bess_hvac"}


def poses(**kw):
    p, yaw, _ = truth_pose()
    e, eyaw, _ = est_pose()
    return dict(truth=[round(float(p[0]), 4), round(float(p[1]), 4), round(math.degrees(yaw), 2)],
                estimate=[round(float(e[0]), 4), round(float(e[1]), 4), round(math.degrees(eyaw), 2)], **kw)


def dock(feature):
    for attempt in (1, 2):
        nav = cap("navigate_to_feature", feature=feature, standoff_m=1.5)
        d = cap("dock_at_feature", feature=feature, standoff_m=0.0)
        log("DOCK", feature=feature, attempt=attempt, navigate=nav["success"], dock=d["success"],
            reason=(d["failure_reason"] or nav["failure_reason"] or "")[:300], work_pose=IXJ["work_poses"].get(feature), **poses())
        if d["success"]:
            return True
    return False


def go_view(view):
    v = IXJ["view_poses"][view]
    r = cap("navigate_to_pose", x=v["xy"][0], y=v["xy"][1], yaw=math.radians(v["yaw_deg"]))
    log("VIEW", view=view, success=r["success"], reason=(r["failure_reason"] or "")[:300], view_pose=[*v["xy"], v["yaw_deg"]], **poses())
    return r["success"]


def count(unit, cover):
    spin(1.0)
    u = mission["units"][unit]
    return u["cover_presses"] if cover else u["reset_presses"]


def press(unit, feature, cover, tag):
    before = count(unit, cover)
    r = cap("press", target=APPEARANCE[feature], feature=feature)
    spin(2.0)
    after = count(unit, cover)
    reason = r["failure_reason"] or ""
    stage = (re.search(r"failed at stage (\w+)", reason) or [None, ""])[1]
    reach = stage == "approach" and "cannot stand in front" in reason or stage == "locate" and "cannot put its camera" in reason
    log("PRESS", tag=tag, unit=unit, feature=feature, success=r["success"], stage=stage or ("done" if r["success"] else "?"),
        found=r["success"] or stage not in ("locate", ""), pressed=after > before, out_of_reach=reach,
        count=[before, after], cover_closed=mission["units"][unit]["cover_closed"], reason=reason[:400], sim_s=r["sim_s"])
    return after > before


def observe(unit, view):
    q = QUESTIONS["bess" if unit == "bess" else "pdu"].format(name=NAMES[unit])
    for i in range(FRAMES):
        closed = mission["units"][unit]["cover_closed"]
        a = query("observe", camera="front", question=q)
        m = re.match(r"\W*\(?\s*([AB])\b", a["answer"], re.I)
        said_open = None if not m else m.group(1).upper() == "B"
        log("OBSERVE", unit=unit, view=view, frame=i, truth="closed" if closed else "open",
            closed_after=mission["units"][unit]["cover_closed"], said_open=said_open,
            correct=said_open is not None and said_open == (not closed), answer=a["answer"][:300], image=a["image"])


log("CHECK_START", units=UNITS, presses=PRESSES, frames=FRAMES, mission=dict(mission))
for unit in UNITS:
    reset = f"{unit}_panel/press"
    if dock(reset):
        for k in range(PRESSES):
            press(unit, reset, False, k)
    if unit in COVER:
        view, button = COVER[unit], f"{COVER[unit]}/cover_button"
        if go_view(view):
            observe(unit, view)
        before = mission["units"][unit]["cover_closed"]
        if dock(button):
            for k in range(PRESSES):
                press(unit, button, True, k)
            extra = 0
            while mission["units"][unit]["cover_closed"] == before and extra < 3:
                extra += 1
                press(unit, button, True, f"extra{extra}")
        if mission["units"][unit]["cover_closed"] != before and go_view(view):
            observe(unit, view)
log("CHECK_DONE", mission=dict(mission))
