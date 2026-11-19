"""The two tests before the rerun, no model planning, every try from a fresh approach.

  GATE=T1   the BESS reset button, TRIES times: drive off to the BESS view pose, navigate to and dock at the button's
            work pose, press naming only the feature (the boundary takes the button's look from the site survey).
            Per try: found (past the locate stage), pressed (the simulator counted it), out of reach.
  GATE=T2   the dust-cover question of configs/akr_vocabulary.yaml for UNIT (pdu_1 or bess), TRIES times: drive off to
            the unit's panel, navigate to the cover's view pose as the executor does (navigate_to_pose), ask, and
            compare with the simulator's cover state. The scenario sets the cover closed or open.

    D0_CAPABILITY_SERVER=1 D0_CHECK=rerun_gate_check.py GATE=T1 TRIES=5 ... run_d0.sh <run>
"""
import re

import yaml

from d0_boundary import cap, query
from d0_common import *  # noqa: F401,F403

GATE, TRIES = os.environ["GATE"], int(os.environ.get("TRIES", 5))
ROOT = Path(__file__).resolve().parents[3]
IXJ = json.loads((ROOT / "space_robotics_bench/assets/lunar_base/usd/scene/lunar_base/interaction.json").read_text())
CAMERA = yaml.safe_load((ROOT / "configs/akr_vocabulary.yaml").read_text())["camera"]
NAMES = {"pdu_1": "PDU-1", "bess": "BESS"}


def poses(**kw):
    p, yaw, _ = truth_pose()
    e, eyaw, _ = est_pose()
    return dict(truth=[round(float(p[0]), 4), round(float(p[1]), 4), round(math.degrees(yaw), 2)],
                estimate=[round(float(e[0]), 4), round(float(e[1]), 4), round(math.degrees(eyaw), 2)], **kw)


def to_view(view):
    v = IXJ["view_poses"][view]
    r = cap("navigate_to_pose", x=v["xy"][0], y=v["xy"][1], yaw=math.radians(v["yaw_deg"]))
    log("VIEW", view=view, success=r["success"], reason=(r["failure_reason"] or "")[:300], view_pose=[*v["xy"], v["yaw_deg"]], **poses())
    return r["success"]


def dock(feature):
    nav = cap("navigate_to_feature", feature=feature, standoff_m=1.5)
    d = cap("dock_at_feature", feature=feature, standoff_m=0.0)
    log("DOCK", feature=feature, navigate=nav["success"], dock=d["success"],
        reason=(d["failure_reason"] or nav["failure_reason"] or "")[:300], **poses())
    return d["success"]


log("GATE_START", gate=GATE, tries=TRIES, unit=os.environ.get("UNIT"), mission=dict(mission))
for k in range(TRIES):
    if GATE == "T1":
        to_view("bess")
        if not dock("bess_panel/press"):
            log("TRY", gate=GATE, k=k, docked=False)
            continue
        spin(1.0)
        before = mission["units"]["bess"]["reset_presses"]
        r = cap("press", feature="bess_panel/press")
        spin(2.0)
        after = mission["units"]["bess"]["reset_presses"]
        reason = r["failure_reason"] or ""
        stage = (re.search(r"failed at stage (\w+)", reason) or [None, ""])[1]
        log("TRY", gate=GATE, k=k, docked=True, success=r["success"], stage=stage or ("done" if r["success"] else "?"),
            found=r["success"] or stage not in ("locate", ""), pressed=after > before,
            out_of_reach="cannot stand in front" in reason or "cannot put its camera" in reason, count=[before, after],
            target=(r["evidence"].get("skill") or r["evidence"]).get("target"), reason=reason[:400])
    else:
        unit = os.environ["UNIT"]
        view = CAMERA["shutter"][unit]
        q = CAMERA["shutter_question"]["bess" if unit == "bess" else "pdu"].format(name=NAMES[unit])
        cap("navigate_to_feature", feature=f"{unit}_panel/press", standoff_m=1.5)   # off the view pose: a fresh approach
        if not to_view(view):
            log("TRY", gate=GATE, k=k, at_view=False)
            continue
        closed = mission["units"][unit]["cover_closed"]
        a = query("observe", camera="front", question=q)
        m = re.match(r"\W*\(?\s*([AB])\b", a["answer"], re.I)
        said_open = None if not m else m.group(1).upper() == "B"
        log("TRY", gate=GATE, k=k, unit=unit, truth="closed" if closed else "open", said_open=said_open,
            correct=said_open is not None and said_open == (not closed), answer=a["answer"][:300], image=a["image"])
log("GATE_DONE", mission=dict(mission))
