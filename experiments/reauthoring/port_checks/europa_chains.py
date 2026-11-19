"""Offline checks of the port of the re-authored task models (baselines/revised/): EUROPA chains one after another,
each baselines/europa/run_offline_mission_{a,b}.sh (or run_offline.sh) over its stand-in boundary, on domain 92.

  europa_chains.py <repo> <out_dir> original [name ...]     the 16 missions as the stand-in shows them to the original
  BASELINE_REVISION=reauth/europa-l2a europa_chains.py <repo> <out_dir> europa-l2a     a revision's own offline checks

One line per chain: the capability calls in order (`!` failed) and how the host ended. norm.py turns the runs' events
(without wall clock) and problem files into one text, for comparing two runs of the same chains."""
import json, os, subprocess, sys
from pathlib import Path

repo, out, which = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
only = sys.argv[4:]
ORU, BAT, CNS = "orbital replacement unit (ORU)", "battery module", "consumables container"
def goal(item, src, dst, **kw): return json.dumps(dict(item=item, source=src, destination=dst, **kw))
A, B = "run_offline_mission_a.sh", "run_offline_mission_b.sh"
ORIGINAL = [  # the 16 missions as the stand-in boundary shows them to the original model
    ("ref_pdu2", A, None, ["--tripped", "pdu_2"]),
    ("ref_bess", A, None, ["--tripped", "bess"]),
    ("ref_isru", A, None, ["--tripped", "isru"]),
    ("ref_misreport", A, None, ["--tripped", "bess"]),
    ("ref_oru_pdu3", B, goal(ORU, "depot_stand/slot_0", "pdu_3_shelf/shelf"), []),
    ("ref_bat_bess", B, goal(BAT, "depot_spares_stand/slot_0", "bess_stand/slot_0"), []),
    ("ref_cons_habitat", B, goal(CNS, "landing_stand/slot_0", "habitat_stand/slot_0"), []),
    ("l2a_pdu1", A, None, ["--tripped", "pdu_1", "--no-hold"]),
    ("l2a_bess", A, None, ["--tripped", "bess", "--no-hold"]),
    ("l3_bess", A, None, ["--tripped", "bess", "--no-close"]),
    ("l3_isru", A, None, ["--tripped", "isru", "--no-close"]),
    ("l4_pdu1", A, None, ["--tripped", "pdu_1", "--no-close"]),
    ("l2b_habitat", B, goal(CNS, "landing_stand/slot_0", "habitat_stand/slot_0"), ["--fail", "place:1", "--fail", "place:2", "--fail", "place:3"]),
    ("seq_bess", A, None, ["--tripped", "bess", "--no-close"]),
    ("rec_mislocated", B, goal(ORU, "depot_stand/slot_0", "pdu_3_shelf/shelf", serial="ORU-0417"),
     ["--at", "ORU-0417=pdu_1_shelf/shelf", "--at", "ORU-0272=depot_stand/slot_0"]),
    ("rec_swapped", B, goal(ORU, "depot_stand/slot_0", "pdu_1_shelf/shelf", serial="ORU-0417"),
     ["--at", "ORU-0272=depot_stand/slot_0", "--at", "ORU-0417=depot_stand/slot_1", "--at", "CRT-0112=landing_stand/slot_1"]),
]
O = "run_offline.sh"
ORU_PDU1 = goal(ORU, "depot_stand/slot_0", "pdu_1_shelf/shelf")
CNS_HAB = goal(CNS, "landing_stand/slot_0", "habitat_stand/slot_0")
# each revision's own offline checks, as its worklog records them (the cases it validated with, then its regressions)
CASES = {
    "original": ORIGINAL,
    "europa-l2a": [
        ("l2a_pdu1_drawn", A, None, ["--drawn", "pdu_1"]),
        ("l2a_bess_drawn", A, None, ["--drawn", "bess"]),
        ("reg_pdu2", A, None, ["--tripped", "pdu_2"]),
        ("reg_oru_pdu1", B, ORU_PDU1, []),
    ],
    "europa-l2b": [
        ("l2b_outbound", B, CNS_HAB, ["--outbound", "RTN-0007=habitat_stand/slot_0"]),
        ("l2b_outbound_fail_pick1_place2", B, CNS_HAB, ["--outbound", "RTN-0007=habitat_stand/slot_0", "--fail", "pick:1", "--fail", "place:2"]),
        ("l2b_outbound_at", B, CNS_HAB, ["--outbound", "RTN-0007=habitat_stand/slot_0", "--at", "RTN-0007=habitat_stand/slot_0"]),
        ("reg_cons_habitat", B, CNS_HAB, []),
        ("reg_oru_pdu1", B, ORU_PDU1, []),
        ("reg_pdu2", A, None, ["--tripped", "pdu_2"]),
    ],
    "europa-l3": [
        ("l3_bess", A, None, ["--failed-module", "BAT-0101"]),
        ("l3_isru", A, None, ["--failed-module", "FLT-0201"]),
        ("reg_pdu2", A, None, ["--tripped", "pdu_2"]),
        ("reg_oru_pdu1", B, ORU_PDU1, []),
        ("reg_bat_bess", B, goal(BAT, "depot_spares_stand/slot_0", "bess_stand/slot_0"), []),
        ("reg_bess_overcurrent", A, None, ["--tripped", "bess"]),
        ("reg_isru_overcurrent", A, None, ["--tripped", "isru"]),
    ],
    "europa-l4": [
        ("l4", A, None, ["--failed", "ORU-0211", "--record-failed", "ORU-0417"]),
        ("l4_fail_pick2", A, None, ["--failed", "ORU-0211", "--record-failed", "ORU-0417", "--fail", "pick:2"]),
        ("l4_fail_place1", A, None, ["--failed", "ORU-0211", "--record-failed", "ORU-0417", "--fail", "place:1"]),
        ("l4_fail_place2", A, None, ["--failed", "ORU-0211", "--record-failed", "ORU-0417", "--fail", "place:2"]),
        ("l4_fail_nav3_far", A, None, ["--failed", "ORU-0211", "--record-failed", "ORU-0417", "--fail", "navigate_to_feature:3:far"]),
        ("reg_pdu2", A, None, ["--tripped", "pdu_2"]),
        ("reg_pdu3", A, None, ["--tripped", "pdu_3"]),
        ("reg_bess_nohold", A, None, ["--tripped", "bess", "--no-hold"]),
        ("reg_isru_press_fails2", A, None, ["--tripped", "isru", "--fail", "press:1", "--fail", "press:2"]),
        ("reg_oru_pdu1", B, ORU_PDU1, []),
        ("reg_bat_bess", B, goal(BAT, "depot_spares_stand/slot_0", "bess_stand/slot_0"), []),
        ("reg_cons_habitat", B, CNS_HAB, []),
        ("reg_oru_pdu1_fail_pick1", B, ORU_PDU1, ["--fail", "pick:1"]),
        ("reg_base_far", O, None, ["--fail", "navigate_to_feature:1:far"]),
    ],
}
cases = [c for c in CASES[which] if not only or c[0] in only]
for name, script, mg, args in cases:
    run = out / name
    env = dict(os.environ, REPO=str(repo))
    env.pop("MISSION_GOAL", None)
    if mg: env["MISSION_GOAL"] = mg
    p = subprocess.Popen(["bash", str(repo / "baselines/europa" / script), str(run), *args], env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        rc = p.wait(timeout=300)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, 9)
        rc = "timeout"
    ev = [json.loads(l) for l in (run / "events.jsonl").read_text().splitlines()] if (run / "events.jsonl").exists() else []
    seq = [(e["capability"], e["success"]) for e in ev if e["kind"] == "result"]
    end = [e["kind"] for e in ev if e["kind"] in ("done", "no_plan", "operator_answered")]
    print(f"{name}: rc={rc} calls={len(seq)} end={end} {' '.join(c + ('' if s else '!') for c, s in seq)}", flush=True)
