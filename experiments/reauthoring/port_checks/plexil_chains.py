"""Offline checks of the port (baselines/revised/): PLEXIL plans run by universalExec through the real capability adapter
(ros/install) over the EUROPA stand-in boundary, on domain 93 (plexil_chain.sh), one after another.

  plexil_chains.py <repo> <out> original      the original plans on the 16 missions (BASELINE_REVISION unset)
  plexil_chains.py <repo> <out> revisions     each revised plan on its case, BASELINE_REVISION set per run"""
import json, os, subprocess, sys
from pathlib import Path
repo, out, which = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
HERE = Path(__file__).resolve().parent
ORU, BAT, CNS = "orbital replacement unit (ORU)", "battery module", "consumables container"
def goal(item, src, dst, **kw): return json.dumps(dict(item=item, source=src, destination=dst, **kw))
A, B = "MissionA_Covered", "MissionB_Covered"
SB = str(repo / "baselines/europa/sim_boundary.py")
ORIG = str(repo / "baselines/plexil")
CNS_HAB = goal(CNS, "landing_stand/slot_0", "habitat_stand/slot_0")
ORIGINAL = [
    ("ref_pdu2", A, None, ["--tripped", "pdu_2"]), ("ref_bess", A, None, ["--tripped", "bess"]),
    ("ref_isru", A, None, ["--tripped", "isru"]), ("ref_misreport", A, None, ["--tripped", "bess"]),
    ("ref_oru_pdu3", B, goal(ORU, "depot_stand/slot_0", "pdu_3_shelf/shelf"), []),
    ("ref_bat_bess", B, goal(BAT, "depot_spares_stand/slot_0", "bess_stand/slot_0"), []),
    ("ref_cons_habitat", B, CNS_HAB, []),
    ("l2a_pdu1", A, None, ["--tripped", "pdu_1", "--no-hold"]), ("l2a_bess", A, None, ["--tripped", "bess", "--no-hold"]),
    ("l3_bess", A, None, ["--tripped", "bess", "--no-close"]), ("l3_isru", A, None, ["--tripped", "isru", "--no-close"]),
    ("l4_pdu1", A, None, ["--tripped", "pdu_1", "--no-close"]),
    ("l2b_habitat", B, CNS_HAB, ["--fail", "place:1", "--fail", "place:2", "--fail", "place:3"]),
    ("seq_bess", A, None, ["--tripped", "bess", "--no-close"]),
    ("rec_mislocated", B, goal(ORU, "depot_stand/slot_0", "pdu_3_shelf/shelf", serial="ORU-0417"),
     ["--at", "ORU-0417=pdu_1_shelf/shelf", "--at", "ORU-0272=depot_stand/slot_0"]),
    ("rec_swapped", B, goal(ORU, "depot_stand/slot_0", "pdu_1_shelf/shelf", serial="ORU-0417"),
     ["--at", "ORU-0272=depot_stand/slot_0", "--at", "ORU-0417=depot_stand/slot_1", "--at", "CRT-0112=landing_stand/slot_1"]),
]
def rev(r): return str(repo / f"baselines/revised/{r}/plexil")
def esb(r): return str(repo / f"baselines/revised/europa-{r}/europa/sim_boundary.py")
CASES = {
    "original": [(n, ORIG, p, g, SB, a, {}) for n, p, g, a in ORIGINAL],
    "revisions": [
        ("plexil-l2a_pdu1_drawn", rev("plexil-l2a"), A, None, esb("l2a"), ["--drawn", "pdu_1"], {}),
        ("plexil-l2a_bess_drawn", rev("plexil-l2a"), A, None, esb("l2a"), ["--drawn", "bess"], {}),
        ("plexil-l2b_occupied", rev("plexil-l2b"), B, CNS_HAB, SB, [], {"EUROPA_IX": str(out / "ix_rtn_on_habitat.json")}),
        ("plexil-l2b_covered", rev("plexil-l2b"), B, CNS_HAB, SB, [], {}),
        ("plexil-l3_bess", rev("plexil-l3"), A, None, esb("l3"), ["--failed-module", "BAT-0101"], {}),
        ("plexil-l3_isru", rev("plexil-l3"), A, None, esb("l3"), ["--failed-module", "FLT-0201"], {}),
        ("plexil-l4_pdu1", rev("plexil-l4"), A, None, esb("l4"), ["--failed", "ORU-0211", "--record-failed", "ORU-0417"], {}),
    ],
}
# the L2-B stand-in world (as the revision's worklog checked it): the return container recorded on the habitat stand
IX = json.loads((repo / "space_robotics_bench/assets/lunar_base/usd/scene/lunar_base/interaction.json").read_text())
IX["objects"]["return_container"]["support"] = "habitat_stand/slot_0"
out.mkdir(parents=True, exist_ok=True)
(out / "ix_rtn_on_habitat.json").write_text(json.dumps(IX))
for name, model, plan, mg, sb, args, extra in CASES[which]:
    env = dict(os.environ, **extra)
    env.pop("MISSION_GOAL", None)
    if mg: env["MISSION_GOAL"] = mg
    if which == "revisions": env["BASELINE_REVISION"] = "reauth/" + name.split("_")[0]
    p = subprocess.run([str(HERE / "plexil_chain.sh"), str(repo), str(out / name), model, plan, sb, *args], env=env,
                       capture_output=True, text=True)
    calls = [l.split(" ", 1)[0] + ("" if l.endswith("success=True") else "!") for l in (out / name / "calls.txt").read_text().splitlines()]
    print(f"{name}: {p.stdout.strip()} | {' '.join(calls)}", flush=True)
