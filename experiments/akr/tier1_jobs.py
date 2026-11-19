"""The tier-1 jobs file from the de-duplicated matrix (experiments/akr/matrix.py): one line per run, for pool.py.

Every model run uses the AKR configuration, the simulator branch (D0_SRB), its strategy, the mission key of the
vocabulary and its injected condition (AKR_INJECTION, plus the variables the console and the link read). Readings of the
plan where it left room:
  - wrong-record sweep: the world is placed as in Experiment II's two cases; the record has 0, half or all of its 13
    entries false (falsify_record over the place supports, seeded), so "0" is a correct record of the same world;
  - proposal faults: a removal at the bay the rover is docked at, inserted the first time knowledge has it docked at an
    energized unit's bay (only the no-spare mission docks there), with the ground's decision fixed to refuse, so that an
    unauthorized removal is unsafe in the world;
  - perception faults: every camera answer about the shutter (shutter mission) or about the jaws (no-spare) is wrong;
  - Earth set: 300 s (the latest request time is then before the start: no request can be answered) and 650 s (latest
    request time 341 s, around the first request of the earlier runs);
  - telemetry removal: breaker, energized and lamp together (the lamp would otherwise give the breaker away);
  - ground decision: grant (what the flight-rule table gives in the no-spare scenario) unless stated.

    python3 -m experiments.akr.tier1_jobs > experiments/launch/lunar_base_d0/jobs/akr_tier1.txt
"""
import json
import shlex

from experiments.akr.matrix import rows
from space_runtime.akr.injection import IX_PATH, Injection, env_for, falsify_record, place_supports, records_knob

P = "${THOTH_ROOT}"
A, B = "scenes/deploy/srb_lunar_base_mission_a.yaml", "scenes/deploy/srb_lunar_base_mission_b.yaml"
H1_PLACE = "spare_oru:pdu_1_shelf/shelf+oru_returned:depot_stand/slot_0"
H2_PLACE = "oru_returned:depot_stand/slot_0+spare_oru:depot_stand/slot_1+spare_crate:landing_stand/slot_1"
# manuscript mission -> (short name, scene, scenario, env, vocabulary mission key)
MISSIONS = {
    "restore PDU-2": ("ref_pdu2", A, "covered", "env.faulted_unit=pdu_2", "a_l0_pdu2_reset"),
    "restore BESS": ("ref_bess", A, "covered", "env.faulted_unit=bess", "a_l1_bess_reset"),
    "restore ISRU": ("ref_isru", A, "covered", "env.faulted_unit=isru", "a_l1_isru_reset"),
    "locate and restore": ("ref_misreport", A, "covered_misreport", "env.faulted_unit=bess env.reported_unit=pdu_2", "a_l1_misreport_bess"),
    "spare module to PDU-3": ("ref_oru_pdu3", B, "covered", "env.cargo=spare_oru env.destination=pdu_3", "b_l0_spare_oru_pdu3"),
    "battery module to BESS": ("ref_bat_bess", B, "covered", "env.cargo=bat_spare env.destination=bess", "b_l1_bat_spare_bess"),
    "consumables to habitat": ("ref_cons_habitat", B, "covered", "env.cargo=consumables_container env.destination=habitat", "b_l1_consumables_habitat"),
    "PDU-1 overheating, shutter closed": ("l2a_pdu1", A, "heldout_radiator_cover", "env.faulted_unit=pdu_1", "a_l2_radiator_cover_pdu1"),
    "BESS overheating, shutter closed": ("l2a_bess", A, "heldout_radiator_cover", "env.faulted_unit=bess", "a_l2_radiator_cover_bess"),
    "BESS failed battery module": ("l3_bess", A, "heldout_module_failure", "env.faulted_unit=bess", "a_l3_module_failure_bess"),
    "ISRU clogged filter": ("l3_isru", A, "heldout_module_failure", "env.faulted_unit=isru", "a_l3_module_failure_isru"),
    "PDU-1 no serviceable spare": ("l4_pdu1", A, "heldout_cannibalize", "env.faulted_unit=pdu_1", "a_l4_cannibalize_pdu1"),
    "habitat destination occupied": ("l2b_habitat", B, "heldout_destination_occupied", "env.cargo=consumables_container env.destination=habitat",
                                     "b_l2_destination_occupied_habitat"),
    "BESS module fault then overheating": ("seq_bess", A, "heldout_compound_fault", "env.faulted_unit=bess", "a_compound_fault_bess"),
    "serial item mislocated": ("rec_mislocated", B, "covered", f"env.cargo=spare_oru env.destination=pdu_3 env.name_serial=true env.placement={H1_PLACE}",
                               "b_exp2_h1_mislocated"),
    "serial records exchanged": ("rec_swapped", B, "covered", f"env.cargo=spare_oru env.destination=pdu_1 env.name_serial=true env.placement={H2_PLACE}",
                                 "b_exp2_h2_swapped"),
}
STRATEGY = {"AKR": "akr", "FF-Replan rule": "ffreplan", "K-replanner rule": "kreplanner"}
BASELINE_CHECK = {"PLEXIL": "d0_plexil_mission_b_check.py", "EUROPA→PLEXIL": "d0_europa_mission_b_check.py"}


def world_inventory(placement: str) -> list[dict]:
    """The true inventory of a placed world: authored records with the placed objects where they actually stand."""
    ix = json.loads(IX_PATH.read_text())
    moved = dict(kv.split(":", 1) for kv in placement.split("+") if kv)
    return [{"serial": o["serial"], "item": o["item"], "recorded_location": moved.get(name, o["support"]),
             "status": o.get("status", "serviceable")} for name, o in ix["objects"].items()], ix


def record_env(env: str, n) -> tuple[str, int]:
    """env.records for a record with n false entries over the placed world (0 = a correct record of it)."""
    placement = next(kv.split("=", 1)[1] for kv in env.split() if kv.startswith("env.placement="))
    inv, ix = world_inventory(placement)
    _, count, changes = falsify_record(inv, n, seed=0, supports=place_supports(ix))
    true = [{"serial": r["serial"], "to": r["recorded_location"]} for r in inv]       # the record that matches the world
    false = {c["serial"]: c["to"] for c in changes}
    knob = records_knob([{"serial": t["serial"], "to": false.get(t["serial"], t["to"])} for t in true], ix)
    return f"env.records={knob}", count


def line(r: dict) -> str:
    short, scene, scenario, env, key = MISSIONS[r["mission"]]
    cond, strat = r["condition"], r["strategy"]
    if strat in BASELINE_CHECK:
        name = f"{strat.split('→')[0].lower()}_{short}_r{r['rep']}"
        return f"{name} D0_CAPABILITY_SERVER=1 D0_CHECK={BASELINE_CHECK[strat]} D0_SCENE={scene} D0_SRB={P}/space_robotics_bench " \
               f"D0_SCENARIO={scenario} D0_ENV={shlex.quote(env)}"
    inj, tag = Injection(ground_decision="grant"), "base"
    if "false records" in cond:
        n = {"0 false records": 0, "half false records": "half", "all false records": "all"}[cond]
        knob, count = record_env(env, n)
        env, inj.record_false, tag = f"{env} {knob}", count, f"rec{count}"
    elif "handover proposal" in cond:
        inj.proposal = {"kind": "irreversible_undecided", "when": "docked_at_destination_holding",
                        "destination": "habitat_stand/slot_0", "cargo": "CNS-0301", "item": "consumables container"}
        tag = "propfault"
    elif "proposal" in cond:
        inj.proposal = {"kind": "restricted_without_grant", "when": "docked_at_energized_bay",
                        "bays": {"pdu_2_bay/tray": "pdu_2", "pdu_3_bay/tray": "pdu_3", "isru_power/tray": "isru"}}
        inj.ground_decision, tag = "refuse", "propfault"
    elif "shutter answer" in cond:
        inj.perception, tag = {"shutter_open": 1.0}, "percshutter"
    elif "held-item answer" in cond:
        inj.perception, tag = {"gripper_empty": 1.0}, "percheld"
    elif "Earth sets" in cond:
        inj.earth_set_s = 300.0 if "before" in cond else 650.0
        tag = f"earth{int(inj.earth_set_s)}"
    elif "telemetry removed" in cond:
        inj.telemetry_removed, tag = ("breaker_closed", "energized", "lamp_green"), "notelemetry"
    name = f"{STRATEGY[strat]}_{short}_{tag}_r{r['rep']}"
    extra = " ".join(f"{k}={shlex.quote(v)}" for k, v in env_for(inj).items())
    return (f"{name} D0_CAPABILITY_SERVER=1 D0_CHECK=ours_mission_check.py D0_SCENE={scene} D0_SRB={P}/space_robotics_bench "
            f"OURS_CONFIG={P}/configs/space_runtime_akr.yaml D0_SCENARIO={scenario} D0_ENV={shlex.quote(env)} "
            f"AKR_STRATEGY={STRATEGY[strat]} AKR_MISSION={key} {extra}")


if __name__ == "__main__":
    print("# Tier-1 runs (generated by experiments/akr/tier1_jobs.py from experiments/akr/matrix.py); run with experiments/akr/pool.py")
    order = ["T1-b", "T1-c", "T1-e", "T1-a", "T1-d", "T1-f", "T1-g", "T1-h", "baselines"]   # the plan's batch order
    for r in sorted(rows(), key=lambda r: order.index(r["block"].split(" ")[0])):
        print(line(r))
