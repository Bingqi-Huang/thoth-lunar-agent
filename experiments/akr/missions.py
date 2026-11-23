"""The sixteen lunar-base missions and the variables that launch one of them in simulation.

Each mission names its scene, the simulator scenario, the task overrides and the mission key of
configs/akr_vocabulary.yaml. `launch` gives the variables `experiments/launch/lunar_base_d0/run_d0.sh` reads.

    python3 -m experiments.akr.missions                               # list the missions
    python3 -m experiments.akr.missions no_spare_pdu1 --rule akr        # one mission under an acting rule
    python3 -m experiments.akr.missions no_spare_pdu1 --baseline plexil [--revised]
"""
import argparse
import shlex
from pathlib import Path

import yaml

from space_runtime.akr.injection import Injection, env_for

P = "${THOTH_ROOT}"
A, B = "scenes/deploy/srb_lunar_base_mission_a.yaml", "scenes/deploy/srb_lunar_base_mission_b.yaml"
MISLOCATED = "spare_oru:pdu_1_shelf/shelf+oru_returned:depot_stand/slot_0"
SWAPPED = "oru_returned:depot_stand/slot_0+spare_oru:depot_stand/slot_1+spare_crate:landing_stand/slot_1"
# name -> (title, scene, simulator scenario, task overrides, vocabulary mission key)
MISSIONS = {
    "restore_pdu2": ("Restore PDU-2", A, "covered", "env.faulted_unit=pdu_2", "a_l0_pdu2_reset"),
    "restore_bess": ("Restore BESS", A, "covered", "env.faulted_unit=bess", "a_l1_bess_reset"),
    "restore_isru": ("Restore ISRU", A, "covered", "env.faulted_unit=isru", "a_l1_isru_reset"),
    "locate_and_restore": ("Locate and restore", A, "covered_misreport", "env.faulted_unit=bess env.reported_unit=pdu_2",
                           "a_l1_misreport_bess"),
    "spare_to_pdu3": ("Spare module to PDU-3", B, "covered", "env.cargo=spare_oru env.destination=pdu_3", "b_l0_spare_oru_pdu3"),
    "battery_to_bess": ("Battery module to BESS", B, "covered", "env.cargo=bat_spare env.destination=bess", "b_l1_bat_spare_bess"),
    "consumables_to_habitat": ("Consumables to habitat", B, "covered", "env.cargo=consumables_container env.destination=habitat",
                               "b_l1_consumables_habitat"),
    "pdu1_shutter": ("PDU-1 overheating, shutter closed", A, "heldout_radiator_cover", "env.faulted_unit=pdu_1",
                     "a_l2_radiator_cover_pdu1"),
    "bess_shutter": ("BESS overheating, shutter closed", A, "heldout_radiator_cover", "env.faulted_unit=bess",
                     "a_l2_radiator_cover_bess"),
    "bess_module": ("BESS failed battery module", A, "heldout_module_failure", "env.faulted_unit=bess", "a_l3_module_failure_bess"),
    "isru_filter": ("ISRU clogged filter", A, "heldout_module_failure", "env.faulted_unit=isru", "a_l3_module_failure_isru"),
    "no_spare_pdu1": ("PDU-1 without a serviceable spare", A, "heldout_cannibalize", "env.faulted_unit=pdu_1", "a_l4_cannibalize_pdu1"),
    "occupied_destination": ("Habitat destination occupied", B, "heldout_destination_occupied",
                             "env.cargo=consumables_container env.destination=habitat", "b_l2_destination_occupied_habitat"),
    "sequential_fault": ("BESS module fault, then overheating", A, "heldout_compound_fault", "env.faulted_unit=bess",
                         "a_compound_fault_bess"),
    "serial_mislocated": ("Item named by serial, mislocated", B, "covered",
                          f"env.cargo=spare_oru env.destination=pdu_3 env.name_serial=true env.placement={MISLOCATED}",
                          "b_serial_mislocated"),
    "serial_swapped": ("Item named by serial, records exchanged", B, "covered",
                       f"env.cargo=spare_oru env.destination=pdu_1 env.name_serial=true env.placement={SWAPPED}",
                       "b_serial_swapped"),
}
RULES = ("akr", "kreplanner", "ffreplan")
BASELINE_CHECK = {"plexil": "d0_plexil_mission_{}_check.py", "europa": "d0_europa_mission_{}_check.py"}
# revised task models of the authored baselines (baselines/revised/<baseline>-<model>), by the missions they cover
REVISION = {"pdu1_shutter": "l2a", "bess_shutter": "l2a", "bess_module": "l3", "isru_filter": "l3", "no_spare_pdu1": "l4",
            "occupied_destination": "l2b"}
EARTH = yaml.safe_load(open(Path(__file__).resolve().parents[2] / "configs/akr_vocabulary.yaml"))["earth_set_s"]


def launch(mission: str, rule: str | None = None, baseline: str | None = None, revised: bool = False,
           injection: Injection | None = None) -> str:
    """The variables for run_d0.sh: a foundation-model run under `rule`, or an authored `baseline` run."""
    _, scene, scenario, env, key = MISSIONS[mission]
    common = f"D0_CAPABILITY_SERVER=1 D0_SCENE={scene} D0_SRB={P}/space_robotics_bench D0_SCENARIO={scenario} D0_ENV={shlex.quote(env)}"
    if baseline:
        los = EARTH.get(key, EARTH["nominal"])
        link = f" SPACE_LINK_LOS_S={los:g}" if los != EARTH["nominal"] else ""
        rev = f" BASELINE_REVISION=reauth/{baseline}-{REVISION[mission]}" if revised else ""
        check = BASELINE_CHECK[baseline].format("a" if scene == A else "b")
        return f"D0_CHECK={check} {common}{link}{rev}"
    inj = injection or Injection(ground_decision="grant", earth_set_s=EARTH.get(key, EARTH["nominal"]))
    extra = " ".join(f"{k}={shlex.quote(v)}" for k, v in env_for(inj).items())
    return (f"D0_CHECK=ours_mission_check.py {common} OURS_CONFIG={P}/configs/space_runtime_akr.yaml "
            f"AKR_STRATEGY={rule} AKR_MISSION={key} {extra}").rstrip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mission", nargs="?", choices=sorted(MISSIONS))
    ap.add_argument("--rule", choices=RULES, default="akr")
    ap.add_argument("--baseline", choices=sorted(BASELINE_CHECK))
    ap.add_argument("--revised", action="store_true", help="the revised task model of the baseline")
    a = ap.parse_args()
    if not a.mission:
        for name, (title, *_rest) in MISSIONS.items():
            print(f"{name:24s} {title}")
        return
    if a.revised and a.mission not in REVISION:
        ap.error(f"{a.mission} has no revised task model")
    print(launch(a.mission, a.rule, a.baseline, a.revised))


if __name__ == "__main__":
    main()
