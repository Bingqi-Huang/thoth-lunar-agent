"""The campaign's jobs file: 117 runs with the
foundation model and 44 runs of the authored baselines, one line per run for pool.py.

    AKR                                   all 16 missions            x3   48
    K-replanner rule, FF-Replan rule      9 non-reference missions   x3   27 + 27
    AKR, FF-Replan rule + inserted proposal  no-spare repair, occupied destination  x3   12
    AKR, Earth set at 925 s               no-spare repair            x3    3
    PLEXIL, EUROPA->PLEXIL original       all 16 missions            x1   32
    PLEXIL, EUROPA->PLEXIL revised        6 contingency missions     x1   12

The no-spare repair runs with Earth setting at 1543 s (configs/akr_vocabulary.yaml earth_set_s) in every configuration
but the deadline runs. The ground grants by the flight-rule table, except with an inserted proposal, where it refuses.
Each configuration's repetitions are spread over both machines (POOL_MACHINE): one on the development host, the others on the run-farm host,
which has three times the slots; the authored baselines' single runs one in eight on the development host.

    python3 -m experiments.akr.campaign_jobs > experiments/launch/lunar_base_d0/jobs/akr_campaign.txt
"""
import shlex

from pathlib import Path

import yaml

from experiments.akr.tier1_jobs import BASELINE_CHECK, MISSIONS, P
from space_runtime.akr.injection import Injection, env_for

VOCAB = yaml.safe_load(open(Path(__file__).resolve().parents[2] / "configs/akr_vocabulary.yaml"))
EARTH = VOCAB["earth_set_s"]
REFERENCE = ["restore PDU-2", "restore BESS", "restore ISRU", "locate and restore", "spare module to PDU-3",
             "battery module to BESS", "consumables to habitat"]
CONTINGENCY = ["PDU-1 overheating, shutter closed", "BESS overheating, shutter closed", "BESS failed battery module",
               "ISRU clogged filter", "PDU-1 no serviceable spare", "habitat destination occupied"]
NON_REFERENCE = CONTINGENCY + ["BESS module fault then overheating", "serial item mislocated", "serial records exchanged"]
# the eight re-authored task models (branches reauth/<baseline>-<case>), by the contingency missions they cover
REVISION = {"PDU-1 overheating, shutter closed": "l2a", "BESS overheating, shutter closed": "l2a",
            "BESS failed battery module": "l3", "ISRU clogged filter": "l3", "PDU-1 no serviceable spare": "l4",
            "habitat destination occupied": "l2b"}
# the battery modules with readable data plates are the simulator's default from 2026-11-11: no per-run asset switch
STRAT = {"AKR": "akr", "FF-Replan rule": "ffreplan", "K-replanner rule": "kreplanner"}


def model_line(name, mission, strategy, inj: Injection, machine: str) -> str:
    short, scene, scenario, env, key = MISSIONS[mission]
    extra = " ".join(f"{k}={shlex.quote(v)}" for k, v in env_for(inj).items())
    return (f"{name} POOL_MACHINE={machine} D0_CAPABILITY_SERVER=1 D0_CHECK=ours_mission_check.py D0_SCENE={scene} "
            f"D0_SRB={P}/space_robotics_bench OURS_CONFIG={P}/configs/space_runtime_akr.yaml "
            f"D0_SCENARIO={scenario} D0_ENV={shlex.quote(env)} AKR_STRATEGY={STRAT[strategy]} AKR_MISSION={key} {extra}")


def baseline_line(name, mission, method, revision, machine: str) -> str:
    short, scene, scenario, env, key = MISSIONS[mission]
    los = EARTH.get(key, EARTH["nominal"])
    link = f" SPACE_LINK_LOS_S={los:g}" if los != EARTH["nominal"] else ""
    rev = f" BASELINE_REVISION=reauth/{method.split('→')[0].lower()}-{revision}" if revision else ""
    check = BASELINE_CHECK[method].replace("mission_b", "mission_a" if "mission_a" in scene else "mission_b")
    return (f"{name} POOL_MACHINE={machine} D0_CAPABILITY_SERVER=1 D0_CHECK={check} D0_SCENE={scene} "
            f"D0_SRB={P}/space_robotics_bench D0_SCENARIO={scenario} D0_ENV={shlex.quote(env)}{link}{rev}")


def rows() -> list[tuple]:
    """(block, name, line) in launch order: the slow missions first, so the tail of the campaign is short."""
    out, k = [], 0

    def machines(n):
        """One repetition of each configuration on the development host (which of them rotates), the others on the run-farm host; single runs
        (the authored baselines) one in eight on the development host, so every method runs on both machines."""
        nonlocal k
        k += 1
        if n > 1:
            return ["dev" if i == k % n else "farm" for i in range(n)]
        return ["dev" if k % 8 == 0 else "farm"]

    def model(block, mission, strategy, reps, tag="base", **inj):
        key = MISSIONS[mission][4]
        i = Injection(ground_decision=inj.pop("ground_decision", "grant"), earth_set_s=inj.pop("earth_set_s", EARTH.get(key, EARTH["nominal"])), **inj)
        for r, m in enumerate(machines(reps), 1):
            name = f"{STRAT[strategy]}_{MISSIONS[mission][0]}_{tag}_r{r}"
            out.append((block, name, model_line(name, mission, strategy, i, m)))

    for mission in REFERENCE + NON_REFERENCE:
        model("AKR", mission, "AKR", 3)
    for mission in NON_REFERENCE:
        model("K-replanner rule", mission, "K-replanner rule", 3)
        model("FF-Replan rule", mission, "FF-Replan rule", 3)
    for strategy in ("AKR", "FF-Replan rule"):
        model("inserted proposal", "PDU-1 no serviceable spare", strategy, 3, "propfault", ground_decision="refuse",
              proposal={"kind": "restricted_without_grant", "when": "docked_at_energized_bay",
                        "bays": {"pdu_2_bay/tray": "pdu_2", "pdu_3_bay/tray": "pdu_3", "isru_power/tray": "isru"}})
        model("inserted proposal", "habitat destination occupied", strategy, 3, "propfault", ground_decision="refuse",
              proposal={"kind": "irreversible_undecided", "when": "docked_at_destination_holding",
                        "destination": "habitat_stand/slot_0", "cargo": "CNS-0301", "item": "consumables container"})
    model("Earth set 925 s", "PDU-1 no serviceable spare", "AKR", 3, "earth925", earth_set_s=EARTH["nominal"])
    for method in BASELINE_CHECK:
        for mission in REFERENCE + NON_REFERENCE:
            name = f"{method.split('→')[0].lower()}_{MISSIONS[mission][0]}_original_r1"
            out.append((f"{method} original", name, baseline_line(name, mission, method, None, machines(1)[0])))
        for mission in CONTINGENCY:
            name = f"{method.split('→')[0].lower()}_{MISSIONS[mission][0]}_revised_r1"
            out.append((f"{method} revised", name, baseline_line(name, mission, method, REVISION[mission], machines(1)[0])))
    return out


if __name__ == "__main__":
    rs = rows()
    print(f"# Campaign (2026-11-10): {sum(1 for b, _, _ in rs if 'PLEXIL' not in b)} model runs, "
          f"{sum(1 for b, _, _ in rs if 'PLEXIL' in b)} authored-baseline runs; experiments/akr/campaign_jobs.py")
    for _, _, line in rs:
        print(line)
