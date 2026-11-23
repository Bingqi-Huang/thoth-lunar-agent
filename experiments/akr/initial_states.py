"""The initial state a mission starts from -- the units' telemetry and the inventory manifest as the simulator publishes
them, and the site map -- generated from the scenario definition without the simulator, for the Monte Carlo trials.

The rules are those of the simulator branch (space_robotics_bench, lunar_base/site.py and the two mission tasks): which
unit trips and why, the heat-sink temperature after an over-temperature trip, which modules sit in which bay, what the
record says (authored, the no-spare scenario's withdrawn spare, the occupied destination's occupier, record faults).
The site map and the equipment catalogue are static on the branch and are taken from the recorded runs.

    python3 -m experiments.akr.initial_states --check     # every recorded run: generated state == its own snapshot
    python3 -m experiments.akr.initial_states --write <dir>   # one JSON per mission of the vocabulary
"""
import os
import argparse
import copy
import glob
import json
import math
import re
from pathlib import Path

import yaml

from space_runtime.akr.injection import IX_PATH
from space_runtime.akr.vocabulary import load_config

from experiments.akr.missions import MISSIONS

RUNS = os.path.expandvars("${THOTH_ROOT}/artifacts/runs/akr")
IX = json.loads(IX_PATH.read_text())
# lunar_base_mission_a/task.py, lunar_base/site.py, lunar_base_mission_b/task.py
FAULT_MODULE = {"pdu_1": "oru_pdu_1", "pdu_2": "oru_pdu_2", "pdu_3": "oru_pdu_3", "bess": "bat_bess", "isru": "flt_isru"}
MODULE_FAULT = {"orbital replacement unit (ORU)": "ORU_SELFTEST_FAIL", "battery module": "BATTERY_MODULE_FAULT",
                "filter cartridge": "FILTER_CLOGGED"}
REMOVED = "failed -- removed from service for failure analysis, awaiting return shipment"
DESTINATION = {**{u: f"{u}_shelf/shelf" for u in ("pdu_1", "pdu_2", "pdu_3")}, "bess": "bess_stand/slot_0",
               "habitat": "habitat_stand/slot_0"}
SINK_AMBIENT_C, SINK_CUTOUT_C, SINK_TAU_DRAWN_S, SINK_TAU_OPEN_S = 25.0, 85.0, 600.0, 120.0
AUTHORED_FAILED = {n for n, o in IX["objects"].items() if o.get("status", "").startswith("failed")}
SOURCE = ("the serviced units' own telemetry: breaker, panel lamp, heat-sink temperature, trip cause, resets, installed "
          "module serial and connector state per bay")


def pairs(spec: str) -> dict:
    return dict(kv.split(":", 1) for kv in (spec or "").split("+") if kv)


def parse_env(env: str) -> dict:
    """"env.a=b env.c=d" -> {a: b, c: d}"""
    return {k[len("env."):]: v for k, v in (kv.split("=", 1) for kv in env.split() if kv.startswith("env."))}


def mission_a(env: dict) -> tuple[dict, list, dict]:
    unit, scen = env["faulted_unit"], env.get("scenario", "covered")
    cause = {"heldout_radiator_cover": "OVERTEMP"}.get(scen)
    if scen in ("heldout_module_failure", "heldout_cannibalize", "heldout_compound_fault"):
        cause = MODULE_FAULT[IX["objects"][FAULT_MODULE[unit]]["item"]]
    trips = {unit: cause or "OVERCURRENT"}
    covers = {unit} if scen in ("heldout_radiator_cover", "heldout_compound_fault") else set()
    where = {n: o["support"] for n, o in IX["objects"].items()}
    record = authored_record()
    if scen == "heldout_cannibalize":
        entry(record, "spare_oru")["status"] = REMOVED
    bad = {"heldout_module_failure": {FAULT_MODULE[unit]}, "heldout_compound_fault": {FAULT_MODULE[unit]},
           "heldout_cannibalize": {FAULT_MODULE[unit], "spare_oru"}}.get(scen, set())
    return trips, record, dict(where=where, covers=covers, bad=bad)


def mission_b(env: dict) -> tuple[dict, list, dict]:
    scen, dest = env.get("scenario", "covered"), DESTINATION[env["destination"]]
    where = {n: o["support"] for n, o in IX["objects"].items()}
    record = authored_record()
    if scen == "heldout_destination_occupied":
        occ = "return_container" if env["destination"] == "habitat" else "spare_crate"
        where[occ] = dest
        r = entry(record, occ)
        r["recorded_location"] = dest
        r["status"] = "outbound return, awaiting stowage" if env["destination"] == "habitat" else "returned cargo, awaiting stowage"
    where.update(pairs(env.get("placement", "")))
    return {}, record, dict(where=where, covers=set(), bad=set())


def authored_record() -> list[dict]:
    return [{"serial": o["serial"], "item": o["item"], "recorded_location": o["support"], "status": o.get("status", "serviceable")}
            for o in IX["objects"].values()]


def entry(record: list, name: str) -> dict:
    return next(r for r in record if r["serial"] == IX["objects"][name]["serial"])


def record_faults(record: list, env: dict) -> list:
    for name, loc in pairs(env.get("records", "")).items():
        entry(record, name)["recorded_location"] = loc
    for a, b in pairs(env.get("record_swaps", "")).items():
        ra, rb = entry(record, a), entry(record, b)
        for k in ("recorded_location", "status"):
            ra[k], rb[k] = rb[k], ra[k]
    return record


def generate(scene: str, env: dict, t: float, site: dict, catalogue: dict) -> dict:
    """{site, telemetry, manifest} at mission time t (the heat sink relaxes towards ambient after a trip)."""
    trips, record, w = (mission_a if scene == "A" else mission_b)(env)
    record = record_faults(record, env)
    units = {}
    for u, spec in IX["units"].items():
        tripped = u in trips
        sink = SINK_AMBIENT_C
        if trips.get(u) == "OVERTEMP":
            tau = SINK_TAU_DRAWN_S if u in w["covers"] else SINK_TAU_OPEN_S
            sink = SINK_AMBIENT_C + (SINK_CUTOUT_C - SINK_AMBIENT_C) * math.exp(-t / tau)
        units[u] = {"name": spec["name"], "load": spec["load"], "criticality": spec["criticality"],
                    "breaker": "tripped" if tripped else "closed", "energized": not tripped, "lamp": "red" if tripped else "green",
                    "heat_sink_temperature_c": round(sink, 1), "trip_cause": trips.get(u), "resets_since_start": 0,
                    "installed_modules": {f"{b}/tray": next((IX["objects"][n]["serial"] for n, k in w["where"].items() if k == f"{b}/tray"), None)
                                          for b in spec["bays"]},
                    "bay_connectors": {f"{b}/tray": "ok" for b in spec["bays"]}}
    return {"site": site, "telemetry": {"source": SOURCE, "units": units},
            "manifest": {"inventory": record, "equipment_catalogue": catalogue},
            "truth": {"where": {IX["objects"][n]["serial"]: k for n, k in w["where"].items()}, "drawn": sorted(w["covers"]),
                      # unserviceable modules: the scenario's and those the scene authors as failed (site.py)
                      "unserviceable": sorted(IX["objects"][n]["serial"] for n in w["bad"] | AUTHORED_FAILED)}}


def recorded() -> list[dict]:
    """Every recorded AKR run with its snapshot (site, akr_inputs) and the scene and env it ran with."""
    out = []
    for f in sorted(glob.glob(f"{RUNS}/*/*/ours/events.jsonl")):
        run = Path(f).parents[1]
        if not (run / "scene_run.yaml").exists():
            continue
        site = inputs = None
        for line in open(f):
            if '"site_knowledge"' in line and site is None:
                site = json.loads(line)["site"]
            elif '"akr_inputs"' in line:
                inputs = json.loads(line)
                break
        if not (site and inputs):
            continue
        sr = (run / "scene_run.yaml").read_text()
        env = {m.group(1): m.group(2) for m in re.finditer(r"^\s*-\s*env\.([a-z_]+)=(.*)$", sr, re.M)}
        scene = "B" if "cargo" in env else "A"
        out.append(dict(run=str(run), site=site, inputs=inputs, env=env, scene=scene))
    return out


def diff(a, b, path="") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        return [d for k in sorted(set(a) | set(b)) for d in diff(a.get(k, "<absent>"), b.get(k, "<absent>"), f"{path}.{k}")]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in diff(x, y, f"{path}[{i}]")]
    return [] if a == b else [f"{path}: generated {a!r} recorded {b!r}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--write", type=Path)
    a = ap.parse_args()
    runs = recorded()
    site, catalogue = runs[0]["site"], runs[0]["inputs"]["manifest"]["equipment_catalogue"]
    if a.check:
        bad, by_mission = 0, {}
        for r in runs:
            assert r["site"] == site and r["inputs"]["manifest"]["equipment_catalogue"] == catalogue, r["run"]
            g = generate(r["scene"], r["env"], r["inputs"]["sim_t"] if "sim_t" in r["inputs"] else 0.0, site, catalogue)
            d = diff(g["telemetry"], r["inputs"]["telemetry"]) + diff(g["manifest"], r["inputs"]["manifest"])
            by_mission.setdefault(r["inputs"]["mission_key"], []).append(not d)
            if d:
                bad += 1
                print(f"MISMATCH {r['run']}: {d[:4]}")
        for m, ok in sorted(by_mission.items()):
            print(f"{m:36s} runs {len(ok):3d}  identical {sum(ok):3d}")
        print(f"{len(runs)} recorded runs, {bad} mismatches; site map and catalogue identical in all")
        raise SystemExit(bad > 0)
    if a.write:
        a.write.mkdir(parents=True, exist_ok=True)
        cfg = load_config()
        for short, scene, scenario, env, key in MISSIONS.values():
            e = {**parse_env(env), "scenario": scenario.replace("covered_misreport", "covered")}
            g = generate("A" if "mission_a" in scene else "B", e, 0.0, site, catalogue)
            (a.write / f"{key}.json").write_text(json.dumps({"mission": key, "scene": scene, "env": e, **g}, indent=1))
        print(f"wrote {len(MISSIONS)} initial states to {a.write} ({len(cfg['missions'])} missions in the vocabulary)")


if __name__ == "__main__":
    main()
