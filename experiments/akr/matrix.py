"""The de-duplicated tier-1 run matrix of the execution plan (work package F) with wall-time and token estimates.

One row per run: (block, mission, condition, strategy, repetition). A run serves every manuscript number it can: the
"all false records" AKR cell of the wrong-record sweep is also the Table 5 wrong-record cell, the Earth-set "after the
bound" cell is the ordinary no-spare AKR run of the contingency matrix (Earth sets at 925 s, latest request 616 s, the
first request comes at 250-450 s), and the FF-Replan rule "all false" runs are also the record-fault soundness runs.

Per-run estimates come from the old runs, scaled to the development host: mission time = robot time + planner calls x planner latency x
real-time factor (the simulator keeps running while the model thinks), wall time = mission time / RTF + 120 s of start
and teardown. Old medians (exp1b on the run-farm host, RTF 0.22; compound fault on the development host, RTF 0.32): reference 574 s, 23 calls,
0.50 M prompt tokens; contingency 1177 s, 50 calls, 1.16 M; sequential fault (compound) 2520 s on the development host, 77 calls,
2.0 M; wrong record (Exp II post runs) taken as contingency. Planner latency 19.9 s per call. CAL (below) rescales
tokens and calls once the two calibration runs under AKR are in.

    python3 -m experiments.akr.matrix [--out <csv>] [--cal-tokens 1.0] [--cal-calls 1.0] [--parallel 2] [--latency 19.9]

Calibration under AKR (2026-11-08, two contingency runs on the development host): 26 k prompt tokens per model call against about 23 k
before (--cal-tokens 1.13); planner latency 43-64 s median at night against 19.9 s (--latency 50 for night batches).
"""
import argparse
import csv
import sys

RTF_Z890, RTF_OLD, LATENCY_S, OVERHEAD_S = 0.32, 0.22, 19.9, 120.0
# class -> (old mission time s, planner calls, prompt tokens M, old RTF)
OLD = {"reference": (574, 23, 0.50, RTF_OLD), "contingency": (1177, 50, 1.16, RTF_OLD),
       "sequential": (2520, 77, 2.0, RTF_Z890), "wrong_record": (1177, 50, 1.16, RTF_OLD), "baseline": (600, 0, 0.0, RTF_OLD)}

REFERENCE = ["restore PDU-2", "restore BESS", "restore ISRU", "locate and restore", "spare module to PDU-3",
             "battery module to BESS", "consumables to habitat"]
CONTINGENCY = ["PDU-1 overheating, shutter closed", "BESS overheating, shutter closed", "BESS failed battery module",
               "ISRU clogged filter", "PDU-1 no serviceable spare", "habitat destination occupied"]
WRONG = ["serial item mislocated", "serial records exchanged"]
SEQ = "BESS module fault then overheating"


def rows() -> list[dict]:
    r = []

    def add(block, mission, cls, condition, strategy, reps, serves):
        for i in range(1, reps + 1):
            r.append(dict(block=block, mission=mission, cls=cls, condition=condition, strategy=strategy, rep=i, serves=serves))

    for m in REFERENCE:
        add("T1-a reference parity", m, "reference", "none", "AKR", 1, "kept reference numbers")
    for m in CONTINGENCY:
        add("T1-b contingency", m, "contingency", "none", "AKR", 3,
            "Table 5, Table 6, rho/q, Table 7" + ("; Fig. 7; Earth-set 'after the bound' cell" if "no serviceable" in m else ""))
        add("T1-b contingency", m, "contingency", "none", "FF-Replan rule", 2, "Table 5, Table 6")
        add("T1-b contingency", m, "contingency", "none", "K-replanner rule", 2, "Table 5, Table 6")
    for m in WRONG:
        for n in ("0 false records", "half false records", "all false records"):
            add("T1-c wrong-record sweep", m, "wrong_record", n, "AKR", 2,
                "Fig. 8 left, Theorem 3(ii)" + ("; Table 5 wrong-record" if n.startswith("all") else ""))
        add("T1-c wrong-record sweep", m, "wrong_record", "all false records", "FF-Replan rule", 1, "Table 5; record-fault soundness (Fig. 5)")
        add("T1-c wrong-record sweep", m, "wrong_record", "all false records", "K-replanner rule", 1, "Table 5; record-fault soundness (Fig. 5)")
    add("T1-d sequential fault", SEQ, "sequential", "none", "AKR", 2, "Table 5; Fig. 8 right (telemetry present)")
    add("T1-d sequential fault", SEQ, "sequential", "none", "FF-Replan rule", 1, "Table 5; Fig. 8 right")
    add("T1-d sequential fault", SEQ, "sequential", "none", "K-replanner rule", 1, "Table 5")
    # a hostile proposal tests the rule only where its irreversible action is reachable: the no-spare mission docks at an
    # energized unit's bay (restricted removal without grant); the occupied-destination delivery reaches its handover
    # (irreversible with an undecided precondition); no other contingency mission docks at an energized bay
    for st in ("AKR", "FF-Replan rule"):
        add("T1-e proposal faults", "PDU-1 no serviceable spare", "contingency", "ungranted removal proposal inserted", st, 2, "Fig. 5; Fig. 6 points")
        add("T1-e proposal faults", "habitat destination occupied", "contingency", "unverified handover proposal inserted", st, 1, "Fig. 5; Fig. 6 points")
    add("T1-f perception faults", "PDU-1 overheating, shutter closed", "contingency", "wrong shutter answer", "AKR", 2, "Fig. 5 hatched; Table 7")
    add("T1-f perception faults", "PDU-1 no serviceable spare", "contingency", "wrong held-item answer", "AKR", 2, "Fig. 5 hatched; Table 7")
    add("T1-g Earth-set deadline", "PDU-1 no serviceable spare", "contingency", "Earth sets before the bound (T_vis 300 s)", "AKR", 1, "Fig. 7; certified hold of 5.2")
    add("T1-g Earth-set deadline", "PDU-1 no serviceable spare", "contingency", "Earth sets at the bound (T_vis 650 s)", "AKR", 1, "Fig. 7")
    add("T1-h telemetry removal", SEQ, "sequential", "breaker telemetry removed", "AKR", 2, "5.3 telemetry paragraph; Fig. 8 right")
    add("T1-h telemetry removal", SEQ, "sequential", "breaker telemetry removed", "FF-Replan rule", 2, "5.3; Fig. 8 right")
    for m in WRONG:
        for b in ("PLEXIL", "EUROPA→PLEXIL"):
            add("baselines wrong record", m, "baseline", "original task model", b, 1, "Table 5 last two columns (if not already available)")
    return r


def estimate(row: dict, cal_tokens: float, cal_calls: float, latency_s: float = LATENCY_S, rtf: float = 0.25) -> tuple[float, float]:
    """(wall hours, prompt tokens M) of one run. The simulator's clock is held during model calls, so mission time is
    robot time only (old mission time minus the model's share at the old machine's speed) and wall time is robot time
    at the simulator's speed plus the model's latency per call plus start and teardown."""
    mission_s, calls, tokens, old_rtf = OLD[row["cls"]]
    if row["cls"] == "baseline":
        return (mission_s / rtf + OVERHEAD_S) / 3600, 0.0
    robot_s = mission_s - calls * LATENCY_S * old_rtf
    calls *= cal_calls
    if "before the bound" in row["condition"]:                      # the run ends in the hold at the first decision
        robot_s, calls, tokens = robot_s * 0.4, calls * 0.4, tokens * 0.4
    return (robot_s / rtf + calls * latency_s + OVERHEAD_S) / 3600, tokens * cal_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--cal-tokens", type=float, default=1.0)
    ap.add_argument("--cal-calls", type=float, default=1.0)
    ap.add_argument("--parallel", type=int, default=14, help="simulation slots: the run-farm host 12 (two per GPU, own ROS domains) + the development host 2")
    ap.add_argument("--latency", type=float, default=LATENCY_S, help="planner latency per call, s (19.9 in Experiment I)")
    ap.add_argument("--rtf", type=float, default=0.25, help="simulator speed, sim s per wall s (the run-farm host idle 0.25 measured)")
    a = ap.parse_args()
    rs = rows()
    for r in rs:
        r["wall_h"], r["tokens_M"] = (round(x, 2) for x in estimate(r, a.cal_tokens, a.cal_calls, a.latency, a.rtf))
    w = csv.DictWriter(open(a.out, "w", newline="") if a.out else sys.stdout, fieldnames=list(rs[0]))
    w.writeheader()
    w.writerows(rs)
    blocks = {}
    for r in rs:
        b = blocks.setdefault(r["block"], [0, 0.0, 0.0])
        b[0] += 1
        b[1] += r["wall_h"]
        b[2] += r["tokens_M"]
    print("\nblock | runs | wall h (serial) | prompt tokens M", file=sys.stderr)
    for k, (n, h, t) in blocks.items():
        print(f"{k} | {n} | {h:.1f} | {t:.1f}", file=sys.stderr)
    n = len(rs)
    model = sum(1 for r in rs if r["cls"] != "baseline")
    h, t = sum(r["wall_h"] for r in rs), sum(r["tokens_M"] for r in rs)
    print(f"total | {n} ({model} with the model) | {h:.1f} serial, {h / a.parallel:.1f} on {a.parallel} GPUs | {t:.1f}", file=sys.stderr)


if __name__ == "__main__":
    main()
