"""Figure material of an AKR run, indexed from what every run already records (no extra cost during the run): the key
events of the executor with the nearest 1 Hz frames of both cameras and the evidence images of the skill that produced
a judgment (e.g. a place's settle frames for the verified / refuted / unverified panels of the realization figure), the
authority timeline (requests, decisions, holds, covers, decision points, irreversible dispatches, latest request time)
and the simulator snapshot taken before each step (`snapshots/before_step_<n>.pt`, for offline high-resolution renders).

    python3 -m experiments.akr.figure_index <run dir> [...]      # writes <run>/ours/figure_index.json
"""
import bisect
import json
import sys
from pathlib import Path

KEY = ("decision_point", "cover_set", "subgoal", "operator_request_sent", "operator_answered", "terminal", "planner_report")


def frames(run: Path) -> dict[str, tuple[list[float], list[str]]]:
    out = {}
    for cam in ("front", "wrist"):
        files = sorted((run / "frames" / cam).glob("*.jpg")) if (run / "frames" / cam).exists() else []
        out[cam] = ([float(f.stem) for f in files], [str(f) for f in files])
    return out


def nearest(fr, cam: str, t: float) -> str | None:
    ts, fs = fr[cam]
    if not ts:
        return None
    i = min(max(bisect.bisect_left(ts, t), 0), len(ts) - 1)
    return fs[min((i, max(i - 1, 0)), key=lambda j: abs(ts[j] - t))]


def index(run: Path) -> dict:
    ev = [json.loads(l) for l in (run / "ours/events.jsonl").read_text().splitlines()]
    fr = frames(run)
    images, last_exec = {}, None
    for e in ev:                                # the evidence images of each boundary call, by its finishing seq
        if e["kind"] == "execution_finished":
            sk = (e.get("evidence") or {}).get("skill") or {}
            imgs = {k: v for blk in ("settle", "hold_check", "contact", "scan") for k, v in (sk.get(blk) or {}).items()
                    if isinstance(v, str) and v.endswith((".png", ".jpg"))}
            sc = (e.get("evidence") or {}).get("scan") or {}
            imgs.update({k: v for k, v in sc.items() if isinstance(v, str) and v.endswith((".png", ".jpg"))})
            last_exec = dict(seq=e["seq"], capability=e["capability"], images=imgs)
        if e["kind"] == "outcome":
            images[e["seq"]] = last_exec
    keyed = []
    for e in ev:
        k, t = e["kind"], e.get("sim_t", e.get("t"))
        if k in KEY or (k == "dispatch" and e.get("action_kind") == "irreversible") or (k == "record_pruned" and e.get("refuted")) \
                or k == "outcome":
            item = {key: e[key] for key in ("seq", "kind", "action", "decision_point", "undecided", "elements", "judgments",
                                            "outcome", "reason", "literal", "request_id") if key in e}
            item.update(sim_t=t, front=nearest(fr, "front", t), wrist=nearest(fr, "wrist", t))
            if k == "outcome":
                item["evidence"] = images.get(e["seq"])
            keyed.append(item)
    timeline = [{key: e.get(key) for key in ("sim_t", "kind", "action", "request_kind", "outcome", "reason", "window",
                                            "elements", "decision_point")}
                for e in ev if e["kind"] in ("operator_request_sent", "operator_answered", "decision_point", "cover_set",
                                              "terminal", "sense_started") or (e["kind"] == "dispatch" and e.get("action_kind") == "irreversible")]
    snaps = sorted(str(p) for p in (run / "snapshots").glob("before_step_*.pt")) if (run / "snapshots").exists() else []
    return dict(run=str(run), key_events=keyed, authority_timeline=timeline, snapshots=snaps)


if __name__ == "__main__":
    for r in sys.argv[1:]:
        out = index(Path(r))
        (Path(r) / "ours/figure_index.json").write_text(json.dumps(out, indent=1))
        print(r, len(out["key_events"]), "key events,", len(out["snapshots"]), "snapshots")
