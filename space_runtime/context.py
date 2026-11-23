"""Dynamic context construction: what the reasoner reads, filled from the record.

Every word the model reads is in `prompts/` (the standing text `system.md`, the per-step template `step.md`);
this module only selects facts from the authoritative record and fills the templates, one JSON record per line. The
standing text is filled once per reasoner session (the authority exceptions from the contracts and the guard
configuration, the link's constants, the surveyed site); the step message is rebuilt from the record every step, so a
restarted or switched reasoner sees exactly what the previous one would have. Nothing here chooses or suggests the
next action.
"""
import json

from openral_reasoner.context import ContextRenderer

from openral_ext.capability_server.contracts import CONTRACTS
from experiments.operator_console.console import DECISION_TIME_S
from space_context import link
from space_runtime.prompts import prompt


def _j(v) -> str:
    return json.dumps(v, separators=(",", ":"))


def _lines(records: list[dict], empty: str = "(none)") -> str:
    # empty fields are left out, except a verdict: `holds` null (not established) is a fact of its own
    return "\n".join(_j({k: v for k, v in r.items() if v not in (None, "", [], {}) or k == "holds"}) for r in records) or empty


def system_text(facts: dict, rules: list[dict]) -> str:
    """The standing text of a reasoner session: rules, the authority model read off the contracts and the guard
    configuration (never written per mission), the link's constants, the surveyed site."""
    exceptions = [f"- {k} ({c['authority_class'].replace('_', ' ')})" for k, c in sorted(CONTRACTS["effects"].items())
                  if c["authority_class"] != "routine"]
    exceptions += [f"- {', '.join(r['applies_to'])} on a {r['target_role']} whose unit reports {_j(r['when_unit'])}: "
                   f"{r['outcome'].replace('_', ' ')} -- {r['why']}" for r in rules]
    site, ls = facts["site"], link.state(0.0)
    return prompt("system.md").read_text().format(
        authority_exceptions="\n".join(exceptions) or "(none)",
        link=ls["link"], one_way_delay_s=ls["one_way_delay_s"], decision_time_s=f"{DECISION_TIME_S:.0f}",
        site_frame=site["frame"], site_point_types=_j(site["point_types"]),
        site_points=_lines([{"point": name, **p} for name, p in site["points"].items()]),
        site_view_poses=_lines([{"view_pose": name, **v} for name, v in site["view_poses"].items()]),
        site_dock=_j(site["dock"]), equipment_catalogue=_j(site.get("equipment_catalogue")))


def _now(mission_t: float) -> dict:
    ls = link.state(mission_t)
    return {"mission_time_s": round(mission_t), "link": "up" if ls["earth_visible"] else "down (Earth below the horizon)",
            "earth_sets_in_s": ls["closes_in_s"], "link_reopens_in_s": ls["reopens_in_s"]}


def _exchange(e: dict) -> dict:
    return ({"seq": e["seq"], "asked": e["text"], "kind": e.get("request_kind")} if e["kind"] == "operator_request_sent"
            else {"seq": e["seq"], "answered": e["text"], "outcome": e.get("outcome") or e.get("link_status")})


def _recent(s: dict) -> dict:
    return {"seq": s["seq"], "tool": s["tool"], "args": s["args"],
            "guard": s["guard"] if "guard" in s and s["guard"]["outcome"] != "allow" else None, "result": s.get("result")}


def akr_sections(akr: dict | None, events: list[dict]) -> dict:
    """GOAL, KNOWLEDGE STATE, INVENTORY RECORD and EXECUTOR from the executor's planner input: the facts
    section of the step text is the knowledge state, each entry with what decided it last."""
    if not akr:
        return dict(goal="", knowledge="", record="", executor="")
    view, ex = akr["view"], akr["executor"]
    by = {}
    for e in events:
        if e["kind"] == "kappa_write":
            by[e["prop"]] = f"{e['source']} {e['by']}" if e["source"] in ("sensing", "judgment") else e["source"]
    val = {"T": True, "F": False}
    return dict(
        goal=_lines([{"prop": p, "want": v, "now": val.get(view.kappa.get(p), "undecided")} for p, v in view.goal]),
        # true entries one per line with what decided them; the false ones (most of a 339-proposition map) on one line
        knowledge="\n".join([*(_j({"prop": p, "value": True, "decided_by": by.get(p)}) for p, v in sorted(view.kappa.items()) if v == "T"),
                              _j({"false": sorted(p for p, v in view.kappa.items() if v == "F")})]),
        record=_lines([{"prop": p, "recorded": v} for p, v in view.record], "(no record entry for an undecided proposition)"),
        executor=_lines([{"subgoal": [f"{p}={v}".lower() for p, v in view.subgoal] if view.subgoal else None,
                          "pending_sensing": [s.name for s in ex.cover], "readings_in_flight": sorted(ex.inflight)},
                         *view.history]))


def step_text(facts: dict, step: int, max_steps: int, events: list[dict], mission_t: float, akr: dict | None = None) -> str:
    f = facts
    now = _now(mission_t)
    exchanges = [_exchange(e) for e in events if e["kind"] in ("operator_request_sent", "operator_answered")][-8:]
    approvals = {"approvals_on_record": f["approvals"],
                 "held_for_approval": [{k: p[k] for k in ("capability", "target")} for p in f["pending_approvals"]]}
    step_line = [{"step": step, "of_budget": max_steps}]
    if len(f["reasoner_history"]) > 1:
        step_line.append({"continued_by_another_reasoner": f["reasoner_history"]})
    recent = [_recent(s) for s in f["recent"]]
    return prompt("step.md").read_text().format(
        mission=f["mission"]["text"], step=_lines(step_line), robot=_j(f["robot"]),
        verified_effects=_lines(f["verified_effects"][-12:]),
        observations=_lines([o for o in f["observations"] if o["query"] != "site_map"][-8:]),
        objects=_lines([{"item": name, **o} for name, o in f["objects"].items()], "(no verified manipulation yet)"),
        labels=_lines([{"support": s, **r} for s, r in f["labels"].items()], "(no support scanned yet)"),
        attempts=_lines(f["attempts"][-10:]), anomalies=_lines(f["anomalies"]),
        mission_control=_lines([now, *exchanges, approvals]),
        status=_j({"unavailable_capabilities": f["unavailable"], "active_faults": {k: v for k, v in f["fdir"].items() if v["active"]}}),
        plan=_lines([{k: t[k] for k in ("task_id", "status", "text", "last_verdict")} for t in f["ledger"]["tasks"]] if f["ledger"] else [],
                    "(no plan written yet)"),
        recent=_lines(recent), **akr_sections(akr, events))


class HarnessRenderer(ContextRenderer):
    """OpenRAL's renderer slot filled with the harness context (ReasonerCore calls `render` each tick)."""

    def __init__(self, store, max_steps: int, clock):
        super().__init__()
        self.store, self.max_steps, self.step, self.clock = store, max_steps, 0, clock
        self.akr = None  # the AKR executor's planner input while it consults the model (space_runtime/akr/live.py)

    def render(self, *, world_state=None) -> str:
        return step_text(self.store.facts, self.step, self.max_steps, self.store.events, self.clock(), self.akr)
