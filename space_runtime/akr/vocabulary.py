"""The formal vocabulary for the lunar rover: propositions, grounded actuation and sensing actions with
precondition / effect / observation sets, the evidence rule that turns verification measurements into judgments, the
trusted initial knowledge K0, the record R and the goal of each mission. The declarations are data in
configs/akr_vocabulary.yaml; this module expands them over the site and the inventory and holds the logic.

The vocabulary never reads simulator truth except in `truth`, which only the evaluator and the accounting call.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import yaml

from .types import INF, IRREVERSIBLE, REVERSIBLE, SENSING, Action, Literal, parse_lit

CONFIG = Path(__file__).resolve().parents[2] / "configs/akr_vocabulary.yaml"


def load_config(path: Path = CONFIG) -> dict:
    return yaml.safe_load(path.read_text())


def _family(p: str) -> str:
    return p.split("(")[0]


class Vocabulary:
    def __init__(self, cfg: dict, site: dict, telemetry: dict, manifest: dict, mission: str, t_vis: float = 925.0,
                 telemetry_removed: tuple[str, ...] = ()):
        self.cfg, self.t_vis, self.removed = cfg, t_vis, tuple(telemetry_removed)
        self.mission = cfg["aliases"].get(mission, mission)
        pts = self.pts = site["points"]
        self.units = list(telemetry["units"])
        self.names = {u: telemetry["units"][u]["name"] for u in self.units}
        self.supports = [k for k, v in pts.items() if v["type"] == "place"]
        self.bays = [k for k in self.supports if pts[k].get("role") == cfg["grant"]["role"]]
        self.kind = {s: item for s, item in cfg["objects"].values()}
        self.kind.update({i["serial"]: i["item"] for i in manifest["inventory"]})
        self.serials = list(self.kind)
        self.serial_of = {name: s for name, (s, _) in cfg["objects"].items()}
        self.recorded = {i["recorded_location"]: i["item"] for i in manifest["inventory"] if i["recorded_location"]}
        self._record = [(i["serial"], i["recorded_location"]) for i in manifest["inventory"] if i["recorded_location"]]
        c = cfg["camera"]
        self.shutter_units = list(c["shutter"])
        # Where a rover can be: docked at a work pose (dock band) or standing at a camera view pose (navigation band).
        d = cfg["dock"]
        self.poses = {"dock": (d["x"], d["y"], d["yaw"]),
                      **{k: (x, y, math.radians(deg)) for k, (x, y, deg) in cfg["work_poses"].items()}}
        views = sorted({*c["lamp"].values(), *c["shutter"].values()})
        self.views = {v: (site["view_poses"][v]["x"], site["view_poses"][v]["y"], site["view_poses"][v]["yaw"]) for v in views}
        self.dockable = [k for k in self.poses if k != "dock"]
        self.locations = tuple([f"base_at({k})" for k in self.poses] + [f"at_view({v})" for v in self.views])
        # propositions, exogenous (what telemetry reports) and grants
        P = list(self.locations) + ["gripper_empty", "arm_stowed"]
        P += [f"held({s})" for s in self.serials]
        P += [f"item_at({s},{k})" for s in self.serials for k in self.supports]
        P += [f"occupied({k})" for k in self.supports if k not in self.bays] + [f"installed({k})" for k in self.bays]
        P += [f"{f}({u})" for f in ("breaker_closed", "energized", "lamp_green", "sink_hot") for u in self.units]
        P += [f"connector_ok({k})" for k in self.bays]
        P += [f"shutter_open({u})" for u in self.shutter_units]
        # a delivery's destination is a handover point (the consignee latches what comes to rest there, SRB feat/akr-sim):
        # placing there is irreversible; cargo(X) is the static fact that X is the item the mission delivers
        goal_at = [parse_lit(t)[0] for t in cfg["missions"][self.mission] if t.startswith("item_at(")]
        self.handover, self.cargo = (goal_at[0][len("item_at("):-1].split(",")[::-1] if goal_at else (None, None))
        if self.handover:
            P += [f"cargo({s})" for s in self.serials]
        self.grants = {f"{cfg['grant']['family']}({k})" for k in self.bays}
        P += sorted(self.grants)
        self.propositions = P
        fams = set(cfg["exogenous"]["families"])
        self.exogenous = {p for p in P if _family(p) in fams}
        if cfg["exogenous"]["serials_in_bays"]:
            self.exogenous |= {f"item_at({s},{k})" for s in self.serials for k in self.bays}
        # a removed telemetry channel takes its propositions out of P_x: judgments and the camera decide them now, while
        # the world still changes them on its own (the observability violation the telemetry-removal runs measure)
        self.exogenous -= {p for p in self.exogenous if p in self.removed or _family(p) in self.removed}
        self._actions: list[Action] | None = None
        self._sensors: list[Action] | None = None
        self.origin: tuple | None = None        # where navigation durations are measured from (effect_actions)

    # ---- construction helpers -----------------------------------------------------------------------------------------
    def _fill(self, templates, **kw) -> tuple[Literal, ...]:
        return tuple(parse_lit(t.format(**kw)) for t in templates)

    def _occ(self, support: str) -> str:
        return f"installed({support})" if support in self.bays else f"occupied({support})"

    def _moved_to(self, at: str | None, omit: str | None = None) -> tuple[Literal, ...]:
        """Location effect: `at` holds, every other location proposition does not, except `omit` (no claim)."""
        return tuple((p, p == at) for p in self.locations if p != omit)

    def _resolve(self, x: float, y: float, yaw: float) -> str | None:
        """The location proposition whose pose a commanded pose names, else None."""
        tol = self.cfg["bands"]["same_pose_m"]
        for p, (px, py, pyaw) in [*((f"base_at({k})", v) for k, v in self.poses.items()),
                                  *((f"at_view({k})", v) for k, v in self.views.items())]:
            if math.hypot(x - px, y - py) <= tol and abs(math.remainder(yaw - pyaw, 2 * math.pi)) <= tol:
                return p
        return None

    def _located(self, x: float, y: float, yaw: float) -> dict[str, bool]:
        """Value of every location proposition for a base pose: docked within the dock band, at a view within the navigation band."""
        b = self.cfg["bands"]
        out = {}
        for k, (px, py, pyaw) in self.poses.items():
            out[f"base_at({k})"] = math.hypot(x - px, y - py) <= b["dock"]["position_m"] and abs(math.remainder(yaw - pyaw, 2 * math.pi)) <= b["dock"]["yaw_rad"]
        for v, (px, py, pyaw) in self.views.items():
            out[f"at_view({v})"] = math.hypot(x - px, y - py) <= b["navigate"]["position_m"] and abs(math.remainder(yaw - pyaw, 2 * math.pi)) <= b["navigate"]["yaw_rad"]
        return out

    # ---- actuation actions --------------------------------------------------------------------------------------------
    def _nav_s(self, kind: str, x: float, y: float) -> float:
        """A navigation's duration: base + per metre of straight distance from `self.origin` (None: unknown, 0 m)."""
        d = self.cfg["durations"][kind]
        return d["base_s"] + d["per_m_s"] * (math.hypot(x - self.origin[0], y - self.origin[1]) if self.origin else 0.0)

    def navigate_to_feature(self, feature: str, args: tuple = (), standoff: float = 1.2) -> Action:
        near = standoff < self.cfg["bands"]["no_claim_standoff_m"]    # may still end within the dock band of the feature
        return Action(f"navigate_to_feature({feature})", "navigate_to_feature", args, REVERSIBLE,
                      eff=self._moved_to(None, f"base_at({feature})" if near else None),
                      duration=self._nav_s("navigate_to_feature", *self.pts[feature]["xyz"][:2]))

    def dock_at_feature(self, feature: str, args: tuple = (), standoff: float = 0.0) -> Action:
        at = f"base_at({feature})" if feature in self.poses and abs(standoff) < 0.01 else None
        near = at is None and standoff < self.cfg["bands"]["no_claim_standoff_m"]
        return Action(f"dock_at_feature({feature})", "dock_at_feature", args, REVERSIBLE,
                      eff=self._moved_to(at, f"base_at({feature})" if near else None), duration=self.cfg["durations"]["dock_at_feature"])

    def navigate_to_pose(self, x: float, y: float, yaw: float, args: tuple = ()) -> Action:
        p = self._resolve(x, y, yaw)    # the navigation band says nothing about a dock band: a work pose is no claim
        return Action(f"navigate_to_pose({round(x, 3)},{round(y, 3)},{round(yaw, 3)})", "navigate_to_pose", args, REVERSIBLE,
                      eff=self._moved_to(p if p and p.startswith("at_view") else None, p if p and p.startswith("base_at") else None),
                      duration=self._nav_s("navigate_to_pose", x, y))

    def dock_to_pose(self, x: float, y: float, yaw: float, args: tuple = ()) -> Action:
        return Action(f"dock_to_pose({round(x, 3)},{round(y, 3)},{round(yaw, 3)})", "dock_to_pose", args, REVERSIBLE,
                      eff=self._moved_to(self._resolve(x, y, yaw)), duration=self.cfg["durations"]["dock_to_pose"])

    def press(self, feature: str, args: tuple = ()) -> Action:
        unit = self.pts[feature]["unit"]
        t = self.cfg["templates"]["press_shutter" if feature.endswith("/cover_button") else "press_reset"]
        kw = dict(feature=feature, unit=unit)
        return Action(f"press({feature})", "press", args, REVERSIBLE, pre=self._fill(t["pre"], **kw),
                      eff=self._fill(t["eff"], **kw), duration=self.cfg["durations"]["press"])

    def pick(self, serial: str, support: str, args: tuple = (), removal: bool = False) -> Action:
        kw = dict(support=support, serial=serial, occ=self._occ(support), unit=self.pts[support].get("unit"),
                  grant=f"{self.cfg['grant']['family']}({support})")
        t, r = self.cfg["templates"]["pick"], self.cfg["templates"]["removal"]
        pre, eff = self._fill(t["pre"], **kw), self._fill(t["eff"], **kw)
        if removal:
            return Action(f"remove({serial},{support})", "pick", args, IRREVERSIBLE, pre=pre + self._fill(r["pre"], **kw),
                          eff=eff + self._fill(r["eff"], **kw), duration=self.cfg["durations"]["removal"], grant=kw["grant"])
        return Action(f"pick({serial},{support})", "pick", args, REVERSIBLE, pre=pre, eff=eff, duration=self.cfg["durations"]["pick"])

    def place(self, serial: str, support: str, args: tuple = ()) -> Action:
        kw = dict(support=support, serial=serial, occ=self._occ(support))
        t = self.cfg["templates"]["place"]
        pre, eff = self._fill(t["pre"], **kw), self._fill(t["eff"], **kw)
        if support == self.handover:            # the handover: irreversible, and only of the mission's cargo
            return Action(f"handover({serial},{support})", "place", args, IRREVERSIBLE, pre=pre + ((f"cargo({serial})", True),),
                          eff=eff, duration=self.cfg["durations"]["place"])
        return Action(f"place({serial},{support})", "place", args, REVERSIBLE, pre=pre, eff=eff, duration=self.cfg["durations"]["place"])

    def posture(self, capability: str) -> Action:
        t = self.cfg["templates"][capability]
        return Action(capability, capability, (), REVERSIBLE, pre=self._fill(t["pre"]), eff=self._fill(t["eff"]),
                      duration=self.cfg["durations"][capability])

    def effect_actions(self, knowledge=None) -> list[Action]:
        """Every grounded actuation action of the site: all navigation and docking targets, every press, every
        (serial, support) pick and place, the removal variant of every bay pick, and the postures. With knowledge, the
        navigations' durations are measured from the location knowledge has the rover at (else from nowhere: 0 m)."""
        here = None
        if knowledge is not None:
            at = [p for p in self.locations if knowledge.value(p) == "T"]
            if at:
                name = at[0][at[0].index("(") + 1:-1]
                here = (self.poses.get(name) or self.views[name])[:2]
        if here != self.origin:
            self.origin, self._actions = here, None
        if self._actions is None:
            A = [self.navigate_to_feature(k) for k in self.dockable] + [self.dock_at_feature(k) for k in self.dockable]
            A += [self.navigate_to_pose(*self.poses["dock"]), self.dock_to_pose(*self.poses["dock"])]
            A += [self.navigate_to_pose(*p) for p in self.views.values()]
            A += [self.press(k) for k in self.dockable if self.pts[k]["type"] == "press"
                  and (not k.endswith("/cover_button") or self.pts[k]["unit"] in self.shutter_units)]
            for s in self.serials:
                for k in self.supports:
                    A += [self.pick(s, k), self.place(s, k)] + ([self.pick(s, k, removal=True)] if k in self.bays else [])
            self._actions = A + [self.posture("stow"), self.posture("arm_ready")]
        return self._actions

    # ---- sensing actions ----------------------------------------------------------------------------------------------
    def request_for(self, grant: str) -> Action:
        """The ground request that reads one grant proposition: precondition-free, costs the round trip plus the
        decision, and must start early enough that the reply arrives before Earth sets."""
        c = 2 * self.cfg["link"]["one_way_s"] + self.cfg["link"]["decision_s"]
        bay = grant[len(self.cfg["grant"]["family"]) + 1:-1]
        return Action(f"ground_request({grant})", "request_operator",
                      (("authorize_capability", self.cfg["grant"]["capability"]), ("authorize_target", bay), ("kind", "authorization")),
                      SENSING, obs=(grant,), cost=c, window=self.t_vis - c)   # w = T_vis - c_g, unclamped: negative =
                                                                                     # never available (Earth sets too soon)

    def scan(self, support: str, item: str | None = None) -> Action:
        c = self.cfg["scan"]
        occ = self._occ(support)
        return Action(f"scan_label({support})", "scan_label", (("item", item or self.recorded.get(support, c["default_item"])), ("support", support)),
                      SENSING, pre=self._fill(["base_at({s})", "gripper_empty"], s=support),
                      obs=tuple(f"item_at({s},{support})" for s in self.serials) + (occ, "arm_stowed"), cost=c["cost_s"])

    def sensors(self, knowledge=None) -> list[Action]:
        """Every grounded sensing action. Telemetry (cost 0, no precondition, always) is implicit and not listed. The
        held-item checks depend on which supports knowledge has free (see held_checks)."""
        if self._sensors is not None:
            return self._sensors + self.held_checks(knowledge)
        c = self.cfg["camera"]
        out = []
        for u, v in c["lamp"].items():
            out.append(Action(f"observe_lamp({u})", "observe", (("camera", c["camera"]), ("question", c["lamp_question"].format(name=self.names[u]))),
                              SENSING, pre=((f"at_view({v})", True),), obs=(f"breaker_closed({u})", f"energized({u})", f"lamp_green({u})"), cost=c["cost_s"]))
        for u, v in c["shutter"].items():
            q = c["shutter_question"]["bess" if u == "bess" else "pdu"].format(name=self.names[u])
            out.append(Action(f"observe(shutter_open({u}))", "observe", (("camera", c["camera"]), ("question", q)), SENSING,
                              pre=((f"at_view({v})", True),), obs=(f"shutter_open({u})",), cost=c["cost_s"]))
        out += [self.scan(k) for k in self.supports]
        out += [self.request_for(g) for g in sorted(self.grants)]
        out.append(Action("base_pose", "base_pose", (), SENSING, obs=self.locations, cost=self.cfg["localization"]["cost_s"]))
        out.append(self.held_sensor())
        out.append(Action("arm_state", "arm_state", (), SENSING, obs=("arm_stowed",), cost=self.cfg["arm_posture"]["cost_s"]))
        self._sensors = out
        return out + self.held_checks(knowledge)

    def ground_call(self, capability: str, args: dict) -> Action:
        """A boundary call the executor makes itself (inside a held-item check), as an Action to dispatch: only its
        capability and arguments are used."""
        return Action(f"{capability}({args.get('support', '')})", capability, tuple(sorted(args.items())), REVERSIBLE)

    def staging(self) -> list[str]:
        """Supports the held item can be set down on for a held-item check: not a bay, not the handover point."""
        return [k for k in self.supports if k not in self.bays and k != self.handover]

    def held_check(self, support: str) -> Action:
        """The held-item check: set the held item down on a free support, read its plate there, pick it up again -- the
        state is the same afterwards when the re-pick succeeds (it can fail: an evidence-assumption violation, counted
        as such). It is the one sensor of held(X): no plate is in the wrist view while carrying."""
        d = self.cfg["durations"]
        return Action(f"held_check({support})", "held_check", (("support", support),), SENSING,
                      pre=((f"base_at({support})", True), ("gripper_empty", False), (self._occ(support), False)),
                      obs=(*(f"held({s})" for s in self.serials), "gripper_empty", self._occ(support),
                           *(f"item_at({s},{support})" for s in self.serials)),
                      cost=d["place"] + self.cfg["scan"]["cost_s"] + d["pick"])

    def held_checks(self, knowledge=None) -> list[Action]:
        """A check at every staging support that knowledge does not have occupied: one known free can be used at once,
        one whose occupancy is undecided after a route that decides or assumes it (its precondition becomes the
        executor's subgoal). Without knowledge (the observability check, the lower bound): every staging
        support."""
        return [self.held_check(k) for k in self.staging()
                if knowledge is None or knowledge.value(self._occ(k)) != "T"]

    def held_sensor(self) -> Action:
        # the held-item camera tells empty jaws from holding something, never which item (no plate is in the wrist view
        # while carrying): it observes gripper_empty only, and held(X) has no
        # sensor of its own; held(X) is observed by the held-item check (held_check) or established by a pick verified
        # with X's identity known
        return Action("held_item", "held_item", (), SENSING, obs=("gripper_empty",),
                      cost=self.cfg["held_item"]["cost_s"])

    def unobservable(self, telemetry: bool = True) -> list[str]:
        """Observability: the propositions outside exogenous and grants that no sensing action with an
        unbounded window observes. With telemetry=False the exogenous propositions must be observable too (telemetry removed)."""
        seen = {p for a in self.sensors() if a.window == INF for p in a.obs}
        skip = (self.exogenous if telemetry else set()) | self.grants | {p for p in self.propositions if p.startswith("cargo(")}
        return [p for p in self.propositions if p not in skip and p not in seen]

    # ---- grounding ----------------------------------------------------------------------------------------------------
    def _serial_to_pick(self, kw: dict, support: str, knowledge) -> str | None:
        if kw.get("serial"):
            return kw["serial"]
        named = re.search("|".join(map(re.escape, self.serials)), kw.get("target", ""))
        if named:
            return named.group()
        known = [s for s in self.serials if knowledge.value(f"item_at({s},{support})") == "T"]
        recorded = [s for s in self.serials if (f"item_at({s},{support})", True) in knowledge.record]
        return (known or recorded or [None])[0]

    def _serial_to_place(self, kw: dict, knowledge) -> str | None:
        if kw.get("serial"):
            return kw["serial"]
        held = [s for s in self.serials if knowledge.value(f"held({s})") == "T"]
        maybe = [s for s in self.serials if knowledge.value(f"held({s})") == "U" and (not kw.get("item") or self.kind[s] == kw["item"])]
        return (held or (maybe if len(maybe) == 1 else [None]))[0]

    def ground(self, proposal, knowledge) -> Action | None:
        """The grounded Action of a capability call, or None when the call is not a modelled action (free-text
        observe, queries, plan bookkeeping, engineering stages). A pick at an equipment bay is the irreversible
        removal unless knowledge has the unit de-energized: undecided counts as energized, the safe classification."""
        n, kw = proposal.name, proposal.kwargs
        if n == "check_proposition":            # the planner's typed sensing: a sensor that observes it, executable first
            own = [a for a in self.sensors(knowledge) if kw.get("proposition") in a.obs]
            return min(own, key=lambda a: (not all(knowledge.models(l) for l in a.pre), a.cost, a.name)) if own else None
        # press names only its feature: the button's look is the site survey's, never the planner's
        args = tuple(sorted((k, v) for k, v in kw.items() if k != "serial" and not (n == "press" and k == "target")))
        f = kw.get("feature")
        if n in ("navigate_to_feature", "dock_at_feature") and f in self.pts:
            if n == "navigate_to_feature":
                return self.navigate_to_feature(f, args, kw.get("standoff_m", 1.2))
            return self.dock_at_feature(f, args, kw.get("standoff_m", 0.0))
        if n == "navigate_to_pose":
            return self.navigate_to_pose(kw["x"], kw["y"], kw["yaw"], args)
        if n == "dock_to_pose":
            return self.dock_to_pose(kw["x"], kw["y"], kw["yaw"], args)
        if n == "press" and f in self.dockable and self.pts[f]["type"] == "press":
            return self.press(f, args)
        if n in ("stow", "arm_ready"):
            return self.posture(n)
        if n == "pick" and kw.get("support") in self.supports:
            s = self._serial_to_pick(kw, kw["support"], knowledge)
            removal = kw["support"] in self.bays and knowledge.value(f"energized({self.pts[kw['support']]['unit']})") != "F"
            return s and self.pick(s, kw["support"], args, removal)
        if n == "place" and kw.get("support") in self.supports:
            s = self._serial_to_place(kw, knowledge)
            return s and self.place(s, kw["support"], args)
        if n == "scan_label" and kw.get("support") in self.supports:
            return self.scan(kw["support"], kw.get("item"))
        if n == "held_item":
            return self.held_sensor()
        if n == "request_operator" and kw.get("kind") == "authorization" and kw.get("grant") in self.grants:
            return self.request_for(kw["grant"])
        return None

    # ---- sensing results ----------------------------------------------------------------------------------------------
    def read(self, action: Action, raw: dict) -> dict[str, bool]:
        """The values a sensing action's raw result decides, over action.obs. Anything the result does not settle is left out."""
        cap = action.capability
        if cap == "observe":
            if "Answer A or B first" in dict(action.args).get("question", ""):     # a two-way choice: (B) the proposition true
                m = re.match(r"\W*\(?\s*([AB])\b", raw["answer"], re.I)
                return {p: m.group(1).upper() == "B" for p in action.obs} if m else {}
            m = re.search(r"\b(yes|no)\b", raw["answer"], re.I)
            return {p: m.group().lower() == "yes" for p in action.obs} if m else {}
        if cap == "request_operator":
            return {action.obs[0]: bool(raw["grant"])}
        if cap == "base_pose":
            return self._located(raw["x"], raw["y"], raw["yaw"])
        if cap == "arm_state":
            return {"arm_stowed": raw["posture"] == "stowed"}
        if cap == "held_check":                 # raw: {"serial": plate read on the support or None, "repicked": bool}
            S, occ = dict(action.args)["support"], self._occ(dict(action.args)["support"])
            if raw["serial"] is None:           # nothing read: only what the re-pick establishes
                return {"gripper_empty": not raw["repicked"]} if raw["repicked"] is not None else {}
            back = raw["repicked"]
            return {**{f"held({s})": back and s == raw["serial"] for s in self.serials}, "gripper_empty": not back,
                    occ: not back, **{f"item_at({s},{S})": (not back) and s == raw["serial"] for s in self.serials}}
        if cap == "held_item":
            jaw, j = raw["jaw_rad"], self.cfg["jaw"]
            if jaw <= j["empty_max_rad"] or jaw >= j["open_min_rad"]:     # closed on nothing, or open: cannot hold
                return {"gripper_empty": True, **{p: False for p in action.obs if p.startswith("held(")}}
            return {"gripper_empty": False} if raw["observed"]["holding"] else {}
        # scan_label: raw is the effect list verify() emitted for the call
        by = {e["effect"]: e for e in raw}
        out = {"arm_stowed": by["arm_stowed"]["posture"] == "stowed"} if "arm_stowed" in by else {}
        e = by["labels_read"]
        if not e["holds"]:                                  # the view was not reached
            return out
        support, occ = dict(action.args)["support"], self._occ(dict(action.args)["support"])
        if e["plates_read"]:                                # the nearest plate to the aim point is the item on the support
            near = e["plates"][0]["serial"]
            return {**out, **{f"item_at({s},{support})": s == near for s in self.serials}, occ: True}
        if e["something_on_support"]:                       # an item whose plate was not read: occupied, identity open
            return {**out, occ: True}
        return {**out, **{f"item_at({s},{support})": False for s in self.serials}, occ: False}

    def telemetry_values(self, power_status: dict) -> dict[str, bool]:
        """True values of the exogenous propositions in a `power_status` value, minus the removed ones."""
        out = {}
        for u, s in power_status["units"].items():
            out[f"breaker_closed({u})"] = s["breaker"] == "closed"
            out[f"energized({u})"] = bool(s["energized"])
            out[f"lamp_green({u})"] = s["lamp"] == "green"
            out[f"sink_hot({u})"] = s["heat_sink_temperature_c"] > self.cfg["sink_hot_c"]
            out.update({f"connector_ok({bay})": state == "ok" for bay, state in s["bay_connectors"].items()})
            for bay, serial in s["installed_modules"].items():
                out[f"installed({bay})"] = serial is not None
                out.update({f"item_at({x},{bay})": serial == x for x in self.serials})
        return {p: v for p, v in out.items() if p not in self.removed and _family(p) not in self.removed}

    # ---- initial knowledge, record, goal ------------------------------------------------------------------------------
    def k0(self) -> dict[str, bool]:
        """Trusted at the start: rover docked at the dock, jaws empty (so nothing held), arm stowed. The surveyed layout is
        the structure of the vocabulary itself; the inventory record is not here."""
        return {**{p: p == "base_at(dock)" for p in self.locations}, "gripper_empty": True, "arm_stowed": True,
                **{f"held({s})": False for s in self.serials},
                **({f"cargo({s})": s == self.cargo for s in self.serials} if self.handover else {})}

    def record(self) -> list[Literal]:
        """The inventory record as literals: each tracked item at its recorded location."""
        return [(f"item_at({s},{loc})", True) for s, loc in self._record]

    def goal(self) -> list[Literal]:
        return [parse_lit(t) for t in self.cfg["missions"][self.mission]]

    # ---- evidence rules -----------------------------------------------------------------------------------------------
    def judge(self, action: Action, verify_effects: list[dict], kappa_pre: dict | None = None,
              telemetry: dict | None = None) -> dict[Literal, str]:
        """A judgment "+", "-" or "?" for every literal of action.eff, from the effects verification.verify emitted.

        "+" only when a measurement establishes the literal, "-" only when one establishes its absence; a missing
        measurement (verify reports holds=False for it too), holds=None, or a proposition the rule does not measure is "?".
        `kappa_pre` maps the action's precondition propositions to "T"/"F"/"U" as knowledge had them at dispatch: an
        effect about the identity of the item (held(X), item_at(X,S)) is "+" only if the identity precondition was "T".
        `telemetry` is the `power_status` read right after a press: its effects that telemetry reports (breaker closed,
        energized, lamp green) are judged by it, "+" when the reading shows the literal, "-" when it shows its negation."""
        out = {l: "?" for l in action.eff}
        if telemetry is not None and action.capability == "press":
            read = self.telemetry_values(telemetry)
            out.update({l: "+" if read[l[0]] == l[1] else "-" for l in action.eff if l[0] in self.exogenous and l[0] in read})
        rule = {"navigate_to_feature": self._nav, "navigate_to_pose": self._nav, "dock_at_feature": self._nav,
                "dock_to_pose": self._nav, "stow": self._posture, "arm_ready": self._posture, "pick": self._pick,
                "place": self._place}.get(action.capability)    # press: its equipment effects by telemetry (above); the
                                                                  # shutter and the folded arm are not measured: "?"
        if rule:
            rule(action, {e["effect"]: e for e in verify_effects}, kappa_pre or {}, out)
        return out

    @staticmethod
    def _identity_known(action: Action, kp: dict) -> bool:
        return all(kp.get(p) == ("T" if v else "F") for p, v in action.pre if p.startswith(("item_at(", "held(")))

    @staticmethod
    def _nav(action, by, kp, out):
        e = by.get("base_at_commanded_pose")
        if e is None:                                       # no base-pose estimate in the evidence
            return
        for l in action.eff:
            if l[1]:                                        # the location the call was for: inside / outside the band
                out[l] = "+" if e["holds"] else "-"
            elif e["holds"]:                                # at the commanded pose, which is not any other location
                out[l] = "+"

    @staticmethod
    def _posture(action, by, kp, out):
        e = by.get("arm_stowed") or by.get("arm_ready")
        if e is not None:
            for l in action.eff:
                out[l] = "+" if (e["posture"] == "stowed") == l[1] else "-"

    def _left_behind(self, e: dict) -> bool:
        """The image flow of the item matched a left-behind item's: a conclusive measurement that it did not rise."""
        h = self.cfg["hold"]
        flow, expect = e.get("image_flow_px"), e.get("left_behind_flow_px")
        return flow is not None and math.isfinite(flow) and e["lift_m"] >= h["min_lift_m"] and flow >= h["flow_fraction"] * expect

    def _pick(self, action, by, kp, out):
        e = by.get("item_held")
        if e is None:                                       # no arm evidence
            return
        empty, rose, stayed = e["jaw"] != "holding_object", e["holds"] is True, "rose_with_gripper" in e and self._left_behind(e)
        ident = self._identity_known(action, kp)
        for l in action.eff:
            p = l[0]
            if p.startswith("held("):
                out[l] = "-" if empty or stayed else "+" if rose and ident else "?"
            elif p == "gripper_empty":
                out[l] = "-" if empty else "+"
            elif p.startswith("item_at("):
                out[l] = "-" if stayed and ident else "+" if rose and ident else "?"
            elif p.startswith(("occupied(", "installed(")):
                out[l] = "-" if stayed else "+" if rose else "?"

    def _place(self, action, by, kp, out):
        e = by.get("item_placed")
        if e is None:
            return
        between = e.get("points_between_open_fingers")     # depth points in the open fingers' grip space; absent without a settle record
        busy = e["jaw"] == "holding_object" or (between or 0) >= self.cfg["place"]["in_jaws_min_points"]   # the item is still with the gripper
        empty = e["jaw"] == "closed_empty" or (e["jaw"] == "open" and between is not None and not busy)
        ident = self._identity_known(action, kp)
        was_free = any(not v and kp.get(p) == "F" for p, v in action.pre if p.startswith(("occupied(", "installed(")))
        for l in action.eff:
            p = l[0]
            if p == "gripper_empty":
                out[l] = "+" if empty else "-" if busy else "?"
            elif p.startswith("held("):
                out[l] = "+" if empty else "-" if busy and ident else "?"
            elif p.startswith("item_at("):
                out[l] = "+" if e["holds"] is True and ident else "-" if e["holds"] is False and busy and ident else "?"
            elif p.startswith(("occupied(", "installed(")):
                out[l] = "+" if e["holds"] is True else "-" if e["holds"] is False and busy and was_free else "?"

    # ---- ground truth (evaluator and accounting only) ---------------------------------------------------------------------
    def truth(self, state: dict) -> dict[str, bool]:
        """True value of the propositions the simulator's evaluator state decides. `state` is the evaluator's
        `mission_state` (units, objects, parked_and_stowed), optionally with "held" (object names between the jaws),
        "arm_posture" and "base_pose" {x, y, yaw}, which the evaluator log does not carry today. Without a truth source,
        a proposition is absent: granted_remove (the ground's decision is run configuration, not simulator state),
        and, unless supplied, held / gripper_empty / arm_stowed / at_view / base_at (only parked_and_stowed fixes the dock)."""
        t = {}
        for u, s in state["units"].items():
            t[f"breaker_closed({u})"] = t[f"energized({u})"] = t[f"lamp_green({u})"] = not s["tripped"]
            if "heat_sink_temperature_c" in s:
                t[f"sink_hot({u})"] = s["heat_sink_temperature_c"] > self.cfg["sink_hot_c"]
                t.update({f"connector_ok({bay}/tray)": bay not in s["damaged_bays"] for bay in s["bays"]})
            if u in self.shutter_units:
                t[f"shutter_open({u})"] = not s["cover_closed"]
            t.update({f"installed({bay}/tray)": module != "" for bay, module in s["bays"].items()})
        where = {self.serial_of[name]: loc for name, loc in state["objects"].items()}
        # a module seated in a bay's tray is not "resting on a support" for the evaluator's object test (where_is gives
        # ""): the units' bay contents are the truth for bays
        for s_ in state["units"].values():
            where.update({self.serial_of[m]: f"{bay}/tray" for bay, m in s_["bays"].items() if m})
        t.update({f"item_at({s},{k})": where.get(s) == k for s in self.serials for k in self.supports})
        t.update({f"occupied({k})": k in where.values() for k in self.supports if k not in self.bays})
        if "held" in state:
            held = {self.serial_of[n] for n in state["held"]}
            t.update({f"held({s})": s in held for s in self.serials})
            t["gripper_empty"] = not held
        if "arm_posture" in state:
            t["arm_stowed"] = state["arm_posture"] == "stowed"
        if "base_pose" in state:
            t.update(self._located(state["base_pose"]["x"], state["base_pose"]["y"], state["base_pose"]["yaw"]))
        elif state["parked_and_stowed"]:
            t.update({"base_at(dock)": True, "arm_stowed": True})
        return t
