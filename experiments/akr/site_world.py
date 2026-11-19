"""The sampled world of the Monte Carlo trials (manuscript Sec. 4.7): the executor of the rover runs unchanged, with the
skill layer and the simulator replaced by this world.

The state is the site's: where each item is, what the rover holds and where it stands, and the units' breakers, trip
causes, reset counts, connectors and shutters, with the equipment rules of the simulator branch (lunar_base/site.py):
a reset closes a tripped breaker at the press's contact; a closed unit with an empty bay trips MODULE_MISSING at once,
with a damaged connector CONNECTOR_DAMAGED, with an unserviceable module its fault code after the 3 s self-test, with
its shutter drawn OVERTEMP after 30 s (opening the shutter in time cancels it); a module taken out of a closed unit
damages the bay's connector and trips the unit. Which modules are unserviceable and which shutters are drawn is the
world's, never a proposition. Telemetry is the units' power_status as the simulator publishes it, read by the
vocabulary's own reader.

Actuation follows Definition 2: effects realize only under a true precondition, and then with probability rho; a press
acts on the equipment as the button does. Judgments: a press's equipment effects by the telemetry after it, its
other effects unverified (manuscript Sec. 4.2); every other effect conclusive with probability q, and then right.
Sensing returns the true values, each wrong with probability eta (not the ground's decision, which is fixed per run).
The ground's decision on each grant proposition is `grant`, or, with p_G > 0, refused with probability p_G independently
per grant (one request per grant: a refusal is final), drawn from its own stream so the rest of the trial is unchanged;
it is the grant's truth, so a removal without a grant is hazardous exactly when the ground would refuse. A grant is a
flight rule, not a physical condition: an ungranted removal acts on the hardware like a granted one.
"""
from __future__ import annotations

import random

from space_runtime.akr.injection import IX_PATH
from space_runtime.akr.types import Outcome

import json

RETRIP_S, SELFTEST_S, SINK_AMBIENT_C, SINK_TAU_S = 30.0, 3.0, 25.0, 600.0
CONTACT = 0.5          # fraction of a press's duration at which the button is pressed
IX = json.loads(IX_PATH.read_text())
FAULT = {"orbital replacement unit (ORU)": "ORU_SELFTEST_FAIL", "battery module": "BATTERY_MODULE_FAULT",
         "filter cartridge": "FILTER_CLOGGED"}


class SiteWorld:
    def __init__(self, vocab, telemetry: dict, truth: dict, *, rho: float = 1.0, q: float = 1.0, eta: float = 0.0,
                 grant: bool = True, p_G: float = 0.0, seed: int = 0):
        self.v, self.rng, self.rho, self.q, self.eta, self.grant = vocab, random.Random(seed), rho, q, eta, grant
        g_rng = random.Random(f"{seed}-ground")
        self.decision = {g: (g_rng.random() >= p_G) if p_G > 0 else grant for g in sorted(vocab.grants)}
        self.t, self.loc, self.held, self.stowed = 0.0, "base_at(dock)", None, True
        self.bad, self.drawn = set(truth.get("unserviceable", ())), set(truth.get("drawn", ()))
        self.where = dict(truth["where"])
        self.bays = {u: [f"{b}/tray" for b in spec["bays"]] for u, spec in IX["units"].items()}
        self.units = {u: dict(closed=s["breaker"] == "closed", cause=s.get("trip_cause"), resets=s.get("resets_since_start", 0),
                              sink=s["heat_sink_temperature_c"], damaged={b for b, c in s["bay_connectors"].items() if c != "ok"})
                      for u, s in telemetry["units"].items()}
        self.timers: list[tuple[float, str, str]] = []
        self.truth_log: list[dict] = []
        self.wrong: list[dict] = []
        for u, s in self.units.items():
            if s["closed"]:
                self._schedule(u)

    # ---- the units ---------------------------------------------------------------------------------------------------
    def _mods(self, u: str) -> dict:
        return {b: next((x for x, k in self.where.items() if k == b), None) for b in self.bays[u]}

    def _trip(self, u: str, cause: str) -> None:
        s = self.units[u]
        if s["closed"]:
            s["closed"], s["cause"] = False, cause
            if cause == "OVERTEMP":
                s["sink"] = 85.0
        self.timers = [x for x in self.timers if x[1] != u]

    def _schedule(self, u: str) -> None:
        s, mods = self.units[u], self._mods(u)
        self.timers = [x for x in self.timers if x[1] != u]
        if any(m is None for m in mods.values()):
            return self._trip(u, "MODULE_MISSING")
        if s["damaged"] & set(mods):
            return self._trip(u, "CONNECTOR_DAMAGED")
        bad = next((m for m in mods.values() if m in self.bad), None)
        if bad:
            self.timers.append((self.t + SELFTEST_S, u, FAULT[self.v.kind[bad]]))
        if u in self.drawn:
            self.timers.append((self.t + RETRIP_S, u, "OVERTEMP"))

    def _advance(self, dt: float) -> None:
        end = self.t + dt
        while True:
            due = sorted(x for x in self.timers if x[0] <= end)
            if not due:
                break
            self.t = due[0][0]
            self._trip(due[0][1], due[0][2])
        self.t = end
        for s in self.units.values():          # the heat sink relaxes towards ambient (site.py, drawn time constant)
            s["sink"] += (SINK_AMBIENT_C - s["sink"]) * min(1.0, dt / SINK_TAU_S)

    # ---- truth -------------------------------------------------------------------------------------------------------
    def props(self) -> dict[str, bool]:
        v, out = self.v, {}
        for p in v.locations:
            out[p] = p == self.loc
        out["gripper_empty"], out["arm_stowed"] = self.held is None, self.stowed
        for x in v.serials:
            out[f"held({x})"] = self.held == x
            for k in v.supports:
                out[f"item_at({x},{k})"] = self.where.get(x) == k
        for k in v.supports:
            out[v._occ(k)] = any(w == k for w in self.where.values())
        for u, s in self.units.items():
            for f in ("breaker_closed", "energized", "lamp_green"):
                out[f"{f}({u})"] = s["closed"]
            out[f"sink_hot({u})"] = s["sink"] > v.cfg["sink_hot_c"]
            for b in self.bays[u]:
                out[f"connector_ok({b})"] = b not in s["damaged"]
        out.update({f"shutter_open({u})": u not in self.drawn for u in v.shutter_units})
        out.update({g: self.decision[g] for g in v.grants})
        if v.handover:
            out.update({f"cargo({x})": x == v.cargo for x in v.serials})
        return out

    def power_status(self) -> dict:
        return {"units": {u: {"name": self.v.names[u], "breaker": "closed" if s["closed"] else "tripped", "energized": s["closed"],
                              "lamp": "green" if s["closed"] else "red", "heat_sink_temperature_c": round(s["sink"], 1),
                              "trip_cause": None if s["closed"] else s["cause"], "resets_since_start": s["resets"],
                              "installed_modules": self._mods(u),
                              "bay_connectors": {b: "damaged" if b in s["damaged"] else "ok" for b in self.bays[u]}}
                          for u, s in self.units.items()}}

    # ---- the World protocol ------------------------------------------------------------------------------------------
    def now(self) -> float:
        return self.t

    def telemetry(self) -> dict[str, bool]:
        return self.v.telemetry_values(self.power_status())

    def poll(self) -> list:
        return []

    def sense(self, a) -> dict[str, bool]:
        self._advance(a.cost)
        if a.capability == "request_operator":
            return {a.obs[0]: self.decision.get(a.obs[0], self.grant)}
        tr, out = self.props(), {}
        for p in a.obs:
            if p not in tr:
                continue
            wrong = self.eta > 0 and self.rng.random() < self.eta
            out[p] = tr[p] != wrong
            if wrong:
                self.wrong.append(dict(t=self.t, action=a.name, prop=p, true=tr[p]))
        return out

    def execute(self, a) -> Outcome:
        tr = self.props()
        holds = all(tr.get(p) == val for p, val in a.pre)
        self.truth_log.append({"t": self.t, "action": a.name, "kind": a.kind, "pre_holds": holds,
                               "pre_truth": {f"{'' if val else 'not '}{p}": tr.get(p) == val for p, val in a.pre}})
        realize = self.rng.random() < self.rho
        if a.capability == "press":
            self._press(a, holds and realize)
            tele = self.v.telemetry_values(self.power_status())
            judg = {(p, val): ("+" if tele[p] == val else "-") if p in self.v.exogenous and p in tele else "?" for p, val in a.eff}
            return Outcome(judg, a.duration)
        # an authorization is a flight rule, not physics: a removal without a grant still acts (and is hazardous)
        physical = all(tr.get(p) == val for p, val in a.pre if p not in self.v.grants)
        if physical and realize:
            self._apply(a)
        self._advance(a.duration)
        after = self.props()
        judg = {(p, val): "?" if self.rng.random() >= self.q else ("+" if after.get(p) == val else "-") for p, val in a.eff}
        return Outcome(judg, a.duration)

    def _press(self, a, ok: bool) -> None:
        f = a.name[len("press("):-1]
        u = self.v.pts[f]["unit"]
        self._advance(CONTACT * a.duration)
        if ok and f.endswith("/cover_button"):
            if u in self.drawn:
                self.drawn.discard(u)
                self.timers = [x for x in self.timers if not (x[1] == u and x[2] == "OVERTEMP")]
            else:
                self.drawn.add(u)
                if self.units[u]["closed"]:
                    self._schedule(u)
        elif ok and not self.units[u]["closed"]:
            s = self.units[u]
            s["closed"], s["cause"], s["resets"] = True, None, s["resets"] + 1
            self._schedule(u)
        if ok:
            self.stowed = True
        self._advance((1 - CONTACT) * a.duration)

    def _apply(self, a) -> None:
        locs = [p for p, val in a.eff if p in self.v.locations and val]
        if any(p in self.v.locations for p, _ in a.eff):
            self.loc = locs[0] if locs else None
        for p, val in a.eff:
            if p == "arm_stowed":
                self.stowed = val
        if a.capability == "pick":
            x = next(p[len("held("):-1] for p, val in a.eff if p.startswith("held(") and val)
            k = self.where.pop(x, None)
            self.held, self.stowed = x, False
            unit = self.v.pts[k].get("unit") if k in self.v.bays else None
            if unit and self.units[unit]["closed"]:     # out of an energized unit: the connector arcs
                self.units[unit]["damaged"].add(k)
                self._trip(unit, "MODULE_MISSING")
        elif a.capability == "place":
            lit = next(p for p, val in a.eff if p.startswith("item_at(") and val)
            x, k = lit[len("item_at("):-1].split(",")
            self.where[x], self.held = k, None
