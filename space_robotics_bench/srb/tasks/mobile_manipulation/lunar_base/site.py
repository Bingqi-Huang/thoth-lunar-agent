"""The lunar base as the mission tasks share it: the power field's electrical and thermal model and its telemetry, the
movable equipment and mission control's inventory record of it, and the robot state every mission ends in.

The serviced units are the scene's `units` (layout.yaml): each has a
breaker behind its control panel, one or more equipment bays each holding a replaceable module (an ORU, a battery
module, a filter cartridge), and possibly a heat-rejection surface with a shutter (a PDU radiator, the BESS HVAC
condenser). The physics is the same in every scenario; a scenario only sets the initial conditions (which breaker is
tripped and why, which shutters are drawn, which modules are unserviceable, where the objects stand, what the
inventory record says):

- a closed breaker whose shutter is drawn overheats and trips OVERTEMP after `retrip_delay_s`; opening the shutter in
  time cancels it;
- a closed breaker with an unserviceable module in a bay trips after the unit's power-on self-test, with the module's
  own fault code (MODULE_FAULT); with a bay empty it trips MODULE_MISSING at once;
- pressing a panel's reset button closes a tripped breaker; pressing a shutter's button opens or closes the shutter;
- a module taken out of a bay while its unit is energized arcs the bay's connector: the connector is damaged for good,
  and a module seated there again does not make contact (the unit trips CONNECTOR_DAMAGED whenever it is closed). The
  damage is a state, not a model of the arc; it makes the removal irreversible for the robot;
- an item set down on a handover point (a delivery's receiving stand) is taken in by the consignee once it rests there:
  it is latched in place and the robot cannot take it back, whichever item it is (items standing there at the start are
  not latched).

The panel lamp shows the breaker (red = tripped, green = closed). The units publish their own telemetry
(`power_telemetry`: breaker, lamp, power-converter heat-sink temperature, trip cause, resets, module serial and
connector state per bay), equipment data any method may read; `mission_state()` in the mission tasks is
evaluator-only. Record faults (`records`, `record_swaps`) change only what the inventory record says, never the world.
"""

import math
from typing import Dict, List, Sequence, Set

import torch
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz

from srb.assets.scenery.lunar_base import OBJECT_PREFIX, load_interaction, set_radiator_cover, set_status_lamp
from srb.utils.cfg import configclass

from .task import Task as _BaseTask
from .task import TaskCfg as _BaseTaskCfg

## The base's serviced units as the scene defines them (layout.yaml `units`): the three power-field PDUs, the battery
## storage unit and the ISRU oxygen plant, each with its load and criticality -- the load-shedding order (Lewis et al.
## 2022, NTRS 20220013667: crew life support is shed last, "technology demonstrations and pilot plant systems" first).
UNIT_SPECS = load_interaction()["units"]
UNITS = tuple(UNIT_SPECS)
#: how mission control writes a unit's name (briefs, reports)
UNIT_NAMES = {u: spec["name"] for u, spec in UNIT_SPECS.items()}
## the fault code a unit's self-test reports for an unserviceable module, by module kind
MODULE_FAULT = {"orbital replacement unit (ORU)": "ORU_SELFTEST_FAIL", "battery module": "BATTERY_MODULE_FAULT",
                "filter cartridge": "FILTER_CLOGGED"}
TRIP_CAUSES = ("", "OVERCURRENT", "OVERTEMP", *MODULE_FAULT.values(), "MODULE_MISSING", "CONNECTOR_DAMAGED")
SELFTEST_S = 3.0  # the unit's power-on self-test: how long a closed breaker runs on an unserviceable module
## the power converter's heat sink (the thermal assumption of `retrip_delay_s` below): ambient with the rejection path
## open, the cutout at 85 C, reopening 5 K below it; with the shutter drawn a running unit climbs those 5 K in
## `retrip_delay_s`; a unit not running, or running with the shutter open, relaxes towards ambient (time constant
## shorter with the shutter open). Reported only: the OVERTEMP trip itself keeps its timer.
SINK_AMBIENT_C, SINK_CUTOUT_C, SINK_HYSTERESIS_K = 25.0, 85.0, 5.0
SINK_TAU_OPEN_S, SINK_TAU_DRAWN_S = 120.0, 600.0
LAMP_RED = (1.0, 0.0, 0.0)
LAMP_GREEN = (0.1, 1.0, 0.2)
PRESS_TOLERANCE_M = 0.03  # TCP within this of a button cap is a press
DOCK_TOLERANCE_M = 0.6
STOW_TOLERANCE_RAD = 0.35
ON_SUPPORT_XY_M = (0.12, 0.18)  # object centre within (along normal, lateral) of the support point
ON_SUPPORT_Z_M = 0.03  # object base within this of the support top
UPRIGHT_COS = math.cos(math.radians(15.0))
RESTING_SPEED_M_S = 0.05
NOT_HELD_M = 0.30  # handle this far from the gripper TCP
_STOW = (0.0, 0.40, 0.0, -0.70, 0.0, -1.27, 1.57079633)  # wrist camera level
_TCP_OFFSET = (0.0, 0.0, 0.140)


def _pairs(spec: str) -> Dict[str, str]:
    """"a:b+c:d" -> {a: b, c: d}"""
    return dict(kv.split(":", 1) for kv in spec.split("+") if kv)


@configclass
class SiteTaskCfg(_BaseTaskCfg):
    ## how long a running unit with its heat-rejection shutter drawn takes to re-trip OVERTEMP after a reset. Fixed
    ## before the freeze from a stated thermal assumption, not tuned to any method: the unit tripped OVERTEMP, so the
    ## thermal cutout on the power converter's heat sink reopens only after the cutout's hysteresis (5 K); with the
    ## rejection path shut, the converter's losses (200 W: a few kW unit at ~95 % efficiency) heat the sink (1.5 kg of
    ## aluminium, 1.35 kJ/K) back through that hysteresis in 1350 J/K * 5 K / 200 W ~ 34 s. Rounded to 30 s.
    retrip_delay_s: float = 30.0
    ## end the episode (and auto-reset) on verifier success; scripted demos turn this off to keep the final state
    terminate_on_success: bool = True
    ## record faults (the inventory record only; the world is untouched), Hydra strings "object:object+...":
    ## `records` "object:place_frame" sets where the record says an object is (false locations); `record_swaps`
    ## "object:object" exchanges what the record says about two objects (false identities: each serial recorded where
    ## the other item stands, with the other's status)
    records: str = ""
    record_swaps: str = ""
    ## handover points (place frames, "a+b") where arriving items are latched; mission tasks may add their own
    handover: str = ""


class SiteTask(_BaseTask):
    cfg: SiteTaskCfg

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        ix = load_interaction()
        self._ix = ix
        dev, n = self.device, self.num_envs
        ## buttons: every unit's panel reset, then the shutter button of every unit that has one
        self._covered_units = [k for k, u in enumerate(UNITS) if UNIT_SPECS[u]["cover"]]
        self._buttons = torch.tensor([ix["points"][f"{UNIT_SPECS[u]['panel']}/press"]["xyz"] for u in UNITS]
                                     + [ix["points"][f"{UNIT_SPECS[UNITS[k]]['cover']}/cover_button"]["xyz"] for k in self._covered_units], device=dev)
        self._objs = {name: self.scene[OBJECT_PREFIX + name] for name in ix["objects"]}
        self._handle_local = {name: torch.tensor(o["handle"]["xyz"], device=dev) for name, o in ix["objects"].items()}
        self._modules = [name for name, o in ix["objects"].items() if o["item"] in MODULE_FAULT]
        self._bays = [bay for unit in UNITS for bay in UNIT_SPECS[unit]["bays"]]
        ## the place frames as tensors: the on-support test of many (object, support) pairs is one batched computation
        ## per tick instead of one small-tensor pass per pair (it was most of `_get_dones`' 16 ms)
        self._place = [k for k, v in ix["points"].items() if v["type"] == "place"]
        self._place_ix = {k: i for i, k in enumerate(self._place)}
        self._place_xyz = torch.tensor([ix["points"][k]["xyz"] for k in self._place], device=dev)
        self._place_nrm = torch.tensor([ix["points"][k]["normal"][:2] for k in self._place], device=dev)
        self._up = torch.tensor([0.0, 0.0, 1.0], device=dev)
        self._arm_ids, _ = self._robot.find_joints([f"joint{i}" for i in range(1, 8)], preserve_order=True)
        self._link7 = self._robot.find_bodies("Link7")[0][0]
        self._stow = torch.tensor(_STOW, device=dev)
        self._tcp_offset = torch.tensor(_TCP_OFFSET, device=dev)
        self._tripped = torch.zeros(n, len(UNITS), dtype=torch.bool, device=dev)
        self._cause = torch.zeros(n, len(UNITS), dtype=torch.long, device=dev)
        self._overtemp_at = torch.full((n, len(UNITS)), math.inf, device=dev)
        self._selftest_at = torch.full((n, len(UNITS)), math.inf, device=dev)
        self._cover_closed = torch.zeros(n, len(UNITS), dtype=torch.bool, device=dev)
        self._sink_c = torch.full((n, len(UNITS)), SINK_AMBIENT_C, device=dev)
        self._damaged = torch.zeros(n, len(self._bays), dtype=torch.bool, device=dev)
        self._last_content = None  # bay contents at the previous step (a removal is a bay going from a module to empty)
        names = list(self._objs)
        self._latched = torch.zeros(n, len(names), dtype=torch.bool, device=dev)
        self._latch_pose = torch.zeros(n, len(names), 7, device=dev)
        self._there_at_start = torch.zeros(n, len(names), dtype=torch.bool, device=dev)
        self._selftest_cause = torch.zeros(n, len(UNITS), dtype=torch.long, device=dev)
        self._serviceable = {name: torch.ones(n, dtype=torch.bool, device=dev) for name in self._modules}
        self._pressing = torch.zeros(n, len(self._buttons), dtype=torch.bool, device=dev)
        self._press_count = torch.zeros(n, len(self._buttons), dtype=torch.long, device=dev)
        self._resets = torch.zeros(n, len(UNITS), dtype=torch.long, device=dev)
        self._success = torch.zeros(n, dtype=torch.bool, device=dev)

    ## -- scenario initial conditions (mission tasks override) ----------------------------------------------------
    def initial_trips(self) -> Dict[str, str]:
        """{unit: trip cause} tripped at reset."""
        return {}

    def initial_closed_covers(self) -> Set[str]:
        """Units whose shutter is drawn at reset."""
        return set()

    def unserviceable_modules(self) -> Set[str]:
        """Module objects that fail their unit's self-test: those the scene authors as failed (a unit removed from
        service); scenarios add their own."""
        return {name for name, o in self._ix["objects"].items() if o.get("status", "").startswith("failed")}

    def object_placement(self) -> Dict[str, str]:
        """{object: place frame} at reset; the authored supports by default."""
        return {name: o["support"] for name, o in self._ix["objects"].items()}

    def inventory_record(self) -> List[Dict[str, str]]:
        """Mission control's inventory record (IMS-like): what it believes each tracked item is, where, and in what
        state. A record, not an observation."""
        return [{"serial": o["serial"], "item": o["item"], "recorded_location": o["support"], "status": o.get("status", "serviceable")}
                for o in self._ix["objects"].values()]

    def with_record_faults(self, record: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """The configured record faults applied to a record (mission tasks call it last in `inventory_record`)."""
        entry = {r["serial"]: r for r in record}
        for name, location in _pairs(self.cfg.records).items():
            entry[self._ix["objects"][name]["serial"]]["recorded_location"] = location
        for a, b in _pairs(self.cfg.record_swaps).items():
            ra, rb = entry[self._ix["objects"][a]["serial"]], entry[self._ix["objects"][b]["serial"]]
            for key in ("recorded_location", "status"):
                ra[key], rb[key] = rb[key], ra[key]
        return record

    ## -- reset ---------------------------------------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        if not hasattr(self, "_success"):  # resets issued while the base class is still initialising
            return
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        id_list = ids.tolist()
        trips, covers, bad = self.initial_trips(), self.initial_closed_covers(), self.unserviceable_modules()
        for k, unit in enumerate(UNITS):
            self._tripped[ids, k] = unit in trips
            self._cause[ids, k] = TRIP_CAUSES.index(trips.get(unit, ""))
            self._cover_closed[ids, k] = unit in covers
            self._sink_c[ids, k] = SINK_CUTOUT_C if trips.get(unit) == "OVERTEMP" else SINK_AMBIENT_C
            set_status_lamp(id_list, UNIT_SPECS[unit]["panel"], LAMP_RED if unit in trips else LAMP_GREEN)
            if UNIT_SPECS[unit]["cover"]:
                set_radiator_cover(id_list, UNIT_SPECS[unit]["cover"], unit in covers)
        for name in self._modules:
            self._serviceable[name][ids] = name not in bad
        self._overtemp_at[ids] = math.inf
        self._selftest_at[ids] = math.inf
        self._damaged[ids] = False
        self._last_content = None
        self._pressing[ids] = False
        self._press_count[ids] = 0
        self._resets[ids] = 0
        self._success[ids] = False
        where = self.object_placement()
        for name, support in where.items():
            self.place_object(id_list, name, support)
        handover = set(self.handover_supports())
        self._latched[ids] = False
        self._there_at_start[ids] = torch.tensor([where.get(name) in handover for name in self._objs], device=self.device)

    def handover_supports(self) -> List[str]:
        """Place frames where the consignee latches arriving items (mission tasks add the delivery's destination)."""
        return [s for s in self.cfg.handover.split("+") if s]

    def _update_handover(self):
        """Latch items that come to rest on a handover point; hold latched items where they were taken in."""
        sup = self.handover_supports()
        if not sup:
            return
        names = list(self._objs)
        arrived = self.on_supports(names, sup, settled=True).any(dim=2) & ~self._latched & ~self._there_at_start
        for j, name in enumerate(names):
            new = arrived[:, j]
            if new.any():
                self._latch_pose[new, j] = self._objs[name].data.root_state_w[new, :7]
                self._latched[new, j] = True
            held = self._latched[:, j]
            if held.any():
                ids = held.nonzero().flatten()
                self._objs[name].write_root_pose_to_sim(self._latch_pose[ids, j], env_ids=ids)
                self._objs[name].write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=self.device), env_ids=ids)

    def place_object(self, env_ids: Sequence[int], name: str, support: str):
        """Put an object on a place frame with the same yaw relative to the frame normal as its authored spawn."""
        spec = self._ix["objects"][name]
        default_n = self._ix["points"][spec["support"]]["normal"]
        rel_yaw = math.radians(spec["yaw_deg"]) - math.atan2(default_n[1], default_n[0])
        pt = self._ix["points"][support]
        yaw = math.atan2(pt["normal"][1], pt["normal"][0]) + rel_yaw
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        pos = torch.tensor([pt["xyz"][0], pt["xyz"][1], pt["xyz"][2] + 0.005], device=self.device)
        zero = torch.zeros(1, device=self.device)
        quat = quat_from_euler_xyz(zero, zero, torch.tensor([yaw], device=self.device))
        pose = torch.cat([pos.unsqueeze(0).repeat(len(ids), 1) + self.scene.env_origins[ids], quat.repeat(len(ids), 1)], dim=1)
        self._objs[name].write_root_pose_to_sim(pose, env_ids=ids)
        self._objs[name].write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=self.device), env_ids=ids)

    ## -- the electrical and thermal model ------------------------------------------------------------------------
    def sim_time(self) -> torch.Tensor:
        return self.episode_length_buf.float() * self.step_dt

    def bay_contents(self, bays: Sequence[str]) -> Dict[str, List[str]]:
        """Module object resting in each bay, per env 0 ... n-1 ('' if empty)."""
        # physically in the bay, gripper on its handle or not, settling or not: a module is out when it is lifted out
        on = self.on_supports(self._modules, [f"{bay}/tray" for bay in bays], settled=False).tolist()
        return {bay: [next((m for i, m in enumerate(self._modules) if on[e][i][j]), "") for e in range(self.num_envs)]
                for j, bay in enumerate(bays)}

    def bay_content(self, bay: str) -> List[str]:
        """Module objects resting in a bay, per env 0 ... n-1 ('' if empty)."""
        return self.bay_contents([bay])[bay]

    def _trip(self, mask: torch.Tensor, k: int, cause: torch.Tensor | str):
        self._tripped[mask, k] = True
        self._cause[mask, k] = TRIP_CAUSES.index(cause) if isinstance(cause, str) else cause[mask]
        self._overtemp_at[mask, k] = math.inf
        self._selftest_at[mask, k] = math.inf
        set_status_lamp(mask.nonzero().flatten().tolist(), UNIT_SPECS[UNITS[k]]["panel"], LAMP_RED)

    def _update_power(self):
        t = self.sim_time()
        dist = torch.linalg.norm(self.tcp_pos_env().unsqueeze(1) - self._buttons.unsqueeze(0), dim=-1)  # (n, 2 * units)
        near = dist < PRESS_TOLERANCE_M
        pressed = near & ~self._pressing
        self._pressing = near | (self._pressing & (dist < 2.0 * PRESS_TOLERANCE_M))  # contact jitter is one press
        self._press_count += pressed.long()
        content = self.bay_contents(self._bays)
        last, self._last_content = self._last_content, content
        dt = self.step_dt
        for k, unit in enumerate(UNITS):
            reset = pressed[:, k] & self._tripped[:, k]
            if reset.any():
                self._tripped[reset, k] = False
                self._cause[reset, k] = 0
                self._resets[reset, k] += 1
                set_status_lamp(reset.nonzero().flatten().tolist(), UNIT_SPECS[unit]["panel"], LAMP_GREEN)
            if k in self._covered_units:
                toggle = pressed[:, len(UNITS) + self._covered_units.index(k)]
                if toggle.any():
                    self._cover_closed[toggle, k] = ~self._cover_closed[toggle, k]
                    for e in toggle.nonzero().flatten().tolist():
                        set_radiator_cover([e], UNIT_SPECS[unit]["cover"], bool(self._cover_closed[e, k]))
            closed = ~self._tripped[:, k]
            # a module lifted out of a bay while the unit is energized arcs the connector (before this step's trip)
            if last is not None:
                for bay in UNIT_SPECS[unit]["bays"]:
                    j = self._bays.index(bay)
                    out = torch.tensor([last[bay][e] != "" and content[bay][e] == "" for e in range(self.num_envs)], device=self.device)
                    self._damaged[out & closed, j] = True
            # thermal: a running unit with its shutter drawn heats until it trips; opening the shutter cancels it
            heating = closed & self._cover_closed[:, k]
            self._overtemp_at[heating & torch.isinf(self._overtemp_at[:, k]), k] = t[heating & torch.isinf(self._overtemp_at[:, k])] + self.cfg.retrip_delay_s
            self._overtemp_at[~heating, k] = math.inf
            # the heat sink as telemetry reports it: on its way from the reopened cutout to the cutout while heating,
            # relaxing to ambient otherwise
            remaining = (self._overtemp_at[:, k] - t).clamp(min=0.0) / self.cfg.retrip_delay_s
            tau = torch.where(self._cover_closed[:, k], torch.full_like(t, SINK_TAU_DRAWN_S), torch.full_like(t, SINK_TAU_OPEN_S))
            relaxed = self._sink_c[:, k] + (SINK_AMBIENT_C - self._sink_c[:, k]) * (dt / tau).clamp(max=1.0)
            self._sink_c[:, k] = torch.where(heating, SINK_CUTOUT_C - SINK_HYSTERESIS_K * remaining, relaxed)
            # modules: an empty bay trips at once, an unserviceable module after the self-test with its own fault code
            contents = [content[bay] for bay in UNIT_SPECS[unit]["bays"]]
            empty = torch.tensor([any(c[e] == "" for c in contents) for e in range(self.num_envs)], device=self.device)
            bad = [next((m for m in (c[e] for c in contents) if m and not bool(self._serviceable[m][e])), "") for e in range(self.num_envs)]
            testing = closed & torch.tensor([m != "" for m in bad], device=self.device)
            start = testing & torch.isinf(self._selftest_at[:, k])
            self._selftest_at[start, k] = t[start] + SELFTEST_S
            self._selftest_at[~testing, k] = math.inf
            self._selftest_cause[:, k] = torch.tensor([TRIP_CAUSES.index(MODULE_FAULT[self._ix["objects"][m]["item"]]) if m else 0
                                                       for m in bad], device=self.device)
            if (closed & empty).any():
                self._trip(closed & empty, k, "MODULE_MISSING")
            damaged = self._damaged[:, [self._bays.index(bay) for bay in UNIT_SPECS[unit]["bays"]]].any(dim=1)
            if (~self._tripped[:, k] & damaged).any():
                self._trip(~self._tripped[:, k] & damaged, k, "CONNECTOR_DAMAGED")
            if (t >= self._selftest_at[:, k]).any():
                self._trip(t >= self._selftest_at[:, k], k, self._selftest_cause[:, k])
            if (t >= self._overtemp_at[:, k]).any():
                self._trip(t >= self._overtemp_at[:, k], k, "OVERTEMP")

    def unit_healthy(self, k: int) -> torch.Tensor:
        """(n,) unit k closed with nothing about to trip it -- what 'restored' means for the evaluator."""
        damaged = self._damaged[:, [self._bays.index(bay) for bay in UNIT_SPECS[UNITS[k]]["bays"]]].any(dim=1)
        return ~self._tripped[:, k] & torch.isinf(self._overtemp_at[:, k]) & torch.isinf(self._selftest_at[:, k]) & ~damaged

    def power_telemetry(self) -> Dict[str, object]:
        """The units' own telemetry for env 0, as the base network carries it to any subscriber."""
        return {
            "source": "the serviced units' own telemetry: breaker, panel lamp, heat-sink temperature, trip cause, resets, "
                      "installed module serial and connector state per bay",
            "units": {unit: {"name": UNIT_SPECS[unit]["name"], "load": UNIT_SPECS[unit]["load"],
                            "criticality": UNIT_SPECS[unit]["criticality"],
                            "breaker": "tripped" if bool(self._tripped[0, k]) else "closed",
                            "energized": not bool(self._tripped[0, k]),
                            "lamp": "red" if bool(self._tripped[0, k]) else "green",
                            "heat_sink_temperature_c": round(float(self._sink_c[0, k]), 1),
                            "trip_cause": TRIP_CAUSES[int(self._cause[0, k])] or None,
                            "resets_since_start": int(self._resets[0, k]),
                            "installed_modules": {f"{bay}/tray": self._ix["objects"][c]["serial"] if (c := self.bay_content(bay)[0]) else None
                                                  for bay in UNIT_SPECS[unit]["bays"]},
                            "bay_connectors": {f"{bay}/tray": "damaged" if bool(self._damaged[0, self._bays.index(bay)]) else "ok"
                                               for bay in UNIT_SPECS[unit]["bays"]}}
                      for k, unit in enumerate(UNITS)},
        }

    def site_state(self) -> Dict[str, object]:
        """Evaluator-side truth of the site for env 0 (the mission tasks add their own fields)."""
        return dict(
            units={unit: dict(tripped=bool(self._tripped[0, k]), cause=TRIP_CAUSES[int(self._cause[0, k])] or None,
                             healthy=bool(self.unit_healthy(k)[0]), cover_closed=bool(self._cover_closed[0, k]),
                             bays={bay: self.bay_content(bay)[0] for bay in UNIT_SPECS[unit]["bays"]},
                             resets=int(self._resets[0, k]), reset_presses=int(self._press_count[0, k]),
                             damaged_bays=[bay for bay in UNIT_SPECS[unit]["bays"] if bool(self._damaged[0, self._bays.index(bay)])],
                             heat_sink_temperature_c=round(float(self._sink_c[0, k]), 1),
                             cover_presses=int(self._press_count[0, len(UNITS) + self._covered_units.index(k)]) if k in self._covered_units else 0)
                   for k, unit in enumerate(UNITS)},
            objects={name: self.where_is(name) for name in self._objs},
            parked_and_stowed=bool(self.parked_and_stowed()[0]),
            record_faults=dict(records=_pairs(self.cfg.records), record_swaps=_pairs(self.cfg.record_swaps)),
            latched=[name for j, name in enumerate(self._objs) if bool(self._latched[0, j])],
        )

    ## -- robot and objects ---------------------------------------------------------------------------------------
    def tcp_pos_env(self) -> torch.Tensor:
        p7 = self._robot.data.body_pos_w[:, self._link7]
        q7 = self._robot.data.body_quat_w[:, self._link7]
        return p7 + quat_apply(q7, self._tcp_offset.expand(self.num_envs, 3)) - self.scene.env_origins

    def parked_and_stowed(self) -> torch.Tensor:
        """(n,) the rover on its dock with the arm in the travel pose -- every mission's safe end state."""
        robot_xy = self._robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
        q = self._robot.data.joint_pos[:, self._arm_ids]
        return (torch.linalg.norm(robot_xy, dim=1) < DOCK_TOLERANCE_M) & ((q - self._stow).abs().max(dim=1).values < STOW_TOLERANCE_RAD)

    def handle_pos_env(self, name: str) -> torch.Tensor:
        obj = self._objs[name]
        return obj.data.root_pos_w + quat_apply(obj.data.root_quat_w, self._handle_local[name].expand(self.num_envs, 3)) - self.scene.env_origins

    def on_supports(self, names: Sequence[str], supports: Sequence[str], settled: bool = True) -> torch.Tensor:
        """(n, objects, supports) each object upright on each place frame; with `settled`, also at rest and not held by
        the gripper."""
        objs = [self._objs[name] for name in names]
        pos = torch.stack([o.data.root_pos_w for o in objs], dim=1) - self.scene.env_origins.unsqueeze(1)  # (n, m, 3)
        quat = torch.stack([o.data.root_quat_w for o in objs], dim=1)  # (n, m, 4)
        j = [self._place_ix[s] for s in supports]
        rel = pos.unsqueeze(2) - self._place_xyz[j]  # (n, m, s, 3)
        nrm = self._place_nrm[j]  # (s, 2)
        along = (rel[..., :2] * nrm).sum(dim=-1)
        lateral = rel[..., 0] * nrm[:, 1] - rel[..., 1] * nrm[:, 0]
        up = quat_apply(quat, self._up.expand(*quat.shape[:2], 3))[..., 2]
        on = (along.abs() < ON_SUPPORT_XY_M[0]) & (lateral.abs() < ON_SUPPORT_XY_M[1]) & (rel[..., 2].abs() < ON_SUPPORT_Z_M) & (up > UPRIGHT_COS).unsqueeze(2)
        if settled:
            speed = torch.linalg.norm(torch.stack([o.data.root_lin_vel_w for o in objs], dim=1), dim=-1)
            handle = pos + quat_apply(quat, torch.stack([self._handle_local[name] for name in names]).expand(*quat.shape[:2], 3))
            released = torch.linalg.norm(handle - self.tcp_pos_env().unsqueeze(1), dim=-1) > NOT_HELD_M
            on &= ((speed < RESTING_SPEED_M_S) & released).unsqueeze(2)
        return on

    def on_support(self, name: str, support: str, settled: bool = True) -> torch.Tensor:
        """(n,) object upright on a place frame; with `settled`, also at rest and not held by the gripper."""
        return self.on_supports([name], [support], settled)[:, 0, 0]

    def place_frames(self) -> List[str]:
        return self._place

    def where_is(self, name: str) -> str:
        """Place frame an object rests on for env 0 ('' if none)."""
        on = self.on_supports([name], self._place)[0, 0].tolist()
        return next((s for s, o in zip(self._place, on) if o), "")

    ## -- step ----------------------------------------------------------------------------------------------------
    def mission_success(self) -> torch.Tensor:
        """(n,) the mission task's success condition."""
        raise NotImplementedError

    def extract_step_return(self):
        ret = super().extract_step_return()
        if hasattr(self, "_success"):
            self._update_power()
            self._update_handover()
            self._success = self.mission_success()
            if self.cfg.terminate_on_success:
                ret = ret._replace(termination=ret.termination | self._success)
        return ret
