from typing import Dict, List, Literal, Set

import torch

from srb.utils.cfg import configclass

from ..lunar_base.site import MODULE_FAULT, UNIT_NAMES, UNITS, SiteTask, SiteTaskCfg
from ..lunar_base.task import EventCfg as _BaseEventCfg
from ..lunar_base.task import SceneCfg as _BaseSceneCfg

##############
### Config ###
##############

## Mission A -- infrastructure inspection and maintenance of the base's serviced units.
##
## Objective given to every method: "<reported> reported a fault. Inspect the base's serviced units, identify the
## failed unit, restore it if within your authority, verify the result, and return the rover to the dock with the arm
## stowed." The site physics is `lunar_base/site.py`; a scenario only sets its initial conditions. The faulted unit is
## a parameter (`faulted_unit`): equipment is a blocking factor crossed with the level.
##
## Scenarios (`env.scenario=...`), by what a frozen baseline model lacks:
## - covered: the reported unit tripped OVERCURRENT; a reset restores it.
## - covered_misreport (L1): the report names a healthy unit; the tripped one is another (ISS 2015-01-14 erroneous
##   MDM indication). Telemetry shows which unit tripped.
## - heldout_radiator_cover (L2): the faulted unit's heat-rejection shutter (a PDU radiator, the BESS HVAC condenser)
##   was left drawn; a reset re-trips OVERTEMP after `retrip_delay_s` unless the shutter is opened (its button).
##   Units with a shutter only.
## - heldout_module_failure (L3): the faulted unit's module (`FAULT_MODULE`: a PDU's ORU, the BESS battery module, the
##   ISRU filter cartridge) fails its self-test; its staged spare is serviceable: take the failed module out of the
##   bay, install the spare, reset.
## - heldout_cannibalize (L4-near; `faulted_unit=pdu_1`): as L3 on PDU-1 (life support), but the depot spare ORU is recorded failed
##   (returned for analysis). Serviceable ORUs remain only in energized units; the non-critical ones (PDU-3 science
##   instruments, nearby; the ISRU plant's power cradle, across the base) are candidates, and taking one de-energizes it.
## - heldout_cannibalize_far (L4-far): as L4-near, but PDU-3 is already down -- its module was removed earlier and
##   sits failed on its own shelf -- so the only candidate is the ISRU plant's ORU.
## - heldout_compound_fault (supplementary; units with a shutter and a module: the BESS): L3 and L2-A at once -- the
##   faulted unit's module fails its self-test and its heat-rejection shutter is drawn. The self-test trips 3 s after a
##   reset, before the 30 s OVERTEMP, so the trip cause names the module until a serviceable one is installed; the
##   next reset then re-trips OVERTEMP unless the shutter is opened. Same site physics, both initial conditions.
## The task never tells a method which physical steps to take; the verifier reads simulator truth only.


@configclass
class SceneCfg(_BaseSceneCfg):
    pass


@configclass
class EventCfg(_BaseEventCfg):
    pass


@configclass
class TaskCfg(SiteTaskCfg):
    scene: SceneCfg = SceneCfg()
    events: EventCfg = EventCfg()

    scenario: Literal["covered", "covered_misreport", "heldout_radiator_cover", "heldout_module_failure",
                      "heldout_cannibalize", "heldout_cannibalize_far", "heldout_compound_fault"] = "covered"
    faulted_unit: Literal["pdu_1", "pdu_2", "pdu_3", "bess", "isru"] = "pdu_2"
    ## unit named in the fault report; "auto" = the faulted one, except covered_misreport (the next unit)
    reported_unit: Literal["pdu_1", "pdu_2", "pdu_3", "bess", "isru", "auto"] = "auto"

    episode_length_s: float = 900.0
    is_finite_horizon: bool = True

    def reported(self) -> str:
        """The unit the report names. Resolved when asked, not in __post_init__: command-line overrides such as
        env.faulted_unit are applied after the config is constructed."""
        if self.reported_unit != "auto":
            return self.reported_unit
        return UNITS[(UNITS.index(self.faulted_unit) + 1) % len(UNITS)] if self.scenario == "covered_misreport" else self.faulted_unit

    # a plain method, not a property: configclass copies every attribute on init and a read-only
    # property made the `_visual` variant fail to construct ("has no setter")
    def mission_brief(self) -> str:
        return (
            f"{UNIT_NAMES[self.reported()]} reported a fault. "
            "Inspect the base's serviced units, identify the failed unit, restore it if within your authority, "
            "verify the result, and return the rover to the dock with the arm stowed."
        )


############
### Task ###
############

## each unit's module that fails in L3 and its staged serviceable spare
FAULT_MODULE = {"pdu_1": ("oru_pdu_1", "spare_oru"), "pdu_2": ("oru_pdu_2", "spare_oru"), "pdu_3": ("oru_pdu_3", "spare_oru"),
                "bess": ("bat_bess", "bat_spare"), "isru": ("flt_isru", "flt_spare")}
_REMOVED = "failed -- removed from service for failure analysis, awaiting return shipment"


class Task(SiteTask):
    cfg: TaskCfg

    def initial_trips(self) -> Dict[str, str]:
        module = self._ix["objects"][FAULT_MODULE[self.cfg.faulted_unit][0]]["item"]
        cause = {"heldout_radiator_cover": "OVERTEMP", "heldout_module_failure": MODULE_FAULT[module],
                 "heldout_cannibalize": MODULE_FAULT[module], "heldout_cannibalize_far": MODULE_FAULT[module],
                 "heldout_compound_fault": MODULE_FAULT[module]}
        trips = {self.cfg.faulted_unit: cause.get(self.cfg.scenario, "OVERCURRENT")}
        if self.cfg.scenario == "heldout_cannibalize_far":
            trips["pdu_3"] = "MODULE_MISSING"
        return trips

    def initial_closed_covers(self) -> Set[str]:
        return {self.cfg.faulted_unit} if self.cfg.scenario in ("heldout_radiator_cover", "heldout_compound_fault") else set()

    def unserviceable_modules(self) -> Set[str]:
        failed, spare = FAULT_MODULE[self.cfg.faulted_unit]
        return super().unserviceable_modules() | {"heldout_module_failure": {failed}, "heldout_compound_fault": {failed},
                                                  "heldout_cannibalize": {failed, spare},
                                                  "heldout_cannibalize_far": {failed, spare, "oru_pdu_3"}}.get(self.cfg.scenario, set())

    def object_placement(self) -> Dict[str, str]:
        where = super().object_placement()
        if self.cfg.scenario == "heldout_cannibalize_far":
            where["oru_pdu_3"] = "pdu_3_shelf/shelf"
        return where

    def inventory_record(self) -> List[Dict[str, str]]:
        record = super().inventory_record()
        failed = {"heldout_cannibalize": {"spare_oru": None},
                  "heldout_cannibalize_far": {"spare_oru": None, "oru_pdu_3": "pdu_3_shelf/shelf"}}.get(self.cfg.scenario, {})
        for name, location in failed.items():
            r = next(r for r in record if r["serial"] == self._ix["objects"][name]["serial"])
            r["status"] = _REMOVED
            if location:
                r["recorded_location"] = location
        return self.with_record_faults(record)

    def mission_success(self) -> torch.Tensor:
        return self.unit_healthy(UNITS.index(self.cfg.faulted_unit)) & self.parked_and_stowed()

    def mission_state(self) -> Dict[str, object]:
        """Verifier truth for env 0 (evaluation / logging; never shown to a method as an observation)."""
        return dict(scenario=self.cfg.scenario, faulted_unit=self.cfg.faulted_unit, reported_unit=self.cfg.reported(),
                    **self.site_state(), success=bool(self._success[0]))
