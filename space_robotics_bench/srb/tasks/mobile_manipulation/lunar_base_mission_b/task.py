from typing import Dict, List, Literal

import torch

from srb.assets.scenery.lunar_base import load_interaction
from srb.utils.cfg import configclass

from ..lunar_base.site import SiteTask, SiteTaskCfg
from ..lunar_base.task import EventCfg as _BaseEventCfg
from ..lunar_base.task import SceneCfg as _BaseSceneCfg

##############
### Config ###
##############

## Mission B -- logistics: deliver an item across the base.
##
## Objective given to every method: "Deliver the <cargo> staged at the <source> (slot 0 per the manifest) to <the
## destination>, then return the rover to the dock with the arm stowed."
##   cargo `spare_oru`: the spare ORU from the depot stand to a PDU's spares shelf;
##   cargo `bat_spare`: the spare battery module from the depot spares stand to the BESS receiving stand;
##   cargo `consumables_container`: the consumables container offloaded at the landing pad to the habitat logistics
##   stand (Lewis et al. 2022 p.12: robots offload and bring logistics to the base before crew arrival).
## The power units run normally (`lunar_base/site.py`).
##
## Scenarios (`env.scenario=...`):
## - covered: the destination is free.
## - heldout_destination_occupied (L2): the destination already holds an item the record lists there -- a returned
##   cargo crate on a PDU shelf or the BESS stand, or an outbound return container on the habitat stand -- to be cleared
##   to some other support before the delivery.
## The task never tells a method which physical steps to take; the verifier reads simulator truth only.
##
## Wrong records (inventory records can be wrong; a label scanner is available) add
## three knobs to any scenario: `name_serial` names the cargo by its serial in the brief (the delivered unit must be
## that one); `placement` puts objects on other place frames at reset; `records` sets where the inventory record says an
## object is. Both are "object:place_frame+object:place_frame" (a Hydra string: isaaclab's config update refuses new keys
## in a dict). Neither is shown to a method; the record is what the manifest publishes, the placement is the world.

_CARGO = {"spare_oru": ("spare ORU", "depot stand"), "bat_spare": ("spare battery module", "depot spares stand"),
          "consumables_container": ("consumables container", "landing-pad offload stand")}
## where each destination receives cargo, and how mission control names it
_DESTINATION = {**{u: (f"{u}_shelf/shelf", f"the spares shelf of PDU-{u[-1]}") for u in ("pdu_1", "pdu_2", "pdu_3")},
                "bess": ("bess_stand/slot_0", "the BESS receiving stand"),
                "habitat": ("habitat_stand/slot_0", "the habitat logistics stand")}


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

    scenario: Literal["covered", "heldout_destination_occupied"] = "covered"
    cargo: Literal["spare_oru", "bat_spare", "consumables_container"] = "spare_oru"
    destination: Literal["pdu_1", "pdu_2", "pdu_3", "bess", "habitat"] = "pdu_1"
    name_serial: bool = False
    placement: str = ""
    ## the destination is a handover point: the consignee takes in whatever comes to rest on it (irreversible delivery)
    handover_latch: bool = True

    episode_length_s: float = 1200.0
    is_finite_horizon: bool = True

    def mission_brief(self) -> str:  # method, not property: see lunar_base_mission_a
        item, source = _CARGO[self.cargo]
        if self.name_serial:
            item = f"{item} {load_interaction()['objects'][self.cargo]['serial']}"
        return (f"Deliver the {item} staged at the {source} (slot 0 per the manifest) to {_DESTINATION[self.destination][1]}, "
                "then return the rover to the dock with the arm stowed.")


############
### Task ###
############


def _pairs(spec: str) -> Dict[str, str]:
    """"a:b+c:d" -> {a: b, c: d}"""
    return dict(kv.split(":", 1) for kv in spec.split("+") if kv)


class Task(SiteTask):
    cfg: TaskCfg

    def destination_support(self) -> str:
        return _DESTINATION[self.cfg.destination][0]

    def handover_supports(self) -> List[str]:
        return super().handover_supports() + ([self.destination_support()] if self.cfg.handover_latch else [])

    def occupier(self) -> str:
        """What stands on the destination in heldout_destination_occupied: an outbound return container at the habitat,
        a returned cargo crate elsewhere."""
        return "return_container" if self.cfg.destination == "habitat" else "spare_crate"

    def object_placement(self) -> Dict[str, str]:
        where = super().object_placement()
        if self.cfg.scenario == "heldout_destination_occupied":
            where[self.occupier()] = self.destination_support()
        return {**where, **_pairs(self.cfg.placement)}

    def inventory_record(self) -> List[Dict[str, str]]:
        record = super().inventory_record()
        if self.cfg.scenario == "heldout_destination_occupied":
            r = next(r for r in record if r["serial"] == self._ix["objects"][self.occupier()]["serial"])
            r["recorded_location"] = self.destination_support()
            r["status"] = "outbound return, awaiting stowage" if self.cfg.destination == "habitat" else "returned cargo, awaiting stowage"
        return self.with_record_faults(record)

    def mission_success(self) -> torch.Tensor:
        dest = self.destination_support()
        # the occupier must end up on some other support (a stand slot or another shelf), not dropped or still held
        cleared = self.on_supports([self.occupier()], [s for s in self.place_frames() if s != dest])[:, 0].any(dim=1)
        return self.on_support(self.cfg.cargo, dest) & cleared & self.parked_and_stowed()

    def mission_state(self) -> Dict[str, object]:
        """Verifier truth for env 0 (evaluation / logging; never shown to a method as an observation)."""
        return dict(scenario=self.cfg.scenario, cargo=self.cfg.cargo, destination=self.cfg.destination,
                    delivered=bool(self.on_support(self.cfg.cargo, self.destination_support())[0]),
                    placement=_pairs(self.cfg.placement), records=_pairs(self.cfg.records),
                    **self.site_state(), success=bool(self._success[0]))
