import json
import os
from functools import lru_cache
from typing import TYPE_CHECKING, ClassVar, Dict, Sequence

from srb.core.asset import AssetBaseCfg, RigidObjectCfg, Terrain
from srb.core.domain import Domain
from srb.core.env import ViewerCfg
from srb.core.manager import EventTermCfg, SceneEntityCfg
from srb.core.mdp import reset_root_state_uniform
from srb.core.sim import CollisionPropertiesCfg, UsdFileCfg
from srb.utils.math import rpy_to_quat
from srb.utils.path import SRB_ASSETS_DIR

if TYPE_CHECKING:
    from srb._typing import AnyEnvCfg

## Built by `assets/lunar_base/scene/build_scene.py` from `scene/layout.yaml`
## (LOLA Site01 DEM near the Shackleton connecting ridge, graded core platform,
## processed lunar base assets). Frame: +x east, +y north, rover dock pad at
## the origin with its surface at z = 0.
LUNAR_BASE_USD = SRB_ASSETS_DIR.joinpath("lunar_base", "usd", "scene", "lunar_base", "lunar_base.usda")

## Interaction layer, written next to the scenery by build_scene.py: interaction frames
## (press / indicator / place / pick / grasp), dynamic objects and their spawn poses, LunarBot work and
## view poses. All in the scenery frame (= env frame).
LUNAR_BASE_INTERACTION = LUNAR_BASE_USD.parent.joinpath("interaction.json")
OBJECT_PREFIX = "lb_"


DEFAULT_USD_OVERRIDE = "bat_bess=power/bat_0101_v2,bat_spare=power/bat_0102_v2"


@lru_cache
def load_interaction() -> Dict:
    data = json.loads(LUNAR_BASE_INTERACTION.read_text())
    root = SRB_ASSETS_DIR.parent
    for obj in data["objects"].values():
        # build_scene.py records absolute build paths; re-root them on this checkout
        obj["usd"] = root.joinpath(obj["usd"][obj["usd"].index("assets/lunar_base/") :]).as_posix()
    # SRB_LB_USD_OVERRIDE="<object>=<category>/<output>,...": a revised asset under its own path. The battery modules
    # with readable data plates (power/bat_010x_v2) are the default; an empty value loads
    # the originals
    for item in filter(None, os.environ.get("SRB_LB_USD_OVERRIDE", DEFAULT_USD_OVERRIDE).split(",")):
        name, rel = item.split("=")
        out = rel.rstrip("/").split("/")[-1]
        data["objects"][name]["usd"] = SRB_ASSETS_DIR.joinpath("lunar_base", "usd", rel, f"{out}.usdc").as_posix()
    return data


def set_status_lamp(env_ids: Sequence[int], prop: str, rgb: Sequence[float], intensity: float = 1.5):
    """Recolour the emissive status lamp of a unit's control panel (e.g. "pdu_2_panel", "bess_panel") in the given envs."""
    import omni.usd
    from pxr import Gf

    stage = omni.usd.get_context().get_stage()
    for i in env_ids:
        shader = stage.GetPrimAtPath(f"/World/envs/env_{i}/scenery/assets/{prop}/_materials/StatusLamp/Principled_BSDF")
        if not shader.IsValid():
            raise KeyError(f"no status lamp under {prop} in env {i}")
        shader.GetAttribute("inputs:diffuseColor").Set(Gf.Vec3f(*rgb))
        shader.GetAttribute("inputs:emissiveColor").Set(Gf.Vec3f(*(c * intensity for c in rgb)))


def set_radiator_cover(env_ids: Sequence[int], prop: str, closed: bool):
    """Show a heat-rejection surface's roll-up shutter drawn (a PDU radiator's fins, the BESS condenser), or hide it, e.g.
    "pdu_2_radiator". The drawn shutter is a visual-only part of the prop (no collider): a look, not a mechanism."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    for i in env_ids:
        base = stage.GetPrimAtPath(f"/World/envs/env_{i}/scenery/assets/{prop}")
        for child in base.GetChildren():
            if child.GetName().startswith("cover_closed"):
                (UsdGeom.Imageable(child).MakeVisible if closed else UsdGeom.Imageable(child).MakeInvisible)()


## Fixed polar sun, matching `sun` in layout.yaml: 12 deg elevation, light
## arriving from bearing 200 deg (SSW). A DistantLight shines along its -Z, so
## tilt it by (90 - elevation) about X, then yaw it by (180 - bearing).
SUN_ELEVATION_DEG = 12.0
SUN_BEARING_DEG = 200.0


class LunarBase(Terrain):
    DOMAINS: ClassVar[Sequence[Domain]] = (Domain.MOON,)

    ## The first Mesh under the prim is `/lunar_base/lidar` (terrain + one box
    ## per static structure, invisible, no collider) because the LunarBot
    ## RayCaster only casts against the first Mesh it finds. Physics uses the
    ## `terrain` mesh collider and each structure's own convex hulls.
    asset_cfg: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/lunar_base",
        spawn=UsdFileCfg(
            usd_path=LUNAR_BASE_USD.as_posix(),
            collision_props=CollisionPropertiesCfg(),
        ),
    )

    def setup_extras(self, env_cfg: "AnyEnvCfg"):
        scene = env_cfg.scene
        events = env_cfg.events

        ## Scene: one 160 m site per env
        scene.env_spacing = 0.0 if env_cfg.stack else 200.0

        ## Viewer: from the south-west over the dock toward the habitat row
        env_cfg.viewer = ViewerCfg(  # type: ignore
            eye=(-14.0, -18.0, 9.0), lookat=(4.0, 6.0, 0.0), origin_type="env"
        )

        ## Light: fixed low sun (no orientation randomization)
        if scene.sunlight is not None:
            scene.sunlight.init_state.rot = rpy_to_quat(
                90.0 - SUN_ELEVATION_DEG, 0.0, 180.0 - SUN_BEARING_DEG
            )
        events.randomize_sunlight_orientation = None
        events.randomize_sunlight_angular_diameter = None
        events.randomize_sunlight_color_temperature = None

        ## Dynamic objects of the interaction layer, reset to their authored spawn poses. Tasks move
        ## them per scenario by overriding `events.reset_lb_<name>` params (pose_range is a delta).
        for name, obj in load_interaction()["objects"].items():
            key = OBJECT_PREFIX + name
            setattr(
                scene,
                key,
                RigidObjectCfg(
                    prim_path=f"/World/{key}" if env_cfg.stack else f"{{ENV_REGEX_NS}}/{key}",
                    spawn=UsdFileCfg(usd_path=obj["usd"]),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=tuple(obj["pos"]),
                        rot=rpy_to_quat(0.0, obj["pitch_deg"], obj["yaw_deg"]),
                    ),
                ),
            )
            setattr(
                events,
                f"reset_{key}",
                EventTermCfg(
                    func=reset_root_state_uniform,
                    mode="reset",
                    params={
                        "asset_cfg": SceneEntityCfg(key),
                        "pose_range": {},
                        "velocity_range": {},
                    },
                ),
            )


class LunarBaseRouteBlock(LunarBase):
    """Route-block variant (`build_scene.py --variant route_block`): a parked crew rover across the west leg of
    the depot loop, baked into the lidar mesh so Nav2 sees it. Same interaction layer as `LunarBase`."""

    asset_cfg: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/lunar_base",
        spawn=UsdFileCfg(
            usd_path=LUNAR_BASE_USD.with_name("lunar_base_route_block.usda").as_posix(),
            collision_props=CollisionPropertiesCfg(),
        ),
    )
