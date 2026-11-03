from typing import Dict, Sequence

import torch

from srb.core.asset import AssetBaseCfg
from srb.core.env import GroundManipulationEnvVisualExtCfg, VisualExt
from srb.core.sim import SphereLightCfg
from srb.utils.cfg import configclass

from .task import Task, TaskCfg


@configclass
class VisualTaskCfg(GroundManipulationEnvVisualExtCfg, TaskCfg):
    ## The rover's RGB-D cameras (RealSense D435i) at 15 Hz, the rate the physical cameras run at: rendering and reading
    ## back three cameras at the 50 Hz agent rate was most of each step (research repo F53, F28).
    camera_update_period: float = 1.0 / 15.0

    def __post_init__(self):
        TaskCfg.__post_init__(self)
        GroundManipulationEnvVisualExtCfg.wrap(self, env_cfg=self)
        ## no overview camera: nothing subscribes to it
        self.cameras_cfg.pop("cam_scene")
        self.scene.cam_scene = None
        ## render only when the cameras sample (physics at env_rate; a multiple of the decimation)
        self.sim.render_interval = round(self.camera_update_period / self.env_rate)
        self.sim.render.enable_reflections = False
        self.sim.render.enable_translucency = False
        ## work lights beside the two cameras, as lunar surface rovers carry for shadowed and night operations: without
        ## them a black handle against the black sky was invisible to the wrist camera (research repo F53). Sphere lights
        ## behind the lens planes (out of view); intensity puts about half of the sunlight's level at 0.3 m.
        for key, body, pos in (("wrist_light", "Link7", (0.0, -0.07, -0.03)), ("front_light", "rgbd_camera_frame", (-0.03, 0.0, 0.05))):
            setattr(self.scene, key, AssetBaseCfg(
                prim_path=f"{self._robot.asset_cfg.prim_path}/{body}/{key}",
                spawn=SphereLightCfg(intensity=1.5e5, radius=0.02, color=(1.0, 0.97, 0.92)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
            ))


class VisualTask(VisualExt, Task):
    cfg: VisualTaskCfg

    def __init__(self, cfg: VisualTaskCfg, **kwargs):
        Task.__init__(self, cfg, **kwargs)
        VisualExt.__init__(self, cfg, **kwargs)

    def _reset_idx(self, env_ids: Sequence[int]):
        Task._reset_idx(self, env_ids)
        VisualExt._reset_idx(self, env_ids)

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        return {
            **Task._get_observations(self),
            **VisualExt._get_observations(self),
        }
