#!/usr/bin/env python3
"""Couple the EG2's finger joints to eg2_joint1 with PhysX mimic joints (Isaac 5.1).

The real EG2 is one actuator driving a linkage that forces both jaws to move together. In the USD the
six joints are an unconstrained tree, so each has its own position drive: under load one side closed
fully while the other was levered open, and the payload squirted out (research repo F49). Mimic joints
restore the mechanical coupling: q_j = ratio * q_joint1 (PhysX solves q_j + gearing*q_ref + offset = 0).
Ratios follow the jaw signs that keep the pads parallel (measured: distal = +proximal, tilt 0 deg).
"""
from srb.core.app import AppLauncher
from srb.utils.path import SRB_APPS_DIR
launcher = AppLauncher(headless=True, enable_cameras=False, experience=SRB_APPS_DIR.joinpath("srb.headless.kit"))
from pxr import Usd, PhysxSchema

USD = os.path.expandvars("${THOTH_ROOT}/space_robotics_bench/assets/srb_assets/robot/mobile_manipulation/lunarbot/lunarbot.usdc")
RATIO = {"eg2_joint2": 1.0, "eg2_joint3": -1.0, "eg2_joint4": -1.0, "eg2_joint5": 1.0, "eg2_joint6": 1.0}
stage = Usd.Stage.Open(USD)
ref = next(p for p in stage.Traverse() if p.GetName() == "eg2_joint1")
for prim in stage.Traverse():
    r = RATIO.get(prim.GetName())
    if r is None:
        continue
    api = PhysxSchema.PhysxMimicJointAPI.Apply(prim, "rotY")
    api.CreateReferenceJointRel().SetTargets([ref.GetPath()])
    api.CreateReferenceJointAxisAttr().Set("rotY")
    api.CreateGearingAttr().Set(-r)  # q_j + gearing*q_ref + offset = 0
    api.CreateOffsetAttr().Set(0.0)
    print("MIMIC", prim.GetName(), "-> eg2_joint1 ratio", r, flush=True)
stage.GetRootLayer().Save()
print("SAVED", USD, flush=True)
launcher.app.close()
import os

os._exit(0)
