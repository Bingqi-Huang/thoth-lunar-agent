"""Give the EG2 a contact material independent of the rover's wheel material.

User-authorized simulation assumption: static/dynamic 0.8/0.6, matching the
existing processed-equipment material range, not a measured hardware calibration.
Keep multiply combination: against equipment 0.8/0.6 the effective coefficients
become 0.64/0.36 instead of 0.40/0.30. Applies to all missions and controllers.
No geometry, mass, actuator force, restitution, or wheel material changes.
"""
from pathlib import Path

from pxr import Sdf, Usd, UsdPhysics, UsdShade

path = Path(__file__).resolve().parents[2] / 'srb_assets/robot/mobile_manipulation/lunarbot/lunarbot.usdc'
stage = Usd.Stage.Open(str(path))
material = UsdShade.Material.Define(stage, '/LunarRover/PhysicsMaterials/gripper_contact')
api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
api.CreateStaticFrictionAttr(0.8)
api.CreateDynamicFrictionAttr(0.6)
api.CreateRestitutionAttr(0.0)
material.GetPrim().AddAppliedSchema('PhysxMaterialAPI')
material.GetPrim().CreateAttribute('physxMaterial:frictionCombineMode', Sdf.ValueTypeNames.Token).Set('multiply')
gripper = stage.GetPrimAtPath('/LunarRover/gripper')
UsdShade.MaterialBindingAPI.Apply(gripper).Bind(material, materialPurpose='physics')
stage.GetRootLayer().Save()
print(path)
