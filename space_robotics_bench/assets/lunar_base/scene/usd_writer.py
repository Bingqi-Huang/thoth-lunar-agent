"""Write the lunar base scenery USD (run by scene/build_scene.py inside Blender for its pxr).

    blender -b --factory-startup -P scene/usd_writer.py -- scene/generated/scene_job.json
"""

import json
import math
import os
import sys

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

job = json.load(open(sys.argv[sys.argv.index("--") + 1]))
g = np.load(job["job"])
out = job["out"]
os.makedirs(out, exist_ok=True)


def mesh(stage, path, pts, tris, nrm=None, uv_tile=None, bake_size=None):
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(tris), 3, np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(tris.astype(np.int32).ravel()))
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    lo, hi = pts.min(0), pts.max(0)
    m.CreateExtentAttr([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))])
    if nrm is not None:
        m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(nrm.astype(np.float32)))
        m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    if uv_tile:
        st = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        st.Set(Vt.Vec2fArray.FromNumpy((pts[:, :2] / uv_tile).astype(np.float32)))
    if bake_size:
        sb = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st_bake", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        sb.Set(Vt.Vec2fArray.FromNumpy((pts[:, :2] / bake_size + 0.5).astype(np.float32)))
    return m


def regolith_material(stage, path, tint, roughness_scale=1.0, baked_diffuse=None, baked_normal=None):
    mat = UsdShade.Material.Define(stage, path)
    pbr = UsdShade.Shader.Define(stage, f"{path}/pbr")
    pbr.CreateIdAttr("UsdPreviewSurface")
    readers = {}
    for var in ("st", "st_bake"):
        readers[var] = UsdShade.Shader.Define(stage, f"{path}/{var}")
        readers[var].CreateIdAttr("UsdPrimvarReader_float2")
        readers[var].CreateInput("varname", Sdf.ValueTypeNames.Token).Set(var)

    def texture(name, file, channels, scale=(1, 1, 1, 1), bias=(0, 0, 0, 0), srgb=True, var="st"):
        t = UsdShade.Shader.Define(stage, f"{path}/{name}")
        t.CreateIdAttr("UsdUVTexture")
        t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./textures/{file}")
        t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(readers[var].ConnectableAPI(), "result")
        t.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        t.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB" if srgb else "raw")
        t.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*scale))
        t.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*bias))
        return t

    if baked_diffuse:
        diff = texture("diffuse", baked_diffuse, 3, scale=(*tint, 1), var="st_bake")
        diff.GetInput("wrapS").Set("clamp")
        diff.GetInput("wrapT").Set("clamp")
    else:
        diff = texture("diffuse", "regolith_diff.png", 3, scale=(*tint, 1))
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(diff.ConnectableAPI(), "rgb")
    if baked_normal:
        nor = texture("normal", baked_normal, 3, scale=(2, 2, 2, 1), bias=(-1, -1, -1, 0), srgb=False, var="st_bake")
        nor.GetInput("wrapS").Set("clamp")
        nor.GetInput("wrapT").Set("clamp")
    else:
        nor = texture("normal", "regolith_nor_gl.png", 3, scale=(2, 2, 2, 1), bias=(-1, -1, -1, 0), srgb=False)
    pbr.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(nor.ConnectableAPI(), "rgb")
    rough = texture("roughness", "regolith_rough.png", 1, scale=(roughness_scale,) * 4, srgb=False)
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(rough.ConnectableAPI(), "r")
    pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
    return mat


# ---- geometry layer ------------------------------------------------------------------------------
geo_path = os.path.join(out, f"terrain{job.get('suffix', '')}.usdc")
geo = Usd.Stage.CreateNew(geo_path)
UsdGeom.SetStageUpAxis(geo, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(geo, 1.0)
root = UsdGeom.Xform.Define(geo, "/lunar_base")
geo.SetDefaultPrim(root.GetPrim())

# 1st Mesh child: ray-cast target for the LunarBot lidar, never rendered, no collider.
lidar = mesh(geo, "/lunar_base/lidar", g["lidar_pts"], g["lidar_tris"])
lidar.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
lidar.CreatePurposeAttr(UsdGeom.Tokens.guide)
# Explicitly non-colliding: in SRB the scenery's structure boxes otherwise acted as colliders (a spare module
# rested on the staging stand's box top, 5 cm above the real table).
UsdPhysics.CollisionAPI.Apply(lidar.GetPrim()).CreateCollisionEnabledAttr(False)

terrain = mesh(geo, "/lunar_base/terrain", g["terrain_pts"], g["terrain_tris"], g["terrain_nrm"], job["tile_m"], job["near_size_m"])
UsdPhysics.CollisionAPI.Apply(terrain.GetPrim())
UsdPhysics.MeshCollisionAPI.Apply(terrain.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.none)
midm = mesh(geo, "/lunar_base/mid_terrain", g["mid_pts"], g["mid_tris"], g["mid_nrm"], job["tile_m"], job["mid_size_m"])
farm = mesh(geo, "/lunar_base/far_terrain", g["far_pts"], g["far_tris"], g["far_nrm"], job["tile_m"])

looks = "/lunar_base/Looks"
UsdShade.MaterialBindingAPI.Apply(terrain.GetPrim()).Bind(
    regolith_material(geo, f"{looks}/regolith_near", (1.0, 1.0, 1.0), baked_diffuse="terrain_near_albedo.jpg"))
UsdShade.MaterialBindingAPI.Apply(midm.GetPrim()).Bind(
    regolith_material(geo, f"{looks}/regolith_mid", (1.0, 1.0, 1.0), baked_diffuse="terrain_mid_albedo.jpg",
                      baked_normal="terrain_mid_normal.jpg"))
UsdShade.MaterialBindingAPI.Apply(farm.GetPrim()).Bind(regolith_material(geo, f"{looks}/regolith", (1.0, 1.0, 1.0)))

phys = UsdShade.Material.Define(geo, f"{looks}/regolith_physics")
pm = UsdPhysics.MaterialAPI.Apply(phys.GetPrim())
pm.CreateStaticFrictionAttr(0.9)
pm.CreateDynamicFrictionAttr(0.7)
pm.CreateRestitutionAttr(0.02)
UsdShade.MaterialBindingAPI(terrain.GetPrim()).Bind(phys, UsdShade.Tokens.weakerThanDescendants, "physics")
geo.GetRootLayer().Save()

# ---- composition layer ---------------------------------------------------------------------------
top_path = os.path.join(out, f"lunar_base{job.get('suffix', '')}.usda")
if os.path.exists(top_path):
    os.remove(top_path)
top = Usd.Stage.CreateNew(top_path)
UsdGeom.SetStageUpAxis(top, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(top, 1.0)
base = top.DefinePrim("/lunar_base", "Xform")
base.GetReferences().AddReference(f"./terrain{job.get('suffix', '')}.usdc")
top.SetDefaultPrim(base)


def place(path, ref, pos, yaw_deg, scale=1.0, tilt=(0.0, 0.0)):
    p = top.DefinePrim(path, "Xform")
    p.GetReferences().AddReference(ref)
    x = UsdGeom.Xformable(p)
    x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(*pos))
    x.AddRotateXYZOp().Set(Gf.Vec3f(tilt[0], tilt[1], yaw_deg))
    if scale != 1.0:
        x.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
    return p


top.DefinePrim("/lunar_base/assets", "Scope")
for a in job["placements"]:
    place(f"/lunar_base/assets/{a['name']}", os.path.relpath(a["usd"], out), a["pos"], a["yaw_deg"])
# Interaction frames (C3): an Xform per frame, +X along its outward normal, for tasks/verifiers and the viewer.
top.DefinePrim("/lunar_base/interaction", "Scope")
for key, pt in job["interaction"]["points"].items():
    path = "/lunar_base/interaction/" + key.replace("/", "__")
    f = UsdGeom.Xform.Define(top, path)
    f.AddTranslateOp().Set(Gf.Vec3d(*pt["xyz"]))
    f.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, math.degrees(math.atan2(pt["normal"][1], pt["normal"][0]))))
    prim = f.GetPrim()
    prim.CreateAttribute("srb:interaction:type", Sdf.ValueTypeNames.Token).Set(pt["type"])
    prim.CreateAttribute("srb:interaction:asset", Sdf.ValueTypeNames.String).Set(pt["asset"])
top.DefinePrim("/lunar_base/rocks", "Scope")
for k, r in enumerate(job["rocks"]):
    place(f"/lunar_base/rocks/rock_{k:04d}", f"./rocks/{r['file']}", r["pos"], r["yaw_deg"], r["scale"], r["tilt_deg"]).SetInstanceable(True)
top.GetRootLayer().Save()
print(f"WROTE {top_path} ({len(job['placements'])} assets, {len(job['rocks'])} rocks)", flush=True)
