"""Turn one raw source into a scene-ready USD and preview renders (driven by process.py).

    blender -b -P blender_process.py -- <job.json>
"""

import json
import re
import sys
from pathlib import Path

import bmesh
import numpy as np
import bpy
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).parent))
from blender_common import add_lunarbot_proxy, import_source, render_views, scene_bounds, setup_render  # noqa: E402

job = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text())
name = job["name"]
out_dir = Path(job["out_dir"])
out_dir.mkdir(parents=True, exist_ok=True)
usd_path = out_dir / f"{name}.usdc"

bpy.ops.wm.read_factory_settings(use_empty=True)
import_source(Path(job["source"]))


def meshes():
    return [o for o in bpy.context.scene.objects if o.type == "MESH"]


# Bake parent transforms into mesh data so every mesh is in world coordinates.
for o in meshes():
    world = o.matrix_world.copy()
    o.parent = None
    o.data = o.data.copy()
    o.data.transform(world)
    o.matrix_world = Matrix.Identity(4)
for o in [o for o in bpy.context.scene.objects if o.type != "MESH"]:
    bpy.data.objects.remove(o)
bpy.context.view_layer.update()

for pattern in job.get("exclude", []):
    for o in [o for o in meshes() if re.search(pattern, o.name)]:
        bpy.data.objects.remove(o)

if "crop" in job:
    x0, y0, z0, x1, y1, z1 = job["crop"]
    for o in meshes():
        bm = bmesh.new()
        bm.from_mesh(o.data)
        doomed = [
            f
            for f in bm.faces
            if not (x0 <= (c := f.calc_center_median()).x <= x1 and y0 <= c.y <= y1 and z0 <= c.z <= z1)
        ]
        bmesh.ops.delete(bm, geom=doomed, context="FACES")
        bm.to_mesh(o.data)
        bm.free()
for o in [o for o in meshes() if len(o.data.polygons) == 0]:
    bpy.data.objects.remove(o)
bpy.context.view_layer.update()

objs = meshes()
lo, hi = scene_bounds(objs)
size = hi - lo
if "target" in job:
    (dim, metres), = job["target"].items()
    ref = size.z if dim == "height" else max(size.x, size.y, size.z)
    s = metres / ref
else:
    s = job.get("scale", 1.0)
# `scale_xyz` stretches one axis of an authored prop without re-modelling it: a shelf's depth is set by what has
# to fit on it, and the same shelf scaled up all round would also get taller, which moves its surface out of the
# arm's working band. Applied in the imported (Z-up) frame, so x is depth, y width, z height, the order `size_m`
# is reported in. frames.json is authored in the same frame and is *not* scaled, so a stretched prop's frames
# have to be re-authored with it -- the build prints both so the mismatch cannot pass silently.
sxyz = [s * float(v) for v in job.get("scale_xyz", (1.0, 1.0, 1.0))]
# Authored props keep their modelled origin (the mount point their frames.json refers to).
shift = Vector((0, 0, 0)) if job.get("keep_origin") else Vector((-(lo.x + hi.x) / 2, -(lo.y + hi.y) / 2, -lo.z))
xf = Matrix.Diagonal((*sxyz, 1.0)) @ Matrix.Translation(shift)
for o in objs:
    o.data.transform(xf)
    o.data.update()
bpy.context.view_layer.update()

faces = sum(len(o.data.polygons) for o in objs)
if faces > job["max_faces"]:
    # Unwelded triangle soups (split vertices) make collapse-decimation throw spikes; weld first.
    for o in objs:
        bm = bmesh.new()
        bm.from_mesh(o.data)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4 * max(hi - lo) * s)
        bm.to_mesh(o.data)
        bm.free()
    faces = sum(len(o.data.polygons) for o in objs)
    ratio = job["max_faces"] / faces
    for o in objs:
        mod = o.modifiers.new("decimate", "DECIMATE")
        mod.ratio = ratio
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.modifier_apply(modifier=mod.name)
faces = sum(len(o.data.polygons) for o in objs)
bpy.context.view_layer.update()
lo, hi = scene_bounds(objs)
size = hi - lo

# Optional grasp handle (dynamic spares): a 25 mm bail bar centred on the top face, on two posts, far enough
# above the lid that LunarBot's open lower jaw passes under it on a side-on approach. Added after scaling, in
# metres; `size_m` stays the body size and the handle frame is written to the asset json.
handle = None
if "top_handle" in job:
    # Captive lifting collar. A flat flange only gave a friction pinch, and a 1 kg payload worked itself out
    # of the jaws during transfers; a T-handle fixed the lift -- the jaws close horizontally around a
    # thin neck and the head sits on top of them, so a vertical pull is taken by geometry, not by grip force.
    # It did not fix rotation. With 55 mm of free neck the fingers had 27 mm of empty travel below them, so a
    # roll about the approach axis was resisted by friction alone: the load rolled 23.7 deg round the
    # neck during a turn, presented its 30 mm face to jaws closed on 12, and came out. The same axes on
    # the two drives that held moved 4.9 and 4.8 deg.
    #
    # The head is the upper stop. This adds the lower one: a collar of the head's own footprint, leaving a
    # slot the fingers sit in. To roll, one finger must rise out of the slot and the other drop into it, and
    # both are stopped within the slack. Neither number is chosen for this payload -- the slot follows the
    # gripper's finger height and the collar reaches as far across as its fingers touch:
    #     finger height 18 mm (eg2_link5/6 span y +-0.009 in their own meshes) + 3 mm each side
    #     -> roll limited to asin(3/27) = 6.4 deg at the collar's 27 mm half-width.
    th = job["top_handle"]
    neck_t = th.get("neck_thickness_m", 0.012)  # across the jaws (they close on this)
    neck_w = th.get("neck_width_m", 0.030)  # along the approach
    neck_h = th.get("neck_height_m", 0.028)  # the slot: the gripper's fingers, plus clearance each side
    head_w = th.get("head_width_m", 0.090)  # across the jaws: overhangs the closed jaws
    # Along the approach the head and the collar are *shallower than the neck*, so the neck is the nearest
    # surface from any direction and has open air on both sides of it. At 45 mm they stood 7.5 mm proud of a
    # 30 mm neck instead, and a look from 10 deg above the horizontal saw the collar's own top face filling the
    # slot: the candidate search read a continuous 90 mm surface where the neck should have been isolated and
    # returned nothing at the declared 12 mm. Standing the neck proud is
    # what the search asks of any part -- narrow, clear on both sides, in front of what is behind it -- so the
    # level look stops being a precondition. It was a 15 mm-wide condition on where the rover parked, against a
    # docking band of 60 mm. The stop against roll is the collar's reach *across* the jaws, which is unchanged.
    # Thin along the approach for a second reason: the bars are what a look from above sees *through* the slot,
    # and each millimetre of their depth costs tan(look) of the neck's visible length. A thicker bar left 8.7 mm of neck
    # showing from 10 deg above and the candidate search needs `pad_height_m` = 14 mm of it, so it found the
    # neck at exactly 12 mm wide with clear air both sides and threw it away as too short. At 12 mm deep the
    # bars cost 2 mm of a 28 mm slot instead of 4, and the neck stands 9 mm proud of them from any direction.
    head_d = th.get("head_depth_m", 0.012)  # along the approach: well under neck_width_m
    head_t = th.get("head_thickness_m", 0.014)
    collar_t = th.get("collar_thickness_m", head_t)  # the lower stop, under the slot
    # Where the jaws bite, above the item's own base, the same for every item: the ORU's neck (its 0.367 m body plus the
    # 0.10 m it stood proud), so every item on a given support is picked at one height and the column is what
    # differs -- a short body gets a longer column, never a stretched body.
    neck_z = lo.z + th.get("neck_above_base_m", 0.4669)
    cx, cy = (lo.x + hi.x) / 2, (lo.y + hi.y) / 2
    hmat = bpy.data.materials.new("HandleBlack")
    hmat.use_nodes = True
    hb = hmat.node_tree.nodes["Principled BSDF"]
    hb.inputs["Base Color"].default_value = (0.05, 0.05, 0.05, 1.0)
    hb.inputs["Roughness"].default_value = 0.7
    along_x = th.get("axis", "X") == "X"

    def _box(nm, size, loc):
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc)
        o = bpy.context.object
        o.name = nm
        o.scale = size
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        o.data.materials.append(hmat)
        objs.append(o)

    # `axis` is the direction the jaws close along (across the neck); the approach is perpendicular to it
    def sized(across, along):
        return (across, along, None) if along_x else (along, across, None)

    # The column stands on the lid, and the lid is not the bounding box: hi.z is the highest point anywhere on the item,
    # a bracket or a fitting somewhere else, and starting the column there left the whole handle hanging above the
    # surface beneath it. A search over the vertices under the column's footprint did no better on a lid of a few large
    # faces with no vertex under the column at all (it fell back to hi.z: the return container's column stood 14.5 mm
    # clear of its lid). Cast straight down onto the body over the column's footprint and stand it on the lowest hit,
    # so no part of its foot is in the air, sunk into the body so the two are one solid, not two touching boxes.
    a_, b_, _ = sized(neck_t * 2.2, neck_w * 1.4)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    hits = [bpy.context.scene.ray_cast(depsgraph, Vector((cx + u * a_ / 2, cy + v * b_ / 2, hi.z + 1.0)), Vector((0, 0, -1)))[1].z
            for u in np.linspace(-1, 1, 5) for v in np.linspace(-1, 1, 5)]
    seat = min(hits)
    base_z = seat - 0.02
    print(f"HANDLE body under the column {seat:.4f}..{max(hits):.4f} m (bounding top {hi.z:.4f}), base sunk to {base_z:.4f}, "
          f"neck centre {neck_z:.4f}", flush=True)
    pedestal_top = neck_z - neck_h / 2 - collar_t
    pedestal_h = pedestal_top - base_z
    _box("handle_pedestal", (a_, b_, pedestal_h), (cx, cy, (base_z + pedestal_top) / 2))
    a_, b_, _ = sized(head_w, head_d)
    _box("handle_collar", (a_, b_, collar_t), (cx, cy, pedestal_top + collar_t / 2))
    a_, b_, _ = sized(neck_t, neck_w)
    # the neck runs into the head above and the collar below, so the three boxes join as one solid
    _box("handle_neck", (a_, b_, neck_h + head_t), (cx, cy, neck_z))
    a_, b_, _ = sized(head_w, head_d)
    _box("handle_head", (a_, b_, head_t), (cx, cy, neck_z + neck_h / 2 + head_t / 2))
    handle = dict(type="grasp", xyz=[round(cx, 4), round(cy, 4), round(neck_z, 4)], axis=th.get("axis", "X"),
                  neck_thickness_m=neck_t, neck_width_m=neck_w, neck_height_m=neck_h,
                  head_width_m=head_w, head_depth_m=head_d, collar_thickness_m=collar_t,
                  collar_width_m=head_w, collar_depth_m=head_d, standoff_m=round(neck_z - seat, 4))
    bpy.context.view_layer.update()

# Serial label (`serial_label` in processing.yaml): the item's data plate, Data Matrix + S/N (NASA-STD-6002), flat on the
# lid beside the handle, read by `scan_label` from a view that looks down on the lid (70 deg below the horizontal, 0.45 m
# from the column's foot, along the approach). One plate each side of the column, across the jaws: the column and the
# head hide the lid behind them and the head overhangs 45 mm each side, so the plate starts `LABEL_GAP` beyond the head's
# end. The plate is the only flat thing that reads well; on each side it takes the flattest seat within half a plate of
# the column's cross line (the lids are moulded, stepped and domed), nearest the column among equals. A side with no seat
# flatter than `LABEL_FLAT` gets no plate; if neither side has one, both get a plate pressed onto the surface (a
# conformed sticker). The seat is found by casting down onto the body only -- the old corner cast landed on the column
# and floated the consumables container's labels on top of it. Portrait 40 x 56 mm: the symbol with its quiet zone fills
# the far 40 x 40 (2.5 mm modules), the serial under it, upright read from the rover (image up = away from the rover,
# image right = the rover's right; the rover side is the jaw axis turned -90 deg for every carried item, interaction.json
# spawn yaws). No collider (processing.yaml `no_collision`).
if "serial_label_png" in job:
    LABEL_W, LABEL_H = 0.040, 0.056
    LABEL_GAP = 0.010
    LABEL_FLAT = 0.004
    lmat = bpy.data.materials.new("SerialLabel")
    lmat.use_nodes = True
    tex = lmat.node_tree.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(job["serial_label_png"])
    lb = lmat.node_tree.nodes["Principled BSDF"]
    lmat.node_tree.links.new(tex.outputs["Color"], lb.inputs["Base Color"])
    lb.inputs["Roughness"].default_value = 0.6
    jaw = Vector((1, 0, 0)) if along_x else Vector((0, 1, 0))
    far = Vector((-jaw.y, jaw.x, 0))  # away from the rover
    half_across = (size.x if along_x else size.y) / 2
    depsgraph = bpy.context.evaluated_depsgraph_get()

    def _down(p):
        """First hit straight down onto the body; None where the handle or nothing is below."""
        ok, loc, _, _, ob, _ = bpy.context.scene.ray_cast(depsgraph, Vector((p.x, p.y, hi.z + 1.0)), Vector((0, 0, -1)))
        return loc.z if ok and not ob.name.startswith("handle") else None

    def _seat(ca, cf):
        zs = [_down(Vector((cx, cy, 0)) + jaw * (ca + u * LABEL_W / 2) + far * (cf + v * LABEL_H / 2))
              for u in np.linspace(-1, 1, 7) for v in np.linspace(-1, 1, 9)]
        return None if None in zs else (max(zs) - min(zs), max(zs))

    seats = {}
    for sgn in (-1, 1):
        cands = []
        for ca in np.arange(head_w / 2 + LABEL_GAP + LABEL_W / 2, half_across - 0.005 - LABEL_W / 2 + 1e-6, 0.005):
            for cf in np.arange(-LABEL_H / 2, LABEL_H / 2 + 1e-6, 0.005):
                s = _seat(sgn * ca, cf)
                if s:
                    cands.append((round(s[0] / 0.002), abs(cf), ca, sgn * ca, cf, s[0], s[1]))
        if cands:
            seats[sgn] = min(cands)
    if job.get("serial_label_seat") == "front":
        # A lid domed across the jaws (the battery module) is flat only at its far ends: the plates there sat low on
        # the sides, at the edge of the scan view and edge-on to it, and did not decode. One
        # plate instead on the crown in front of the column (the rover's side), where the lid runs flat along the
        # approach and faces the scan camera; the column's footprint (`handle_pedestal`) is 1.4 neck widths deep.
        col_half = job["top_handle"].get("neck_width_m", 0.030) * 1.4 / 2
        cf0 = -(col_half + LABEL_GAP + LABEL_H / 2)
        cands = []
        for ca in np.arange(-0.010, 0.010 + 1e-6, 0.005):
            for cf in np.arange(cf0 - 0.020, cf0 + 1e-6, 0.005):
                s = _seat(ca, cf)
                if s:
                    cands.append((round(s[0] / 0.002), abs(cf - cf0) + abs(ca), abs(ca), ca, cf, s[0], s[1]))
        seats = {1: min(cands)} if cands else seats
    flat = {sgn: s for sgn, s in seats.items() if s[5] <= LABEL_FLAT}
    use, conform = (flat, False) if flat else (seats, True)
    for sgn, (_, _, _, ca, cf, spread, top) in sorted(use.items()):
        centre = Vector((cx, cy, 0)) + jaw * ca + far * cf
        if conform:
            bpy.ops.mesh.primitive_grid_add(x_subdivisions=16, y_subdivisions=22, size=1.0, location=(0, 0, 0))
        else:
            bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, 0))
        o = bpy.context.object
        o.data.transform(Matrix.Diagonal((LABEL_W, LABEL_H, 1.0, 1.0)))
        o.data.transform(Matrix((jaw, far, Vector((0, 0, 1)))).transposed().to_4x4())  # x -> rover's right, y -> away
        o.data.transform(Matrix.Translation((centre.x, centre.y, top + 0.001)))
        if conform:  # pressed onto the surface under each vertex
            for vtx in o.data.vertices:
                z = _down(vtx.co)
                vtx.co.z = (z if z is not None else top) + 0.001
        o.name = f"serial_label_{'a' if sgn < 0 else 'b'}"
        o.data.materials.append(lmat)
        objs.append(o)
        print(f"LABEL {o.name}: across {ca:+.3f} along {cf:+.3f} m from the column, seat spread {spread * 1000:.1f} mm"
              f"{' (conformed)' if conform else ''}", flush=True)
    bpy.context.view_layer.update()

# The USD exporter only follows Principled BSDF graphs; rebuild unlit/emission materials
# (glTF KHR_materials_unlit) as Principled with the same colour texture.
for mat in {m for o in objs for m in o.data.materials if m and m.use_nodes}:
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    out = next(n for n in nodes if n.bl_idname == "ShaderNodeOutputMaterial")
    if out.inputs["Surface"].is_linked and out.inputs["Surface"].links[0].from_node.bl_idname == "ShaderNodeBsdfPrincipled":
        continue
    image = next((n for n in nodes if n.bl_idname == "ShaderNodeTexImage"), None)
    emission = next((n for n in nodes if n.bl_idname == "ShaderNodeEmission"), None)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.8
    if image:
        links.new(image.outputs["Color"], bsdf.inputs["Base Color"])
    elif emission:
        bsdf.inputs["Base Color"].default_value = emission.inputs["Color"].default_value
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

# glTF BLEND materials whose texture alpha never cuts out (e.g. a baked 0.44 minimum) render as
# see-through in RTX; keep alpha only when the texture has real cut-out pixels.
for mat in {m for o in objs for m in o.data.materials if m and m.use_nodes}:
    for bsdf in (n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"):
        alpha = bsdf.inputs["Alpha"]
        if not alpha.is_linked or alpha.links[0].from_node.bl_idname != "ShaderNodeTexImage":
            continue
        img = alpha.links[0].from_node.image
        px = np.empty(len(img.pixels), np.float32)
        img.pixels.foreach_get(px)
        if (px[3::4] < 0.1).mean() < 0.01:
            mat.node_tree.links.remove(alpha.links[0])
            alpha.default_value = 1.0
            print(f"ALPHA dropped on {mat.name} ({img.name})", flush=True)

# Per-material fixes: surfaces coplanar with another material (z-fighting in RTX) are pushed out
# along their normals; PBR scalars are set where the source material is wrong for the look.
# Patterns match Blender material names (USD prefixes names that start with a digit with "_").
for pattern, spec in job.get("material_overrides", {}).items():
    hits = 0
    for o in objs:
        slots = {i for i, m in enumerate(o.data.materials) if m and re.search(pattern, m.name)}
        if not slots:
            continue
        hits += 1
        if spec.get("offset_m"):
            bm = bmesh.new()
            bm.from_mesh(o.data)
            sel = [f for f in bm.faces if f.material_index in slots]
            # A flat sheet (e.g. PV cells) moves rigidly along its area-weighted normal; averaging
            # per-vertex normals would drag cell borders sideways.
            mean = sum((f.normal * f.calc_area() for f in sel), Vector())
            flat = mean.length > 0.9 * sum(f.calc_area() for f in sel)
            push = {}
            for f in sel:
                for v in f.verts:
                    push[v] = push.get(v, Vector()) + f.normal
            for v, n in push.items():
                v.co += (mean if flat else n).normalized() * spec["offset_m"]
            bm.to_mesh(o.data)
            bm.free()
        for i in slots:
            bsdf = next((n for n in o.data.materials[i].node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None)
            if spec.get("tint") and bsdf.inputs["Base Color"].is_linked and not o.data.materials[i].get("tinted"):
                o.data.materials[i]["tinted"] = True  # a material shared by many meshes is tinted once
                # the texture itself multiplied by a colour, saved as a new image: a variant of the same item that keeps
                # its surface detail (USD preview surfaces cannot carry a mix node)
                src = bsdf.inputs["Base Color"].links[0].from_node.image
                px = np.empty(len(src.pixels), np.float32)
                src.pixels.foreach_get(px)
                px = px.reshape(-1, 4)
                px[:, :3] *= np.asarray(spec["tint"], np.float32)
                img = bpy.data.images.new(f"{src.name}_{job['name']}", src.size[0], src.size[1], alpha=True)
                img.pixels.foreach_set(px.ravel())
                img.filepath_raw = str(out_dir / f"{job['name']}_tinted.png")
                img.file_format = "PNG"
                img.save()
                bsdf.inputs["Base Color"].links[0].from_node.image = img
            for key, socket in (("base_color", "Base Color"), ("roughness", "Roughness"), ("metallic", "Metallic")):
                if key in spec and bsdf:
                    for link in list(bsdf.inputs[socket].links):
                        o.data.materials[i].node_tree.links.remove(link)
                    bsdf.inputs[socket].default_value = (*spec[key], 1.0) if key == "base_color" else spec[key]
    print(f"OVERRIDE {pattern} on {hits} meshes: {spec}", flush=True)

bpy.ops.object.select_all(action="DESELECT")
for o in objs:
    o.select_set(True)
bpy.ops.wm.usd_export(
    filepath=str(usd_path),
    selected_objects_only=True,
    export_materials=True,
    generate_preview_surface=True,
    export_textures_mode="NEW",
    overwrite_textures=True,
    relative_paths=True,
    root_prim_path=f"/{name}",
    export_animation=False,
    export_lights=False,
    export_cameras=False,
    convert_world_material=False,
)

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

stage = Usd.Stage.Open(str(usd_path))
root = stage.GetPrimAtPath(f"/{name}")
stage.SetDefaultPrim(root)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
# `no_collision`: parts that are only a look (e.g. a cover shown in one of two states) get no collider
for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Mesh) and not any(re.search(p, str(prim.GetPath())) for p in job.get("no_collision", [])):
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
phys_mat = UsdShade.Material.Define(stage, f"/{name}/PhysicsMaterial")
mat_api = UsdPhysics.MaterialAPI.Apply(phys_mat.GetPrim())
mat_api.CreateStaticFrictionAttr(job["friction"][0])
mat_api.CreateDynamicFrictionAttr(job["friction"][1])
mat_api.CreateRestitutionAttr(job["restitution"])
UsdShade.MaterialBindingAPI.Apply(root).Bind(phys_mat, UsdShade.Tokens.weakerThanDescendants, "physics")
if job["physics"] == "dynamic":
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(job["mass_kg"])
    root.AddAppliedSchema("PhysxRigidBodyAPI")
    root.CreateAttribute("physxRigidBody:maxDepenetrationVelocity", Sdf.ValueTypeNames.Float).Set(job["max_depenetration_velocity"])
stage.GetRootLayer().Save()
textures = sum(1 for p in stage.Traverse() if p.GetTypeName() == "Shader" and p.GetAttribute("info:id").Get() == "UsdUVTexture")

# Previews: the asset beside a LunarBot-sized proxy for scale, low south-pole sun.
asset_meshes = meshes()  # includes an added handle
asset_lo, asset_hi = scene_bounds(asset_meshes)  # before the scale proxy joins the scene
add_lunarbot_proxy(Vector((lo.x - 1.6, lo.y - 1.2, 0)))  # opposite the main (az 30) camera
setup_render(samples=48, resolution=(960, 720))
render_views(asset_meshes, out_dir.parent.parent.parent / "previews" / name, views=[(30, 18), (120, 18), (210, 18), (300, 35)])

(out_dir / f"{name}.json").write_text(
    json.dumps(dict(name=name, faces=faces, size_m=[round(v, 3) for v in size],
                    # Where the prop actually is in its own frame, not just how big it is. A prop that keeps its
                    # modelled origin need not start at z = 0: the service shelf spans 0.16..0.53, and a collision
                    # box built as [0, size_m[2]] put a phantom obstacle under it and left the real slab uncovered
                    # (the same mistake as bounding a stand at its lip instead of its table).
                    z_range_m=[round(float(asset_lo.z), 4), round(float(asset_hi.z), 4)],
                    physics=job["physics"],
                    mass_kg=job.get("mass_kg"), friction=job["friction"], restitution=job["restitution"],
                    **(dict(handle=handle) if handle else {})), indent=1)
)
print(f"PROCESSED {name}: {faces} faces, {textures} texture inputs, size {[round(v, 2) for v in size]} m", flush=True)
