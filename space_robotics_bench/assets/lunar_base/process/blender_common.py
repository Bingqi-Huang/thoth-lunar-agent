"""Shared Blender helpers for the lunar base asset pipeline."""

import math
from pathlib import Path

import bpy
from mathutils import Vector


def import_source(src_dir: Path):
    files = sorted(src_dir.rglob("*"))
    glbs = [f for f in files if f.suffix.lower() in (".glb", ".gltf")]
    if glbs:
        for f in glbs:
            bpy.ops.import_scene.gltf(filepath=str(f))
        return
    daes = [f for f in files if f.suffix.lower() == ".dae"]
    objs = [f for f in files if f.suffix.lower() == ".obj"]
    if objs:
        for f in objs:
            bpy.ops.wm.obj_import(filepath=str(f))
        return
    if daes:
        for f in daes:
            bpy.ops.wm.collada_import(filepath=str(f))
        return
    usds = [f for f in files if f.suffix.lower() in (".usd", ".usdc", ".usda", ".usdz")]
    bpy.ops.wm.usd_import(filepath=str(usds[0]))


def scene_bounds(objs):
    lo = Vector((math.inf,) * 3)
    hi = Vector((-math.inf,) * 3)
    for o in objs:
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            lo = Vector(map(min, lo, w))
            hi = Vector(map(max, hi, w))
    return lo, hi


def setup_render(samples=32, resolution=(960, 720)):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type == "OPTIX"
    scene.cycles.device = "GPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.film_transparent = False
    world = bpy.data.worlds.new("preview") if not scene.world else scene.world
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.02, 0.022, 0.026, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 1.0


def render_views(objs, stem: Path, views, sun_elevation=12.0):
    """Render one image per (azimuth_deg, elevation_deg) framing all objs; low sun like a lunar pole."""
    scene = bpy.context.scene
    lo, hi = scene_bounds(objs)
    center = (lo + hi) / 2
    radius = max((hi - lo).length / 2, 1e-3)

    sun = bpy.data.objects.get("preview_sun")
    if sun is None:
        sun = bpy.data.objects.new("preview_sun", bpy.data.lights.new("preview_sun", "SUN"))
        scene.collection.objects.link(sun)
    sun.data.energy = 4.0
    sun.data.angle = math.radians(0.53)
    sun.rotation_euler = (math.radians(90 - sun_elevation), 0, math.radians(135))

    fill = bpy.data.objects.get("preview_fill")
    if fill is None:
        fill = bpy.data.objects.new("preview_fill", bpy.data.lights.new("preview_fill", "SUN"))
        scene.collection.objects.link(fill)
    fill.data.energy = 0.6
    fill.rotation_euler = (math.radians(50), 0, math.radians(-60))

    ground = bpy.data.objects.get("preview_ground")
    if ground is None:
        bpy.ops.mesh.primitive_plane_add(size=1)
        ground = bpy.context.active_object
        ground.name = "preview_ground"
        mat = bpy.data.materials.new("preview_regolith")
        mat.use_nodes = True
        mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.32, 0.31, 0.3, 1)
        mat.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.95
        ground.data.materials.append(mat)
    ground.scale = (max(radius, 3.0) * 12, max(radius, 3.0) * 12, 1)
    ground.location = (center.x, center.y, lo.z)

    cam = bpy.data.objects.get("preview_cam")
    if cam is None:
        cam = bpy.data.objects.new("preview_cam", bpy.data.cameras.new("preview_cam"))
        scene.collection.objects.link(cam)
    scene.camera = cam
    cam.data.lens = 50
    cam.data.clip_end = radius * 100
    dist = radius / math.tan(cam.data.angle / 2) * 1.15

    paths = []
    for i, (az, el) in enumerate(views):
        a, e = math.radians(az), math.radians(el)
        cam.location = center + Vector((math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e))) * dist
        direction = center - cam.location
        cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        path = f"{stem}.png" if len(views) == 1 else f"{stem}_{i}.png"
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        paths.append(path)
    return paths


def add_lunarbot_proxy(location: Vector):
    """Scale reference built from measured LunarBot dimensions (chassis, wheels, arm stowed upright)."""
    mat = bpy.data.materials.new("lunarbot_proxy")
    mat.use_nodes = True
    mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.85, 0.55, 0.12, 1)
    parts = []
    bpy.ops.mesh.primitive_cube_add(size=1, location=location + Vector((0, 0, (0.365 + 0.952) / 2)))
    chassis = bpy.context.active_object
    chassis.scale = (1.322, 1.052, 0.952 - 0.365)
    parts.append(chassis)
    for x in (0.4925, -0.5225):
        for y in (0.42705, -0.42705):
            bpy.ops.mesh.primitive_cylinder_add(radius=0.1453, depth=0.12, location=location + Vector((x, y, 0.1453)),
                                                rotation=(math.pi / 2, 0, 0))
            parts.append(bpy.context.active_object)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.05, depth=1.721 - 0.727, location=location + Vector((-0.4475, 0, (0.727 + 1.721) / 2)))
    parts.append(bpy.context.active_object)
    for p in parts:
        p.name = "lunarbot_proxy_" + p.name
        p.data.materials.append(mat)
    return parts
