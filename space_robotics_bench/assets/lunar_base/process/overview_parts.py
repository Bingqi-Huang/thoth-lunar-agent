"""Top-down and side orthographic overviews plus each mesh's index position (label with label_parts.py).

    blender -b -P overview_parts.py -- <raw_dir> <out_dir> <category/slug> ...

Writes <out_dir>/<slug>_parts_top.png, <slug>_parts_side.png and <slug>_parts.txt (index -> object name).
"""

import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector
import json

sys.path.insert(0, str(Path(__file__).parent))
from blender_common import import_source, scene_bounds, setup_render  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1 :]
raw, out = Path(argv[0]), Path(argv[1])
out.mkdir(parents=True, exist_ok=True)

for rel in argv[2:]:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    import_source(raw / rel)
    slug = rel.split("/")[-1]
    objs = sorted((o for o in bpy.context.scene.objects if o.type == "MESH"), key=lambda o: o.name)
    setup_render(samples=8, resolution=(1400, 1050))
    scene = bpy.context.scene
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 3
    sun.rotation_euler = (math.radians(35), 0, math.radians(30))
    scene.collection.objects.link(sun)
    lo, hi = scene_bounds(objs)
    center, size = (lo + hi) / 2, hi - lo
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam.data.type = "ORTHO"
    cam.data.clip_end = size.length * 10
    lines = [f"{i:3d} {o.name}" for i, o in enumerate(objs)]
    (out / f"{slug}_parts.txt").write_text("\n".join(lines) + "\n")

    for view, loc, rot, span in (
        ("top", center + Vector((0, 0, size.z + size.length)), (0, 0, 0), max(size.x, size.y * 4 / 3)),
        ("side", center + Vector((0, -(size.y + size.length), 0)), (math.radians(90), 0, 0), max(size.x, size.z * 4 / 3)),
    ):
        cam.location = loc
        cam.rotation_euler = rot
        cam.data.ortho_scale = span * 1.08
        path = out / f"{slug}_parts_{view}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        # image u in [0,1] maps to world horizontal axis, v (top=0) to world vertical axis
        axis_h, axis_v = ("x", "y") if view == "top" else ("x", "z")
        half_w = cam.data.ortho_scale / 2
        half_h = half_w * scene.render.resolution_y / scene.render.resolution_x
        c_h, c_v = getattr(center, axis_h), getattr(center, axis_v)
        (out / f"{slug}_parts_{view}.json").write_text(json.dumps(dict(
            axes=[axis_h, axis_v], h_range=[c_h - half_w, c_h + half_w], v_range=[c_v + half_h, c_v - half_h])))
    print(f"OVERVIEW {rel}: {len(objs)} parts", flush=True)
