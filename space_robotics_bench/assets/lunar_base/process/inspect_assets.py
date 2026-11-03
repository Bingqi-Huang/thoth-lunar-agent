"""Import raw source assets in Blender and report their structure.

    blender -b -P inspect_assets.py -- <raw_dir> <out_dir> [<category/slug> ...]

Writes <out_dir>/<slug>.json (objects, face counts, world bounds, materials)
and <out_dir>/<slug>.png (one 3/4 view render) for choosing processing settings.
"""

import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).parent))
from blender_common import import_source, render_views, scene_bounds, setup_render  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1 :]
raw, out = Path(argv[0]), Path(argv[1])
out.mkdir(parents=True, exist_ok=True)

for rel in argv[2:]:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    src_dir = raw / rel
    import_source(src_dir)
    objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    lo, hi = scene_bounds(objs)
    report = dict(
        asset=rel,
        size=[round(v, 3) for v in (hi - lo)],
        min=[round(v, 3) for v in lo],
        faces=sum(len(o.data.polygons) for o in objs),
        materials=sorted({m.name for o in objs for m in o.data.materials if m}),
        objects=sorted(
            (
                dict(
                    name=o.name,
                    parent=o.parent.name if o.parent else None,
                    faces=len(o.data.polygons),
                    size=[round(v, 3) for v in (Vector(scene_bounds([o])[1]) - Vector(scene_bounds([o])[0]))],
                    center=[round(v, 3) for v in (scene_bounds([o])[0] + scene_bounds([o])[1]) / 2],
                )
                for o in objs
            ),
            key=lambda d: -d["faces"],
        ),
    )
    slug = rel.split("/")[-1]
    (out / f"{slug}.json").write_text(json.dumps(report, indent=1))
    setup_render(samples=16, resolution=(640, 480))
    render_views(objs, out / slug, views=[(35, 25)])
    print(f"INSPECTED {rel}: {len(objs)} meshes, {report['faces']} faces, size {report['size']}", flush=True)
