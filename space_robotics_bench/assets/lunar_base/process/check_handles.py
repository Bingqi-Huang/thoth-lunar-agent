#!/usr/bin/env python3
"""Offline check of every pickable item's lifting handle, from the processed USDs and the scene's interaction.json.

    uv run --no-project --with usd-core --with numpy python process/check_handles.py

Per item: the handle's parts have the ORU's shape (boxes relative to the neck centre, in the handle's own frame, within
0.5 mm; the column only down to where it stands), the neck centre sits at the ORU's height above the item's base, the
column's foot is inside the body everywhere under it and no deeper than the 20 mm it is sunk below the lowest surface
there (straight-down rays onto the body), and the jaws close across the support's normal.
"""

import json
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom

ROOT = Path(__file__).resolve().parent.parent
PARTS = ("handle_pedestal", "handle_collar", "handle_neck", "handle_head")
TOL = 0.0005
SINK = 0.02

ix = json.loads((ROOT / "usd/scene/lunar_base/interaction.json").read_text())


def meshes(usd):
    """{part name: (points, triangles)} in the item's frame; body meshes under their own prim names."""
    stage = Usd.Stage.Open(str(usd))
    xc = UsdGeom.XformCache()
    out = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        m = UsdGeom.Mesh(prim)
        M = np.array(xc.GetLocalToWorldTransform(prim))
        pts = np.array(m.GetPointsAttr().Get(), float) @ M[:3, :3] + M[3, :3]
        counts, idx = m.GetFaceVertexCountsAttr().Get(), np.array(m.GetFaceVertexIndicesAttr().Get())
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        tris = np.array([[idx[s], idx[s + k], idx[s + k + 1]] for s, c in zip(starts, counts) for k in range(1, c - 1)])
        name = next((p for p in PARTS if f"/{p}/" in str(prim.GetPath())), str(prim.GetPath()))
        out[name] = (pts, tris)
    return out


def surface_z(body, x, y):
    """Height of the first body surface a ray straight down through (x, y) meets."""
    best = -np.inf
    for pts, tris in body:
        a, b, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
        e1, e2, d = b - a, c - a, np.array([x, y]) - a[:, :2]
        den = e1[:, 0] * e2[:, 1] - e2[:, 0] * e1[:, 1]
        den = np.where(den == 0.0, np.nan, den)
        u = (d[:, 0] * e2[:, 1] - e2[:, 0] * d[:, 1]) / den
        v = (e1[:, 0] * d[:, 1] - d[:, 0] * e1[:, 1]) / den
        hit = (u >= 0) & (v >= 0) & (u + v <= 1)
        if hit.any():
            best = max(best, float((a[:, 2] + u * e1[:, 2] + v * e2[:, 2])[hit].max()))
    return best


def handle_frame(lo, hi, centre, axis):
    """A part's box relative to the neck centre, as (across the jaws, along the approach, up)."""
    lo, hi = lo - centre, hi - centre
    return (lo, hi) if axis == "X" else (lo[[1, 0, 2]], hi[[1, 0, 2]])


ref, failures, rows = None, [], []
for name, o in sorted(ix["objects"].items(), key=lambda kv: "oru" not in kv[1]["usd"]):  # an ORU first: the reference
    h = o["handle"]
    parts = meshes(ROOT / "usd" / Path(*Path(o["usd"]).parts[-3:]))
    body = [v for k, v in parts.items() if k not in PARTS and "serial_label" not in k]
    base = min(p.min(0)[2] for p, _ in body)
    box = {p: (parts[p][0].min(0), parts[p][0].max(0)) for p in PARTS}
    centre = (box["handle_neck"][0] + box["handle_neck"][1]) / 2
    shape = {p: handle_frame(*box[p], centre, h["axis"]) for p in PARTS}
    foot = box["handle_pedestal"][0][2]
    shape["handle_pedestal"] = (np.r_[shape["handle_pedestal"][0][:2], 0.0], shape["handle_pedestal"][1])  # its top only
    ref = ref or (name, shape, centre[2] - base)
    bad = [p for p in PARTS if max(np.abs(shape[p][0] - ref[1][p][0]).max(), np.abs(shape[p][1] - ref[1][p][1]).max()) > TOL]
    if bad:
        failures.append(f"{name}: {bad} differ from {ref[0]}'s")
    neck = centre[2] - base
    if abs(neck - ref[2]) > TOL or abs(h["xyz"][2] - neck) > TOL:
        failures.append(f"{name}: neck {neck:.4f} above base (json {h['xyz'][2]}), {ref[0]}'s {ref[2]:.4f}")
    (x0, y0, _), (x1, y1, _) = box["handle_pedestal"]
    under = [surface_z(body, x, y) for x in np.linspace(x0, x1, 5) for y in np.linspace(y0, y1, 5)]
    if foot > min(under):
        failures.append(f"{name}: column foot {foot:.4f} above the body under it ({min(under):.4f})")
    if min(under) - foot > SINK + TOL:
        failures.append(f"{name}: column foot {min(under) - foot:.4f} below the lowest surface under it (sunk {SINK})")
    yaw = np.radians(o["yaw_deg"])
    jaws = np.array([np.cos(yaw), np.sin(yaw)]) if h["axis"] == "X" else np.array([-np.sin(yaw), np.cos(yaw)])
    normal = np.array(ix["points"][o["support"]]["normal"][:2])
    if abs(jaws @ normal) > 1e-6:
        failures.append(f"{name}: jaws close along {jaws.round(3)}, not across the support normal {normal.round(3)}")
    rows.append(f"{name:22s} {o['support']:26s} neck {neck:.4f} above base, world {o['pos'][2] + neck:.4f} m; "
                f"body under the column {min(under):.4f}..{max(under):.4f}, foot {foot:.4f}")

print("\n".join(rows))
print("\n".join(failures) if failures else f"OK: {len(rows)} items, handles as {ref[0]}'s, neck {ref[2]:.4f} m above base")
