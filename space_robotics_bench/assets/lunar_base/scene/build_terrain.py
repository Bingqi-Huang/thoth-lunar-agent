#!/usr/bin/env python3
"""Heightfields for the lunar base from the LOLA Site01 DEM + procedural detail + graded pads/roads.

    python3 scene/build_terrain.py

Three nested fields share one world-space surface function (DEM + noise octaves + craters), so they
agree wherever they overlap and differ only in resolution:
  near  160 m @ 0.4 m  collider + lidar; graded platform, pads and roads carved in
  mid   600 m @ 2 m    visual; same noise and craters, fading to the plain DEM at its border
  far   6 km  @ 20 m   visual horizon; plain DEM
Each coarser field is sunk a little where a finer one covers it, reaching zero offset on the finer
field's border.

Writes scene/generated/{terrain_near,terrain_mid,terrain_far}.npy + masks + terrain.json; heights in
metres relative to the dock pad (z = 0 at the dock).
"""

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "scene/generated"
OUT.mkdir(parents=True, exist_ok=True)
L = yaml.safe_load(open(ROOT / "scene/layout.yaml"))
T = L["terrain"]
rng = np.random.default_rng(T["seed"])

dem = cv2.imread(str(ROOT / L["dem"]["file"]), cv2.IMREAD_UNCHANGED).astype(np.float32)
px = L["dem"]["pixel_m"]
r0, c0 = L["dem"]["center_px"]
NEAR, MID, FAR = T["near_size_m"] / 2, T["mid_size_m"] / 2, T["far_size_m"] / 2


def sample_dem(xs, ys):
    """Bicubic DEM lookup; +x = +col, +y = -row, origin at center_px."""
    mx = (c0 + xs / px).astype(np.float32)
    my = (r0 - ys / px).astype(np.float32)
    return cv2.remap(dem, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT).astype(np.float64)


def grid(size, res):
    n = int(round(size / res)) + 1
    v = np.linspace(-size / 2, size / 2, n)
    return np.meshgrid(v, v[::-1])  # row 0 = north edge


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


class Noise:
    """Octaves of bicubic-interpolated white noise on world-anchored lattices covering the mid field."""

    def __init__(self, octaves, seed):
        g = np.random.default_rng(seed)
        self.octaves = []
        for wavelength, amp in octaves:
            n = int(np.ceil(2 * (MID + 4 * wavelength) / wavelength)) + 1
            self.octaves.append((wavelength, amp, g.standard_normal((n, n)).astype(np.float32)))

    def __call__(self, xs, ys):
        out = np.zeros(xs.shape)
        for wavelength, amp, lattice in self.octaves:
            origin = MID + 4 * wavelength
            mx = ((xs + origin) / wavelength).astype(np.float32)
            my = ((origin - ys) / wavelength).astype(np.float32)
            out += amp * cv2.remap(lattice, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        return out


def crater_field():
    """Power-law crater population: small craters over the mid field, the rare large ones over the whole
    far field so the horizon carries the same population (N(>R) ~ R^-1.8 throughout)."""
    cr = T["craters"]
    a, b = cr["radius_m"]
    big_max = cr["large_radius_max_m"]
    density_big = cr["density_per_m2"] * (b / a) ** -1.8  # areal density of craters larger than b
    pops = [(a, b, int(cr["density_per_m2"] * (2 * MID) ** 2), MID), (b, big_max, int(density_big * (2 * FAR) ** 2), FAR)]
    centres, radii = [], []
    for lo, hi, count, extent in pops:
        radii.append(lo * (1 - rng.random(count) * (1 - (lo / hi) ** 1.8)) ** (-1 / 1.8))
        centres.append(rng.uniform(-extent, extent, (count, 2)))
    return np.concatenate(centres), np.concatenate(radii)


def add_craters(h, xs, ys, res, centres, radii, weight=None, keep_out=None):
    """Add bowl + rim profiles in place, touching only each crater's window."""
    cr = T["craters"]
    x0, y1 = xs[0, 0], ys[0, 0]
    rows, cols = xs.shape
    for (cx, cy), R in zip(centres, radii):
        reach = 1.6 * R
        j0, j1 = int(np.floor((cx - reach - x0) / res)), int(np.ceil((cx + reach - x0) / res)) + 1
        i0, i1 = int(np.floor((y1 - cy - reach) / res)), int(np.ceil((y1 - cy + reach) / res)) + 1
        j0, i0, j1, i1 = max(j0, 0), max(i0, 0), min(j1, cols), min(i1, rows)
        if j0 >= j1 or i0 >= i1:
            continue
        if keep_out is not None and keep_out(cx, cy, R):
            continue
        r = np.hypot(xs[i0:i1, j0:j1] - cx, ys[i0:i1, j0:j1] - cy) / R
        d = cr["depth_ratio"] * 2 * R
        prof = np.where(r < 1, d * (r ** 2 - 1), 0.0) + 0.12 * d * np.exp(-((r - 1) / 0.35) ** 2)
        h[i0:i1, j0:j1] += prof if weight is None else prof * weight[i0:i1, j0:j1]


noise = Noise(T["noise"], T["seed"] + 100)
centres, radii = crater_field()

# ---- near field -------------------------------------------------------------------------------
X, Y = grid(T["near_size_m"], T["near_res_m"])
res = T["near_res_m"]
base = sample_dem(X, Y)
h = base + noise(X, Y)


def to_px(x, y):
    return int(round((x + NEAR) / res)), int(round((NEAR - y) / res))


pad_masks = {}
for name, pad in L["pads"].items():
    m = np.zeros(X.shape, np.uint8)
    if "radius" in pad:
        cv2.circle(m, to_px(*pad["center"]), int(pad["radius"] / res), 1, -1)
    else:
        (cx, cy), (sx, sy) = pad["center"], pad["size"]
        cv2.rectangle(m, to_px(cx - sx / 2, cy + sy / 2), to_px(cx + sx / 2, cy - sy / 2), 1, -1)
    pad_masks[name] = m
road_mask = np.zeros(X.shape, np.uint8)
core_roads = np.zeros(X.shape, np.uint8)
for name, pts in L["roads"].items():
    for m in (road_mask,) if name == "ship_access" else (road_mask, core_roads):
        cv2.polylines(m, [np.array([to_px(*p) for p in pts], np.int32)], False, 1,
                      thickness=max(1, int(T["road_width_m"] / res)))
site = np.maximum(road_mask, np.max(np.stack(list(pad_masks.values())), axis=0))
dist_site = cv2.distanceTransform((1 - site).astype(np.uint8), cv2.DIST_L2, 5) * res


def near_site(cx, cy, R):
    """Craters whose rim would reach a pad or road are left out (in every field, so they stay consistent)."""
    i, j = int(round((NEAR - cy) / res)), int(round((cx + NEAR) / res))
    ii, jj = np.clip(i, 0, X.shape[0] - 1), np.clip(j, 0, X.shape[1] - 1)
    outside = np.hypot(max(abs(cx) - NEAR, 0), max(abs(cy) - NEAR, 0))
    return dist_site[ii, jj] + outside < R * 1.4 + T["craters"]["keep_out_m"]


add_craters(h, X, Y, res, centres, radii, keep_out=near_site)

# Base platform: the core site (hull of the pads + perimeter road) is cut-and-filled to one level,
# like a graded construction site; natural terrain resumes through a berm of width platform_blend_m.
core = np.zeros(X.shape, np.uint8)
core_pts = [np.argwhere(m.T > 0) for n, m in pad_masks.items() if n != "ship"]
core_pts.append(np.argwhere(core_roads.T > 0))
cv2.fillConvexPoly(core, cv2.convexHull(np.concatenate(core_pts).astype(np.int32)), 1)
core_z = float(np.median(base[core > 0]))
d_core = cv2.distanceTransform((1 - core).astype(np.uint8), cv2.DIST_L2, 5) * res
graded = core_z + Noise(T["platform_noise"], T["seed"] + 200)(X, Y)
pad_z = {n: core_z for n in pad_masks}

# Landing pad and its access road are cut to the platform level too: a 1.6 m graded ramp measured
# 14.5 deg tilt where it crossed the side slope and LunarBot slid back at 0.6 m/s (F43). The cut
# walls get a wider berm (cut_blend_m) than the in-platform roads.
ship = pad_masks["ship"]
d_ship = cv2.distanceTransform((1 - ship).astype(np.uint8), cv2.DIST_L2, 5) * res
access = np.zeros(X.shape, np.uint8)
cv2.polylines(access, [np.array([to_px(*p) for p in L["roads"]["ship_access"]], np.int32)], False, 1,
              thickness=max(1, int(T["road_width_m"] / res)))
d_access = cv2.distanceTransform((1 - access).astype(np.uint8), cv2.DIST_L2, 5) * res
w_road = smoothstep(1 - cv2.distanceTransform((1 - road_mask).astype(np.uint8), cv2.DIST_L2, 5) * res / T["road_blend_m"])
w_cut = np.maximum(smoothstep(1 - d_access / T["cut_blend_m"]), smoothstep(1 - d_ship / T["cut_blend_m"]))
w = np.maximum(w_road, w_cut)
h = h * (1 - w) + core_z * w

w_core = smoothstep(1 - d_core / T["platform_blend_m"])
h = h * (1 - w_core) + graded * w_core
# Pads themselves are exactly flat (no micro noise) so static assets sit without gaps.
for n, m in pad_masks.items():
    w = smoothstep(1 - cv2.distanceTransform((1 - m).astype(np.uint8), cv2.DIST_L2, 5) * res / 1.0)
    h = h * (1 - w) + pad_z[n] * w
site = np.maximum(site, core)

z0 = pad_z["dock"]
h -= z0
np.save(OUT / "terrain_near.npy", h.astype(np.float32))
np.save(OUT / "sintered_mask.npy", np.maximum(road_mask, np.max(np.stack(list(pad_masks.values())), axis=0)))
np.save(OUT / "platform_mask.npy", (d_core < T["platform_blend_m"]) | (d_ship < T["cut_blend_m"]) | (d_access < T["cut_blend_m"]))


def sink_under(xs, ys, inner_half, depth, ramp):
    """Offset below a finer field: `depth` inside it, rising to 0 exactly on its border over `ramp` metres."""
    edge = inner_half - np.maximum(np.abs(xs), np.abs(ys))
    return depth * np.clip(edge / ramp, 0, 1)


# ---- mid field (visual) -------------------------------------------------------------------------
MX, MY = grid(T["mid_size_m"], T["mid_res_m"])
mid_detail = smoothstep((MID - np.maximum(np.abs(MX), np.abs(MY))) / T["mid_fade_m"])
mid = noise(MX, MY) * mid_detail
small = radii < T["craters"]["radius_m"][1]
add_craters(mid, MX, MY, T["mid_res_m"], centres[small], radii[small], weight=mid_detail, keep_out=near_site)
add_craters(mid, MX, MY, T["mid_res_m"], centres[~small], radii[~small], keep_out=near_site)
mid += sample_dem(MX, MY) - z0
mid -= sink_under(MX, MY, NEAR, 0.15, T["mid_res_m"])
np.save(OUT / "terrain_mid.npy", mid.astype(np.float32))

# ---- far field (visual horizon) ----------------------------------------------------------------
FX, FY = grid(T["far_size_m"], T["far_res_m"])
far = np.zeros(FX.shape)
large = radii >= T["craters"]["radius_m"][1]  # only these are resolvable at 20 m and absent from mid's fade
add_craters(far, FX, FY, T["far_res_m"], centres[large], radii[large], keep_out=near_site)
far += sample_dem(FX, FY) - z0 - sink_under(FX, FY, MID, 0.4, T["far_res_m"])
np.save(OUT / "terrain_far.npy", far.astype(np.float32))

# Seam check: on each finer field's border the coarser field must match it.
edge_n = mid[np.isclose(np.maximum(np.abs(MX), np.abs(MY)), NEAR)]
edge_near = h[np.isclose(np.maximum(np.abs(X), np.abs(Y)), NEAR)]
meta = dict(
    near=dict(size_m=T["near_size_m"], res_m=res, shape=list(h.shape)),
    mid=dict(size_m=T["mid_size_m"], res_m=T["mid_res_m"], shape=list(mid.shape)),
    far=dict(size_m=T["far_size_m"], res_m=T["far_res_m"], shape=list(far.shape)),
    craters=int(len(radii)),
    pad_z_m={k: round(v - z0, 3) for k, v in pad_z.items()},
    dem_center_px=[r0, c0],
    dock_dem_height_m=round(z0, 2),
    near_relief_m=[round(float(h.min()), 2), round(float(h.max()), 2)],
    near_border_z_range_m=[round(float(edge_near.min()), 2), round(float(edge_near.max()), 2)],
    mid_on_near_border_z_range_m=[round(float(edge_n.min()), 2), round(float(edge_n.max()), 2)],
)
(OUT / "terrain.json").write_text(json.dumps(meta, indent=1))
print(json.dumps(meta, indent=1))

# Quick looks: near hillshade with the site outlined; mid hillshade with the near field outlined.
def hillshade(z, r):
    gy, gx = np.gradient(z, r)
    shade = np.clip(0.5 + 0.5 * (-gx * 0.6 + gy * 0.4) / np.sqrt(1 + gx ** 2 + gy ** 2), 0, 1)
    return cv2.cvtColor((shade * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


img = hillshade(h, res)
contours, _ = cv2.findContours(site, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (60, 160, 255), 1)
cv2.imwrite(str(OUT / "terrain_near_hillshade.png"), img)
img = hillshade(mid, T["mid_res_m"])
k = int(round((MID - NEAR) / T["mid_res_m"]))
cv2.rectangle(img, (k, k), (img.shape[1] - 1 - k, img.shape[0] - 1 - k), (60, 160, 255), 1)
cv2.imwrite(str(OUT / "terrain_mid_hillshade.png"), img)


# ---- mid-field normal map ------------------------------------------------------------------------
# The mid mesh (2 m) cannot show the 1.2/4 m noise octaves or small craters that the 0.4 m near field
# does, which read as a square seam from altitude. Bake the full surface normal as a normal map on the
# mid ring's st_bake (u = +x, v = +y).
def bake_mid_normal():
    A = L["albedo"]
    n_px = A["mid_px"]
    ppm = n_px / T["mid_size_m"]
    mres = T["mid_res_m"]
    tile_src = cv2.imread(str(ROOT / "raw/terrain/omnilrs_regolith_rocks/LunarRegolith8k/LunarRegolith8k_nor_gl.png"),
                          cv2.IMREAD_UNCHANGED)[..., ::-1].astype(np.float32) / 65535.0 * 2 - 1  # RGB = xyz
    tile_n = cv2.resize(tile_src, (max(8, int(round(ppm * A["tile_m"]))),) * 2, interpolation=cv2.INTER_AREA)
    tile_n /= np.linalg.norm(tile_n, axis=2, keepdims=True)
    out = np.empty((n_px, n_px, 3), np.uint8)
    xs = -MID + (np.arange(n_px) + 0.5) / ppm
    chunk = 1024
    for r in range(0, n_px, chunk):
        rows = np.arange(max(r - 1, 0), min(r + chunk + 1, n_px))
        ys = MID - (rows + 0.5) / ppm
        x, y = np.meshgrid(xs, ys)
        detail = smoothstep((MID - np.maximum(np.abs(x), np.abs(y))) / T["mid_fade_m"])
        hi = noise(x, y) * detail
        add_craters(hi, x, y, 1 / ppm, centres[small], radii[small], weight=detail, keep_out=near_site)
        add_craters(hi, x, y, 1 / ppm, centres[~small], radii[~small], keep_out=near_site)
        hi += sample_dem(x, y)
        dzdy, dzdx = np.gradient(hi, -1 / ppm, 1 / ppm)
        n_hi = np.dstack([-dzdx, -dzdy, np.ones_like(hi)])
        n_hi /= np.linalg.norm(n_hi, axis=2, keepdims=True)
        # The mid mesh's vertex normals are authored straight up (build_scene.py), so the tangent frame RTX
        # shades in is exactly (+x, +y, +z) and the map carries all relief: no mismatch with RTX's own
        # interpolation of the 2 m triangles. Regolith micro normals (tile prefiltered to this texel size)
        # are UDN-blended on top.
        u = ((x / A["tile_m"]) % 1.0 * tile_n.shape[1]).astype(np.float32)
        v = ((1.0 - (y / A["tile_m"]) % 1.0) * tile_n.shape[0]).astype(np.float32)
        micro = cv2.remap(tile_n, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        nb = np.dstack([n_hi[..., 0] + micro[..., 0], n_hi[..., 1] + micro[..., 1], n_hi[..., 2] * micro[..., 2]])
        nb /= np.linalg.norm(nb, axis=2, keepdims=True)
        keep = slice(r - rows[0], r - rows[0] + min(chunk, n_px - r))
        out[r : r + min(chunk, n_px - r)] = np.clip((nb[keep] * 0.5 + 0.5) * 255 + 0.5, 0, 255).astype(np.uint8)[..., ::-1]
    tex = ROOT / "usd/scene/lunar_base/textures"
    tex.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(tex / "terrain_mid_normal.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(OUT / "terrain_mid_normal_preview.jpg"), cv2.resize(out, (1600, 1600), interpolation=cv2.INTER_AREA))
    print(f"BAKED terrain_mid_normal.jpg {n_px}x{n_px}")


bake_mid_normal()
