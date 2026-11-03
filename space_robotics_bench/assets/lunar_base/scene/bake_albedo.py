#!/usr/bin/env python3
"""Bake the terrain albedo textures and the tiling regolith maps.

    python3 scene/bake_albedo.py      (after build_terrain.py; build_scene.py calls it)

Both bakes sample one world-space regolith function: the 4 m LunarRegolith8k tile, a rotated 13 m
copy blended in through a noise mask (anti-tiling), and a macro tone variation, all on lattices
anchored in world coordinates. So the near bake (160 m, 1.96 cm/px, plus roads, wheel tracks and
sintered pads) and the mid bake (600 m ring, 7.3 cm/px) agree where they meet, and the mid bake
reduces to the plain tile at its border, where the far field's tiled material takes over.
"""

from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "scene/generated"
OUT = ROOT / "usd/scene/lunar_base/textures"
SRC = ROOT / "raw/terrain/omnilrs_regolith_rocks/LunarRegolith8k/LunarRegolith8k_diff.png"

L = yaml.safe_load(open(ROOT / "scene/layout.yaml"))
T, A = L["terrain"], L["albedo"]
MID = T["mid_size_m"] / 2
CHUNK = 1024


def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def resize_srgb(img, size):
    """Downsample colour in linear light, as the GPU's sRGB mipmaps do; averaging the encoded values
    darkens the result, which showed as a brightness step between a bake and the tiled far field."""
    return linear_to_srgb(cv2.resize(srgb_to_linear(img), (size, size), interpolation=cv2.INTER_AREA)).astype(np.float32)


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


class WorldNoise:
    """Bicubic white noise on a lattice anchored in world coordinates over the mid field."""

    def __init__(self, wavelength, seed):
        self.w = wavelength
        self.origin = MID + 4 * wavelength
        n = int(np.ceil(2 * self.origin / wavelength)) + 1
        self.lattice = np.random.default_rng(seed).standard_normal((n, n)).astype(np.float32)

    def __call__(self, x, y):
        return cv2.remap(self.lattice, ((x + self.origin) / self.w).astype(np.float32),
                         ((self.origin - y) / self.w).astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def tiled(tex, x, y, tile_m, rot_deg=0.0):
    """Sample a repeating texture laid in world space (USD st = xy / tile, image row 0 at v = 1)."""
    a = np.radians(rot_deg)
    u = (np.cos(a) * x - np.sin(a) * y) / tile_m
    v = (np.sin(a) * x + np.cos(a) * y) / tile_m
    n = tex.shape[0]
    return cv2.remap(tex, ((u % 1.0) * n).astype(np.float32), ((1.0 - v % 1.0) * n).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


OUT.mkdir(parents=True, exist_ok=True)
src = cv2.imread(str(SRC), cv2.IMREAD_COLOR).astype(np.float32) / 255.0
cv2.imwrite(str(OUT / "regolith_diff.png"), (resize_srgb(src, 2048) * 255 + 0.5).astype(np.uint8))
for name, size in (("nor_gl", 2048), ("rough", 1024)):  # data maps: plain averaging
    data = cv2.imread(str(SRC.with_name(f"LunarRegolith8k_{name}.png")), cv2.IMREAD_UNCHANGED)
    cv2.imwrite(str(OUT / f"regolith_{name}.png"), cv2.resize(data, (size, size), interpolation=cv2.INTER_AREA))

overlay_mask_noise = WorldNoise(18.0, 11)
tone_noise = (WorldNoise(30.0, 21), WorldNoise(7.0, 22))


def regolith(size_m, n_px, border_fade_m=None):
    """The shared regolith albedo over a size_m square centred on the dock, as an (n, n, 3) sRGB image."""
    ppm = n_px / size_m
    tile = resize_srgb(src, max(8, int(round(ppm * A["tile_m"]))))
    big = resize_srgb(src, max(8, int(round(ppm * A["overlay_tile_m"]))))
    img = np.empty((n_px, n_px, 3), np.float32)
    xs = (-size_m / 2 + (np.arange(n_px, dtype=np.float32) + 0.5) / ppm)[None, :]
    for r0 in range(0, n_px, CHUNK):
        ys = (size_m / 2 - (np.arange(r0, min(n_px, r0 + CHUNK), dtype=np.float32) + 0.5) / ppm)[:, None]
        x, y = np.broadcast_to(xs, (len(ys), n_px)), np.broadcast_to(ys, (len(ys), n_px))
        detail = np.ones(x.shape, np.float32)
        if border_fade_m:
            detail = smoothstep((size_m / 2 - np.maximum(np.abs(x), np.abs(y))) / border_fade_m).astype(np.float32)
        m = (np.clip(0.5 + 0.9 * overlay_mask_noise(x, y), 0, 1) * A["overlay_strength"] * detail)[..., None]
        chunk = tiled(tile, x, y, A["tile_m"]) * (1 - m) + tiled(big, x, y, A["overlay_tile_m"], A["overlay_rotation_deg"]) * m
        tone = 1 + (A["macro_tone"][0] * tone_noise[0](x, y) + A["macro_tone"][1] * tone_noise[1](x, y)) * detail
        img[r0 : r0 + len(ys)] = chunk * tone[..., None]
    return img


def save(img, name):
    out = np.clip(img * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(str(OUT / name), out, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(str(GEN / name.replace(".jpg", "_preview.jpg")), cv2.resize(out, (1600, 1600), interpolation=cv2.INTER_AREA))


# ---- mid ring: regolith only, plain tile at its border ------------------------------------------
n_mid = A["mid_px"]
save(regolith(T["mid_size_m"], n_mid, border_fade_m=T["mid_fade_m"]), "terrain_mid_albedo.jpg")
print(f"BAKED terrain_mid_albedo.jpg {n_mid}x{n_mid} ({100 * T['mid_size_m'] / n_mid:.2f} cm/px)")

# ---- near field: regolith + site surfaces ---------------------------------------------------------
size_m = T["near_size_m"]
N = int(round(size_m / A["tile_m"])) * A["tile_px"]
ppm = N / size_m
rng = np.random.default_rng(T["seed"] + 2)
img = regolith(size_m, N)


def to_px(p):
    return (int(round((p[0] + size_m / 2) * ppm * 16)), int(round((size_m / 2 - p[1]) * ppm * 16)))  # 4-bit subpixel


darken = np.ones((N, N), np.float32)  # overlapping surfaces take the darkest factor, not the product


def multiply(mask, factor, blur_m, combine="mul"):
    global darken
    m = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=max(0.5, blur_m * ppm))
    if combine == "min":
        darken = np.minimum(darken, 1 + (factor - 1) * m)
    else:
        img[:] *= (1 + (factor - 1) * m)[..., None]


# Compacted roads + wheel tracks at the LunarBot track width.
roads = np.zeros((N, N), np.uint8)
tracks = np.zeros((N, N), np.uint8)
for pts in L["roads"].values():
    pts = np.asarray(pts, float)
    cv2.polylines(roads, [np.array([to_px(p) for p in pts], np.int32)], False, 1, int(T["road_width_m"] * ppm), cv2.LINE_AA, 4)
    for lane in (-1.1, 1.1):  # one lane each way
        for wheel in (-A["track_half_width_m"], A["track_half_width_m"]):
            off = []
            for k, p in enumerate(pts):
                d = pts[min(k + 1, len(pts) - 1)] - pts[max(k - 1, 0)]
                nrm = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-9)
                off.append(p + nrm * (lane + wheel + rng.normal(0, 0.03)))
            cv2.polylines(tracks, [np.array([to_px(p) for p in off], np.int32)], False, 1, max(1, int(0.16 * ppm)), cv2.LINE_AA, 4)
multiply(roads, A["road_factor"], 0.25, "min")

# Sintered pads and their markings.
pads = np.zeros((N, N), np.uint8)
marks = np.zeros((N, N), np.uint8)
sintered = np.zeros((N, N), np.uint8)
for name, pad in L["pads"].items():
    target = sintered if pad.get("sintered") else pads
    if "radius" in pad:
        ctr = to_px(pad["center"])
        cv2.circle(target, ctr, int(pad["radius"] * ppm * 16), 1, -1, cv2.LINE_AA, 4)
        for r in (pad["radius"] - 2.5, pad["radius"] * 0.45):
            cv2.circle(marks, ctr, int(r * ppm * 16), 1, int(0.45 * ppm), cv2.LINE_AA, 4)
    else:
        (cx, cy), (sx, sy) = pad["center"], pad["size"]
        p0, p1 = to_px((cx - sx / 2, cy + sy / 2)), to_px((cx + sx / 2, cy - sy / 2))
        cv2.rectangle(target, p0, p1, 1, -1, cv2.LINE_AA, 4)
        if name == "dock":
            q0, q1 = to_px((cx - sx / 2 + 0.6, cy + sy / 2 - 0.6)), to_px((cx + sx / 2 - 0.6, cy - sy / 2 + 0.6))
            cv2.rectangle(marks, q0, q1, 1, int(0.2 * ppm), cv2.LINE_AA, 4)
multiply(pads, A["pad_factor"], 0.35, "min")
multiply(sintered, A["sintered_factor"], 0.25, "min")
img *= darken[..., None]
# Tracks show on roads and graded pads, fading in and out along their length.
xs = (-size_m / 2 + (np.arange(N) + 0.5) / ppm)[None, :]
ys = (size_m / 2 - (np.arange(N) + 0.5) / ppm)[:, None]
track_noise = WorldNoise(2.5, 31)(np.broadcast_to(xs, (N, N)), np.broadcast_to(ys, (N, N)))
tracks = (tracks * (1 - sintered)).astype(np.float32) * np.clip(0.6 + 0.8 * track_noise, 0, 1)
multiply(tracks, A["track_factor"], 0.06)
multiply(marks, A["marking_factor"], 0.02)

save(img, "terrain_near_albedo.jpg")
print(f"BAKED terrain_near_albedo.jpg {N}x{N} ({100 / ppm:.2f} cm/px)")
