"""Overlay a labeled world-coordinate grid on overview renders: python3 label_parts.py <dir>."""

import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw


def nice_step(span):
    raw = span / 10
    mag = 10 ** math.floor(math.log10(raw))
    return min((m * mag for m in (1, 2, 5, 10)), key=lambda s: abs(s - raw))


for js in sorted(Path(sys.argv[1]).glob("*_parts_*.json")):
    meta = json.loads(js.read_text())
    img = Image.open(js.with_suffix(".png")).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    (h0, h1), (v0, v1) = meta["h_range"], meta["v_range"]
    for lo, hi, horizontal in ((h0, h1, True), (v1, v0, False)):
        step = nice_step(hi - lo)
        k = math.ceil(lo / step)
        while k * step <= hi:
            val = k * step
            if horizontal:
                x = (val - h0) / (h1 - h0) * w
                draw.line((x, 0, x, h), fill=(90, 160, 255), width=1)
                draw.text((x + 2, 2), f"{meta['axes'][0]}={val:g}", fill=(120, 190, 255))
            else:
                y = (val - v0) / (v1 - v0) * h
                draw.line((0, y, w, y), fill=(255, 140, 90), width=1)
                draw.text((2, y + 2), f"{meta['axes'][1]}={val:g}", fill=(255, 170, 120))
            k += 1
    img.save(js.with_name(js.stem + "_grid.png"))
