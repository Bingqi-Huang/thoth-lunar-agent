#!/usr/bin/env python3
"""Process raw sources into scene-ready USDs per processing.yaml (one Blender run per output).

    python3 process/process.py                 # all outputs
    python3 process/process.py --only cargo_crate radome
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
BLENDER = os.environ.get("BLENDER", "blender")

ap = argparse.ArgumentParser()
ap.add_argument("--only", nargs="*")
args = ap.parse_args()

cfg = yaml.safe_load(open(ROOT / "processing.yaml"))


def serial_label_png(serial: str) -> str:
    """An item's data plate (NASA-STD-6002: Data Matrix plus the serial in plain text), black on white, portrait
    40 x 56 mm at 10 px/mm: the symbol with its one-module quiet zone fills the top 40 x 40 (2.5 mm modules, what the
    scan_label view resolves at ~3 px), the serial under it. Written next to the processing previews; Blender maps it
    onto the item with the image's top away from the rover."""
    import cv2
    import numpy as np
    import zxingcpp

    w, h = 400, 560
    img = np.full((h, w), 255, np.uint8)
    dm = np.array(zxingcpp.create_barcode(serial, zxingcpp.BarcodeFormat.DataMatrix).to_image(scale=1))  # quiet zone incl.
    side = w // dm.shape[0] * dm.shape[0]
    o = (w - side) // 2
    img[o:o + side, o:o + side] = cv2.resize(dm, (side, side), interpolation=cv2.INTER_NEAREST)
    text = f"S/N {serial}"
    scale = 1.0
    while cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale + 0.05, 2)[0][0] < w - 30:
        scale += 0.05
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 2)
    cv2.putText(img, text, ((w - tw) // 2, w + (h - w + th) // 2), cv2.FONT_HERSHEY_DUPLEX, scale, 0, 2, cv2.LINE_AA)
    out = ROOT / "previews" / "labels" / f"{serial}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    return str(out)
(ROOT / "previews").mkdir(exist_ok=True)
for name, spec in cfg["outputs"].items():
    if args.only and name not in args.only:
        continue
    # `like`: an instance of another output (same source and processing), e.g. one ORU per serial number
    spec = {**cfg["outputs"][spec["like"]], **spec} if "like" in spec else spec
    job = dict(cfg["defaults"], **spec)
    if "serial_label" in spec:
        job["serial_label_png"] = serial_label_png(spec["serial_label"])
    job.update(
        name=name,
        source=str(ROOT / spec.get("source_root", "raw") / spec["source"]),
        out_dir=str(ROOT / "usd" / spec["category"] / name),
    )
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(job, f)
    run = subprocess.run([BLENDER, "-b", "-P", str(ROOT / "process/blender_process.py"), "--", f.name],
                         capture_output=True, text=True)
    lines = [l for l in run.stdout.splitlines() + run.stderr.splitlines() if l.startswith(("PROCESSED", "ALPHA", "OVERRIDE", "HANDLE")) or "Error" in l or "Traceback" in l]
    print("\n".join(lines) if lines else f"{name}: no output (exit {run.returncode})", flush=True)
