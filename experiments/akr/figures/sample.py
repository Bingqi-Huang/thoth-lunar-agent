"""Throwaway demo of the manuscript style: one bar chart, four strategies, synthetic numbers (not results).

    python3 sample.py [outdir]     # default: the session scratch dir; writes style_demo.pdf, .png, _grey.png
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from PIL import Image

import style

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else
           "/tmp/claude-1000/-mnt-SharedData-Research-Space-Robot-Harness--claude-worktrees-"
           "bridge-cse-014QkimHvr6rSwpRiV57XxiU/867c4525-4090-5ab4-b1b0-ac6618c64efa/scratchpad")
OUT.mkdir(parents=True, exist_ok=True)

style.use()
STRATS = ["AKR", "FF-Replan rule", "K-replanner rule", "PLEXIL"]
CASES = ["case 1", "case 2", "case 3", "case 4", "case 5"]
rng = np.random.default_rng(3)
nominal = {"AKR": [.92, .88, .96, .84, .90], "FF-Replan rule": [.80, .72, .84, .60, .68],
           "K-replanner rule": [.70, .58, .76, .52, .60], "PLEXIL": [.46, .40, .52, .30, .36]}
faulty = {k: np.clip(np.array(v) - rng.uniform(.04, .30, 5) * (0 if k == "AKR" else 1) - (.05 if k == "AKR" else 0), 0, 1)
          for k, v in nominal.items()}

fig, axes = plt.subplots(1, 2, figsize=(style.PAGE_W, 2.45), sharey=True)
x = np.arange(len(CASES))
w = 0.2
for ax, data, hatched in ((axes[0], nominal, False), (axes[1], faulty, True)):
    for i, s in enumerate(STRATS):
        kw = style.hatch_kw(s) if hatched else dict(facecolor=style.COLORS[s], linewidth=0)
        bars = ax.bar(x + (i - 1.5) * (w + 0.03), data[s], w, **kw)
    ax.set_xticks(x, CASES)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, .25, .5, .75, 1.0])
    ax.tick_params(axis="x", length=0)
axes[0].set_ylabel("success rate (demo data)")
axes[0].spines["bottom"].set_color(style.GREY_RULE)
style.panel_label(axes[0], "a", "Nominal faults")
style.panel_label(axes[1], "b", "Perception faults")
axes[1].tick_params(axis="y", labelleft=False, length=0)
axes[1].spines["left"].set_visible(False)
# annotation in the 'note' size, with an SemiBold lane-style label
axes[0].annotate("AKR", xy=(0 - 1.5 * (w + .03), .92), xytext=(0, 3), textcoords="offset points", ha="center",
                 fontsize=style.FS["label"], fontweight=style.SEMIBOLD, color=style.COLORS["AKR"])
handles = [Patch(facecolor=style.COLORS[s], linewidth=0, label=s) for s in STRATS] + [style.hatch_handle()]
fig.legend(handles=handles, loc="outside lower center", ncol=5, handlelength=1.36)

pdf = OUT / "style_demo.pdf"
fig.savefig(pdf)
fig.savefig(OUT / "style_demo.png")

# greyscale PNG through CIE luminance (the same L* the style comments quote), not a gamma-space average
im = np.asarray(Image.open(OUT / "style_demo.png").convert("RGB"), float) / 255
lin = np.where(im <= .04045, im / 12.92, ((im + .055) / 1.055) ** 2.4)
y = lin @ np.array([.2126729, .7151522, .0721750])
g = np.where(y <= .0031308, 12.92 * y, 1.055 * y ** (1 / 2.4) - .055)
Image.fromarray((np.clip(g, 0, 1) * 255 + .5).astype(np.uint8)).save(OUT / "style_demo_grey.png")

print("wrote", pdf)
print("L* table:")
names = list(style.COLORS)
for n in names:
    print(f"  {n:15s} {style.COLORS[n]} L*={style.lstar(style.COLORS[n]):5.1f}")
print("pairwise |dL*|:")
for i, a in enumerate(names):
    for b in names[i + 1:]:
        print(f"  {a:15s} {b:15s} {abs(style.lstar(style.COLORS[a]) - style.lstar(style.COLORS[b])):5.1f}")
