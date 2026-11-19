"""Manuscript figure style for the AKR paper, read off two finished figures.

Sources (the development host): <local figure sources> and
.../pdf/fig_authority_timeline.pdf, with their generators in .../figures_v3/src/ (fig*.html, common.js).
Both are drawn in CSS px at 657 px = 173.8 mm (Elsevier cas-dc text width; one column = 316 px = 83.7 mm).
1 px = 0.75 pt, so the type sizes are 8.8 / 9.2 / 9.4 / 10.4 px = 6.6 / 6.9 / 7.05 / 7.8 pt at print size.
Draw a matplotlib figure at figsize=(PAGE_W, h) or (COLUMN_W, h) and it prints at exactly these sizes.

Usage:
    import style
    style.use()
    fig, axes = plt.subplots(1, 2, figsize=(style.PAGE_W, 2.4))
    style.panel_label(axes[0], "a", "Success rate")
"""
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager as fm
from matplotlib.patches import Patch
from matplotlib.textpath import text_to_path

HERE = Path(__file__).resolve().parent
MPLSTYLE = HERE / "style.mplstyle"

# ---- geometry (inches) -------------------------------------------------------------------------------------
PX = 0.75                          # pt per source px
PAGE_W = 173.8 / 25.4              # 6.843 in, cas-dc double-column text width (source: common.js W_TEXT = 657 px)
COLUMN_W = 83.7 / 25.4             # 3.295 in, one cas-dc column (common.js W_COL = 316 px)
PAGE_W_NOMINAL = 190 / 25.4        # 7.480 in, nominal figure given in the brief; the source figures use PAGE_W above
COLUMN_W_NOMINAL = 90 / 25.4       # 3.543 in

# ---- greys (common.js C.*) -------------------------------------------------------------------------------------
GREY_INK = "#1D2329"               # all primary text
GREY_SUB = "#4A535C"               # tick labels, legend text, secondary text
GREY_MUTED = "#7B848D"             # axis title, tick marks, dashed event line
GREY_RULE = "#C7CDD3"              # axis line, thin rules
GREY_HAIR = "#E3E7EB"              # grid and lane hairlines
GREY_PANEL = "#F6F7F8"             # pill / panel background
GREY = {"ink": GREY_INK, "sub": GREY_SUB, "muted": GREY_MUTED, "rule": GREY_RULE, "hair": GREY_HAIR,
        "panel": GREY_PANEL}

# ---- other source colours with a fixed meaning (not strategies) ----------------------------------------------
GROUND = "#6A4C93"                 # mission control, human authority, HOLD
OK = "#2F8551"                     # verified evidence
FAULT = "#C2463B"                  # fault indicator only

# ---- strategy colours -------------------------------------------------------------------------------------------
# AKR (called Thoth in the source figures), PLEXIL and EUROPA->PLEXIL are the exact hexes of common.js
# (C.thoth, C.plexil, C.europa). FF-Replan rule and K-replanner rule are new: same blue hue as AKR (Lab hue ~255),
# one much lighter, one much darker, so that they sit outside the L* band 23..68 that AKR, PLEXIL and EUROPA
# occupy together. CIE L* (D65) of each colour:
#   AKR            #2E5E80  L* 38.1
#   FF-Replan rule     #9FC0D9  L* 76.1   (light blue, C* 17)
#   K-replanner rule #1B3344  L* 20.1   (deep blue-slate, C* 14)
#   PLEXIL         #B06F3C  L* 52.9
#   EUROPA->PLEXIL #7A736B  L* 48.8
# L* differences among the four strategies AKR / FF-Replan rule / K-replanner rule / PLEXIL: 18.0 (AKR-know.),
# 38.0 (AKR-opt.), 14.8 (AKR-PLEXIL, inherited from the fixed source hexes), 56.0 (opt.-know.), 23.2 (opt.-PLEXIL),
# 32.8 (know.-PLEXIL). Both new tones are >= 18 from AKR, >= 23 from PLEXIL and EUROPA. AKR-PLEXIL (14.8) and
# PLEXIL-EUROPA (4.0) come from the source hexes, which were separated by hue and hatch, not by lightness.
COLORS = {
    "AKR": "#2E5E80",
    "FF-Replan rule": "#9FC0D9",
    "K-replanner rule": "#1B3344",
    "PLEXIL": "#B06F3C",
    "EUROPA→PLEXIL": "#7A736B",
}
# light tints used by the sources for fills behind the strategy colour (common.js C.*L)
COLORS_LIGHT = {"AKR": "#DCE7EF", "PLEXIL": "#F2E3D6", "EUROPA→PLEXIL": "#E7E3DE"}

# ---- type sizes in pt (source px * 0.75) and line widths in pt ------------------------------------------------
FS = {
    "note": 8.8 * PX,              # 6.6: annotations, ticks, legend (timeline FS 8.8; fig4 notes 8.8)
    "label": 9.2 * PX,             # 6.9: SemiBold callout labels in fig4
    "lane": 9.4 * PX,              # 7.05: SemiBold lane names / bold sub-heads
    "head": 10.4 * PX,             # 7.8: panel heads in fig4
}
LW = {
    "axis": 0.7 * PX,              # axis line and ticks
    "hair": 0.8 * PX,              # grid, lane hairlines, outlines, leader lines
    "line": 1.2 * PX,              # data line (timeline baseline 1.2, arrows 1.1)
    "bold": 1.3 * PX,              # pole arrow in fig4
}
SEMIBOLD = 600                     # source weight 600/650 renders as Inter SemiBold (pdffonts: Inter-SemiBold)

# ---- hatch for "perception faults (assumption violated)" ---------------------------------------------------------
# Source hatch (common.js hatch()): lines at 45 deg, perpendicular gap 3.4 px = 2.55 pt, stroke 0.9 px = 0.675 pt.
# '///' gives 18 lines per 72 pt, i.e. 2.8 pt perpendicular gap, the nearest matplotlib density; hatch.linewidth in the
# mplstyle is the 0.675 pt stroke.
HATCH = "///"
HATCH_LABEL = "perception faults (assumption violated)"

_FONT_DIRS = [
    Path.home() / ".local/share/fonts/inter/extras/ttf",
    Path("/usr/share/fonts/truetype/inter"),
    Path("/usr/share/fonts/opentype/inter"),
]
_FONT_FILES = ("Inter-Regular.ttf", "Inter-Medium.ttf", "Inter-SemiBold.ttf")


def _hex_rgb(h):
    return tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))


def lstar(hex_colour):
    """CIE L* (D65) of a #RRGGBB colour."""
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in _hex_rgb(hex_colour)]
    y = 0.2126729 * lin[0] + 0.7151522 * lin[1] + 0.0721750 * lin[2]
    return 116 * y ** (1 / 3) - 16 if y > 216 / 24389 else 24389 / 27 * y


def _mix(hex_a, hex_b, t):
    a, b = _hex_rgb(hex_a), _hex_rgb(hex_b)
    return "#%02X%02X%02X" % tuple(round((x + (y - x) * t) * 255) for x, y in zip(a, b))


def use():
    """Register Inter and apply style.mplstyle. Returns True if Inter was found."""
    found = False
    for d in _FONT_DIRS:
        for f in _FONT_FILES:
            if (d / f).exists():
                fm.fontManager.addfont(str(d / f))
                found = True
    mpl.style.use(str(MPLSTYLE))
    if not found:
        print("style.use(): Inter not found, falling back to Liberation Sans / Arial / DejaVu Sans")
    return found


def panel_label(ax, letter, title=None, dx=0.0):
    """Source panel head: bold '(a)' (Inter SemiBold, 10.4 px = 7.8 pt, ink) followed by one space and the title in
    Inter Regular, baseline 7 px (5.25 pt) above the panel top, flush with the panel's left edge (fig4_platform).
    dx shifts the label right (pt), e.g. to clear y tick labels."""
    size = FS["head"]
    lab = f"({letter})"
    t = ax.annotate(lab, xy=(0, 1), xycoords="axes fraction", xytext=(dx, 5.25), textcoords="offset points",
                    ha="left", va="baseline", fontsize=size, fontweight=SEMIBOLD, color=GREY_INK,
                    annotation_clip=False)
    if title:
        prop = fm.FontProperties(weight=SEMIBOLD, size=size)
        w_pt = text_to_path.get_text_width_height_descent(lab, prop, False)[0]
        space = 0.28 * size            # one Inter space (the SVG collapses the double space to one)
        ax.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(dx + w_pt + space, 5.25),
                    textcoords="offset points", ha="left", va="baseline", fontsize=size, color=GREY_INK,
                    annotation_clip=False)
    return t


def hatch_kw(strategy):
    """bar() kwargs for a strategy under perception faults: solid strategy colour with a hatch drawn over it.
    White hatch on dark colours (L* < 60), darkened hatch on the light one, so that the strategy's lightness is kept."""
    c = COLORS[strategy]
    h = "#FFFFFF" if lstar(c) < 60 else _mix(c, GREY_INK, 0.55)
    return dict(facecolor=c, edgecolor=h, hatch=HATCH, linewidth=0)


def hatch_handle(label=HATCH_LABEL):
    """Legend handle for the hatch (grey hatch on white, outlined in the rule grey)."""
    return Patch(facecolor="#FFFFFF", edgecolor=GREY_MUTED, hatch=HATCH, linewidth=LW["hair"], label=label)
