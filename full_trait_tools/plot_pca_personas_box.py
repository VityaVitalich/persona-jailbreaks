"""Figure 4 (fig:pca-personas): PCA of jailbreak activations drawn as persona axes in a perspective box.

Recovered 2026-09-15 from the 2026-08-26 session scratchpad (fig_pca_personas_final.py);
PC1 of Llama renamed irreverence -> unrestraint. Usage:
    python3 full_trait_tools/plot_pca_personas_box.py <output_dir>
writes fig_pca_personas_{llama,gemma}.pdf/.png into <output_dir>.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Polygon, FancyBboxPatch

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "figure.dpi": 200,
    "pdf.fonttype": 42,
})

PC_COLORS = ["#4477AA", "#228833", "#B5A32E"]
INK = "#333333"
EDGE_C = "#B3BCC7"
LEAD_C = "#AAB4BF"
SPACE_TEXT = "#6B7888"
SPACE_NAME = "jailbreak-persona space"

DATA = {
    "llama": [
        dict(name="unrestraint",
             pos=["edgy", "sycophantic", "goofy"],
             neg=["diplomatic", "calm", "methodical"], var=17.7),
        dict(name="morality",
             pos=["principled", "descriptive", "conciliatory"],
             neg=["evil", "condescending", "misanthropic"], var=13.2),
        dict(name="expressiveness",
             pos=["introspective", "witty", "entertaining"],
             neg=["reserved", "dispassionate", "detached"], var=9.2),
    ],
    "gemma": [
        dict(name="restlessness",
             pos=["impatient", "competitive", "pessimistic"],
             neg=["generalist", "strategic", "methodical"], var=27.0),
        dict(name="opacity",
             pos=["enigmatic", "obsessive", "concise"],
             neg=["accessible", "systems thinker", "meticulous"], var=14.8),
        dict(name="impulsivity",
             pos=["impulsive", "nonchalant", "essentialist"],
             neg=["exploratory", "systems thinker", "cautious"], var=8.6),
    ],
}
import os, sys; FIGDIR = sys.argv[1]
SCRATCH = FIGDIR

DY = 0.215
NAME_FS, TRAIT_FS = 10.0, 8.8

# ---------------------------------------------------------------- camera
D_CAM, FL = 7.5, 6.6
EL, AZ = np.deg2rad(16), np.deg2rad(27)
C = D_CAM * np.array([np.cos(EL) * np.sin(AZ), -np.cos(EL) * np.cos(AZ),
                      np.sin(EL)])
_f = -C / np.linalg.norm(C)
_r = np.cross(_f, np.array([0.0, 0.0, 1.0]))
_r /= np.linalg.norm(_r)
_u = np.cross(_r, _f)


def cam(p):
    q = np.asarray(p, dtype=float) - C
    return np.array([q @ _r, q @ _u]), q @ _f


_center = cam([0, 0, 0])[0] / cam([0, 0, 0])[1] * FL
_corn = [cam([sx, sy, sz]) for sx in (-1, 1) for sy in (-1, 1)
         for sz in (-1, 1)]
_raw = [xy / z * FL - _center for xy, z in _corn]
SCALE = 1.18 / max(abs(p[0]) for p in _raw)


def proj(p):
    xy, z = cam(p)
    return (xy / z * FL - _center) * SCALE, z


EXT = 0.96  # arrowheads meet the faces from inside the box
# depth axis: + pole recedes into the space, - pole toward the viewer
AX3D = [np.array([1, 0, 0]), np.array([0, 0, 1]), np.array([0, 1, 0])]

_sil = [proj((sx, sy, sz))[0] for sx in (-1, 1) for sy in (-1, 1)
        for sz in (-1, 1)]
SIL_TOP = max(p[1] for p in _sil)
SIL_BOT = min(p[1] for p in _sil)
X0 = 1.46


def tint(hexcol, frac=0.09):
    rgb = mpl.colors.to_rgb(hexcol)
    return tuple(1 - frac * (1 - c) for c in rgb)


def draw_scene(ax, pcs):
    corners = [(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1)
               for sz in (-1, 1)]
    for keys, fc in [
            ([(1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)], "#F2F5F9"),
            ([(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)],
             "#F4F7FA"),
            ([(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)], "#F7F9FC")]:
        ax.add_patch(Polygon([proj(k)[0] for k in keys], closed=True,
                             fc=fc, ec="none", zorder=1))
    hidden_corner = (-1, 1, -1)
    for a in corners:
        for b in corners:
            if sum(x != y for x, y in zip(a, b)) == 1 and a < b:
                hidden = hidden_corner in (a, b)
                ax.plot(*zip(proj(a)[0], proj(b)[0]), color=EDGE_C,
                        lw=0.55 if hidden else 0.7,
                        ls=(0, (4, 3)) if hidden else "-",
                        alpha=0.55 if hidden else 1.0, zorder=2)

    vmax = max(pc["var"] for pc in pcs)
    for i, pc in enumerate(pcs):
        d = AX3D[i]
        lw = 1.0 + 1.2 * pc["var"] / vmax
        near = 1 if (C @ d) > 0 else -1
        p_near, _ = proj(near * EXT * d)
        p_far, _ = proj(-near * EXT * d)
        p_mid, _ = proj([0, 0, 0])
        ax.annotate("", xy=p_far, xytext=p_mid, zorder=4,
                    arrowprops=dict(arrowstyle="-|>", color=PC_COLORS[i],
                                    lw=0.85 * lw, alpha=0.62,
                                    mutation_scale=7, shrinkA=0, shrinkB=0))
        ax.annotate("", xy=p_near, xytext=p_mid, zorder=5,
                    arrowprops=dict(arrowstyle="-|>", color=PC_COLORS[i],
                                    lw=lw, mutation_scale=9,
                                    shrinkA=0, shrinkB=0))

    # letterspaced caps lying on the box floor, front area
    p1, _ = proj((-1, -1, -1))
    p2, _ = proj((1, -1, -1))
    pos, _ = proj((0.12, -0.44, -1.0))
    angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
    ax.text(pos[0], pos[1], " ".join(SPACE_NAME.upper()),
            ha="center", va="center", fontsize=7.6, color=SPACE_TEXT,
            rotation=angle, rotation_mode="anchor", zorder=3)


def onaxis(i, pos_side):
    """Name line sits exactly on the axis' outward continuation."""
    d = AX3D[i] * (1 if pos_side else -1)
    fp, _ = proj(1.04 * d)
    fa, _ = proj(1.40 * d)
    v = fa - fp
    v = v / np.linalg.norm(v)
    if i == 1:  # vertical axis
        if pos_side:  # exception: name first, traits below, above the box
            name_y = SIL_TOP + 0.24 + 3 * DY
            return (0.0, name_y, "center", fp,
                    (0.0, name_y - 3 * DY - 0.14), -1)
        name_y = SIL_BOT - 0.24
        return (0.0, name_y, "center", fp, (0.0, name_y + 0.14), -1)
    tx = X0 if v[0] > 0 else -X0
    yc = fp[1] + v[1] / v[0] * (tx - fp[0])
    align = "left" if v[0] > 0 else "right"
    lead_end = (tx - (0.07 if v[0] > 0 else -0.07), yc)
    return (tx, yc, align, fp, lead_end, -1)


def block(ax, x0, name_y, align, pc, i, pos_side, trait_dir=-1):
    words = pc["pos"] if pos_side else pc["neg"]
    sign = "$+$" if pos_side else "$-$"
    col = PC_COLORS[i]
    ax.text(x0, name_y, pc["name"].upper() + "  " + sign, ha=align,
            va="center", fontsize=NAME_FS, color=col,
            fontweight="bold", zorder=8)
    W = 0.058 * max(len(w) for w in words) + 0.14
    ys = [name_y + trait_dir * DY * (k + 1) for k in range(len(words))]
    y_hi, y_lo = max(ys) + 0.105, min(ys) - 0.105
    xl = {"left": x0 - 0.08, "right": x0 - W - 0.08,
          "center": x0 - W / 2 - 0.08}[align]
    ax.add_patch(FancyBboxPatch(
        (xl, y_lo), W + 0.16, y_hi - y_lo,
        boxstyle="round,pad=0,rounding_size=0.06",
        fc=tint(col), ec="none", zorder=6.5))
    for y, w in zip(ys, words):
        ax.text(x0, y, w, ha=align, va="center", fontsize=TRAIT_FS,
                color=INK, fontstyle="italic", zorder=8)


def leader(ax, p_from, p_to):
    ax.plot([p_from[0], p_to[0]], [p_from[1], p_to[1]], color=LEAD_C,
            lw=0.7, alpha=0.9, ls=(0, (1.4, 2.2)), zorder=6)


XLIM = (-3.25, 3.25)
YLIM = (-2.26, 2.14)
W_IN = 5.0
H_IN = W_IN * (YLIM[1] - YLIM[0]) / (XLIM[1] - XLIM[0])

for model, pcs in DATA.items():
    fig, ax = plt.subplots(figsize=(W_IN, H_IN))
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.axis("off")
    draw_scene(ax, pcs)
    for i, pc in enumerate(pcs):
        for pos_side in (True, False):
            x0, name_y, align, pf, pt, tdir = onaxis(i, pos_side)
            block(ax, x0, name_y, align, pc, i, pos_side, trait_dir=tdir)
            leader(ax, np.asarray(pf), np.asarray(pt))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(f"{FIGDIR}/fig_pca_personas_{model}.pdf",
                bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{SCRATCH}/fig_pca_personas_{model}.png",
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
print("done")
