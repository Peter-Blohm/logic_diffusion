import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.lines import Line2D

# ICLR paper typography/layout assumptions
ICLR_TEXT_WIDTH_IN = 6.75
ICLR_FONT_SIZE_PT = 10
SVG_PX_PER_PT = 96 / 72
ICLR_FONT_SIZE_SVG = ICLR_FONT_SIZE_PT * SVG_PX_PER_PT

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "text.usetex": False,
        "mathtext.fontset": "dejavuserif",
        "font.size": ICLR_FONT_SIZE_SVG,
        "axes.labelsize": ICLR_FONT_SIZE_SVG,
        "axes.titlesize": ICLR_FONT_SIZE_SVG,
        "legend.fontsize": ICLR_FONT_SIZE_SVG,
        "xtick.labelsize": ICLR_FONT_SIZE_SVG,
        "ytick.labelsize": ICLR_FONT_SIZE_SVG,
        "lines.linewidth": 3,
        "figure.figsize": (ICLR_TEXT_WIDTH_IN / 2, 2.1),
        "svg.fonttype": "path",
    }
)

# --- Data Generation ---
x = np.linspace(0, 1, 1000)

# Three striped Gaussians (approximate target figure)
mu1, sig1 = 0.42, 0.14
y1 = 1.0 * norm.pdf(x, mu1, sig1)

mu2, sig2 = 0.50, 0.09
y2 = 1.0 * norm.pdf(x, mu2, sig2)

mu3, sig3 = 0.58, 0.14
y3 = 1.0 * norm.pdf(x, mu3, sig3)

# --- Calculations ---
# PoE: product of all three input Gaussians
y_poe = y1 * y2 * y3
poe_area = np.trapezoid(y_poe, x)
if poe_area > 0:
    y_poe = y_poe / poe_area

# Dombi-like smooth composition via inverse-power sum
lam = 5.0
eps = 1e-16
term_inner = y1 ** (-lam) + y2 ** (-lam) + y3 ** (-lam)
term_inner_clamped = np.maximum(term_inner, eps)
y_dombi = term_inner_clamped ** (-1 / lam)

# --- Plotting ---
base_max = max(np.max(y1), np.max(y2), np.max(y3))
max_val = base_max * 1.12

fig, ax = plt.subplots(1, 1)

# Inputs: three striped Gaussians
ax.plot(x, y1, color="#555555", linestyle=":", alpha=0.8, lw=3)
ax.plot(x, y2, color="#555555", linestyle="--", alpha=0.8, lw=3)
ax.plot(x, y3, color="#555555", linestyle="-.", alpha=0.8, lw=3)

# Outputs
ax.plot(x, y_poe, color="#d62728", lw=5)
ax.fill_between(x, y_poe, color="#d62728", alpha=0.15)
ax.plot(x, y_dombi, color="#9467bd", lw=5)
ax.fill_between(x, y_dombi, color="#9467bd", alpha=0.15)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(2)
ax.spines["bottom"].set_linewidth(2)
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylim(0, max_val)
ax.set_xlim(0, 1)
ax.margins(x=0, y=0)
ax.set_ylabel("Density", fontsize=ICLR_FONT_SIZE_SVG, labelpad=2)

legend_row1 = [
    Line2D([0], [0], color="#555555", linestyle=":", lw=3, label="$p$"),
    Line2D([0], [0], color="#555555", linestyle="--", lw=3, label="$q$"),
    Line2D([0], [0], color="#d62728", lw=5, label=r"$\mathrm{Prob:}\ pqr$"),
]
legend_row2 = [
    Line2D([0], [0], color="#555555", linestyle="-.", lw=3, label="$r$"),
    Line2D([0], [0], color="#9467bd", lw=5, label=r"$\mathrm{Dombi:}\ p\wedge_\lambda q\wedge_\lambda r$"),
]

leg1 = ax.legend(
    handles=legend_row1,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.05),
    ncol=3,
    fontsize=ICLR_FONT_SIZE_SVG,
    columnspacing=0.9,
    handlelength=1.4,
    handletextpad=0.4,
)
ax.add_artist(leg1)

ax.legend(
    handles=legend_row2,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.1),
    ncol=2,
    fontsize=ICLR_FONT_SIZE_SVG,
    columnspacing=0.9,
    handlelength=1.4,
    handletextpad=0.4,
)

plt.subplots_adjust(left=0.11, right=1.0, top=1.0, bottom=0.25)
plt.savefig("neurips_demorgan_stability.svg", format="svg", bbox_inches="tight", pad_inches=0)
plt.show()
