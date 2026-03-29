import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.lines import Line2D

# ICLR paper typography/layout assumptions
ICLR_TEXT_WIDTH_IN = 6.75
ICLR_FONT_SIZE_PT = 10
SVG_PX_PER_PT = 96 / 72
ICLR_FONT_SIZE_SVG = ICLR_FONT_SIZE_PT * SVG_PX_PER_PT

# --- Configuration ---
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

# Scenario: Negation Stability
# y1: The Base knowledge (Broad, Low Precision)
mu1, sig1 = 0.5, 0.15
y1 = norm.pdf(x, mu1, sig1)

# y2: The Constraint to "Unlearn" (Narrow, High Precision)
# We place it slightly off-center to see the "push" effect
mu2, sig2 = 0.65, 0.08
y2 = norm.pdf(x, mu2, sig2)

# Gain factor g
g = 1.0

# 1. PoE Negation: y1^(g+1) / y2^g
# Since sig2 < sig1, the precision of y2 is higher.
# Subtracting it leads to negative precision -> Convex curve -> Explosion
y_poe_raw = y1 * (y1 ** (g + 1)) / (y2**g)
y_poe = y_poe_raw  # No normalization possible for exploding curve

# 2. Dombi Negation: Weighted generalized mean with negative weight
# w1 = g+1, w2 = -g. Sum weights = 1.
lam = 5.0
eps = 1e-16

# Formula: ((w1 * y1^-lam + w2 * y2^-lam) / (w1+w2)) ^ (-1/lam)
# With w1+w2 = 1, it simplifies.
# Note: We clamp the inner term to be at least eps to avoid complex numbers
# if the subtraction goes negative (which would define the instability boundary).
term_inner = y1 ** (-lam) + ((g + 1) * y1 ** (-lam)) / (g * y2 ** (-lam))

# Check stability: If term_inner < 0, Dombi is undefined (or complex).
# However, usually y^-lam is huge for small y.
# If y2 is narrow (high peak), y2^-lam is SMALL.
# If y1 is broad (low peak), y1^-lam is LARGE.
# So (Large - Small) is usually positive. Dombi is inherently stable for unlearning sharp constraints!
term_inner_clamped = np.maximum(term_inner, eps)
y_dombi = term_inner_clamped ** (-1 / lam)

# --- Plotting ---
max_val = 5

fig, ax = plt.subplots(1, 1)

# Inputs: Base (y1) and Negation Target (y2)
ax.plot(x, y1, color="#555555", linestyle=":", alpha=0.8, lw=3)
ax.plot(x, y2, color="#555555", linestyle="--", alpha=0.8, lw=3)

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
    Line2D([0], [0], color="#d62728", lw=5, label=r"$\mathrm{Prob:}\ p/q^\gamma$"),
]
legend_row2 = [
    Line2D([0], [0], color="#9467bd", lw=5, label=r"$\mathrm{Dombi:}\ p\wedge_\lambda\neg_{p}q$"),
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
    ncol=1,
    fontsize=ICLR_FONT_SIZE_SVG,
    columnspacing=0.9,
    handlelength=1.4,
    handletextpad=0.4,
)

plt.subplots_adjust(left=0.11, right=1.0, top=1.0, bottom=0.25)
plt.savefig("neurips_negation_stability.svg", format="svg", bbox_inches="tight", pad_inches=0)
plt.show()
