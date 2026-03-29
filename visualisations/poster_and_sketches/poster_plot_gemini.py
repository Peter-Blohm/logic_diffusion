import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.lines import Line2D

# --- Configuration for Publication Quality ---
# plt.rcParams.update(
#     {
#         "font.family": "sans-serif",
#         "font.sans-serif": ["Inter", "Arial", "DejaVu Sans", "sans-serif"],
#         "font.size": 18,
#         "axes.labelsize": 22,
#         "axes.titlesize": 24,
#         "legend.fontsize": 20,
#         "lines.linewidth": 3,
#         "figure.figsize": (14, 6),
#         # CRITICAL for SVGs: Keeps text as text (editable) rather than paths
#         "svg.fonttype": "none",
#     }
# )


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Arial", "DejaVu Sans", "sans-serif"],
        "font.size": 24,
        "axes.labelsize": 32,
        "axes.titlesize": 24,
        "legend.fontsize": 24,
        "lines.linewidth": 3,
        "figure.figsize": (14 / 2, 6 / 2),
        "svg.fonttype": "none",
    }
)


# --- Data Generation ---
x = np.linspace(0, 1, 1000)

# Distributions
mu1, sig1 = 0.9, 0.16  # Prior
mu2, sig2 = 0.36, 0.07  # Obs A
mu3, sig3 = 0.54, 1.07  # Obs B

y1 = norm.pdf(x, mu1, sig1) + norm.pdf(x, mu1 - 0.8, sig1)
y2 = norm.pdf(x, mu2, sig2) * 0 + 10
y3 = norm.pdf(x, mu3, sig3) * 0 + 10
# y3 = norm.pdf(x, mu3, sig3) * 100

# Product (PoE)
y_product = (y1 * y2 * y3) ** (1 / 3)


def score(x):
    # return x
    return np.diff(np.log(x), prepend=0.0) * 500


# Tempered Harmonic Mean (Dombi)
lam = 5.0
eps = 1e-10
y_inputs = [y1, y2, y3]
y_sum_pow = np.zeros_like(x)
for y in y_inputs:
    y_clamped = np.maximum(y, eps)
    y_sum_pow += y_clamped ** (-lam)
y_tempered = y_sum_pow ** (-1 / lam)

# Y-limit
max_reasonable_y = max(np.max(y1), np.max(y2), np.max(y3), np.max(y_tempered))
y_limit_view = max_reasonable_y * 1.2

# --- Plotting ---
fig, axes = plt.subplots(1, 2, sharey=True)


def plot_inputs(ax):
    # Unified "Priors" style
    ax.plot(x, score(y1), color="#555555", linestyle=":", alpha=0.8, lw=3)
    ax.plot(x, score(y2), color="#555555", linestyle="--", alpha=0.8, lw=3)
    ax.plot(x, score(y3), color="#555555", linestyle="--", alpha=0.8, lw=3)

    # Clean spines and ticks
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(-10, y_limit_view)


# Left: PoE
plot_inputs(axes[0])
axes[0].plot(x, score(y_product), color="#d62728", lw=5)
axes[0].fill_between(x, y_product, color="#d62728", alpha=0.2)
axes[0].set_ylabel("Density", fontsize=24, labelpad=10)

legend_elements_1 = [
    Line2D([0], [0], color="#555555", linestyle="--", lw=3, label="Priors"),
    Line2D([0], [0], color="#d62728", lw=5, label="PoE"),
]
axes[0].legend(handles=legend_elements_1, frameon=False, loc="upper right")

# Right: Dombi
plot_inputs(axes[1])
axes[1].plot(x, score(y_tempered), color="#9467bd", lw=5)
axes[1].fill_between(x, y_tempered, color="#9467bd", alpha=0.2)

legend_elements_2 = [
    Line2D([0], [0], color="#555555", linestyle="--", lw=3, label="Priors"),
    Line2D([0], [0], color="#9467bd", lw=5, label="Dombi"),
]
axes[1].legend(handles=legend_elements_2, frameon=False, loc="upper right")

plt.tight_layout()

# Save as SVG
plt.savefig("neurips_qualitative_final.svg", format="svg", bbox_inches="tight")
print("Figure saved as neurips_qualitative_final.svg")
