import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch

# --- Configuration ---
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Arial", "DejaVu Sans", "sans-serif"],
        "font.size": 11,
        "axes.labelsize": 12,
        "svg.fonttype": "none",
    }
)

# --- Data Generation ---
range_val = 2
x_left = np.linspace(-range_val, -0.1, 300)
x_right = np.linspace(0.1, range_val, 300)
x = np.concatenate([x_left, x_right])
y = np.logspace(-1, 1, 400)
X, Y = np.meshgrid(x, y)

# --- Math Calculations (Compute BOTH) ---
with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
    # 1. Likelihood Data
    A = X * np.log(Y)
    logF_gen = (1.0 / X) * np.logaddexp(A, 0.0)
    Z_like = np.where(np.isfinite(logF_gen), logF_gen, np.nan)

    # Asymptotes Likelihood
    left_asym_like = np.log(np.minimum(y, 1.0))
    right_asym_like = np.log(np.maximum(y, 1.0))

    # Y: density. log(Y): log-likelihood Y.
    # log likelihood C: 1(?)

    # 2. Softmax Data
    logF_softmax = np.exp(np.log(Y) * np.sign(X) - logF_gen * np.sign(X))
    logF_softmax = np.exp(np.log(Y) * X) / (np.exp(np.log(Y) * X) + 1)
    Z_soft = np.where(np.isfinite(logF_softmax), logF_softmax, np.nan)
    # Z_soft = A / (A + 1)
    # Asymptotes Softmax (Step functions)
    left_asym_soft = 1.0 - (y > 1.0).astype(float)
    right_asym_soft = 1.0 - (y < 1.0).astype(float)

# --- Plotting Setup ---
# Height reduced relative to width (9x6 for 2 stacked plots = 3.0 height per plot)
fig = plt.figure(figsize=(9 / 3 * 2, 6 / 3 * 2))

# Grid: 2 Rows, 3 Columns
gs = gridspec.GridSpec(2, 3, width_ratios=[0.08, 0.84, 0.08], wspace=0.05, hspace=0.25)

# Definitions for the two modes to iterate over
modes = [
    {
        "name": "Softmax",
        "Z": Z_soft,
        "L_asym": left_asym_soft,
        "R_asym": right_asym_soft,
        "colors": ["#9fbfd6", "#ffffff", "#d69f9f"],  # Pastel Blue/Red
        "vmin": 0.0,
        "vcenter": 0.5,
        "vmax": 1.0,
        "ticks": [0, 0.5, 1],
        "tick_labels": ["0", "0.5", "1"],
        "ylabel": "Mixing Weight",
        "row_idx": 0,
    },
    {
        "name": "Likelihood",
        "Z": Z_like,
        "L_asym": left_asym_like,
        "R_asym": right_asym_like,
        "colors": ["#2166ac", "#ffffff", "#b2182b"],  # Strong Blue/Red
        "vmin": -2.0,
        "vcenter": 0.0,
        "vmax": 2.0,
        "ticks": [-2, 0, 2],
        "tick_labels": ["Low", "0", "High"],
        "ylabel": "Log Likelihood",
        "row_idx": 1,
    },
]

axes_matrix = []

for mode in modes:
    row = mode["row_idx"]

    # Create Subplots for this row
    axL = fig.add_subplot(gs[row, 0])
    axM = fig.add_subplot(gs[row, 1], sharey=axL)
    axR = fig.add_subplot(gs[row, 2], sharey=axL)
    axes_matrix.append((axL, axM, axR))

    # Colormap & Norm
    cmap = LinearSegmentedColormap.from_list(f"cmap_{row}", mode["colors"], N=256)
    norm = TwoSlopeNorm(vmin=mode["vmin"], vcenter=mode["vcenter"], vmax=mode["vmax"])
    levels = np.linspace(mode["vmin"], mode["vmax"], 51)

    # 1. Left Strip
    xL = np.array([-1.0, 0.0])
    XL, YL = np.meshgrid(xL, y)
    ZL = np.tile(mode["L_asym"][:, None], (1, 2))
    axL.pcolormesh(XL, YL, ZL, cmap=cmap, norm=norm, shading="auto")

    # 2. Middle Heatmap
    cf = axM.contourf(
        X, Y, mode["Z"], levels=levels, cmap=cmap, norm=norm, extend="both"
    )

    # 3. Right Strip
    xR = np.array([0.0, 1.0])
    XR, YR = np.meshgrid(xR, y)
    ZR = np.tile(mode["R_asym"][:, None], (1, 2))
    axR.pcolormesh(XR, YR, ZR, cmap=cmap, norm=norm, shading="auto")

    # --- Styling ---
    for ax in [axL, axM, axR]:
        ax.set_yscale("log")
        ax.set_ylim(1e-1, 1e1)
        ax.tick_params(axis="y", left=False, right=False, labelleft=False)
        ax.tick_params(axis="x", bottom=False, top=False, labelbottom=False)

    # Y-Axis Label (Left)
    axL.set_yticks([1.0])
    axL.set_yticklabels(["$c$"], fontsize=12, fontweight="bold")
    axL.tick_params(axis="y", left=True, labelleft=True, length=4)
    axL.set_ylabel(mode["ylabel"], fontsize=12, labelpad=5)

    # Annotations specific to rows
    if row == 0:  # Top Row (Softmax)
        # Top Labels
        y_top = 10.0
        labels = [(-1.0, "HM"), (0.0, "PoE"), (1.0, "MoE")]
        for xv, txt in labels:
            axM.annotate(
                txt,
                xy=(xv, y_top),
                xycoords="data",
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                clip_on=False,
            )
            axM.axvline(x=xv, linestyle="--", linewidth=1, color="black", alpha=0.3)

        # Limit Labels
        for ax, txt in zip([axL, axR], ["min", "max"]):
            ax.annotate(
                txt,
                xy=(0.5, 1.0),
                xycoords="axes fraction",
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                clip_on=False,
            )

    if row == 1:  # Bottom Row (Likelihood)
        axM.set_xlabel("Mixing Parameter $\lambda$", fontsize=12, labelpad=8)

        # Grid lines for reference
        for xv in [-1.0, 0.0, 1.0]:
            axM.axvline(x=xv, linestyle="--", linewidth=1, color="black", alpha=0.3)

        # Lambda Limits (Bottom)
        axL.text(
            0.5,
            -0.15,
            "$\lambda \\to -\infty$",
            transform=axL.transAxes,
            ha="center",
            va="top",
            fontsize=10,
        )
        axR.text(
            0.5,
            -0.15,
            "$\lambda \\to +\infty$",
            transform=axR.transAxes,
            ha="center",
            va="top",
            fontsize=10,
        )

    # Colorbar per row
    cbar = fig.colorbar(cf, ax=[axL, axM, axR], fraction=0.04, pad=0.02)
    cbar.set_ticks(mode["ticks"])
    cbar.set_ticklabels(mode["tick_labels"])

# --- Global Arrows ---
# Adjust coordinates for the new stacked layout
plt.subplots_adjust(top=0.90, bottom=0.15, left=0.12, right=0.90)

arrow_color = "#555555"
arrow_kw = dict(
    arrowstyle="<->,head_width=4,head_length=4",
    lw=1.5,
    color=arrow_color,
    clip_on=False,
    zorder=10,
)

# 1. Negation (Left side, spanning both plots to indicate symmetry)
# Coordinates are (x, y) in Figure fraction (0,0 is bottom-left, 1,1 is top-right)
arrow_left = FancyArrowPatch(
    posA=(0.04, 0.85),
    posB=(0.04, 0.15),
    connectionstyle="arc3,rad=0.15",
    transform=fig.transFigure,
    **arrow_kw,
)
fig.patches.append(arrow_left)
fig.text(
    0.01,
    0.5,
    "Negation Symmetry",
    ha="left",
    va="center",
    rotation=90,
    fontsize=11,
    fontweight="bold",
    color=arrow_color,
)

# 2. Dual Operator (Bottom)
arrow_bottom = FancyArrowPatch(
    posA=(0.20, 0.06),
    posB=(0.80, 0.06),
    connectionstyle="arc3,rad=0.2",
    transform=fig.transFigure,
    **arrow_kw,
)
fig.patches.append(arrow_bottom)
fig.text(
    0.5,
    0.01,
    "Dual Operator",
    ha="center",
    va="bottom",
    fontsize=11,
    fontweight="bold",
    color=arrow_color,
)

# Save
fname = "neurips_stacked.svg"
plt.savefig(fname, bbox_inches="tight")
print(f"Saved {fname}")
plt.show()
