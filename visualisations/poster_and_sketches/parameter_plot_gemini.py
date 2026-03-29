import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch

# --- Configuration ---
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Arial", "DejaVu Sans", "sans-serif"],
        "font.size": 12,
        "axes.labelsize": 14,
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

# Softmax Calculation
with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
    A = X * np.log(Y)
    logF_gen = (1.0 / X) * np.logaddexp(A, 0.0)
    logF = np.exp(np.log(Y) * np.sign(X) - logF_gen * np.sign(X))

logF = np.where(np.isfinite(logF), logF, np.nan)

# Asymptotes
left_asym = 1.0 - (y > 1.0).astype(float)
right_asym = 1.0 - (y < 1.0).astype(float)

# --- Custom "Muted Purple-Orange" Color Palette ---
color_low = "#8c6bb1"  # Muted Purple
color_mid = "#ffffff"  # White
color_high = "#e69f00"  # Muted Orange (Colorblind friendly)

cmap_muted = LinearSegmentedColormap.from_list(
    "muted_div", [color_low, color_mid, color_high], N=256
)

norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
levels = np.linspace(0, 1, 51)
cbar_ticks = [0, 0.5, 1]
cbar_labels = ["0 (Low)", "0.5", "1 (High)"]

# --- Plotting ---
fig = plt.figure(figsize=(9 / 3, 4 / 3))
gs = gridspec.GridSpec(1, 3, width_ratios=[0.08, 0.84, 0.08], wspace=0.05)

axL = fig.add_subplot(gs[0, 0])
axM = fig.add_subplot(gs[0, 1], sharey=axL)
axR = fig.add_subplot(gs[0, 2], sharey=axL)

# 1. Left Strip
xL = np.array([-1.0, 0.0])
XL, YL = np.meshgrid(xL, y)
ZL = np.tile(left_asym[:, None], (1, 2))
axL.pcolormesh(XL, YL, ZL, cmap=cmap_muted, norm=norm, shading="auto")

# 2. Middle Heatmap
cf = axM.contourf(X, Y, logF, levels=levels, cmap=cmap_muted, norm=norm, extend="both")

# 3. Right Strip
xR = np.array([0.0, 1.0])
XR, YR = np.meshgrid(xR, y)
ZR = np.tile(right_asym[:, None], (1, 2))
axR.pcolormesh(XR, YR, ZR, cmap=cmap_muted, norm=norm, shading="auto")

# --- Formatting ---
for ax in [axL, axM, axR]:
    ax.set_yscale("log")
    ax.set_ylim(1e-1, 1e1)
    ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

# Y-Axis Labels (Left)
axL.set_yticks([1.0])
axL.set_yticklabels(["$c$"], fontsize=14, fontweight="bold")
axL.tick_params(axis="y", which="major", left=True, labelleft=True, length=5)
axL.set_ylabel("Operand Likelihood", fontsize=14, labelpad=5)

# X-Axis Label
axM.set_xlabel("Mixing Parameter $\lambda$", fontsize=14, labelpad=10)

# Annotations (Top)
y_top = 10.0
labels = [(-1.0, "HM"), (0.0, "PoE"), (1.0, "MoE")]
for xv, txt in labels:
    axM.annotate(
        txt,
        xy=(xv, y_top),
        xycoords="data",
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        clip_on=False,
    )
    axM.axvline(x=xv, linestyle="--", linewidth=1, color="black", alpha=0.3)

# Limit Labels
axL.annotate(
    "min",
    xy=(0.5, 1.0),
    xycoords="axes fraction",
    xytext=(0, 10),
    textcoords="offset points",
    ha="center",
    va="bottom",
    fontsize=12,
    fontweight="bold",
    clip_on=False,
)
axR.annotate(
    "max",
    xy=(0.5, 1.0),
    xycoords="axes fraction",
    xytext=(0, 10),
    textcoords="offset points",
    ha="center",
    va="bottom",
    fontsize=12,
    fontweight="bold",
    clip_on=False,
)

# Lambda Limits (Bottom)
axL.text(
    0.5,
    -0.08,
    "$\lambda \\to -\infty$",
    transform=axL.transAxes,
    ha="center",
    va="top",
    fontsize=11,
)
axR.text(
    0.5,
    -0.08,
    "$\lambda \\to +\infty$",
    transform=axR.transAxes,
    ha="center",
    va="top",
    fontsize=11,
)

# --- Improved Arrows ---
plt.subplots_adjust(top=0.85, bottom=0.22, left=0.12, right=0.90)

arrow_color = "#555555"  # Soft black for arrows

# Style: Double-headed, heads properly scaled.
# We remove 'tail_width' and use 'lw' (linewidth) for the shaft thickness.
arrow_kw = dict(
    arrowstyle="<->,head_width=0.4,head_length=0.6",
    color=arrow_color,
    clip_on=False,
    zorder=10,
    lw=2,  # Thicker shaft
    mutation_scale=20,
)  # Controls overall arrow size

# Left Arrow (Negation) - Vertical Arc
arrow_left = FancyArrowPatch(
    posA=(0.05, 0.80),
    posB=(0.05, 0.20),
    connectionstyle="arc3,rad=0.25",
    transform=fig.transFigure,
    **arrow_kw
)
fig.patches.append(arrow_left)
fig.text(
    0.01,
    0.5,
    "Negation",
    ha="left",
    va="center",
    rotation=90,
    fontsize=12,
    fontweight="bold",
    color=arrow_color,
)

# Bottom Arrow (Dual Operator) - Horizontal Arc
arrow_bottom = FancyArrowPatch(
    posA=(0.20, 0.10),
    posB=(0.80, 0.10),
    connectionstyle="arc3,rad=0.25",
    transform=fig.transFigure,
    **arrow_kw
)
fig.patches.append(arrow_bottom)
fig.text(
    0.5,
    0.02,
    "Dual Operator",
    ha="center",
    va="top",
    fontsize=12,
    fontweight="bold",
    color=arrow_color,
)

# Colorbar
cbar = fig.colorbar(cf, ax=[axL, axM, axR], fraction=0.04, pad=0.02)
cbar.set_ticks(cbar_ticks)
cbar.set_ticklabels(cbar_labels)

plt.savefig("neurips_softmax_purple_orange_fixed.svg", bbox_inches="tight")
print("Saved fixed plot as neurips_softmax_purple_orange_fixed.svg")
plt.show()
