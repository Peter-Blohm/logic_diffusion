import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path
from scipy.special import softmax  # still optional; you can remove if unused

# Data
range = 2
x_left = np.linspace(-range, -0.1, 300)
x_right = np.linspace(0.1, range, 300)
x = np.concatenate([x_left, x_right])
y = np.logspace(-1, 1, 400)
X, Y = np.meshgrid(x, y)
show_softmax = not True  # same as not not not True

with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
    A = X * np.log(Y)
    logF = (1.0 / X) * np.logaddexp(A, 0.0)

    # for displaying softmax instead
    if show_softmax:
        logF = np.exp(np.log(Y) * np.sign(X) - logF * np.sign(X))

logF = np.where(np.isfinite(logF), logF, np.nan)

left_asym = np.log(np.minimum(y, 1.0))  # x→-∞
right_asym = np.log(np.maximum(y, 1.0))  # x→+∞

if show_softmax:
    left_asym = 1 - (y > 1.0)
    right_asym = 1 - (y < 1.0)

norm = TwoSlopeNorm(vmin=-2, vcenter=0.0, vmax=2)
cmap = mpl.colormaps["coolwarm"]
levels = np.linspace(-2, 2, 41)

fig = plt.figure(figsize=(7.5, 3.2))
gs = gridspec.GridSpec(1, 3, width_ratios=[0.08, 0.84, 0.08], wspace=0.04)

axL = fig.add_subplot(gs[0, 0])
axM = fig.add_subplot(gs[0, 1], sharey=axL)
axR = fig.add_subplot(gs[0, 2], sharey=axL)

# Left asymptotic strip
xL = np.array([-1.0, 0.0])
XL, YL = np.meshgrid(xL, y)
ZL = np.tile(left_asym[:, None], (1, 2))
axL.pcolormesh(XL, YL, ZL, cmap=cmap, norm=norm, shading="auto")

# Middle contour
cf = axM.contourf(X, Y, logF, levels=levels, cmap=cmap, norm=norm, extend="both")

# Right asymptotic strip
xR = np.array([0.0, 1.0])
XR, YR = np.meshgrid(xR, y)
ZR = np.tile(right_asym[:, None], (1, 2))
axR.pcolormesh(XR, YR, ZR, cmap=cmap, norm=norm, shading="auto")

# Shared y (log) + limits, and remove all y tick marks / numbers
for ax in [axL, axM, axR]:
    ax.set_yscale("log")
    ax.set_ylim(1e-1, 1e1)
    ax.tick_params(
        axis="y",
        which="both",
        left=False,
        right=False,
        labelleft=False,
        labelright=False,
    )
axL.set_yticks([1.0])
axL.set_yticklabels(["c"])
axL.tick_params(
    axis="y",
    which="both",
    left=True,
    right=False,  # tick only on left side
    labelleft=True,
    labelright=False,
)

# Remove x ticks from LEFT and RIGHT panels
axL.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
axR.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

# Axis labels
axL.set_ylabel("Operand Likelihood")
axM.set_ylabel("")
axR.set_ylabel("")
axM.set_xlabel("λ")

# --- removed add_break_marks & calls entirely ---

# Middle labels
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

# Guides
for xv in [-1.0, 1.0]:
    axM.axvline(
        x=xv,
        ymin=0.0,
        ymax=1.0,
        linestyle="--",
        linewidth=1.2,
        color="black",
        alpha=0.45,
    )

# Bottom "λ→±∞" labels
axL.text(
    0.5, -0.10, "λ→−∞", transform=axL.transAxes, ha="center", va="top", fontsize=10
)
axR.text(
    0.5, -0.10, "λ→+∞", transform=axR.transAxes, ha="center", va="top", fontsize=10
)

# Top "min"/"max"
axL.annotate(
    "min",
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
axR.annotate(
    "max",
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

# Margins to fit outside arrows (left and bottom)
plt.subplots_adjust(top=0.86, bottom=0.20, left=0.12, right=0.90)

# === Outside arrows (Bézier) ===
# Left arrow (negation)
P0L = (0.08, 0.78)  # top
P3L = (0.08, 0.22)  # bottom
C1L = (0.04, 0.70)
C2L = (0.04, 0.30)
pathL = Path(
    [P0L, C1L, C2L, P3L],
    [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4],
)
arrow_left = FancyArrowPatch(
    path=pathL,
    transform=fig.transFigure,
    arrowstyle="<->",
    linewidth=1.6,
    mutation_scale=13,
    clip_on=False,
    zorder=10,
)
fig.patches.append(arrow_left)
fig.text(
    0.042,
    0.50,
    "negation",
    ha="right",
    va="center",
    rotation=90,
    fontsize=11,
    fontweight="bold",
)

# Bottom arrow (dual operator)
P0 = (0.28, 0.12)
P3 = (0.72, 0.12)
C1 = (0.31, 0.055)
C2 = (0.69, 0.055)
pathB = Path(
    [P0, C1, C2, P3],
    [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4],
)
arrow_bottom = FancyArrowPatch(
    path=pathB,
    transform=fig.transFigure,
    arrowstyle="<->",
    linewidth=1.6,
    mutation_scale=13,
    clip_on=False,
    zorder=10,
)
fig.patches.append(arrow_bottom)
fig.text(
    0.50,
    0.055,
    "dual operator",
    ha="center",
    va="top",
    fontsize=11,
    fontweight="bold",
)

# Colorbar with categorical labels
cbar = fig.colorbar(
    cf,
    ax=[axL, axM, axR],
    fraction=0.04,
    pad=0.02,
    label="result likelihood",
)
cbar.set_ticks([-2, 0, 2])
cbar.set_ticklabels(["low", "neutral", "high"])
for label in cbar.ax.get_yticklabels():
    label.set_rotation(90)
    # label.set_ha("center")  # optional: center-align
fig.savefig("likelihoods_no_margins.pdf", bbox_inches="tight", pad_inches=0)
plt.show()
