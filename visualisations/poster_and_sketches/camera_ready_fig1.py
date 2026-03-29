import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import time

# --- Configuration (stable paper style across machines) ---
plt.rcParams.update(
    {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "text.usetex": False,
        "font.size": 11,
        "axes.labelsize": 12,
        "svg.fonttype": "path",
    }
)

# --- Data Generation ---
range_val = 2.0  # finite plot range; ends will be labeled as +/- infinity

# x = mixing parameter λ
x = np.linspace(-range_val, range_val, 900)

# y = likelihood gap Δ = log(p1/p2)
gap_max = 2.0
y = np.linspace(-gap_max, gap_max, 600)

X, Y = np.meshgrid(x, y)

# Softmax mixing weights: α1 = σ(λ * Δ)
with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
    A = X * Y
    Z_soft = 1.0 / (1.0 + np.exp(-A))  # sigmoid

# Asymptotes for λ→-∞ (left) and λ→+∞ (right)
# λ→+∞: α1 -> 1 if Δ>0 else 0
# λ→-∞: α1 -> 1 if Δ<0 else 0
left_asym_soft = (y < 0.0).astype(float)
right_asym_soft = (y > 0.0).astype(float)

# --- Plotting Setup (single row: left strip, main, right strip) ---
fig = plt.figure(figsize=(9 / 3 * 2, 1.6875))

# Add a 4th narrow column for the colorbar outside (prevents overlap)
gs = gridspec.GridSpec(
    1, 4, width_ratios=[0.08, 0.84, 0.08, 0.04], wspace=0.10
)

axL = fig.add_subplot(gs[0, 0])
axM = fig.add_subplot(gs[0, 1], sharey=axL)
axR = fig.add_subplot(gs[0, 2], sharey=axL)
cax = fig.add_subplot(gs[0, 3])

# Strong diverging colormap (your "likelihood" colors)
cmap = LinearSegmentedColormap.from_list(
    "strong_div", ["#2166ac", "#ffffff", "#b2182b"], N=512
)
norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)

# --- Left strip (λ → -∞) ---
xL = np.array([-1.0, 0.0])
XL, YL = np.meshgrid(xL, y)
ZL = np.tile(left_asym_soft[:, None], (1, 2))
axL.pcolormesh(XL, YL, ZL, cmap=cmap, norm=norm, shading="auto", rasterized=True)

# --- Middle heatmap (smooth; avoids contour banding/stripes) ---
pcm = axM.pcolormesh(X, Y, Z_soft, cmap=cmap, norm=norm, shading="auto", rasterized=True)

# Subtle semantic regions: conjunction (left) and disjunction (right)
axM.axvspan(-range_val, 0.0, facecolor="#2166ac", alpha=0.06, zorder=2)
axM.axvspan(0.0, range_val, facecolor="#b2182b", alpha=0.06, zorder=2)

t_left = axM.text(
    0.22,
    0.90,
    r"Conjunction $\wedge$",
    transform=axM.transAxes,
    ha="center",
    va="top",
    fontsize=12,
    fontweight="bold",
    bbox=dict(
        boxstyle="round,pad=0.22",
        facecolor="white",
        edgecolor="#ffffff",
        linewidth=0.8,
        alpha=0.62,
    ),
)
t_right = axM.text(
    0.78,
    0.90,
    r"Disjunction $\vee$",
    transform=axM.transAxes,
    ha="center",
    va="top",
    fontsize=12,
    fontweight="bold",
    bbox=dict(
        boxstyle="round,pad=0.22",
        facecolor="white",
        edgecolor="#ffffff",
        linewidth=0.8,
        alpha=0.62,
    ),
)
t_left.get_bbox_patch().set_path_effects(
    [pe.withStroke(linewidth=4.0, foreground="white", alpha=0.35)]
)
t_right.get_bbox_patch().set_path_effects(
    [pe.withStroke(linewidth=4.0, foreground="white", alpha=0.35)]
)

axM.annotate(
    "",
    xy=(0.62, 0.865),
    xytext=(0.38, 0.865),
    xycoords="axes fraction",
    arrowprops=dict(arrowstyle="<->", color="black", lw=1.0, alpha=0.70),
)
axM.text(
    0.50,
    0.842,
    "Dual Operator",
    transform=axM.transAxes,
    ha="center",
    va="top",
    fontsize=10,
    fontweight="bold",
    alpha=0.85,
)

# --- Right strip (λ → +∞) ---
xR = np.array([0.0, 1.0])
XR, YR = np.meshgrid(xR, y)
ZR = np.tile(right_asym_soft[:, None], (1, 2))
axR.pcolormesh(XR, YR, ZR, cmap=cmap, norm=norm, shading="auto", rasterized=True)

# --- Styling ---
for ax in [axL, axM, axR]:
    ax.set_ylim(-gap_max, gap_max)
    ax.tick_params(axis="y", left=False, right=False, labelleft=False)

# Y-axis label + center tick "0" on left axis (as requested)
axL.set_yticks([0.0])
axL.set_yticklabels([r"$0$"], fontsize=12, fontweight="bold")
axL.tick_params(axis="y", left=True, labelleft=True, length=4)
axL.set_ylabel(r"Density gap ($\log\frac{p_1}{p_2}$)", fontsize=12, labelpad=3)

# X label + ticks only at (-1, 0, 1)
axM.set_xlabel(r"Composition Parameter ($\lambda$)", fontsize=12, labelpad=4)
axM.set_xlim(-range_val, range_val)
axM.set_xticks([-1.0, 0.0, 1.0])
axM.set_xticklabels(["-1", "0", "1"])

# No x tick labels for side strips
axL.tick_params(axis="x", bottom=False, top=False, labelbottom=False)
axR.tick_params(axis="x", bottom=False, top=False, labelbottom=False)

# Optional: very light reference lines for HM / PoE / MoE (kept subtle)
for xv, txt in [(-1.0, "HM"), (0.0, "PoE"), (1.0, "MoE")]:
    axM.axvline(x=xv, linestyle="--", linewidth=1, color="black", alpha=0.15)
    axM.annotate(
        txt,
        xy=(xv, gap_max),
        xycoords="data",
        xytext=(0, 6),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        clip_on=False,
        alpha=0.85,
    )

# Side labels
for ax, txt in [(axL, "min"), (axR, "max")]:
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

# Bottom asymptote labels (plain text; no math mode)
axL.text(
    0.5,
    -0.08,
    r"$-\infty$",
    transform=axL.transAxes,
    ha="center",
    va="top",
    fontsize=10,
)
axR.text(
    0.5,
    -0.08,
    r"$+\infty$",
    transform=axR.transAxes,
    ha="center",
    va="top",
    fontsize=10,
)

# Colorbar OUTSIDE, with label
cbar = fig.colorbar(pcm, cax=cax)
cbar.set_ticks([0, 0.5, 1])
cbar.set_ticklabels(["0", "0.5", "1"])
cbar.set_label(r"Mixture Weight ($\alpha_1$)", rotation=90, labelpad=6)
cbar.ax.yaxis.set_ticks_position("left")
cbar.ax.yaxis.set_label_position("right")
cbar.ax.set_facecolor("#f5f5f5")
cbar.outline.set_edgecolor("#666666")
cbar.outline.set_linewidth(0.9)
cbar.ax.tick_params(length=3, width=0.8, colors="#444444", direction="in", pad=5)
for tick_label in cbar.ax.get_yticklabels():
    tick_label.set_rotation(90)
    tick_label.set_va("center")
    tick_label.set_ha("center")
for spine in cbar.ax.spines.values():
    spine.set_linestyle((0, (2, 2)))
    spine.set_alpha(0.9)

# Tighten margins (avoid bbox tight pulling colorbar inward)
plt.subplots_adjust(top=0.88, bottom=0.22, left=0.10, right=0.95)

# Nudge colorbar (legend) slightly to the right
_cax_pos = cax.get_position()
cax.set_position([_cax_pos.x0 + 0.01, _cax_pos.y0, _cax_pos.width, _cax_pos.height])

fname = "mixing_weights_clean_iclr.svg"
start = time.time()
print("Saving figure...", flush=True)
plt.savefig(fname, dpi=300, bbox_inches="tight", pad_inches=0)
print(f"Saved {fname} in {time.time() - start:.2f}s", flush=True)
