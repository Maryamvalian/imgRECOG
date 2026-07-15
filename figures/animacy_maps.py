# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.image import imread
import matplotlib as mpl
sys.path.append(str(Path.cwd().parent))
from ncrf_analysis import *


# %%
# Data locations
model_dir = Path("../models/all_runs")
dc_dir = model_dir / "ncrf-dc"
ec_dir = model_dir / "ncrf-ec"

# Where to save the figure
fig_dir = Path("figures")
fig_dir.mkdir(parents=True, exist_ok=True)
tmp_dir = fig_dir / "tmp_glassbrain"
tmp_dir.mkdir(parents=True, exist_ok=True)

# Configure the matplotlib figure style
FONT = 'Arial'
FONT_SIZE = 14
RC = {
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.transparent': True,
    # Font
    'font.family': 'sans-serif',
    'font.sans-serif': FONT,
    'font.size': FONT_SIZE,
    'figure.labelsize': FONT_SIZE,
    'figure.titlesize': FONT_SIZE,
    'axes.labelsize': FONT_SIZE,
    'axes.titlesize': FONT_SIZE,
    'xtick.labelsize': FONT_SIZE,
    'ytick.labelsize': FONT_SIZE,    
    'legend.fontsize': FONT_SIZE,
}
plt.rcParams.update(RC)

# %%
# Load ncrf models
data_ec = create_ncrf_dataset(mod="effect", path=ec_dir) 
data_dc = create_ncrf_dataset(mod="dummy", path=dc_dir) 

# Statistical Tests
result_dc = ncrf_stats(data=data_dc, comparison="paired", mod="dummy")
result_ec = ncrf_stats(data=data_ec, comparison="condition", mod="effect", condition="contrast",)

diff_dc = result_dc.masked_difference()
diff_ec = result_ec.masked_difference()

# %%
TIME_MARKERS = (120, 150, 250, 350)

# Plot Butterfly comparison 
fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

for ax, diff, title in zip(
    axes,
    [diff_dc, diff_ec],
    ["NCRF-DC", "NCRF-EC"],
):
    y = diff.norm("space").x
    times = diff.time.times

    ax.plot(times, y.T, color="k", linewidth=0.8)

    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v / 1e-12:.1f}"))
    ax.set_ylabel("NCRF")
    ax.set_title(title)

    for t in TIME_MARKERS:
        ax.axvline(t, color="#1f77b4", linewidth=1)

    ax.text(
        -0.10,
        1.02,
        r"$1e^{-12}$",
        transform=ax.transAxes,
    )

axes[-1].set_xlabel("Time (ms)")

plt.tight_layout()
plt.savefig("figures/dc_ec_BF.svg")
plt.show()

# %%
# Plot Glass brain comparison of NCRF-DC and NCRF-EC

# Save temporary GlassBrain images
vmax=3e-12
for model_name, diff in {
    "dc": diff_dc,
    "ec": diff_ec,
}.items():
    for t in TIME_MARKERS:
        f = plot.GlassBrain(
            diff.sub(time=t),
            display_mode="yz",
            title="",
            vmax=vmax,
        )
        f.save(tmp_dir / f"{model_name}_{t}.png")
        f.close()

# Merge into one figure
fig, axes = plt.subplots(
    nrows=len(TIME_MARKERS),
    ncols=2,
    figsize=(9, 11),
)

for i, t in enumerate(TIME_MARKERS):
    for j, model_name in enumerate(("dc", "ec")):
        img = imread(tmp_dir / f"{model_name}_{t}.png")
        axes[i, j].imshow(img)
        axes[i, j].axis("off")

    axes[i, 0].text(
        -0.08,
        0.5,
        f"{t} ms",
        transform=axes[i, 0].transAxes,
        va="center",
        ha="right",
    )

axes[0, 0].set_title("A) NCRF-DC", loc="left")
axes[0, 1].set_title("B) NCRF-EC", loc="left")

fig.subplots_adjust(
    left=0.10,
    right=0.98,
    top=0.97,
    bottom=0.10,
    hspace=0.08,
    wspace=0.0,
)

# Shared colorbar
norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
sm = mpl.cm.ScalarMappable(norm=norm, cmap="hot_r")
sm.set_array([])

cax = fig.add_axes([0.35, 0.04, 0.30, 0.02])
cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")

cbar.set_label("Norm of estimated response")
cbar.set_ticks(np.arange(0, vmax + 1e-12, 1e-12))
cbar.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x / 1e-12:.0f}"))
cbar.ax.xaxis.get_offset_text().set_visible(False)

cbar.ax.text(
    1.08,
    -0.2,
    r"$1e^{-12}$",
    transform=cbar.ax.transAxes,
    ha="left",
    va="center",
)

fig.savefig(fig_dir / "DC_EC_GlassBrain.png")
fig.savefig(fig_dir / "DC_EC_GlassBrain.svg")
fig.savefig(fig_dir / "DC_EC_GlassBrain.pdf")
       

# %%
