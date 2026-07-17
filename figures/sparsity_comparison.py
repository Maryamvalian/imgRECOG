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

# Configure the matplotlib figure style
FONT = 'Arial'
FONT_SIZE = 14
RC = {
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.transparent': True,
    'savefig.bbox': 'tight',
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
def plot_l1(ax, times, mean_ec, sem_ec, mean_dc, sem_dc):
    
    ax.plot( times, mean_ec, linewidth=2, label="NCRF-EC")
    ax.fill_between( times, mean_ec - sem_ec, mean_ec + sem_ec, alpha=0.25)

    ax.plot( times,  mean_dc, linewidth=2, label="NCRF-DC")
    ax.fill_between( times,mean_dc - sem_dc,mean_dc + sem_dc, alpha=0.25)

    ax.set(xlabel="Time (ms)", ylabel="Mean L1 norm")
    ax.legend()


def plot_sig_sources(ax, times, counts_ec, counts_dc):
    
    ax.plot(times,counts_ec,linewidth=2, label="NCRF-EC" )
    ax.plot(times, counts_dc,linewidth=2, label="NCRF-DC")

    ax.set(xlabel="Time (ms)",ylabel="Significant sources (count)")
    ax.legend()


# Main------------

# Load ncrf models
data_ec = create_ncrf_dataset(mod="effect", path=ec_dir) 
data_dc = create_ncrf_dataset(mod="dummy", path=dc_dir) 

# Statistical Tests
result_dc = ncrf_stats(data=data_dc, comparison="paired", mod="dummy")
result_ec = ncrf_stats(data=data_ec, comparison="condition", mod="effect", condition="contrast",)

# Compute L1 time courses
times_ec, ec_by_subject = compute_l1(data_ec,mod="effect")
times_dc, dc_by_subject = compute_l1(data_dc, mod="dummy")

if not np.allclose(times_ec, times_dc):
    raise ValueError("EC and DC L1 time axes do not match.")

subjects = sorted(set(ec_by_subject) & set(dc_by_subject))

if not subjects:
    raise ValueError("No matching subjects between EC and DC.")

l1_ec = np.stack([ec_by_subject[subject] for subject in subjects])
l1_dc = np.stack([dc_by_subject[subject] for subject in subjects])

mean_ec = l1_ec.mean(axis=0)
mean_dc = l1_dc.mean(axis=0)

sem_ec = (l1_ec.std(axis=0, ddof=1)/ np.sqrt(len(subjects)))
sem_dc = (l1_dc.std(axis=0, ddof=1) / np.sqrt(len(subjects)))

# Compute significant source counts
times_sig_ec, counts_ec = significant_source_timecourse(result_ec)
times_sig_dc, counts_dc = significant_source_timecourse(result_dc)

if not np.allclose(times_sig_ec, times_sig_dc):
    raise ValueError("EC and DC significance time axes do not match.")

# Plots
fig, axes = plt.subplots(2, 1,figsize=(7, 8), constrained_layout=True)

plot_l1( axes[0], times_ec,mean_ec,sem_ec, mean_dc, sem_dc)
plot_sig_sources( axes[1], times_sig_ec, counts_ec, counts_dc)

axes[0].set_title("A) Contrast Model Size",loc="left")
axes[1].set_title("B) Significant Source Count",loc="left")

fig.savefig(f"{fig_dir}/quantitative_comparison.pdf")

plt.show()

# %%
