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

sys.path.append(str(Path.cwd().parent))
from ncrf_dataset import *
sys.path.append(str(Path.cwd().parent))
from vol2surf import *

# %%
# Data locations
root = Path("~/Data/ds005810")
subjects_dir = str(Path('~/Data/ds005810/derivatives/freesurfer/subjects').expanduser())

# Configure the matplotlib figure style
FONT = 'Arial'
FONT_SIZE = 8
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
pyplot.rcParams.update(RC)

# Load ncrf models
data_ec = create_ncrf_dataset(mod="effect", path="../models/all_runs/ncrf-ec") 
data_dc = create_ncrf_dataset(mod="dummy", path="../models/all_runs/morphed") 

# Statistical Tests
animate_data = data_dc.sub("animacy == 'animate'")
inanimate_data = data_dc.sub("animacy == 'inanimate'")

result_anim = testnd.Vector(
    'ncrf',
    match='subject',
    data=animate_data,
    tfce=True,
    tstart=80,
    tstop=600,
    samples=1000
)

result_inanim = testnd.Vector(
    'ncrf',
    match='subject',
    data=inanimate_data,
    tfce=True,
    tstart=80,
    tstop=600,
    samples=1000
)

# Keep significants
anim_sig = result_anim.masked_difference().sub(time=(0,400))
inanim_sig = result_inanim.masked_difference().sub(time=(0,400))

# Aggreagate over time points
anim_map = anim_sig.norm("space").sum("time")
inanim_map = inanim_sig.norm("space").sum("time")

# Get data from NDVAR
anim_x = anim_map.x
inanim_x = inanim_map.x

# Normalize
anim_norm = (anim_x - anim_x.min()) / (anim_x.max() - anim_x.min())
inanim_norm = (inanim_x - inanim_x.min()) / (inanim_x.max() - inanim_x.min())

# Animacy Contrast
x = anim_norm - inanim_norm

# Plot volume on the surface
fig=Plot_vol2surf(
    x, subject="fsaverage2",
    title="Aggregation NCRF-DC",
    save="figures/Aggregation NCRF-DC",
)

