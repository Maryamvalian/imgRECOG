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
root = Path("~/Data/ds005810")
subjects_dir = str(Path('~/Data/ds005810/derivatives/freesurfer/subjects').expanduser())

#Create dataset of ncrf models
data_ec = create_ncrf_dataset(mod="effect", path="../models/all_runs/ncrf-ec") 
data_dc = create_ncrf_dataset(mod="dummy", path="../models/all_runs/morphed") 
#Statistical Tests
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

# Keep sig
anim_sig = result_anim.masked_difference().sub(time=(0,400))
inanim_sig = result_inanim.masked_difference().sub(time=(0,400))


anim_map = anim_sig.norm("space").sum("time")
inanim_map = inanim_sig.norm("space").sum("time")

anim_x = anim_map.x
inanim_x = inanim_map.x


anim_norm = (anim_x - anim_x.min()) / (anim_x.max() - anim_x.min())
inanim_norm = (inanim_x - inanim_x.min()) / (inanim_x.max() - inanim_x.min())


x = anim_norm - inanim_norm


# %%
source_label_names_fixed = np.load(
    "../source_label_names_fixed.npy",
    
)
len(source_label_names_fixed)
ROI_LABELS = {
    "ctx-lh-lateraloccipital",
    "ctx-rh-lateraloccipital",
    "ctx-lh-cuneus",
    "ctx-rh-cuneus",
    "ctx-lh-lingual",
    "ctx-rh-lingual",
    "ctx-lh-pericalcarine",
    "ctx-rh-pericalcarine",
    "ctx-lh-fusiform",
    "ctx-rh-fusiform",
    "ctx-lh-inferiortemporal",
    "ctx-rh-inferiortemporal",
    "ctx-lh-middletemporal",
    "ctx-rh-middletemporal",
    "ctx-lh-temporalpole",
    "ctx-rh-temporalpole",
    "ctx-lh-occipitalpole",
    "ctx-rh-occipitalpole",
    "ctx-lh-superiorparietal",
    "ctx-rh-superiorparietal",
    "ctx-lh-precuneus",
    "ctx-rh-precuneus",
    "ctx-lh-lateraloccipital",
    "ctx-rh-lateraloccipital",
    "ctx-lh-inferiorparietal",
    "ctx-rh-inferiorparietal",
    "ctx-lh-supramarginal",
    "ctx-rh-supramarginal",
    
}

# Boolean mask 
mask = np.isin(source_label_names_fixed, list(ROI_LABELS))


contrast_dc = x.copy()
contrast_dc[~mask] = 0

# %%
fig=Plot_vol2surf(
    contrast_dc, subject="fsaverage2",
    views=("lateral", "ventral"),
    title="Aggregation NCRF-DC",
    save="figures/Aggregation NCRF-DC",
)

# %%

# %%
