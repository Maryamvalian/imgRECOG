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
import mne
import pandas as pd
from mne import *
from ncrf import fit_ncrf
from pathlib import Path
import numpy as np

# %%
subject="sub-01"
root = Path("~/Data/ds005810")
#clean_fif = root / f"derivatives/preprocessed/raw/{subject}_ses-ImageNet01_task-ImageNet_run-01_meg_clean.fif"
raw_fif=root/f"{subject}/ses-ImageNet01/meg/{subject}_ses-ImageNet01_task-ImageNet_run-01_meg.fif"

# %%
raw = mne.io.read_raw_fif(raw_fif, preload=False)

events = find_events(raw, stim_channel="UPPT001")
#raw.info["bads"]  
#raw
raw.n_times

# %%
stim = events[events[:, 2] == 2]   #ID2 : stim_on

stim_samples = stim[:, 0]

stim_times = stim_samples / raw.info["sfreq"]
len(stim) 

# %%
meta = pd.read_csv(f"/Users/maryamvalian/Data/ds005810/derivatives/detailed_events/{subject}_events.csv")
meta_r1 = meta[(meta['session'] == 'ImageNet01') & (meta['run'] == 1)].reset_index(drop=True)

# %%
n=len(meta_r1)

# %%
anim_flags = meta_r1.loc[:n-1, 'stim_is_animate'].to_numpy().astype(bool)

# %%
event_table = pd.DataFrame({
           
    "time": stim_times,               
    "animate": anim_flags           
})

# %%
event_table

# %%
