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
from eelbrain import NDVar, UTS
from eelbrain import plot, combine

# %%
subject="sub-01"
root = Path("~/Data/ds005810")
#clean_fif = root / f"derivatives/preprocessed/raw/{subject}_ses-ImageNet01_task-ImageNet_run-01_meg_clean.fif"
raw_fif=root/f"{subject}/ses-ImageNet01/meg/{subject}_ses-ImageNet01_task-ImageNet_run-01_meg.fif"
empty_room=root/"sub-emptyroom/ses-20211114/meg/sub-emptyroom_ses-20211114_task-noise_meg.fif"

# %%
raw = mne.io.read_raw_fif(raw_fif, preload=True)

events = find_events(raw, stim_channel="UPPT001")
#raw.info["bads"]  
raw
#raw.n_times

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
meg.first_time   

# %%
meg=raw.pick("meg")
meg.filter(1., 40., phase="zero-double", verbose=False)
meg.resample(100, npad="auto", verbose=False)

sfreq = meg.info['sfreq']         
n_times = meg.n_times              
     


time = UTS(0, 1/sfreq, n_times)


stim1 = np.zeros(n_times, dtype=float)  #inanimate
stim2 = np.zeros(n_times, dtype=float)  #animate


for t, is_anim in zip(event_table['time'], event_table['animate']):
    idx = int(round((t ) * sfreq))
    if 0 <= idx < n_times:
        if is_anim:
            stim2[idx] = 1.0
        else:
            stim1[idx] = 1.0


stim1 = NDVar(stim1, time)
stim2 = NDVar(stim2, time)


p=plot.LineStack(combine([stim1, stim2]), ylabels=["inanimate", "animate"], offset=1.5)

# %%
stim1.x

# %%
stim1.x[738]

# %% [markdown]
# ## NOISE COv

# %%
raw_er = mne.io.read_raw_fif(empty_room, preload=True, verbose=False).pick('meg')
raw_er.filter(1., 40., phase="zero-double", verbose=False)
raw_er.resample(100, npad="auto", verbose=False)

noise_cov = mne.compute_raw_covariance(raw_er, method='shrunk', rank=None,verbose=False)
noise_cov

# %%
raw_er.filter(1., 40., phase="zero-double", verbose=False)
