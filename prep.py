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

from pathlib import Path
import os, mne

# %%
raw = mne.io.read_raw_fif("/Users/maryamvalian/Data/ds005810/sub-01/ses-ImageNet01/meg/sub-01_ses-ImageNet01_task-ImageNet_run-01_meg.fif", preload=False)
#raw.info["bads"]  
raw

# %%
clean= mne.io.read_raw_fif("/Users/maryamvalian/Data/ds005810/derivatives/preprocessed/raw/sub-01_ses-ImageNet01_task-ImageNet_run-01_meg_clean.fif", preload=False)
#clean

# %%
#p=clean.plot(n_channels=40, duration=10)

# %%
events = find_events(raw, stim_channel="UPPT001")


fig = mne.viz.plot_events(events, sfreq=raw.info["sfreq"], first_samp=raw.first_samp)
#ID:  
#1 begin
#2 stim_on
#4 response
#8 end

# %%
fig = mne.viz.plot_events(
    events[3:20],
    sfreq=raw.info["sfreq"],
    
)
#2: stim_om
#4 resp

# %%
events[120]

# %%

epochs = mne.Epochs(raw, events, tmin=-0.1, tmax=0.8, baseline=(-0.1, 0))
#epochs[:5].plot()
evoked = epochs.average()


p=evoked.plot()


evoked.plot_topomap(times=[0.1, 0.2, 0.3], ch_type="mag")

# %%
meta = pd.read_csv("/Users/maryamvalian/Data/ds005810/derivatives/detailed_events/sub-01_events.csv")
meta.head()

# %%
meta.tail()

# %%
meta.iloc[3995,7]

# %%
meta['session'].value_counts()

# %%
meta.groupby(['session','run'])['image_id'].count().sort_index()

# %%
meta[['stim_is_animate','resp_is_right']].mean(numeric_only=True)  

# %%
m01r1 = meta[(meta['session']=='ImageNet01') & (meta['run']==1)].copy()

m01r1.tail()

# %%
len(m01r1)

# %%
m01r1[['stim_is_animate','resp_is_right']].mean(numeric_only=True)  

# %%
