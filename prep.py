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

# %%
raw = mne.io.read_raw_fif("/Users/maryamvalian/Data/ds005810/sub-01/ses-ImageNet01/meg/sub-01_ses-ImageNet01_task-ImageNet_run-01_meg.fif", preload=False)
#raw.info["bads"]  
raw

# %%
p=raw.plot(n_channels=40, duration=10)

# %%
events = mne.find_events(raw, stim_channel="UPPT001")
epochs = mne.Epochs(raw, events, tmin=-0.1, tmax=0.8, baseline=(-0.1, 0))
#epochs[:5].plot()
evoked = epochs.average()


p=evoked.plot()


evoked.plot_topomap(times=[0.1, 0.2, 0.3], ch_type="mag")

# %%
meta = pd.read_csv("/Users/maryamvalian/Data/ds005810/derivatives/detailed_events/sub-01_events.csv")
meta.head()

# %%
