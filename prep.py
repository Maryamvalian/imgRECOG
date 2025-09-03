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
import os
from mne.coreg import Coregistration

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
root = Path("~/Data/ds005810").expanduser()
subjects_dir = root / "derivatives" / "freesurfer" / "subjects"
subjects_dir.mkdir(parents=True, exist_ok=True)

# Make sure libraries see it
os.environ["SUBJECTS_DIR"] = str(subjects_dir)

# 1) Get fsaverage (MNE will download the official template if missing)
mne.datasets.fetch_fsaverage(subjects_dir=str(subjects_dir))

# %% [markdown]
# # to create fsaverage-vol-7-src.fif in the fsaverage/bem : 

# %%
#to create fsaverage-vol-7-src.fif in the fsaverage/bem : 
src_vol = mne.setup_volume_source_space(
    subject="fsaverage",
    subjects_dir=str(subjects_dir),
    pos=7.0,            
    mri="aparc+aseg.mgz",
    verbose=True
)
mne.write_source_spaces(
    subjects_dir / "fsaverage" / "bem" / "fsaverage-vol-7-src.fif",
    src_vol, overwrite=True
)

# %% [markdown]
# # AFTER RECON-all by freesurfer

# %% [markdown]
# ### 1. Set up home 

# %%
#set up Ffreesurfer home environment

import  shutil

subject="sub-01"
root = Path("~/Data/ds005810")
subjects_dir = root / "derivatives" / "freesurfer" / "subjects"

FS_HOME = "/Applications/freesurfer/8.1.0"
SUBJECTS_DIR = "/Users/maryamvalian/Data/ds005810/derivatives/freesurfer/subjects"

os.environ["FREESURFER_HOME"] = FS_HOME
os.environ["SUBJECTS_DIR"] = SUBJECTS_DIR
os.environ["FS_LICENSE"] = f"{FS_HOME}/.license"  # you put it here; keeps things explicit
os.environ["PATH"] = f"{FS_HOME}/bin:" + os.environ.get("PATH", "")

# Tell MNE as well (sets both mne config and env)
mne.set_config("FREESURFER_HOME", FS_HOME, set_env=True)
mne.set_config("SUBJECTS_DIR", SUBJECTS_DIR, set_env=True)

# Sanity checks
print("FREESURFER_HOME =", os.environ.get("FREESURFER_HOME"))
print("SUBJECTS_DIR    =", os.environ.get("SUBJECTS_DIR"))
print("mri_watershed   =", shutil.which("mri_watershed"))  # should NOT be None

# %% [markdown]
# ### 2. MAKE bem sol

# %%
subject = "sub-01"
subjects_dir = os.environ["SUBJECTS_DIR"]

# 1) Create scalp/outer-/inner-skull surfaces via FreeSurfer's mri_watershed
mne.bem.make_watershed_bem(subject=subject, subjects_dir=subjects_dir, overwrite=True)

# 2) Build a single-layer MEG BEM (brain, 0.3 S/m) and the solution
model = mne.make_bem_model(subject=subject, subjects_dir=subjects_dir,
                           ico=5, conductivity=(0.3,))
bem = mne.make_bem_solution(model)

# %%
out_bem = f"{subjects_dir}/{subject}/bem/{subject}-bem-sol.fif"
mne.write_bem_solution(out_bem, bem, overwrite=True)

# %% [markdown]
# ### 3. Coreg

# %%
#First set fiducials with mne coreg GUI and save as fiducials.fif
from mne.io import read_fiducials
fid_path = Path("/Users/maryamvalian/Data/ds005810/derivatives/mne/sub-01/sub-01-fiducials.fif")
fids_list, coord_frame = read_fiducials(str(fid_path))
fids_list

# %%
coord_frame

# %%
raw_fif=root/f"{subject}/ses-ImageNet01/meg/{subject}_ses-ImageNet01_task-ImageNet_run-01_meg.fif"
info = mne.io.read_info(raw_fif)

coreg = Coregistration(
    subject=subject,
    subjects_dir=subjects_dir,
    info=info,
    fiducials=fids_list,
)

coreg.set_scale_mode("uniform")                #"uniform"


coreg.fit_fiducials()

coreg.fit_icp(n_iterations=50)


out_trans = f"{root}/derivatives/trans/{subject}-trans.fif"
mne.write_trans(out_trans, coreg.trans, overwrite=True)

# %%
coreg.trans

# %%
coreg.scale

# %%
scale=coreg.scale
mne.scale_mri('fsaverage', subject, scale, subjects_dir=SUBJECTS_DIR, labels=False)

# %% [markdown]
# ### 4. create src

# %%
mne_out = Path("/Users/maryamvalian/Data/ds005810/derivatives/mne")
src = mne.setup_volume_source_space(subject=subject, 
                                     subjects_dir=subjects_dir,
                                     pos=7.0, mri="aseg.mgz")
mne.write_source_spaces(mne_out/subject/f"{subject}-vol7-src.fif", src, overwrite=True)

# %% [markdown]
# ### 5. Biuld FWD

# %%
trans=coreg.trans
trans

# %%
raw_fif

# %%
raw = mne.io.read_raw_fif(str(raw_fif), preload=True)

# %%
bem

# %%
fwd = mne.make_forward_solution(
    info=raw.info, trans=trans, src=src, bem=bem,
    meg=True, eeg=False, mindist=0.0
)
mne.write_forward_solution(mne_out/subject/f"{subject}-fwd.fif", fwd, overwrite=True)

# %% [markdown]
# ## after scale subject ( renamed old folder to sun-01_old)

# %%
src_sb01 = mne.read_source_spaces(
    f"{SUBJECTS_DIR}/sub-01/bem/sub-01-vol-7-src.fif"
)

# %%
src_sb01

# %%
src_2 = mne.read_source_spaces(
    f"{root}/derivatives/freesurfer/sub-01_old/bem/sub-01-vol-7-src.fif" #befor scale_subject
)

# %%
src_2

# %%
