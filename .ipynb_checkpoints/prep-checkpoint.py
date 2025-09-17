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
#Download FsAVERAGE by MNE
"""
root = Path("~/Data/ds005810").expanduser()
subjects_dir = root / "derivatives" / "freesurfer" / "subjects"
subjects_dir.mkdir(parents=True, exist_ok=True)


os.environ["SUBJECTS_DIR"] = str(subjects_dir)


mne.datasets.fetch_fsaverage(subjects_dir=str(subjects_dir))
"""

# %% [markdown]
# # AFTER RECON-all by freesurfer

# %% [markdown]
# ### 1. Set up home 

# %%
import  shutil

SUBJECTS_DIR = "/Users/maryamvalian/Data/ds005810/derivatives/freesurfer/subjects"

os.environ["FREESURFER_HOME"] = FS_HOME
os.environ["SUBJECTS_DIR"] = SUBJECTS_DIR
os.environ["FS_LICENSE"] = f"{FS_HOME}/.license"  
os.environ["PATH"] = f"{FS_HOME}/bin:" + os.environ.get("PATH", "")

mne.set_config("FREESURFER_HOME", FS_HOME, set_env=True)
mne.set_config("SUBJECTS_DIR", SUBJECTS_DIR, set_env=True)

print("FREESURFER_HOME =", os.environ.get("FREESURFER_HOME"))
print("SUBJECTS_DIR    =", os.environ.get("SUBJECTS_DIR"))
print("mri_watershed   =", shutil.which("mri_watershed"))  

# %% [markdown]
# ### 2. MAKE bem sol

# %%
#Creats BEM folder and files

subjects_dir = os.environ["SUBJECTS_DIR"]

# for all subjects after recon-all is done! range (1,31)
for i in range(1, 2):
    
    subject = f"sub-{i:02d}"
    print(f"Creating BEM for {subject}...")
    
    
    #Creats scalp/outer-/inner-skull surfaces
    mne.bem.make_watershed_bem(subject=subject, subjects_dir=subjects_dir, overwrite=True,verbose=False)
    
    # single-layer for MEG! 
    model = mne.make_bem_model(subject=subject, subjects_dir=subjects_dir,
                               ico=5, conductivity=(0.3,),verbose=False)
    bem = mne.make_bem_solution(model)
    
    out_bem = f"{subjects_dir}/{subject}/bem/{subject}-bem-sol.fif"
    mne.write_bem_solution(out_bem, bem, overwrite=True,verbose=False)
    print(f"{subject}-bem-sol.fif saved successfully.")


# %% [markdown]
# ### 3. Coreg

# %%
subject="sub-03"
raw = mne.io.read_raw_fif(f"/Users/maryamvalian/Data/ds005810/{subject}/ses-ImageNet01/meg/{subject}_ses-ImageNet01_task-ImageNet_run-01_meg.fif", preload=False)


subjects_dir

# %% [markdown]
# # COREG GUI

# %%
#BASH: (load needed raw and subject in GUI) save fiducials and save trans.fif
"""
mne coreg --subject sub-02 --subjects-dir /Users/maryamvalian/Data/ds005810/derivatives/freesurfer/subjects --fif /Users/maryamvalian/Data/ds005810/sub-02/ses-ImageNet02/meg/sub-02_ses-ImageNet02_task-ImageNet_run-01_meg.fif 
"""
#BASH mne coreg : setting fiducials dosent need meg info

# %% [markdown]
# # COREG Code (only fiducials with gui) then run this code

# %%
session="ImageNet03"
root = Path("~/Data/ds005810")
subjects_dir = "/Users/maryamvalian/Data/ds005810/derivatives/freesurfer/subjects"


for i in range(6,7 ):
    
    subject = f"sub-{i:02d}"

    raw_fif=f"/Users/maryamvalian/Data/ds005810/{subject}/ses-{session}/meg/{subject}_ses-{session}_task-ImageNet_run-08_meg.fif"
    info = mne.io.read_info(raw_fif) 
    
    coreg = mne.coreg.Coregistration( subject=subject,info=info, subjects_dir=subjects_dir)
    coreg.fit_fiducials()
    coreg.fit_icp(n_iterations=50)
    
    out_trans = f"{root}/derivatives/trans/{subject}-{session}-trans.fif"
    mne.write_trans(out_trans, coreg.trans, overwrite=True)

    print(f"{subject}-{session}-trans.fif saved successfully." )

# %%
#BASH : mne coreg , set fiducials, save MRI Fid. then Fit fiducials,Fit ICP and save as trans!
"""
from mne.io import read_fiducials
fid_path = Path(f"/Users/maryamvalian/Data/ds005810/derivatives/freesurfer/subjects/{subject}/bem/{subject}-fiducials.fif")
fids_list, coord_frame = read_fiducials(str(fid_path))
fids_list


raw_fif=f"/Users/maryamvalian/Data/ds005810/{subject}/ses-{session}/meg/{subject}_ses-{session}_task-ImageNet_run-01_meg.fif"
info = mne.io.read_info(raw_fif)         #=====> connect meg coord to mri coord :trans.fif

coreg = Coregistration(
    subject=subject,
    subjects_dir=subjects_dir,
    info=info,
    fiducials=fids_list,
)

coreg.set_scale_mode("None")                


coreg.fit_fiducials()

coreg.fit_icp(n_iterations=50)


out_trans = f"{root}/derivatives/trans/{subject}-{session}-trans.fif"
mne.write_trans(out_trans, coreg.trans, overwrite=True)
"""

# %%
#scale to fsaverage
"""
scale=coreg.scale
mne.scale_mri('fsaverage', subject, scale, subjects_dir=subjects_dir, labels=False)
"""

# %% [markdown]
# ### 4. create src FORCE INNER SKULL

# %%
#Drop sources outside of Inner_skull.surf otherwise not consistent with source in fwd later we will make)
for i in range(11, 31):
    
    subject = f"sub-{i:02d}"

    surf_path = f"{subjects_dir}/{subject}/bem/inner_skull.surf"
    
    src = mne.setup_volume_source_space(subject, subjects_dir=subjects_dir,
                                    pos=7.0, mri="aparc+aseg.mgz",surface=surf_path,verbose=False)
    
    
    mne.write_source_spaces(f"{subjects_dir}/{subject}/bem/{subject}-vol-7-src.fif", src, overwrite=True,verbose=False)
    print(f"Source created for {subject}: {src}")


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
# # WRITE SRC FSAVERAGE 
# ## force inside inner_skull
#

# %%
"""
subjects_dir = "/Users/maryamvalian/Data/ds005810/derivatives/freesurfer/subjects"
surf = Path(subjects_dir) / "fsaverage2" / "bem" / "inner_skull.surf"

src_fs = mne.setup_volume_source_space(
    subject="fsaverage2",
    subjects_dir=subjects_dir,
    pos=7.0,
    mri="aseg.mgz",
    surface=str(surf),          # constrain to inner skull
)
mne.write_source_spaces(f"{subjects_dir}/fsaverage2/bem/fsaverage-vol-7-src.fif",
                        src_fs, overwrite=True)

#___________________
#befor force: n_used=8925
#after: n_used=5222

"""

# %%
