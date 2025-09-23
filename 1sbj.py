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
from eelbrain import *

from eelbrain import NDVar
from eelbrain._data_obj import VolumeSourceSpace
import os

# %%
i=8
subject = f"sub-{i:02d}"
session="ImageNet02"
run="01"
recompute=True

# %%
root = Path("~/Data/ds005810")
subjects_dir = str(Path('~/Data/ds005810/derivatives/freesurfer/subjects').expanduser())
empty_room=root/"sub-emptyroom/ses-20211114/meg/sub-emptyroom_ses-20211114_task-noise_meg.fif"


#Noise
raw_er = mne.io.read_raw_fif(empty_room, preload=True, verbose=False).pick('meg')
raw_er.filter(1., 40., phase="zero-double", verbose=False)
raw_er.resample(100, npad="auto", verbose=False)
noise_cov = mne.compute_raw_covariance(raw_er, method='shrunk', rank=None,verbose=False)

# %%
root_epochs = Path("/Users/maryamvalian/Data/ds005810/derivatives/preprocessed/epochs")
fwd_dir=Path("/Users/maryamvalian/Data/ds005810/derivatives/eelbrain/cache/raw")
fwd_dir.mkdir(parents=True, exist_ok=True)

# %%
epo_file = root_epochs / f"{subject}_meg_epo.fif"
clean_fif = root / f"derivatives/preprocessed/raw/{subject}_ses-{session}_task-ImageNet_run-{run}_clean_meg.fif"
fwd_file=fwd_dir / f"{subject}_ses-{session}/{subject}-fwd.fif"
fwd_file.parent.mkdir(parents=True, exist_ok=True)
modelfile = f"models/mne/{subject}-mne.pickle"



print(f"Loading {epo_file} ...")
epochs = mne.read_epochs(str(epo_file), preload=True, verbose=False)
clean = mne.io.read_raw_fif(clean_fif, preload=True, verbose=False)


epochs_resamp = epochs.copy().resample(100, npad="auto", verbose=False)
meta = epochs_resamp.metadata

    
mask_in = (
    (meta["subject"] == i) &
    (meta["session"] == session) &
    (meta["run"] == int(run))&
    (meta["stim_is_animate"] ==False)
)

epochs_in = epochs_resamp[mask_in]
print(f"#Inanim={len(epochs_in)}")


mask_an = (
    (meta["subject"] == i) &
    (meta["session"] == session) &
    (meta["run"] == int(run))&
    (meta["stim_is_animate"] ==True)
)

epochs_an = epochs_resamp[mask_an]
print(f"#Animate={len(epochs_an)}")

evoked_anim   = epochs_an.average()
evoked_inanim = epochs_in.average()


src_file = f"{subjects_dir}/{subject}/bem/{subject}-vol-7-src.fif"       
src = mne.read_source_spaces(str(src_file),verbose=False)

bem_sol_fif=f"{subjects_dir}/{subject}/bem/{subject}-bem-sol.fif"
bem_sol = mne.read_bem_solution(bem_sol_fif,verbose=False)

trans_fif= f"{root}/derivatives/trans/{subject}-{session}-trans.fif"
trans=mne.read_trans(trans_fif)

if fwd_file.exists() and not(recompute):
    print(" Loading FWD ")
    fwd = mne.read_forward_solution(str(fwd_file), verbose=False)
else:
    print(" Computing FWD...")
    fwd = mne.make_forward_solution(
        clean.info, trans, src, bem_sol,
        meg=True, eeg=False, mindist=0, verbose=False
    )
    mne.write_forward_solution(str(fwd_file), fwd, overwrite=True, verbose=False)
    print(f"   Saved FWD to {fwd_file}")

print("   Inverse...")
inv = mne.minimum_norm.make_inverse_operator(
    info=clean.info,
    forward=fwd,
    noise_cov=noise_cov,
    loose=1.0,    
    depth=0.8,
    fixed=False,
    verbose=False,
)

snr = 3.0
lambda2 = 1.0 / snr**2


stc_anim_vec = mne.minimum_norm.apply_inverse(
    evoked_anim, inv, lambda2=lambda2, method='MNE', pick_ori='vector', verbose=False)

stc_inan_vec = mne.minimum_norm.apply_inverse(
    evoked_inanim, inv, lambda2=lambda2, method='MNE', pick_ori='vector', verbose=False)





# %%
src_fs2 = mne.read_source_spaces(f"{subjects_dir}/fsaverage2/bem/fsaverage2-vol-7-src.fif",verbose=False)
src = mne.read_source_spaces(f"{subjects_dir}/{subject}/bem/{subject}-vol-7-src.fif",verbose=False)
src_fs2,src

# %%
print(f"   Morphing 1/2")
#src_from=fwd['src'] # <When mismatch between src and fwd

morph = mne.compute_source_morph(
    
    src=src, 
    subject_from=subject,
    subject_to="fsaverage2",     
    subjects_dir=subjects_dir,
    spacing=7.0, 
    src_to=src_fs2,                                     
    precompute=True,
    verbose=False,
)
stc_anim_vec_fs = morph.apply(stc_anim_vec)

print(f"   Morphing 2/2")
morph = mne.compute_source_morph(
                        
    src=src, 
    subject_from=subject,
    subject_to="fsaverage2",     
    subjects_dir=subjects_dir,
    spacing=7.0, 
    src_to=src_fs2,                                    
    precompute=True,
    verbose=False,
    )
stc_inan_vec_fs = morph.apply(stc_inan_vec)

stc_inan_vec_fs_nd = load.mne.stc_ndvar(
    stc_inan_vec_fs,
    src='vol-7',                  
    subjects_dir=subjects_dir,
    subject='fsaverage2')
stc_anim_vec_fs_nd = load.mne.stc_ndvar(
    stc_anim_vec_fs,
    src='vol-7',                  
    subjects_dir=subjects_dir,
    subject='fsaverage2')
print("Morphing Finished")   

# %%
p = plot.Butterfly(stc_anim_vec_fs_nd.norm('space'), color='k',title='anim ')
times = [0.15,0.24,0.3,0.35,0.5,0.6]

for t in times:
    p.add_vline(t)
for t in times:
    f = plot.GlassBrain(stc_anim_vec_fs_nd.sub(time=t),title=f"anim ), {t}s")  
   

# %% [markdown]
# # SEPERATE HEMI

# %%
#vol-7-lr-src: LH and RH *****not merged"""" 

src_file = f"{subjects_dir}/{subject}/bem/{subject}-vol-7-lr-src.fif"       
src_lr = mne.read_source_spaces(str(src_file),verbose=False)


if fwd_file.exists() and not(recompute):
    print(" Loading FWD ")
    fwd = mne.read_forward_solution(str(fwd_file), verbose=False)
else:
    print(" Computing FWD...")
    fwd = mne.make_forward_solution(
        clean.info, trans, src_lr, bem_sol,
        meg=True, eeg=False, mindist=0, verbose=False
    )
    mne.write_forward_solution(str(fwd_file), fwd, overwrite=True, verbose=False)
    print(f"   Saved FWD to {fwd_file}")

print("   Inverse...")
inv = mne.minimum_norm.make_inverse_operator(
    info=clean.info,
    forward=fwd,
    noise_cov=noise_cov,
    loose=1.0,    
    depth=0.8,
    fixed=False,
    verbose=False,
)

snr = 3.0
lambda2 = 1.0 / snr**2


stc_anim_vec = mne.minimum_norm.apply_inverse(
    evoked_anim, inv, lambda2=lambda2, method='MNE', pick_ori='vector', verbose=False)

stc_inan_vec = mne.minimum_norm.apply_inverse(
    evoked_inanim, inv, lambda2=lambda2, method='MNE', pick_ori='vector', verbose=False)
print("inverse done!")

# %%
# Split STC by hemi
src_L = mne.read_source_spaces(f"{subjects_dir}/{subject}/bem/{subject}-vol-7-L-src.fif",verbose=False)

print("Animate")
nL = len(src_L[0]['vertno'])  
X  = stc_anim_vec.data        
vL, vR = stc_anim_vec.vertices  

stc_L = mne.VolVectorSourceEstimate(X[:nL],   [vL], stc_anim_vec.tmin, stc_anim_vec.tstep, stc_anim_vec.subject)
stc_R = mne.VolVectorSourceEstimate(X[nL:],   [vR], stc_anim_vec.tmin, stc_anim_vec.tstep, stc_anim_vec.subject)


print("morphing..")

src_fs2_L = mne.read_source_spaces(f"{subjects_dir}/fsaverage2/bem/fsaverage2-vol-7-L-src.fif",verbose=False)


morph =mne.compute_source_morph(
    
    src=src_L,
    subject_from=subject,
    subject_to="fsaverage2",     
    subjects_dir=subjects_dir,
    spacing=7.0, 
    src_to=src_fs2_L,                                     
    precompute=True,
    verbose=False,
)
an_l_fs = morph.apply(stc_L)
print("    LH morphed")

src_fs2_R = mne.read_source_spaces(f"{subjects_dir}/fsaverage2/bem/fsaverage2-vol-7-R-src.fif",verbose=False)
src_R = mne.read_source_spaces(f"{subjects_dir}/{subject}/bem/{subject}-vol-7-R-src.fif",verbose=False)

morph = mne.compute_source_morph(
    
    src=src_R,
    subject_from=subject,
    subject_to="fsaverage2",     
    subjects_dir=subjects_dir,
    spacing=7.0, 
    src_to=src_fs2_R,                                     
    precompute=True,
    verbose= False,
)
an_R_fs = morph.apply(stc_R)
print("    RH morphed")
#convert to ndvar morphed stc
an_L= load.mne.stc_ndvar(an_l_fs, src='vol-7-L', subjects_dir=subjects_dir, subject="fsaverage2")
an_R=load.mne.stc_ndvar(an_R_fs, src='vol-7-R', subjects_dir=subjects_dir, subject="fsaverage2")

# %%
p = plot.Butterfly(an_L.norm('space'), color='k',title='')
times = [0.15,0.24,0.3,0.35,0.5,0.6]

for t in times:
    p.add_vline(t)
for t in times:
    f = plot.GlassBrain(an_L.sub(time=t),title=f"anim LH), {t}s")

p = plot.Butterfly(an_R.norm('space'), color='k',title='')
times = [0.15,0.24,0.3,0.35,0.5,0.6]

for t in times:
    p.add_vline(t)
for t in times:
    f = plot.GlassBrain(an_R.sub(time=t),title=f"anim Rh(MNE), {t}s")  
    

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
