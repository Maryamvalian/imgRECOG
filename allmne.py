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

# %% [markdown]
# # Create STC ans save model

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

for i in range(1, 31):
    if (i<10):
        run="01"
        session="ImageNet02" 
    elif (i==11) or (i==30): 
        run="01" 
        session="ImageNet01" 
    elif (i==22): 
        run="04" 
        session="ImageNet01"     
    else:
        run="05" 
        session="ImageNet01" 
    
    subject = f"sub-{i:02d}"
    
    epo_file = root_epochs / f"{subject}_meg_epo.fif"
    clean_fif = root / f"derivatives/preprocessed/raw/{subject}_ses-{session}_task-ImageNet_run-{run}_clean_meg.fif"
    fwd_file=fwd_dir / f"{subject}_ses-{session}/{subject}-fwd.fif"
    fwd_file.parent.mkdir(parents=True, exist_ok=True)
    modelfile = f"models/mne/{subject}-mne.pickle"

    if os.path.exists(modelfile):
        print(f"{subject} loaded from file.")
        continue
    try:
        

        if not epo_file.exists():
            print(f" Skipping {subject}: no epo file at {epo_file}")
            continue
    
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
    
    
        src_file = f"{subjects_dir}/{subject}/bem/{subject}-vol-7-src.fif"       #************
        src = mne.read_source_spaces(str(src_file),verbose=False)
    
        bem_sol_fif=f"{subjects_dir}/{subject}/bem/{subject}-bem-sol.fif"
        bem_sol = mne.read_bem_solution(bem_sol_fif,verbose=False)
        
        trans_fif= f"{root}/derivatives/trans/{subject}-{session}-trans.fif"
        trans=mne.read_trans(trans_fif)
    
        if fwd_file.exists():
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
        
        
        print(f"   Morphing 1/2")
        src_fs2 = mne.read_source_spaces(f"{subjects_dir}/fsaverage2/bem/fsaverage2-vol-7-src.fif",verbose=False)
        src_from=fwd['src'] # <==========When mismatch between src and fwd
        
        morph = mne.compute_source_morph(
            
            src=src_from, # <===========
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
                                
            src=src_from, # <==============
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
        
        save.pickle((stc_inan_vec_fs_nd,stc_anim_vec_fs_nd), modelfile)                 
        print(f"=====>{subject } done!")
    except Exception as e:
        print(f"Error processing {subject}: {e}")   
        

# %% [markdown]
# # Group Analysis

# %%
cases = []
for i in range(1, 22):
    
    subject = f"sub-{i:02d}"
    modelfile = Path(f"models/mne/{subject}-mne.pickle")    

    if modelfile.exists():       
    
        inanim, anim = load.unpickle(modelfile)    
        
        cases.append([subject, 'inanimate', inanim])
        cases.append([subject, 'animate', anim])
    else:
        print(f"{subject} Skipped")
    
data = Dataset.from_caselist(['subject', 'animacy', 'stc'], cases)
data.head()

# %% [markdown]
# ## Paired Test

# %%
res = testnd.VectorDifferenceRelated(
    'stc',             
    'animacy',           
    'inanimate',     
    'animate',   
    match='subject',     
    data=data,     
    tfce=True,           
    tstart=0.1,
    tstop=0.7,
    samples=1000
)
save.pickle(res, "Tests/mne_paired_test.pickle")

# %%
diff= res.masked_difference()
p = plot.Butterfly(diff.norm('space'), color='k',title='anim VS inanim')
times = [0.15,0.24,0.3,0.35,0.5,0.6]

for t in times:
    p.add_vline(t)
for t in times:
    f = plot.GlassBrain(diff.sub(time=t),title=f"anim vs inan(MNE), {t}s")  

# %% [markdown]
# # One sample test
#

# %%
data_inan = data.sub("animacy == 'inanimate'")
result_inan = testnd.Vector('stc', match='subject', data=data_inan, tfce=True, tstart=0.1, tstop=0.6,samples=1000)

data_an = data.sub("animacy == 'animate'")
result_an = testnd.Vector('stc', match='subject', data=data_an, tfce=True, tstart=0.1, tstop=0.6,samples=1000)

save.pickle((result_an,result_inan), "Tests/mne_1sampletest.pickle")

# %%
p = plot.Butterfly(result_inan.masked_difference().norm('space'), color='k')
times = [0.13,0.25,0.4]
for t in times:
    p.add_vline(t)
for t in times:
    f = plot.GlassBrain(result_inan.masked_difference().sub(time=t),title=f"Inanimate, {t}s")  

p = plot.Butterfly(result_an.masked_difference().norm('space'), color='k')
times = [0.13,0.25,0.4]
for t in times:
    p.add_vline(t)
for t in times:
    f = plot.GlassBrain(result_an.masked_difference().sub(time=t),title=f"animate, {t}s")     

# %%

# %%

# %%

# %%
