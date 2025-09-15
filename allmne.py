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
from morph_nd import morph_nd

# %%
run="01"
session="ImageNet01"            #1-9 :Net02, 10-31:Net01 run4

root = Path("~/Data/ds005810")
subjects_dir = str(Path('raw = mne.io.read_raw_fif(raw_fif, preload=True, verbose=False)/derivatives/freesurfer/subjects').expanduser())



empty_room=root/"sub-emptyroom/ses-20211114/meg/sub-emptyroom_ses-20211114_task-noise_meg.fif"

# %%
run="04"
session="ImageNet01" 
for i in range(1, 31):
    if (i == 28) or (i==11):            #corrupted MEGs in all runs
        continue
    subject = f"sub-{i:02d}"
    modelfile = f"clean_models/{subject}.pickle"
    raw_fif=root/f"{subject}/ses-{session}/meg/{subject}_ses-{session}_task-ImageNet_run-{run}_meg.fif"
    clean_fif = root / f"derivatives/preprocessed/raw/{subject}_ses-{session}_task-ImageNet_run-{run}_meg_clean.fif"

    
    if os.path.exists(modelfile):
        print(f"{subject} loaded from file.")
        continue
    try:
        print(f"Start {subject}...")
        raw = mne.io.read_raw_fif(raw_fif, preload=True, verbose=False)
        events = mne.find_events(raw, stim_channel='UPPT001')
        
        stim = events[events[:, 2] == 2]   #ID2 : stim_on
        stim_samples = stim[:, 0]
        
        stim_times = stim_samples / raw.info["sfreq"]
        meta = pd.read_csv(f"/Users/maryamvalian/Data/ds005810/derivatives/detailed_events/{subject}_events.csv")
        meta_sub = meta[(meta['session'] == session) & (meta['run'] ==int(run))].reset_index(drop=True)
        n=len(meta_sub)
        anim_flags = meta_sub['stim_is_animate'].astype(str).str.lower().eq("true").to_numpy()
        event_table = pd.DataFrame({  "time": stim_times,  "animate": anim_flags })
        #print(event_table)
        
        raw_er = mne.io.read_raw_fif(empty_room, preload=True, verbose=False).pick('meg')
        raw_er.filter(1., 40., phase="zero-double", verbose=False)
        raw_er.resample(100, npad="auto", verbose=False)
        noise_cov = mne.compute_raw_covariance(raw_er, method='shrunk', rank=None,verbose=False)
        
        
        src_file = f"{subjects_dir}/{subject}/bem/{subject}-vol-7-src.fif"
        src = mne.read_source_spaces(str(src_file),verbose=False)
        
        bem_sol_fif=f"{subjects_dir}/{subject}/bem/{subject}-bem-sol.fif"
        bem_sol = mne.read_bem_solution(bem_sol_fif,verbose=False)
        
        
        trans_fif= f"{root}/derivatives/trans/{subject}-{session}-trans.fif"
        trans=mne.read_trans(trans_fif)
        
        meg=raw.pick("meg")
        meg.filter(1., 40., phase="zero-double", verbose=False)
        meg.resample(100, npad="auto", verbose=False)
        
        print("   Computing FWD...")
        fwd = mne.make_forward_solution(meg.info,trans ,
                                        src, bem_sol,
                                        meg=True, eeg=False,
                                        mindist=0,               #======same size with src
                                        verbose=False
                                       )
        
        raw = mne.io.read_raw_fif(raw_fif, preload=True, verbose=False)
        
        raw.filter(1., 40., phase='zero-double', verbose=False)
        
        rs_raw,rs_events = raw.resample(100, npad='auto', events=events, stim_picks='UPPT001', verbose=False)
        
        epochs = mne.Epochs(
            rs_raw, rs_events, event_id={'stim_on': 2}, tmin=-0.1, tmax=0.8, verbose=False
        )
        
        
        
        anim_flags = meta_sub['stim_is_animate'].astype(str).str.lower().eq('true').to_numpy()
        idx_anim = anim_flags
        idx_inanim = ~anim_flags
        
        
        evoked_anim   = epochs[idx_anim].average()
        evoked_inanim = epochs[idx_inanim].average()
        
        print("   Inverse...")
        inv = mne.minimum_norm.make_inverse_operator(
            info=rs_raw.info,
            forward=fwd,
            noise_cov=noise_cov,
            loose=1.0,     # free orientation
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
        
        #________morph______
        print(f"   Morphing 1/2")
        src_fs2 = mne.read_source_spaces(f"{subjects_dir}/fsaverage2/bem/fsaverage2-vol-7-src.fif",verbose=False)
        
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
        
        save.pickle((stc_inan_vec_fs_nd,stc_anim_vec_fs_nd), modelfile)                 #morphed to fsaverage2 stc saved
        print(f"=====>{subject } done!")



             
    except Exception as e:
         print(f"Error processing {subject}: {e}")                 



# %% [markdown]
# # Group Analysis

# %%
cases = []
for i in range(1, 21):
    if (i == 11) or (i==21):           
        continue
    subject = f"sub-{i:02d}"
    modelfile = f"stc/{subject}.pickle"    

    inanim, anim = load.unpickle(modelfile)    
    
    cases.append([subject, 'inanimate', inanim])
    cases.append([subject, 'animate', anim])
    
data = Dataset.from_caselist(['subject', 'animacy', 'stc'], cases)
data.tail()

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

# %%
diff= res.masked_difference()
p = plot.Butterfly(diff.norm('space'), color='k',title='anim VS inanim')
times = [0.35,0.45,0.48,0.52,0.6]
for t in times:
    p.add_vline(t)
for t in times:
    f = plot.GlassBrain(diff.sub(time=t),title=f"anim vs inan(MNE), {t}s")  

# %%
