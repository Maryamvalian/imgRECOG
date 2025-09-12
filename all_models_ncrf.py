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
root = Path("~/Data/ds005810")
subjects_dir = str(Path('~/Data/ds005810/derivatives/freesurfer/subjects').expanduser())

empty_room=root/"sub-emptyroom/ses-20211114/meg/sub-emptyroom_ses-20211114_task-noise_meg.fif"

# %%

for i in range(1, 10):
    if (i<10):
        run="01"
        session="ImageNet02" 
    else:
        run="04"
        session="ImageNet01" 
    
    subject = f"sub-{i:02d}"
    modelfile = f"clean_models/ncrf/{subject}.pickle"
    raw_fif=root/f"{subject}/ses-{session}/meg/{subject}_ses-{session}_task-ImageNet_run-{run}_meg.fif"
    clean_fif = root / f"derivatives/preprocessed/raw/{subject}_ses-{session}_task-ImageNet_run-{run}_meg_clean.fif"
    
    if os.path.exists(modelfile):
        print(f"{subject} loaded from file.")
        continue
    try:
        raw = mne.io.read_raw_fif(raw_fif, preload=True,verbose=False)
        clean = mne.io.read_raw_fif(clean_fif, preload=True,verbose=False)
        events = find_events(raw, stim_channel="UPPT001")
        stim = events[events[:, 2] == 2]   #ID2 : stim_on
        stim_samples = stim[:, 0]
        
        stim_times = stim_samples / raw.info["sfreq"]
        meta = pd.read_csv(f"/Users/maryamvalian/Data/ds005810/derivatives/detailed_events/{subject}_events.csv")
        meta_sub = meta[(meta['session'] == session) & (meta['run'] ==int(run))].reset_index(drop=True)
        n=len(meta_sub)
        anim_flags = meta_sub['stim_is_animate'].astype(str).str.lower().eq("true").to_numpy()
        event_table = pd.DataFrame({  "time": stim_times,  "animate": anim_flags })
        print(event_table)
        
        meg=clean
        meg.filter(1., 40., phase="zero-double", verbose=False)
        meg.resample(100, npad="auto", verbose=False)
        meg_ndvar = load.fiff.raw_ndvar(meg)

        
        
        sfreq = meg.info['sfreq']     #100       
        n_times = meg.n_times         #32000     
        
        

        time = UTS(0, 1/sfreq, n_times)      #from 0 to 32000/100=320 s
        stim1 = np.zeros(n_times, dtype=float)  
        stim2 = np.zeros(n_times, dtype=float) 
        
        for t, is_anim in zip(event_table['time'], event_table['animate']):
            idx = int(round((t ) * sfreq))
            if 0 <= idx < n_times:
                if is_anim:
                    stim2[idx] = 1.0
                else:
                    stim1[idx] = 1.0
        stim1 = NDVar(stim1, time)
        stim2 = NDVar(stim2, time)
        #p=plot.LineStack(combine([stim1, stim2]), ylabels=["inanimate", "animate"], offset=1.5)

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
        
        fwd = mne.make_forward_solution(meg.info,trans ,
                                        src, bem_sol,
                                        meg=True, eeg=False,
                                        mindist=0,               #======same size with src
                                        verbose=False
                                       )
        
        #convert fwd to ndvar
        lf = load.mne.forward_operator(fwd,src='vol-7',
                                       subjects_dir=subjects_dir,
                                       adjacency=False,parc='aparc+aseg') 
        lf  = lf.sub(sensor=meg_ndvar.sensor)  

        print(len(src[0]['vertno']))
        print(len(fwd['src'][0]['vertno']))  

        print(f"Fitting NCRF")
        args = (
                    meg_ndvar,
                    [stim1,stim2],
                    lf,
                    noise_cov,
                    0,
                    0.8,
                )
        kwargs = {'normalize': 'l1',
                          'in_place': False,
                          'mu':'auto',
                          'verbose': True, 
                          'n_iter': 10,
                          'n_iterc': 10,
                          'n_iterf': 100} 
        model = fit_ncrf(*args, **kwargs)  
        
        #---------------
        
        save.pickle(model, modelfile)
        
        hlist=model.h
        print(f"==================>{subject} done!")
        
    except Exception as e:
         print(f"Error processing {subject}: {e}")                 


# %%
