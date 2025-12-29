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
import numpy as np
#from mne import *
from ncrf import fit_ncrf
from eelbrain import NDVar, UTS
from eelbrain import plot, combine
from eelbrain import *
from eelbrain._data_obj import VolumeSourceSpace
import os
from pathlib import Path
from Beyond import *
import eelbrain as eb
import seaborn as sns
import random

# %%
mod="dummy"  #common, effect, dummy, ortho (effect with unbalanced trial counts)
rewrite=True
root = Path("~/Data/ds005810")
subjects_dir = str(Path('~/Data/ds005810/derivatives/freesurfer/subjects').expanduser())
fwd_dir=Path("/Users/maryamvalian/Data/ds005810/derivatives/eelbrain/cache/raw")
model_dir = "models/consist"

# noise
empty_room=root/"sub-emptyroom/ses-20211114/meg/sub-emptyroom_ses-20211114_task-noise_meg.fif"
raw_er = mne.io.read_raw_fif(empty_room, preload=True, verbose=False).pick('meg')
raw_er.filter(1., 40., phase="zero-double", verbose=False)
raw_er.resample(100, npad="auto", verbose=False)
noise_cov = mne.compute_raw_covariance(raw_er, method='shrunk', rank=None,verbose=False)

#functions
def compute_fwd_ndvar(subject, session,subject_dir,meg_info,sensor):
    

    src_file = f"{subjects_dir}/{subject}/bem/{subject}-vol-7-src.fif"
    src = mne.read_source_spaces(str(src_file),verbose=False)
    
    bem_sol_fif=f"{subjects_dir}/{subject}/bem/{subject}-bem-sol.fif"
    bem_sol = mne.read_bem_solution(bem_sol_fif,verbose=False)
    
    
    trans_fif= f"{root}/derivatives/trans/{subject}-{session}-trans.fif"
    trans=mne.read_trans(trans_fif)

    fwd = mne.make_forward_solution(meg_info, trans, src, bem_sol,
                                    meg=True, eeg=False, mindist=0, verbose=False)
    fwd_file=fwd_dir / f"{subject}_ses-{session}/{subject}-fwd.fif"
    fwd_file.parent.mkdir(parents=True, exist_ok=True)
    mne.write_forward_solution(str(fwd_file), fwd, overwrite=True, verbose=False)
    #print(f"   Saved FWD to {fwd_file}")

        
    #convert to ndvar
    lf = load.mne.forward_operator(fwd,src='vol-7',subjects_dir=subjects_dir,
                                   adjacency=False,parc='aparc+aseg') 
    lf  = lf.sub(sensor=sensor)  
    #print(len(src[0]['vertno']))           #check sanity
    #print(len(fwd['src'][0]['vertno']))  


    return lf
    
#-----------------------------------------------

def load_meg_ndvar(subject, session, run):
    
    clean_fif = root / f"derivatives/preprocessed/raw/{subject}_ses-{session}_task-ImageNet_run-{run}_clean_meg.fif"
    clean = mne.io.read_raw_fif(clean_fif, preload=True,verbose=False)
    meg=clean
    meg.filter(1., 40., phase="zero-double", verbose=False)
    meg.resample(100, npad="auto", verbose=False)
    meg_nd = load.fiff.raw_ndvar(meg)
    
    return meg_nd


 
#-----------------------------------------

def make_predictors_for_run(meg_ndvar, event_table,mod):

    sfreq = 100  #meg_ndvar.info['sfreq']     #100       
    n_times = len(meg_ndvar.time)        #32000     
    
    time = UTS(0, 1/sfreq, n_times)      #from 0 to 32000/100=320 s
    stim1 = np.zeros(n_times, dtype=float)  
    stim2 = np.zeros(n_times, dtype=float) 
    
    for t, is_anim in zip(event_table['time'], event_table['animate']):
        idx = int(round((t ) * sfreq))
        if 0 <= idx < n_times:
            if mod=='dummy':
                
                if is_anim:
                    stim2[idx]= 1.0
                else:
                    stim1[idx]= 1.0
                    
            elif mod=='common':
                
                stim1[idx] = 1.0
                
            elif mod=='effect':
                
                stim1[idx] = 1.0
                stim2[idx] = 1.0 if is_anim else -1.0

            elif mod=='ortho':
                
                n_a =int(event_table['animate'].sum())
                n_i =int((~event_table['animate']).sum())
                N=n_a+n_i
                code_anim,code_inanim = (n_i / N),-(n_a / N)
                stim1[idx]= 1.0
                stim2[idx]= code_anim if is_anim else code_inanim
                    
    stim1 = NDVar(stim1, time)
    stim2 = NDVar(stim2, time)

    return stim1,stim2
    


# %%
for i in range (1,2):                # one subject
    session="ImageNet03"
    subject = f"sub-{i:02d}"
    subset=["02", "05" ,"03"]
    
    print(f"computing fwd for {subject}-{session}... ")
    clean_fif = root / f"derivatives/preprocessed/raw/{subject}_ses-{session}_task-ImageNet_run-01_clean_meg.fif"
    clean = mne.io.read_raw_fif(clean_fif, preload=False,verbose=False)
    info= clean.info           
    meg_ndvar = load.fiff.raw_ndvar(clean)
    sensor=meg_ndvar.sensor
    fwd = compute_fwd_ndvar(subject, session,subjects_dir,info,sensor)
    
    
    
    modelfile = f"models/reduce/{subject}_ncrf.pickle"
    if os.path.exists(modelfile):
        print(f"{subject} model file exists.")
        continue
    try:        
        
        meg_all = []
        stim_all = []
        for run in subset:
            
            print(f"Loading run-{run} MEG ...")
            meg= load_meg_ndvar(subject, session, run)
            meg_all.append(meg)
            event_table= make_event_table(subject, session, run)
            stim1,stim2= make_predictors_for_run(meg, event_table,mod=mod)
            predictors=[stim1,stim2]
            stim_all.append(predictors)  
            
        args = (meg_all , stim_all, fwd , noise_cov , 0,0.7)
        kwargs = {'normalize': 'l1','in_place': False,'mu':1e-3,
                  'verbose': True,'n_iter': 5,'n_iterc':5,'n_iterf': 10}        
        model = fit_ncrf(*args, **kwargs)  
        save.pickle(model, modelfile)
        print(f"\nModel saved to {modelfile}\n")  
        
    except Exception as e:
     print(f"\n----------- Error processing {subject}: {e}\n")   

# %%
model.mu

# %%
model._data

# %%
type(model._data)

# %%
import pickle
for attr in model._PICKLE_ATTRS:
    s = pickle.dumps(getattr(model, attr))
    print(len(s), ' ', attr)

# %%
model._data.__dict__.keys()


# %%
type(model._data._bE)     #b^T E   : B time-lagged stimulus matrix, E whitened MEG for each run

# %%
len(model._data._bE)      # be tedade run-ha hast subset = 2,5,3

# %%
x = model._data._bE[0]             #First run
x.shape ,x.dtype

# %% [markdown]
# # It is not necessary to store BE
# 1. Build RegressionData(meg_all, stim_all, ...)
# 2. That constructor recomputes:
#    - covariates
#    - whitened MEG
#    - BᵀE → _bE
#    - BᵀB → _bbt
#    - EᵀE → _EtE

# %%
what=model._data.tstart
what

# %%
len(model._data.meg)

# %%
len(model._data.covariates)

# %%
len(model._data.tstart) , len(model._data.tstop)    # Per predictor ehtemalan


# %%
model._data.tstep

# %%
len(model._data._bE)

# %%
len(model._data._stim_names) , model._data._stim_names[0] ,model._data._stim_names[0]

# %%
model._data.nlevel

# %%
