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
import os
from pathlib import Path
import mne
import numpy as np
from ncrf import fit_ncrf
from eelbrain import NDVar, UTS
from utils import *

# %%
# Data locations
root = Path("~/Data/ds005810").expanduser()
subjects_dir = root /"derivatives"/"freesurfer"/"subjects"
fwd_dir = root /"derivatives"/ "eelbrain"/"cache"/"raw"


# Noise Covariance
empty_room=root/"sub-emptyroom/ses-20211114/meg/sub-emptyroom_ses-20211114_task-noise_meg.fif"
raw_er = mne.io.read_raw_fif(empty_room, preload=True, verbose=False).pick('meg')
raw_er.filter(1., 40., phase="zero-double", verbose=False)
raw_er.resample(100, npad="auto", verbose=False)
noise_cov = mne.compute_raw_covariance(raw_er, method='shrunk', rank=None,verbose=False)


# %%
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
    print(f"   Saved FWD to {fwd_file}")

        
    #convert to ndvar
    lf = load.mne.forward_operator(fwd,src='vol-7',subjects_dir=subjects_dir,
                                   adjacency=False,parc='aparc+aseg') 
    lf  = lf.sub(sensor=sensor) 

    return lf


def load_meg_ndvar(subject, session, run):
    
    clean_fif = root / f"derivatives/preprocessed/raw/{subject}_ses-{session}_task-ImageNet_run-{run}_clean_meg.fif"
    clean = mne.io.read_raw_fif(clean_fif, preload=True,verbose=False)
    meg=clean
    meg.filter(1., 40., phase="zero-double", verbose=False)
    meg.resample(100, npad="auto", verbose=False)
    meg_nd = load.fiff.raw_ndvar(meg)
    
    return meg_nd
    

def make_predictors_for_run(meg_ndvar, event_table,mod):
    
    # create stimululs with two different encoding mod : "dummy" or "effect"
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
                
            elif mod=='effect':
                
                stim1[idx] = 1.0
                stim2[idx] = 1.0 if is_anim else -1.0
                    
    stim1 = NDVar(stim1, time)
    stim2 = NDVar(stim2, time)

    return stim1,stim2


def fit_subject(subject, subject_dir,session,runs , mod, model_dir):
      
            modelfile = f"{model_dir}/{subject}-{session}-ncrf.pickle"
            if os.path.exists(modelfile):
                print(f"{mod}-{subject}-{session} model file exists.")
                return
            try:
                
                print(f"computing fwd for {subject}-{session}... ")
                
                clean_fif = root / f"derivatives/preprocessed/raw/{subject}_ses-{session}_task-ImageNet_run-01_clean_meg.fif"
                clean = mne.io.read_raw_fif(clean_fif, preload=False,verbose=False)
                info= clean.info           
                meg_ndvar = load.fiff.raw_ndvar(clean)
                sensor=meg_ndvar.sensor
                fwd = compute_fwd_ndvar(subject, session,subjects_dir,info,sensor)
        
                meg_all = []
                stim_all = []
                for run in runs:
                    
                    print(f"Loading run-{run} MEG ...")
                    meg= load_meg_ndvar(subject, session, run)
                    meg_all.append(meg)
                    event_table= make_event_table(subject, session, run)
                    stim1,stim2= make_predictors_for_run(meg, event_table,mod=mod)
                    predictors=[stim1,stim2]
                    stim_all.append(predictors)  
                    
                args = (meg_all , stim_all, fwd , noise_cov , 0,0.7)
                kwargs = {'normalize': 'l1','in_place': False,'mu':'auto',
                          'verbose': True,'n_iter': 10,'n_iterc': 10,'n_iterf': 100}        
                model = fit_ncrf(*args, **kwargs)  
                save.pickle(model, modelfile)
                print(f"\n {subject} Model saved to {modelfile}\n")  
                
            except Exception as e:
             print(f"\n----------- Error processing {subject}: {e}\n")  


def morph_subject(subject, subject_dir,session,mod, model_dir):
    
                    
            morphed_file = f"{model_dir}/M{subject}-{session}-ncrf.pickle"  # Morphed
            if os.path.exists(morphed_file):
                print(f"Morphed {mod} {subject}-{session} Exists.")
            else:
                try:
                    
                    print(f"Morphing {subject}-{session}...")
                    modelfile = f"{model_dir}/{subject}-{session}-ncrf.pickle"
                    model = load.unpickle(modelfile)
                    hlist = model.h
                    inanim,anim = hlist[0],hlist[1]       # if mod is effect : anim: contrast, inanim: general
                     
                    #morph  
                    fwd_file=fwd_dir / f"{subject}_ses-{session}/{subject}-fwd.fif"
                    fwd = mne.read_forward_solution(str(fwd_file), verbose=False)
        
                    stc_vec_anim = ndvar_merged_to_stc_lr(
                        
                        ndvar=anim,
                        fwd=fwd,
                        subject=subject,
                        subjects_dir=subjects_dir,
                        src_tag="vol-7",
                    )
                    stc_vec_inanim = ndvar_merged_to_stc_lr(
                        
                        ndvar=inanim,
                        fwd=fwd,
                        subject=subject,
                        subjects_dir=subjects_dir,
                        src_tag="vol-7",
                    )
                
                    print("     1/2")        
                    _, _, anim_fs = morph_hemi(
                        stc_vec_anim,
                        subject=subject,
                        subject_to="fsaverage2",
                        subjects_dir=subjects_dir,
                        src_tag="vol-7",
                    )
                    
                    print("     2/2")        
                    _, _, inanim_fs = morph_hemi(
                        stc_vec_inanim,
                        subject=subject,
                        subject_to="fsaverage2",
                        subjects_dir=subjects_dir,
                        src_tag="vol-7",
                    )
                                   
                    
                    anim= anim_fs.smooth('source', 0.01, 'gaussian')
                    inanim= inanim_fs.smooth('source', 0.01, 'gaussian')
                
                    save.pickle((inanim, anim), morphed_file)                               
                    print(f"\n{subject}-{session}-Morphed Saved \n ")
                    
                except Exception as e:
                    print(f"\n----------- Error processing {subject}: {e}\n")   
    

# %% [markdown]
# # Main

# %%
for i in range(1, 31):          # Subjects                              
        if (i>9):
            sessions=["ImageNet01"]             # all subjects>9 have 1 session of length 5 runs.
            lastruns = [5]                     #run01,run02,..,run05      
            
        else:
            
            sessions = ["ImageNet01","ImageNet02","ImageNet03","ImageNet04"] 
            lastruns= [2,2,8,8]      
            
        subject = f"sub-{i:02d}"
        for idx, session in enumerate(sessions):
            
            lastrun= lastruns[idx]          
            runs = [f"{i:02d}" for i in range(1, lastrun+1)] 
            
            fit_subject(subject=subject, subject_dir=subjects_dir, mod="dummy",session= session, runs= runs, model_dir="models/all_runs//ncrf-dc")
            fit_subject(subject=subject, subject_dir=subjects_dir, mod="effect",session= session, runs= runs, model_dir="models/all_runs//ncrf-ec")

            morph_subject(subject=subject, subject_dir=subjects_dir, mod="dummy",session= session, model_dir="models/all_runs//ncrf-dc")
            morph_subject(subject=subject, subject_dir=subjects_dir, mod="effect",session= session, model_dir="models/all_runs//ncrf-ec")

# %%

# %%

# %%
