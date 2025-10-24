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
import eelbrain
import numpy as np
import mne
import os
from eelbrain import load, save, Dataset





mod="dummy"  #common, effect, dummy, ortho (effect with unbalanced trial counts)
rewrite=True
root = Path("~/Data/ds005810")
subjects_dir = str(Path('~/Data/ds005810/derivatives/freesurfer/subjects').expanduser())
fwd_dir=Path("/Users/maryamvalian/Data/ds005810/derivatives/eelbrain/cache/raw")

empty_room=root/"sub-emptyroom/ses-20211114/meg/sub-emptyroom_ses-20211114_task-noise_meg.fif"
raw_er = mne.io.read_raw_fif(empty_room, preload=True, verbose=False).pick('meg')
raw_er.filter(1., 40., phase="zero-double", verbose=False)
raw_er.resample(100, npad="auto", verbose=False)
noise_cov = mne.compute_raw_covariance(raw_er, method='shrunk', rank=None,verbose=False)


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
import os
import concurrent.futures
import mne
from eelbrain import load, save

# Your constants
model_dir = "models/WR"
size = 1

def process_subject_subset(i, subset):
    subject = f"sub-{i:02d}"
    session = "ImageNet04" if i == 7 else "ImageNet03"
    morphed_file = f"{model_dir}/M{subset}-{size}-{subject}-FN.pickle"

    if os.path.exists(morphed_file):
        print(f"{subset}-{size}-{subject} exists.")
        return

    try:
        print(f"Morphing {subset}-{size}-{subject}...")
        modelfile = f"{model_dir}/{subset}-{size}-{subject}-FN.pickle"
        model = load.unpickle(modelfile)
        hlist = model.h

        if mod == "dummy":
            inanim, anim = hlist
        elif mod == "effect":
            h_mean, h_contrast = hlist
            anim = h_mean + h_contrast
            inanim = h_mean - h_contrast
        elif mod == "ortho":
            h_mean, h_contrast = hlist
            anim = h_mean + code_anim * h_contrast
            inanim = h_mean + code_inanim * h_contrast

        fwd_file = f"{fwd_dir}/{subject}_ses-{session}/{subject}-fwd.fif"
        fwd = mne.read_forward_solution(fwd_file, verbose=False)

        stc_vec_anim = ndvar_merged_to_stc_lr(anim, fwd, subject, subjects_dir, "vol-7")
        stc_vec_inanim = ndvar_merged_to_stc_lr(inanim, fwd, subject, subjects_dir, "vol-7")

        print("     1/2")
        _, _, anim_fs = morph_hemi(stc_vec_anim, subject, "fsaverage2", subjects_dir, "vol-7")
        print("     2/2")
        _, _, inanim_fs = morph_hemi(stc_vec_inanim, subject, "fsaverage2", subjects_dir, "vol-7")

        anim = anim_fs.smooth("source", 0.01, "gaussian")
        inanim = inanim_fs.smooth("source", 0.01, "gaussian")

        save.pickle((inanim, anim), morphed_file)
        print(f"\n{modelfile} Saved\n")

    except Exception as e:
        print(f"\n----------- Error processing {subject}: {e}\n")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(process_subject_subset, i, subset)
            for i in range(1, 10)
            for subset in range(1, 3)
        ]
        for f in concurrent.futures.as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print("Worker failed:", e)


# %%
