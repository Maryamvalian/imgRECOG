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

# %%
mod="dummy"  #common, effect, dummy, ortho (effect with unbalanced trial counts)
rewrite=True
root = Path("~/Data/ds005810")
subjects_dir = str(Path('~/Data/ds005810/derivatives/freesurfer/subjects').expanduser())
fwd_dir=Path("/Users/maryamvalian/Data/ds005810/derivatives/eelbrain/cache/raw")


# %%
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
import random
from typing import List, Dict

def independent_subsets(runs: List[str], k_max: int, n_repeats: int = 2, seed: int = 11) -> Dict[int, List[List[str]]]:
    """
    Make random subsets for each size k = 1..k_max.
    - Each subset has k distinct runs (sampled without replacement).
    - Subsets for different k are independent (not nested).
    """
    rng = random.Random(seed)
    out: Dict[int, List[List[str]]] = {}
    for k in range(1, k_max + 1):
        out[k] = [rng.sample(runs, k) for _ in range(n_repeats)]
    return out


# %%
runs = [f"0{i}" for i in range(1, 9)]
runs

# %%
len(runs)

# %%
subs = independent_subsets(runs, k_max=8, n_repeats=1, seed=11)          #k_max=len(runs)

# %%
subs

# %%
subject="sub-05"
session="ImageNet03"

# %%
#FWD for Session
print(f"computing fwd for {subject}-{session}... ")
clean_fif = root / f"derivatives/preprocessed/raw/{subject}_ses-{session}_task-ImageNet_run-01_clean_meg.fif"
clean = mne.io.read_raw_fif(clean_fif, preload=False,verbose=False)
info= clean.info           
meg_ndvar = load.fiff.raw_ndvar(clean)
sensor=meg_ndvar.sensor
fwd = compute_fwd_ndvar(subject, session,subjects_dir,info,sensor)



for size in range (1,8):            #(1,9)
    subset=subs[size][0]
    #print(f"{subset}")
    modelfile = f"models/consistency/{size}-{subject}-{session}-ncrf.pickle"
    if os.path.exists(modelfile):
        print(f"{size}-{subject}-{session} model file exists.")
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
        kwargs = {'normalize': 'l1','in_place': False,'mu':'auto',
                  'verbose': True,'n_iter': 10,'n_iterc': 10,'n_iterf': 100}        
        model = fit_ncrf(*args, **kwargs)  
        save.pickle(model, modelfile)
        print(f"\nModel saved to {modelfile}\n")  
        
    except Exception as e:
     print(f"\n----------- Error processing {subject}: {e}\n")   

# %% [markdown]
# # morphing
#

# %%
for size in range (1,9):
    
    morphed_file = f"models/consistency/M{size}-{subject}-{session}-ncrf.pickle"  #M stands for Morphed
    if os.path.exists(morphed_file):
        
        print(f"Loading{size}-{subject}-{session} from file.")
        #inanim, anim = load.unpickle(morphed_file)
    else:
        try:
                
            print(f"Morphing {size}-{subject}-{session}...")
            modelfile = f"models/consistency/{size}-{subject}-{session}-ncrf.pickle"
            model= load.unpickle(modelfile)
            hlist = model.h
            
            if mod=="dummy":
                inanim = hlist[0]
                anim = hlist[1]
            elif mod=="effect":
                h_mean,h_contrast = hlist[0],hlist[1]
                anim = h_mean+ h_contrast
                inanim = h_mean- h_contrast 
            
            elif mod=="ortho":
                h_mean,h_contrast= hlist[0],hlist[1]
                anim = h_mean+ code_anim* h_contrast
                inanim = h_mean+ code_inanim* h_contrast
            
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
            an_L_fs, an_R_fs, anim_fs = morph_hemi(
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
            print(f"\n{size}-{subject}-{session}-Morphed Saved \n ")
    
        except Exception as e:
            print(f"\n----------- Error processing {subject}: {e}\n")   

# %% [markdown]
# # Plot morphed model

# %%
size=7

morphed_file = f"models/consistency/M{size}-{subject}-{session}-ncrf.pickle" 
inan, an = load.unpickle(morphed_file)

p = plot.Butterfly(an.norm('space'), color='k',title=f"Anim Size={size}")
times = [120,170,240,370,480]
for t in times:
    p.add_vline(t)
for t in times:
    f = plot.GlassBrain(an.sub(time=t),title=f"{size}-{subject}{session}, anim (NCRF), {t}ms") 
    
p = plot.Butterfly(inan.norm('space'), color='k',title=f" Inanim Size={size}")
times = [120,170,240,370,480]
for t in times:
    p.add_vline(t)
for t in times:
    f = plot.GlassBrain(inan.sub(time=t),title=f"{size}-{subject}{session}, inanim (NCRF), {t}ms") 


# %%

# %%
import os
import numpy as np
import matplotlib.pyplot as plt


subject = "sub-05"
session = "ImageNet03"
model_dir = "models/consistency"
sizes = list(range(1, 9))  # 1..8
TRIALS_PER_RUN = 200  

#-------------------------------------------------------
def load_model(size):
    
    morphed_file = f"{model_dir}/M{size}-{subject}-{session}-ncrf.pickle"
    inan, anim = load.unpickle(morphed_file)
    return inan, anim
#---------------------------

def ndvar_cosine(nd1, nd2, eps=1e-12):
    
    B = np.asarray(nd2.get_data())
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: {A.shape} vs {B.shape}")
    a= A.reshape(-1).astype(float)
    b =B.reshape(-1).astype(float)
    na =np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return np.nan
    return float(np.dot(a, b) / (na * nb))

def compute_pairwise_cosine(models):
    
    n = len(models)
    M = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        M[i, i] = 1.0
        for j in range(i + 1, n):
            c = ndvar_cosine(models[i], models[j])
            M[i, j] = M[j, i] = c
    return M

#------------------------------------
def plot_half_heatmap_with_trials(matrix, kept_sizes, title, outfile):
    
    n = matrix.shape[0]
    trials = [s * TRIALS_PER_RUN for s in kept_sizes]

    # Mask lower triangle and diagonal
    mask = np.tri(n, n, k=0, dtype=bool)  # True on lower incl. diag
    m = np.ma.array(matrix, mask=mask)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(m, vmin=0.0, vmax=1.0, aspect="equal", origin="lower")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(trials); ax.set_yticklabels(trials)
    ax.set_xlabel("Number of trials"); ax.set_ylabel("Number of trials")
    ax.set_title(title)

    # Optional: annotate only unmasked upper-triangle cells
    for i in range(n):
        for j in range(n):
            if i < j and np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}",
                        ha="center", va="center", fontsize=8)

    # Make masked (lower) region appear white
    cmap = im.get_cmap().copy()
    cmap.set_bad(color="white")
    im.set_cmap(cmap)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine similarity")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
#------------------------------------


anim_models = []
inan_models = []

missing = []
for k in sizes:
    try:
        inan, anim = load_model(k)
        inan_models.append(inan)
        anim_models.append(anim)
    except FileNotFoundError:
        missing.append(k)
        inan_models.append(None)
        anim_models.append(None)

if missing:
    print(f"Warning: missing model files for sizes: {missing}")


kept = [i for i, (a, b) in enumerate(zip(anim_models, inan_models)) if a is not None and b is not None]
anim_models = [anim_models[i] for i in kept]
inan_models = [inan_models[i] for i in kept]
kept_sizes = [sizes[i] for i in kept]



anim_cos = compute_pairwise_cosine(anim_models)
inan_cos = compute_pairwise_cosine(inan_models)

plot_half_heatmap_with_trials(
    anim_cos, kept_sizes,
    title=f"Cosine similarity (ANIMATE) — {subject} {session}",
    outfile=f"{subject}_{session}_cosine_anim_tri.png"
)

plot_half_heatmap_with_trials(
    inan_cos, kept_sizes,
    title=f"Cosine similarity (INANIMATE) — {subject} {session}",
    outfile=f"{subject}_{session}_cosine_inanim_tri.png"
)



# %%

# %%
