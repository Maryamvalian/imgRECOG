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


# %%
mod="dummy"  #common, effect, dummy, ortho (effect with unbalanced trial counts)
rewrite=True
root = Path("~/Data/ds005810")
subjects_dir = str(Path('~/Data/ds005810/derivatives/freesurfer/subjects').expanduser())
fwd_dir=Path("/Users/maryamvalian/Data/ds005810/derivatives/eelbrain/cache/raw")
model_dir = "models/consist"

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
   
    rng = random.Random(seed)
    out: Dict[int, List[List[str]]] = {}
    for k in range(1, k_max + 1):
        out[k] = [rng.sample(runs, k) for _ in range(n_repeats)]
    return out
#-------------------------------------------
def nested_subsets(runs: List[str], k_max: int, n_repeats: int = 2, seed: int = 11) -> Dict[int, List[List[str]]]:
    
    rng = random.Random(seed)
    out: Dict[int, List[List[str]]] = {}
    
    for _ in range(n_repeats):
        rng.shuffle(runs)  # random order of runs
        seq = []
        for k in range(1, k_max + 1):
            seq.append(runs[:k])
            out.setdefault(k, []).append(runs[:k])
            
    return out
#-------------------------------------------    



# %% [markdown]
# # Make random subset 

# %%
runs = [f"0{i}" for i in range(1, 9)]
#subs = independent_subsets(runs, k_max=8, n_repeats=1, seed=11)      
subs = nested_subsets(runs, k_max=8, n_repeats=1, seed=11)
subs

# %% [markdown]
# # Fit NCRF Models

# %%
#Consistency folder for independant subsets , Consist for nested subsets model output
for i in range (1,10):                 #first 9 subjects
    if i==7 :
        session="ImageNet04"
    else:
        session="ImageNet03"
    subject = f"sub-{i:02d}"
    
    #FWD for Session
    print(f"computing fwd for {subject}-{session}... ")
    clean_fif = root / f"derivatives/preprocessed/raw/{subject}_ses-{session}_task-ImageNet_run-01_clean_meg.fif"
    clean = mne.io.read_raw_fif(clean_fif, preload=False,verbose=False)
    info= clean.info           
    meg_ndvar = load.fiff.raw_ndvar(clean)
    sensor=meg_ndvar.sensor
    fwd = compute_fwd_ndvar(subject, session,subjects_dir,info,sensor)
    
    
    
    for size in range (1,9):           
        subset=subs[size][0]
        #print(f"{subset}")
        modelfile = f"models/consist/{size}-{subject}-{session}-ncrf.pickle"
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
subject="sub-07"
session="ImageNet04"
model_dir="models/consist"

for size in range (1,9):
    
    morphed_file = f"{model_dir}/M{size}-{subject}-{session}-ncrf.pickle"
    if os.path.exists(morphed_file):
        
        print(f"Loading{size}-{subject}-{session} from file.")
        #inanim, anim = load.unpickle(morphed_file)
    else:
        try:
                
            print(f"Morphing {size}-{subject}-{session}...")
            modelfile = f"{model_dir}/{size}-{subject}-{session}-ncrf.pickle"
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
size=0.5
subject="sub-03"
session="ImageNet03"
model_dir="models/samesize"


morphed_file = f"{model_dir}/M1-{size}-{subject}-{session}-ncrf.pickle"
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


# %% [markdown]
# # Save plot AW-COSINE SIMILARITY - all_paired

# %%
import os
import numpy as np
import matplotlib.pyplot as plt


subject = "sub-04"
session = "ImageNet03"

sizes = list(range(1, 9))  # 1..8
TRIALS_PER_RUN = 200  

#-------------------------------------------------------
def load_model(size):
    
    morphed_file = f"{model_dir}/M{size}-{subject}-{session}-ncrf.pickle"
    inan, anim = load.unpickle(morphed_file)
    return inan, anim
#---------------------------

def ndvar_cosine(nd1, nd2, eps=1e-12):
    
    A = np.asarray(nd1.get_data())
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
#------------------------------------------------
def ndvar_AWcosine(nd1, nd2, thr=1e-12, mode="or"):
    
    A = np.asarray(nd1.get_data(), dtype=float)  # (V, 3, T)
    B = np.asarray(nd2.get_data(), dtype=float)
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: {A.shape} vs {B.shape}")

    V, _, T = A.shape
    cos_aw_t = np.full(T, np.nan)
    n_used   = np.zeros(T, dtype=int)

    
    tmin  = float(nd1.time.tmin)
    tstep = float(nd1.time.tstep)
    times = tmin + np.arange(T) * tstep

    for t in range(T):
        Ai = A[:, :, t]                          
        Bi = B[:, :, t]
        aN = np.linalg.norm(Ai, axis=1)         
        bN = np.linalg.norm(Bi, axis=1)

        if mode.lower() == "and":
            consider = (aN > thr) & (bN > thr)
        else: 
            consider = (aN > thr) | (bN > thr)

        
        valid = consider & (aN > 0) & (bN > 0)
        if not np.any(valid):
            continue

        Ai_v = Ai[valid]
        Bi_v = Bi[valid]
        aN_v = aN[valid]
        bN_v = bN[valid]

        # voxel cosine 
        dots  = np.einsum('ij,ij->i', Ai_v, Bi_v)   # <A_i, B_i>
        cos_i = dots / (aN_v * bN_v)                # (n_vox_t,)

        #voxel amplitude weights 
        w = 0.5 * (aN_v + bN_v)
        wsum = w.sum()
        if wsum == 0:
            continue

        cos_aw_t[t] = float(np.sum(w * cos_i) / wsum)
        n_used[t]   = int(valid.sum())

    return cos_aw_t, n_used, times

#--------------------------------------------

def compute_pairwise_cosine(models):
    
    n = len(models)
    M = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        M[i, i] = 1.0
        for j in range(i + 1, n):
            cos_aw_t, n_used, times = ndvar_AWcosine(models[i],models[j], thr=1e-12, mode="or")              #AWCOSINE
            c = cos_aw_t.mean()
            
            #c = ndvar_cosine(models[i], models[j])                                                         #cosine
            
            M[i, j] = M[j, i] = c
    return M
    

#------------------------------------
def plot_half_heatmap_with_trials(matrix, kept_sizes, title, outfile):
    
    n = matrix.shape[0]
    trials = [s * TRIALS_PER_RUN for s in kept_sizes]

    # Mask lower triangle and diagonal
    mask = np.tri(n, n, k=0, dtype=bool) 
    m = np.ma.array(matrix, mask=mask)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(m, vmin=0.0, vmax=1.0, aspect="equal", origin="lower")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(trials); ax.set_yticklabels(trials)
    ax.set_xlabel("Number of trials"); ax.set_ylabel("Number of trials")
    ax.set_title(title)
    
    for i in range(n):
        for j in range(n):
            if i < j and np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}",
                        ha="center", va="center", fontsize=8)

    
    cmap = im.get_cmap().copy()
    cmap.set_bad(color="white")
    im.set_cmap(cmap)
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("AWCosine similarity")
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
    title=f"AWCosine similarity (ANIMATE) — {subject} {session}",
    outfile=f"{subject}_{session}_c_anim_tri.png"
)

plot_half_heatmap_with_trials(
    inan_cos, kept_sizes,
    title=f"AWCosine similarity (INANIMATE) — {subject} {session}",
    outfile=f"{subject}_{session}_c_inanim_tri.png"
)



# %% [markdown]
# # plot pair-wise compare AW-Cosine

# %%
subject

# %%
k1, k2 = 4 ,8             #pick model by size


inan1,an1 = load_model(k1)
inan2,an2 = load_model(k2)
cos_aw_t, n_used, times = ndvar_AWcosine(an1,an2, thr=1e-12, mode="or")

# quick plot
"""
import matplotlib.pyplot as plt
plt.plot(times, cos_aw_t, marker='o', ms=3)
plt.xlabel("Time (s)")
plt.ylabel("AW-cosine")
plt.title(f"{subject}- an1 vs an8")
plt.ylim(0.8, 1)
plt.grid(True, alpha=0.3)
plt.show()
"""

#plot with n_used
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize


m = np.isfinite(cos_aw_t) & np.isfinite(times) & (n_used > 0)
t = times[m]
y = cos_aw_t[m]
w = n_used[m]

if len(t) < 2:
    raise RuntimeError("Need at least two valid points to draw segments.")
# build line segments
points = np.column_stack([t, y])                       # (N, 2)
segs = np.stack([points[:-1], points[1:]], axis=1)     # (N-1, 2, 2)

# color by mean n_used of segment endpoints
w_mid = 0.5 * (w[:-1] + w[1:])

# blue -> orange colormap
cmap = LinearSegmentedColormap.from_list("blue_orange", ["#1f77b4", "#ff7f0e"])
norm = Normalize(vmin=w_mid.min(), vmax=w_mid.max())

fig, ax = plt.subplots(figsize=(8, 4))

lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=2.0)
lc.set_array(w_mid)
ax.add_collection(lc)

# colored markers at the points
sc = ax.scatter(t, y, c=w, cmap=cmap, norm=norm, s=18, zorder=3, edgecolor="none")

ax.set_xlim(t.min(), t.max())
ax.set_ylim(0.7, 1.0)  # adjust as you like
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude-weighted cosine")
ax.set_title(f"AW-Cosine(an{k1},an{k2})")

cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("# active sources (n_used)")

ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()

# %% [markdown]
# # PEARSON R

# %%
len(anim_models)

# %%
subject="sub-07"
session="ImageNet04"
animacy="anim"
model_dir="models/consist"
def load_model(size):
    
    morphed_file = f"{model_dir}/M{size}-{subject}-{session}-ncrf.pickle"
    inan, anim = load.unpickle(morphed_file)
    return inan, anim
#--------------    

anim_models = []
inan_models = []

for k in range(1,9):
    try:
        inan, anim = load_model(k)
        inan_models.append(inan)
        anim_models.append(anim)      
        
    except FileNotFoundError:
        print(f"Load Model Failed! M{k}")
        
models = anim_models if animacy == "inanim" else inan_models
names = [f'M{i+1}' for i in range(len(models))]

flattened = []
for m in models:
    data = np.asarray(m.get_data())  # (vox, 3, time)
    flat = data.reshape(-1)          
    flattened.append(flat)


corr_mat = np.corrcoef(flattened)

#plot
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr_mat, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=45)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names)

ax.invert_yaxis()

plt.colorbar(im, ax=ax, label='Pearson r')
ax.set_title(f'{subject}- Correlation,  {animacy}  ')
plt.tight_layout()
plt.show()

print(corr_mat)        


# %% [markdown]
# # R2 Measure for each voxel 
# Compare Model A with true model B<br>
# ∥⋅∥ denotes the Euclidean norm $\sqrt{x^2 + y^2 + z^2}$

# %% [markdown]
# $$
# R_i^{2} = 1 -
# \frac{\displaystyle\sum_{t} \left\lVert \mathbf{A}_i(t) - \mathbf{B}_i(t) \right\rVert^{2}}
# {\displaystyle\sum_{t} \left\lVert \mathbf{B}_i(t) - \bar{\mathbf{B}}_i \right\rVert^{2}}
# $$
#

# %%
def ndvar_r2(model1_tuple, model2_tuple, animacy="anim"):
    
    inan1, an1 = model1_tuple
    inan2, an2 = model2_tuple

   
    A = np.asarray(an1.get_data())  if animacy == "anim" else np.asarray(inan1.get_data()) 
    B = np.asarray(an2.get_data())  if animacy == "anim" else np.asarray(inan2.get_data()) 

    scale = 1e12
    A_scaled = A * scale
    B_scaled = B * scale

   
    B_mean = B_scaled.mean(axis=2, keepdims=True)

    
    ss_res = np.sum((B_scaled - A_scaled) ** 2, axis=(1, 2))
    ss_tot = np.sum((B_scaled - B_mean) ** 2, axis=(1, 2))

   
    R2_vox = np.where(ss_tot != 0, 1 - ss_res / ss_tot, np.nan)

    #for ploting
    B_power = np.linalg.norm(B_scaled, axis=(1, 2))

    return R2_vox, B_power
#-----------------------------------------------------------------
k1, k2 = 6, 8
animacy = "anim"
model1_tuple = load_model(k1) 
model2_tuple = load_model(k2) 

R2_vox, B_power = ndvar_r2(model1_tuple, model2_tuple, animacy=animacy)

print(f"Mean R² (vector): {np.nanmean(R2_vox):.3f}")

#plot
# --- compute overall signal norm per voxel (strength) ---
B_power = np.linalg.norm(B_scaled, axis=(1, 2))  # total magnitude per voxel

# --- threshold for activity ---
thr = np.percentile(B_power, 70)  # or set manually, e.g. thr = 1e-10

active_mask = B_power > thr

# --- assign colors ---
colors = np.where(active_mask, 'red', 'lightgray')

# --- plot ---
title=f'R2- {subject}:{animacy}{k1} vs {animacy}{k2} '
plt.figure(figsize=(8, 4))
plt.scatter(np.arange(len(R2_vox)), R2_vox, c=colors, s=10, alpha=0.8)
plt.axhline(0, color='k', linestyle='--', linewidth=0.8)
plt.xlabel('Voxel index')
plt.ylabel('$R^2$')
plt.title(title)
plt.show()

#plot on brain
R2_nd = NDVar(R2_vox,src, name='R2')
plot.GlassBrain(
    R2_nd,
    vmin=-0.0, vmax=1.0,
    cmap='inferno',
    threshold=0.6,          
    colorbar=True,
    title=title
)

# %% [markdown]
# ## plot R2 (model i , and Model 8 )

# %%
subject="sub-08"

# %%
animacy="inanim"

names = [f'M{i}' for i in range(1,9)]

n = len(names)
M = np.full((n), np.nan, dtype=float)
model2_tuple = load_model(n)
for i in range(n):
   
    model1_tuple = load_model(i+1) 
    R2_vox, B_power = ndvar_r2(model1_tuple, model2_tuple, animacy=animacy)
    M[i] = R2_vox.mean()


# Create model indices (X-axis values)
model_indices = np.arange(1, n + 1)

fig, ax = plt.subplots(figsize=(7, 5))

# Plot 
ax.plot(model_indices, M, marker='o', linestyle='-', color='C0', label='Mean $R^2$')


max_r2_index = np.argmax(M)
ax.plot(model_indices[max_r2_index], M[max_r2_index],
        marker='*', markersize=12, color='red',
        label=f'Max $R^2$ = {M[max_r2_index]:.3f}')
ax.set_xlabel('Model Index (Size $i*200$)')
ax.set_ylabel('Mean $R^2$ (across voxels)')
ax.set_title(f'({subject}, {animacy})')
ax.set_xticks(model_indices)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()
plt.tight_layout()
plt.show()       
                                                             
         

    

# %% [markdown]
# # GROUP LEVEL

# %% [markdown]
# ## Anova Test

# %%
animacy="anim"
model_dir="models/consist"


def load_model(size):
    
    morphed_file = f"{model_dir}/M{size}-{subject}-{session}-ncrf.pickle"
    inan, anim = load.unpickle(morphed_file)
    return inan, anim

    

R_data = np.full((9,7), np.nan, dtype=float)             #subject* models (0,1,2,3,4,5,6) : R-value for m1,..,m7 corelation with m8

for i in range (1,10):    #subjects
    if i==7 :
        session="ImageNet04"
    else:
        session="ImageNet03"
    subject = f"sub-{i:02d}"

    
    anim_models = []
    inan_models = []
    
    for k in range(1,9):
        try:
            inan, anim = load_model(k)
            inan_models.append(inan)
            anim_models.append(anim)      
            
        except FileNotFoundError:
            print("Load Model Failed!")
            
    models = anim_models if animacy == "anim" else inan_models
    
    flattened = []
    for m in models:
        data = np.asarray(m.get_data()) 
        flat = data.reshape(-1)          
        flattened.append(flat)
    
    
    corr_mat = np.corrcoef(flattened)
    R_data[i-1]=corr_mat[7,:7].round(4)             # corr_mat[7] : correlation of all models with model8 last numbeer is always 1 corr(model8,model8)=1
       


R_data=np.array(R_data)
Z_data = fisher_r_to_z(R_data)

n_subjects, n_conditions = Z_data.shape        #condition: size of subset
    
    
cases = []
for i in range(n_subjects):
    
    subject_id = i + 1 
    for j in range(n_conditions):
       
        trial_size = f"{j+1}"
        z_score = Z_data[i, j]
        cases.append([str(subject_id), trial_size, z_score])


ds = eb.Dataset.from_caselist(['Subject','Trial_Size', 'Fisher_Z'], cases, random='Subject')
print(ds.head())

anova_results = eb.test.ANOVA(y='Fisher_Z',
                              x='Trial_Size*Subject', 
                              data=ds, 
                              title=" Trial Size Effect")
print(anova_results)


# %%
ds.tail()


# %% [markdown]
# # SECOND EXPERIMENT : Non-overlapping Data
# ## FIT NCRF

# %%
#TRIM MEG TO HALF SIZE (SHOULD TRIM MEG ONLY TRIMING STIM doesn't WORKS)
model_dir = "models/samesize/effect"
sizes = [0.25, 0.5, 1, 2,  3, 4, 6 ,8] 
for size in sizes:
    print(f"=================== Size={size}==================")
    for i in range (1,10):                 #first 9 subjects
        if i==7 or i==6 :
            continue
           
        
        subject = f"sub-{i:02d}"
        
        #FWD for Session
        
        clean_fif = root / f"derivatives/preprocessed/raw/{subject}_ses-{session}_task-ImageNet_run-01_clean_meg.fif"
        clean = mne.io.read_raw_fif(clean_fif, preload=False,verbose=False)
        info= clean.info           
        meg_ndvar = load.fiff.raw_ndvar(clean)
        sensor=meg_ndvar.sensor
        
        
        ordered_runs = ["02", "01", "03", "06", "04", "05", "07", "08"]
        run_select= size if size>=1 else 1
        subset = ordered_runs[:run_select]
    
        
        for model in range (1,3):           #M1, M2
    
            session="ImageNet03" if model==1 else "ImageNet04"            
            run_list = [r for r in subset]
            #subset=subsets[model-1]
            #print(f"{subset}")
            modelfile = f"{model_dir}/{model}-{size}-{subject}.pickle"
            morphfile = f"{model_dir}/M{model}-{size}-{subject}.pickle"
            if os.path.exists(modelfile):
                print(f"{model}-{size}-{subject} model file exists.")
                continue
            if os.path.exists(morphfile):
                print(f"{model}-{size}-{subject} Morph file exists.")
                continue
            try:        
                print(f"computing fwd for {subject}-{session}... ")
                fwd = compute_fwd_ndvar(subject, session,subjects_dir,info,sensor)
                   
                meg_all = []
                stim_all = []
                for run in subset:
                    
                    print(f"Loading session-{session}, run-{run} MEG ...")
                    meg= load_meg_ndvar(subject, session, run)
                    #meg_all.append(meg)
                    event_table= make_event_table(subject, session, run)
                    #-----------half
    
                    event_table = event_table.sort_values(by='time').reset_index(drop=True)
    
                    cut = int(200 * size)
    
                    if size==0.5 or size==0.25:
                        
                        if len(event_table) > cut:
                            event_table = event_table.iloc[:cut]
                        else:
                            print(f"failed event table")
        
                        t_cut = float(event_table.iloc[-1]['time']) 
                        meg_trim = meg.sub(time=(0, t_cut))
                        meg = meg_trim
    
                    meg_all.append(meg)     #meg or Trimed meg for less than one run
    
        
                    stim1,stim2= make_predictors_for_run(meg, event_table,mod="effect")  # <=============== Effect, dummy
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

# %%
n_anim = event_table['animate'].sum()          # True values = animate trials
n_inanim = (~event_table['animate']).sum()     # False values = inanimate trials
n_total = len(event_table)

print(f"Total trials: {n_total}")
print(f"Animate trials: {n_anim}")
print(f"Inanimate trials: {n_inanim}")

# %% [markdown]
# ## Morph

# %%
model_dir="models/samesize"
size=3
for i in range (1,10):
    subject = f"sub-{i:02d}"
    if i==7:
        session="ImageNet04"
    else:
        session="ImageNet03"
    for subset in range (1,3):
        
        morphed_file = f"{model_dir}/M{subset}-{size}-{subject}-{session}-ncrf.pickle"  #M stands for Morphed
        if os.path.exists(morphed_file):
            
            print(f" {subset}-{size}-{subject}-{session} exists.")
            #inanim, anim = load.unpickle(morphed_file)
        else:
            try:
                    
                print(f"Morphing {subset}-{size}-{subject}-{session}...")
                modelfile = f"{model_dir}/{subset}-{size}-{subject}-{session}-ncrf.pickle"
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
                print(f"\n{modelfile} Saved \n ")
        
            except Exception as e:
                
                print(f"\n----------- Error processing {subject}: {e}\n")   

# %% [markdown]
# # R_Data from morphed files

# %%
model_dir="models/samesize"
n_subject=9                  #Loop sub-01,sub-02,...,sub-n_subjects
size=4

def load_model_subset(subset,subject,session,size):
    
    morphed_file = f"{model_dir}/M{subset}-{size}-{subject}-{session}-ncrf.pickle"
    inan, anim = load.unpickle(morphed_file)
    return inan, anim

#-----------
def corr_data(animacy, n_subject,size):
    
    R_data = np.full(n_subject, np.nan)

    def fisher_r_to_z(r):
        r = np.clip(r, -0.999999, 0.999999)  # avoid infinities
        return 0.5 * np.log((1 + r) / (1 - r))

    for i in range(1, n_subject + 1):
        subject = f"sub-{i:02d}"
        session = "ImageNet04" if i == 7 else "ImageNet03"

        try:
                
    
            inan1, anim1 = load_model_subset(1, subject, session,size)
            inan2, anim2 = load_model_subset(2, subject, session,size)
            m1= anim1 if animacy == "anim" else inan1
            m2= anim2 if animacy == "anim" else inan2
            d1= np.asarray(m1.get_data()).reshape(-1)
            d2= np.asarray(m2.get_data()).reshape(-1)
    
            r = np.corrcoef(d1, d2)[0, 1]
            #print(f"{subject}: Corr(m1,m2)={r:.2f}, Fisher_z={fisher_r_to_z(r):.2f}")
            R_data[i - 1] = r
        except Exception as e:
            print(f"failed: {e}")

    Z_data = fisher_r_to_z(R_data)
    return R_data, Z_data

#------------------------------------------
R_data_anim, _ = corr_data("anim", n_subject, size)
R_data_inanim, _ = corr_data("inanim", n_subject, size)


import matplotlib.pyplot as plt
import seaborn as sns


subjects = [f"sub-{i:02d}" for i in range(1, n_subject+1)]

df = pd.DataFrame({
    'Subject': subjects * 2,
    'Animacy': ['Animate'] * (n_subject) + ['Inanimate'] * (n_subject),
    'Corr': list(R_data_anim) + list(R_data_inanim)
})


plt.figure(figsize=(7,4))
sns.barplot(
    data=df,
    x='Subject', y='Corr',
    hue='Animacy',
    palette={'Animate': '#66BB66', 'Inanimate': '#3399FF'},  
    edgecolor='black',
    width=0.5
)
plt.ylim(0, 1.0)
plt.ylabel('Pearson R')
plt.title(f'Model Stability\n({int(size*200)} Trials, Non-overlapping Data)')
plt.legend(bbox_to_anchor=(1.02, 1),loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


print(f" anim:{R_data_anim}")

print(f" inanim:{R_data_inanim}")
print(f" \n\nMean \nanim:{R_data_anim.mean().round(2)}\ninanim:{R_data_inanim.mean().round(2)}")

# %% [markdown]
# # TRail size data set

# %%
model_dir = "models/samesize"
n_subject = 9
sizes = [0.25, 0.5, 1, 2, 3, 4, 8]   
trials_per_subset = 200
#------------------------

results = []
for size in sizes:
    R_anim, _ = corr_data("anim", n_subject, size)
    R_inanim, _ = corr_data("inanim", n_subject, size)

    mean_r_anim = np.nanmean ( R_anim)
    mean_r_inan = np.nanmean( R_inanim)

    results.append({
        "Subset Size": int(size * trials_per_subset),
        "Animate": mean_r_anim,
        "Inanimate": mean_r_inan
    })

df = pd.DataFrame(results)
print("\nSummary:")
print(df)



# ----------------------------------------
# Melt for seaborn plotting
df_melted = df.melt(id_vars="Subset Size", var_name="Animacy", value_name="Mean r")
#plot
plt.figure(figsize=(6, 4))
sns.barplot(
    data=df_melted,
    x="Subset Size", y="Mean r",
    hue="Animacy",
    palette={"Animate": "#66BB66", "Inanimate": "#3399FF"},
    edgecolor="black", width=0.5
)

plt.ylim(0, 1.0)
plt.ylabel("Mean Pearson r")
plt.title("NCRF Model Consistency Across Subset Sizes")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# %% [markdown]
# # Paired Test on Tiral size condition

# %%
model_dir = "models/samesize"
n_subject = 9
sizes = [0.25, 0.5, 1, 2, 4, 8]    
trails_per_run = 200


# --------------------
def fisher_r_to_z(r):
    r = np.clip(r, -0.999999, 0.999999)
    return 0.5 * np.log((1 + r) / (1 - r))


# --------------------

all_R_anim = []
all_R_inanim = []

for size in sizes:
   
    R_anim, _ = corr_data("anim", n_subject, size)
    R_inanim, _ = corr_data("inanim", n_subject, size)

    all_R_anim.append(R_anim)
    all_R_inanim.append(R_inanim)

#convert 1D array to 2D (rows: subjects, column: sizes
R_anim_all = np.column_stack(all_R_anim)
R_inan_all = np.column_stack(all_R_inanim)


Z_anim_all = fisher_r_to_z(R_anim_all)
Z_inan_all = fisher_r_to_z(R_inan_all)

subset_sizes = [int(size * trails_per_run) for size in sizes]                #size*200



#Paired Test 
#Pairs : (50,1600)(100,1600)...(800,1600)

for j, size in enumerate(subset_sizes[:-1]):  #skip 1600
    print(f"\n \nPaired ({size}, 1600)")

    ds_anim = eb.Dataset({
        'Subject': [f"sub-{i+1:02d}" for i in range(n_subject)],
        'Z1': Z_anim_all[:, j],
        'Z2': Z_anim_all[:, -1]         #m8=1600
    })
    ds_anim['Diff'] = ds_anim['Z2'] - ds_anim['Z1']
    ttest_anim = eb.test.ttest('Diff', data=ds_anim, tail=0)

    ds_inan = eb.Dataset({
        'Subject': [f"sub-{i+1:02d}" for i in range(n_subject)],
        'Z1': Z_inan_all[:, j],
        'Z2': Z_inan_all[:, -1]
    })
    ds_inan['Diff'] = ds_inan['Z2'] - ds_inan['Z1']
    ttest_inan = eb.test.ttest('Diff', data=ds_inan, tail=0)
    
    #print(f"\n DataSet ANIM \n {ds_anim}")
    print(f"Anim: \n {ttest_anim}")
    print(f"Inanim: \n {ttest_inan}")
    


# %% [markdown]
# # Correlation (diff(anim,inan)_m1 , diff(anim,inan)_m2)

# %%
def corr_diff(n_subject,size):
    
    R_data = np.full(n_subject, np.nan)

    for i in range(1, n_subject + 1):
        subject = f"sub-{i:02d}"
        session = "ImageNet04" if i == 7 else "ImageNet03"

        try:
            inan1, anim1 = load_model_subset(1, subject, session,size)
            inan2, anim2 = load_model_subset(2, subject, session,size)
            d1 = anim1 - inan1
            d2 = anim2 - inan2

            #Pearson R
            diff1= np.asarray(d1.get_data()).reshape(-1)
            diff2= np.asarray(d2.get_data()).reshape(-1)
            r = np.corrcoef(diff1, diff2)[0, 1]
            R_data[i - 1] = r
            
        except Exception as e:
            print(f"failed: {e}")

    Z_data = fisher_r_to_z(R_data)
    return R_data, Z_data

#------------------------------------------
model_dir = "models/samesize"
n_subject = 9
sizes = [0.25, 0.5, 1, 2, 3, 4, 8]   
trials_per_subset = 200
#------------------------

results = []
for size in sizes:
    R_diff, _ = corr_diff(n_subject, size)
    

    r_mean = np.nanmean ( R_diff)
    

    results.append({
        "Subset Size": int(size * trials_per_subset),
        "mean_r": r_mean,
        
    })

df = pd.DataFrame(results)
print("\nSummary:")
print(df)

#-------plot
plt.figure(figsize=(6, 4))
sns.barplot(
    data=df,
    x="Subset Size", y="mean_r",
    color="maroon", edgecolor="black", width=0.5
)
plt.ylim(0, 1.0)
plt.ylabel("Mean Pearson r")
plt.title("Consistency of the pired difference (Animate-Inanimate)")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


# %%
