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
from ncrf import fit_ncrf
from eelbrain import NDVar, UTS
from eelbrain import plot, combine
from eelbrain import *
from eelbrain._data_obj import VolumeSourceSpace
import os
from pathlib import Path
from Beyond import *
import seaborn as sns
from scipy.spatial import distance

# %%
mod="dummy"  
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
    


# %% [markdown]
# # Generate Auto mu
#

# %%
#sizes = [0.25, 0.5, 1, 2]
sizes=[2]
mu_rows = []      
     
for size in sizes:
    
    for i in range(1, 31):
        
        #if i in [4, 6, 7]: continue
        subject = f"sub-{i:02d}"
        try:
            model_dir=  "models/samesize/dc" 
            
            model1= load.unpickle(f"{model_dir}/1-{size}-{subject}.pickle")
            model2= load.unpickle(f"{model_dir}/2-{size}-{subject}.pickle")

            mu_rows.append({
                "size": size,
                "subject": subject,
                "mu_m1": float(model1.mu),
                "mu_m2": float(model2.mu),
            })
            
        except Exception as e:
             continue
import pandas as pd

df_mu = pd.DataFrame(mu_rows)
print(df_mu.to_string(index=False, float_format=lambda x: f"{x:.2e}"))   

# %%
sizes = [2]
subjects = range(1, 31)

conds = {
    "dummy": Path("models/samesize/dc"),
    "effect": Path("models/samesize/effect"),
}

mu_rows = []

for cond, model_dir in conds.items():
    for size in sizes:
        for i in subjects:
            subject = f"sub-{i:02d}"
            f1 = model_dir / f"1-{size}-{subject}.pickle"
            f2 = model_dir / f"2-{size}-{subject}.pickle"

            if not f1.exists() or not f2.exists():
                continue

            try:
                model1 = load.unpickle(str(f1))
                model2 = load.unpickle(str(f2))

                mu_rows.append({
                    "condition": cond,
                    "size": size,
                    "subject": subject,
                    "mu_m1": float(model1.mu),
                    "mu_m2": float(model2.mu),
                })
            except Exception:
                continue

df_mu = pd.DataFrame(mu_rows).sort_values(["condition", "size", "subject"]).reset_index(drop=True)

print(df_mu.to_string(index=False, float_format=lambda x: f"{x:.2e}"))

# %%
mu_factors = [0.25, 0.5, 2.0, 4.0]    #factors to be multiplied by auto mu

sizes = [2]
subjects = range(10, 31)

mods = ["dummy", "effect"]
#mods = ["dummy",]
for mod in mods:
    print(f"{mod}")
    model_dir = f"models/newmu/{mod}" 

    ordered_runs = {1: ["02", "05"], 2: ["03", "01"]}
    auto_mu = {}
    df_cond = df_mu[df_mu["condition"].astype(str).str.lower() == mod].copy()
    
    for _, row in df_cond.iterrows():
        subj = str(row["subject"])
        auto_mu[(subj, 1)] = float(row["mu_m1"])
        auto_mu[(subj, 2)] = float(row["mu_m2"])
      
    
    for size in sizes:
        order_cut = int(size) if size >= 1 else 1
    
        for i in subjects:
            subject = f"sub-{i:02d}"
            session = "ImageNet01" if i > 9 else "ImageNet03"
    
            for model in [1, 2]:
                if (subject, model) not in auto_mu:
                    print(f"Skipping {subject} M{model}: no auto mu in df_mu")
                    continue
    
                mu_auto = auto_mu[(subject, model)]
                run_list = ordered_runs[model][:order_cut]
    
                
                subj_mu_list = [(fac, fac * mu_auto) for fac in mu_factors]
      
                for fac, mu in subj_mu_list:
                    modelfile = f"{model_dir}/{model}-{fac}-{subject}.pickle"
                    
                    if os.path.exists(modelfile):
                        continue
    
                    try:
                        print(f"  Fitting: {subject} M{model} factor={fac} ")
    
                        clean_fif = root / f"derivatives/preprocessed/raw/{subject}_ses-{session}_task-ImageNet_run-01_clean_meg.fif"
                        clean = mne.io.read_raw_fif(clean_fif, preload=False, verbose=False)
                        info = clean.info
                        meg_ndvar = load.fiff.raw_ndvar(clean)
                        sensor = meg_ndvar.sensor
    
                        fwd = compute_fwd_ndvar(subject, session, subjects_dir, info, sensor)
    
                        meg_all, stim_all = [], []
    
                        for run in run_list:
                            meg = load_meg_ndvar(subject, session, run)
                            event_table = make_event_table(subject, session, run)
                            event_table = event_table.sort_values(by="time").reset_index(drop=True)
    
                            meg_all.append(meg)
    
                            stim1, stim2 = make_predictors_for_run(meg, event_table, mod=mod)
                            stim_all.append([stim1, stim2])
    
                        args = (meg_all, stim_all, fwd, noise_cov, 0, 0.7)
                        kwargs = {
                            "normalize": "l1",
                            "in_place": False,
                            "mu": mu,
                            "verbose": True,
                            "n_iter": 10,
                            "n_iterc": 10,
                            "n_iterf": 100,
                        }
    
                        model_fit = fit_ncrf(*args, **kwargs)
                        save.pickle(model_fit, modelfile)
                        print(f"  Saved: {modelfile}")
    
                    except Exception as e:
                        print(f"Error: {subject} M{model} factor={fac}: {e}")

# %% [markdown]
# # Morphing

# %%
fwd_dir=Path("/Users/maryamvalian/Data/ds005810/derivatives/eelbrain/cache/raw")
subjects_dir = str(Path('~/Data/ds005810/derivatives/freesurfer/subjects').expanduser())
sizes=[0.25 ,0.5,2.0,4.0]         #mu factors : 2.0 means 2*auto mu

for mod in ["dummy", "effect"]:
    
    model_dir = f"models/newmu/{mod}"      
    print(f"mod={mod}")
    for size in sizes:
        for i in range (10,31):
            subject = f"sub-{i:02d}"
            session = "ImageNet01" if i > 9 else "ImageNet03"
                       
            for model in range (1,3):
                morphed_file = f"{model_dir}/M{model}-{size}-{subject}.pickle"   # M stands for Morphed 
                if not os.path.exists(morphed_file):
                    try:
                            
                        print(f"Morphing {mod}-{model}-{size}-{subject}...")
                        modelfile = f"{model_dir}/{model}-{size}-{subject}.pickle"
                        model= load.unpickle(modelfile)
                        hlist = model.h
                        inanim,anim = hlist[0],hlist[1]       # if mod is effect : anim: contrast, inanim: general
                        
                         
                        fwd_file=fwd_dir / f"{subject}_ses-{session}/{subject}-fwd.fif"
                        fwd = mne.read_forward_solution(str(fwd_file), verbose=False)
                        
                        #morphing 
                        stc_vec_anim = ndvar_merged_to_stc_lr(ndvar=anim, fwd=fwd, subject=subject,
                                                              subjects_dir=subjects_dir,src_tag="vol-7")
                        stc_vec_inanim = ndvar_merged_to_stc_lr(ndvar=inanim,fwd=fwd, subject=subject,
                                                                subjects_dir=subjects_dir, src_tag="vol-7")     
                        _,_,anim_fs = morph_hemi(stc_vec_anim, subject=subject, subject_to="fsaverage2",
                                             subjects_dir=subjects_dir, src_tag="vol-7")
                        _,_,inanim_fs = morph_hemi(stc_vec_inanim, subject=subject, subject_to="fsaverage2",
                                               subjects_dir=subjects_dir, src_tag="vol-7")   
                        # Smoothing
                        anim= anim_fs.smooth('source', 0.01, 'gaussian')
                        inanim= inanim_fs.smooth('source', 0.01, 'gaussian')
                        save.pickle((inanim, anim), morphed_file)   
                       
                    except Exception as e:
                        print(f"\n Error processing {subject}: {e}\n") 

# %%
#consistency NCRF_DC

sizes = [0.25, 0.5, 1 , 2.0,4.0]
subjects = [f"sub-{i:02d}" for i in range(10, 31)]



mean_rs = []
all_rs = {}

for size in sizes:
    if size==1 :        # Auto mu
        model_dir = "models/samesize/dc"
        size=2         #size in this folder is trial size which we fixed at 400 trials, meaning size 2 of runs
    else:
        model_dir = "models/newmu/dummy"
        
    subj_rs = []

    for subject in subjects:
        try:
            inan1, anim1 = load.unpickle(f"{model_dir}/M1-{size}-{subject}.pickle")
            inan2, anim2 = load.unpickle(f"{model_dir}/M2-{size}-{subject}.pickle")

            d1 = anim1 - inan1
            d2 = anim2 - inan2

            d1_flat = d1.get_data().ravel()
            d2_flat = d2.get_data().ravel()

            r_contrast = np.corrcoef(d1_flat, d2_flat)[0, 1]
            #print(f"size{size}-sub:{subject},r:{r_contrast}\n")

            if not np.isnan(r_contrast):
                subj_rs.append(r_contrast)

        except Exception as e:
            print(f"Error for {subject}, size={size}: {e}")

    all_rs[size] = subj_rs
    mean_rs.append(np.mean(subj_rs) if len(subj_rs) > 0 else np.nan)

plt.figure(figsize=(6, 4))
plt.plot(sizes, mean_rs, marker='o')
plt.xticks(sizes)
plt.xlabel("Mu Factor ( Mu Factor=1 : Auto mu )")
plt.ylabel("Average Pearson r")
plt.title("consistency within subjects NCRF_DC")
plt.grid(True)
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

sizes = [0.25, 0.5, 1,2.0, 4.0]
subjects = [f"sub-{i:02d}" for i in range(10, 31)]
mods = ["dummy", "effect"]

results = {}

for mod in mods:
    mean_rs = []
    all_rs = {}

    for size in sizes:
        if size == 1:           # Auto mu
            model_dir = f"models/samesize/{'dc' if mod == 'dummy' else 'effect'}"
            file_size = 2   # in this folder, the size is trial size, we fixed it to 400 which is size 2
        else:
            model_dir = f"models/newmu/{mod}"
            file_size = size

        subj_rs = []

        for subject in subjects:
            try:
                inan1, anim1 = load.unpickle(f"{model_dir}/M1-{file_size}-{subject}.pickle")
                inan2, anim2 = load.unpickle(f"{model_dir}/M2-{file_size}-{subject}.pickle")

                if mod=="dummy":
                    d1 = anim1 - inan1
                    d2 = anim2 - inan2
                else:
                    d1=inan1 if size==1 else anim1
                    
                    d2=inan2 if size==1 else anim2
                    
                   

                d1_flat = d1.get_data().ravel()
                d2_flat = d2.get_data().ravel()

                r_contrast = np.corrcoef(d1_flat, d2_flat)[0, 1]

                if not np.isnan(r_contrast):
                    subj_rs.append(r_contrast)

            except Exception as e:
                print(f"Error for mod={mod}, subject={subject}, size={size}: {e}")

        all_rs[size] = subj_rs
        mean_rs.append(np.mean(subj_rs) if len(subj_rs) > 0 else np.nan)

    results[mod] = {
        "mean_rs": mean_rs,
        "all_rs": all_rs
    }


plt.figure(figsize=(7, 5))

for mod in mods:
    marker='^' if mod=="dummy" else 'o'
    plt.plot(sizes, results[mod]["mean_rs"], marker=marker, label=mod)

plt.xticks(sizes)
plt.xlabel("Mu Factor (Mu Factor = 1 means auto mu)")
plt.ylabel("Average Pearson r")
plt.title("Average M1,M2 similarity across subjects")
plt.grid(True)
plt.legend()
plt.show()

# %%

sizes = [0.25, 0.5, 1, 2.0, 4.0]
subjects = [f"sub-{i:02d}" for i in range(10, 31)]

thr = 0
mean_rs = []
mean_awcoses = []

all_rs = {}
all_awcoses = {}

for size in sizes:
    if size == 1:   # auto mu
        model_dir = "models/samesize/dc"
        file_size = 2   # this is trial size 400 means 2 runs.
    else:
        model_dir = "models/newmu/dummy"
        file_size = size

    subj_rs = []
    awcos_subjects = []

    for subject in subjects:
        try:
            inan1, anim1 = load.unpickle(f"{model_dir}/M1-{file_size}-{subject}.pickle")
            inan2, anim2 = load.unpickle(f"{model_dir}/M2-{file_size}-{subject}.pickle")

            d1 = anim1 - inan1
            d2 = anim2 - inan2

            # Pearson r
            d1_flat = d1.get_data().ravel()
            d2_flat = d2.get_data().ravel()
            r_contrast = np.corrcoef(d1_flat, d2_flat)[0, 1]

            if not np.isnan(r_contrast):
                subj_rs.append(r_contrast)

            # AW-cosine
            cos_aw_t, n_used, times = ndvar_AWcosine(d1, d2, thr=thr, mode="or")
            if cos_aw_t is None or len(cos_aw_t) == 0 or np.all(np.isnan(cos_aw_t)):
                awcos = 0.0
            else:
                awcos = np.nanmean(cos_aw_t)

            awcos_subjects.append(awcos)

        except Exception as e:
            print(f"Error for {subject}, size={size}: {e}")

    all_rs[size] = subj_rs
    all_awcoses[size] = awcos_subjects

    mean_rs.append(np.nanmean(subj_rs) if len(subj_rs) > 0 else np.nan)
    mean_awcoses.append(np.nanmean(awcos_subjects) if len(awcos_subjects) > 0 else np.nan)

plt.figure(figsize=(7, 5))
plt.plot(sizes, mean_rs, marker='o', label='Average Pearson r')
plt.plot(sizes, mean_awcoses, marker='s', label='Average AW-cosine')
plt.xticks(sizes)
plt.xlabel("Mu Factor (Mu Factor = 1: Auto mu)")
plt.ylabel("Average similarity")
plt.title("Consistency within subjects - NCRF_DC")
plt.grid(True)
plt.legend()
plt.show()

# %%

sizes = [0.25, 0.5, 1, 2.0, 4.0]
subjects = [f"sub-{i:02d}" for i in range(10, 31)]


mean_rs = []
mean_awcoses = []

all_rs = {}
all_awcoses = {}

for size in sizes:
    if size == 1:   # auto mu
        model_dir = "models/samesize/effect"
        file_size = 2   # this folder uses trial-size 2 for the auto-mu case
    else:
        model_dir = "models/newmu/effect"
        file_size = size

    subj_rs = []
    awcos_subjects = []

    for subject in subjects:
        try:
            inan1, anim1 = load.unpickle(f"{model_dir}/M1-{file_size}-{subject}.pickle")
            inan2, anim2 = load.unpickle(f"{model_dir}/M2-{file_size}-{subject}.pickle")

            # effect logic
            d1 = inan1 if size == 1 else anim1
            d2 = inan2 if size == 1 else anim2

            # Pearson r
            d1_flat = d1.get_data().ravel()
            d2_flat = d2.get_data().ravel()
            r_contrast = np.corrcoef(d1_flat, d2_flat)[0, 1]

            if not np.isnan(r_contrast):
                subj_rs.append(r_contrast)

            # AW-cosine
            cos_aw_t, n_used, times = ndvar_AWcosine(d1, d2, thr=0.000, mode="or") #0.00000000001
            if cos_aw_t is None or len(cos_aw_t) == 0 or np.all(np.isnan(cos_aw_t)):
                awcos = 0.0
            else:
                awcos = np.nanmean(cos_aw_t)

            awcos_subjects.append(awcos)

        except Exception as e:
            print(f"Error for {subject}, size={size}: {e}")

    all_rs[size] = subj_rs
    all_awcoses[size] = awcos_subjects

    mean_rs.append(np.nanmean(subj_rs) if len(subj_rs) > 0 else np.nan)
    mean_awcoses.append(np.nanmean(awcos_subjects) if len(awcos_subjects) > 0 else np.nan)

plt.figure(figsize=(7, 5))
plt.plot(sizes, mean_rs, marker='o', label='Average Pearson r')
plt.plot(sizes, mean_awcoses, marker='s', label='Average AW-cosine')
plt.xticks(sizes)
plt.xlabel("Mu Factor (Mu Factor = 1: Auto mu)")
plt.ylabel("Average similarity")
plt.title("Consistency within subjects: effect")
plt.grid(True)
plt.legend()
plt.show()


# %%
sizes = [0.25, 0.5, 1, 2.0, 4.0]
subjects = [f"sub-{i:02d}" for i in range(10, 31)]

mean_rs = []
mean_cityblocks = []
mean_chebyshevs = []

all_rs = {}
all_cityblocks = {}
all_chebyshevs = {}
print("Distances")
for size in sizes:
    original_size = size

    if size == 1:   # Auto mu
        model_dir = "models/samesize/dc"
        file_size = 2   # in this folder, size 2 means 400 trials
    else:
        model_dir = "models/newmu/dummy"
        file_size = size

    subj_rs = []
    subj_cityblocks = []
    subj_chebyshevs = []
    
    for subject in subjects:
        try:
            inan1, anim1 = load.unpickle(f"{model_dir}/M1-{file_size}-{subject}.pickle")
            inan2, anim2 = load.unpickle(f"{model_dir}/M2-{file_size}-{subject}.pickle")

            d1 = anim1 - inan1
            d2 = anim2 - inan2

            d1_flat = d1.get_data().ravel()
            d2_flat = d2.get_data().ravel()

            r_contrast = np.corrcoef(d1_flat, d2_flat)[0, 1]
            cityblock_dist = distance.cityblock(d1_flat, d2_flat)
            chebyshev_dist = distance.chebyshev(d1_flat, d2_flat)

            if not np.isnan(r_contrast):
                subj_rs.append(r_contrast)

            if not np.isnan(cityblock_dist):
                subj_cityblocks.append(cityblock_dist)

            if not np.isnan(chebyshev_dist):
                subj_chebyshevs.append(chebyshev_dist)

        except Exception as e:
            print(f"Error for {subject}, size={original_size}: {e}")

    all_rs[original_size] = subj_rs
    all_cityblocks[original_size] = subj_cityblocks
    all_chebyshevs[original_size] = subj_chebyshevs

    mean_rs.append(np.mean(subj_rs) if len(subj_rs) > 0 else np.nan)
    mean_cityblocks.append(np.mean(subj_cityblocks) if len(subj_cityblocks) > 0 else np.nan)
    mean_chebyshevs.append(np.mean(subj_chebyshevs) if len(subj_chebyshevs) > 0 else np.nan)
    
    print(f"Size={original_size}, Avg cityblock={np.mean(subj_cityblocks)}, Avg chebyshev={np.mean(subj_chebyshevs)}")

def distance_to_similarity(distances):
    return 1 / (1 + np.array(distances, dtype=float))
"""
def distance_to_similarity(distances):
    distances = np.array(distances, dtype=float)
    dmin = np.nanmin(distances)
    dmax = np.nanmax(distances)

    if dmax == dmin:
        return np.ones_like(distances)

    return 1 - (distances - dmin) / (dmax - dmin)
"""

cityblock_similarity = distance_to_similarity(mean_cityblocks)
chebyshev_similarity = distance_to_similarity(mean_chebyshevs)

# %%
plt.figure(figsize=(6, 4))
plt.plot(sizes, mean_rs, marker='o')
plt.xticks(sizes)
plt.xlabel("Mu Factor (Mu Factor = 1: Auto mu)")
plt.ylabel("Average Pearson r")
plt.title("Pearson Similarity of (M1, M2) NCRF_DC")
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(sizes, cityblock_similarity, marker='s')
plt.xticks(sizes)
plt.xlabel("Mu Factor (Mu Factor = 1: Auto mu)")
plt.ylabel("Cityblock Similarity")
plt.title("Cityblock Similarity of (M1, M2) NCRF_DC")
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(sizes, chebyshev_similarity, marker='^')
plt.xticks(sizes)
plt.xlabel("Mu Factor (Mu Factor = 1: Auto mu)")
plt.ylabel("Chebyshev Similarity")
plt.title("Chebyshev Similarity of (M1, M2) NCRF_DC")
plt.grid(True)
plt.show()

# %%

sizes = [0.25, 0.5, 1, 2.0, 4.0]
subjects = [f"sub-{i:02d}" for i in range(10, 31)]
mods = ["dummy", "effect"]

def distance_to_similarity(distances):
    return 1 / (1 + np.array(distances, dtype=float))

results = {}

for mod in mods:
    mean_rs = []
    mean_cityblocks = []
    mean_chebyshevs = []

    all_rs = {}
    all_cityblocks = {}
    all_chebyshevs = {}

    for size in sizes:
        if size == 1:   # Auto mu
            model_dir = f"models/samesize/{'dc' if mod == 'dummy' else 'effect'}"
            file_size = 2   # in this folder, size 2 means 400 trials
        else:
            model_dir = f"models/newmu/{mod}"
            file_size = size

        subj_rs = []
        subj_cityblocks = []
        subj_chebyshevs = []

        for subject in subjects:
            try:
                inan1, anim1 = load.unpickle(f"{model_dir}/M1-{file_size}-{subject}.pickle")
                inan2, anim2 = load.unpickle(f"{model_dir}/M2-{file_size}-{subject}.pickle")

                if mod == "dummy":
                    d1 = anim1 - inan1
                    d2 = anim2 - inan2
                else:
                    d1 = inan1 if size == 1 else anim1
                    d2 = inan2 if size == 1 else anim2

                d1_flat = d1.get_data().ravel()
                d2_flat = d2.get_data().ravel()

                r_contrast = np.corrcoef(d1_flat, d2_flat)[0, 1]
                cityblock_dist = distance.cityblock(d1_flat, d2_flat)
                chebyshev_dist = distance.chebyshev(d1_flat, d2_flat)

                if not np.isnan(r_contrast):
                    subj_rs.append(r_contrast)

                if not np.isnan(cityblock_dist):
                    subj_cityblocks.append(cityblock_dist)

                if not np.isnan(chebyshev_dist):
                    subj_chebyshevs.append(chebyshev_dist)

            except Exception as e:
                print(f"Error for mod={mod}, subject={subject}, size={size}: {e}")

        all_rs[size] = subj_rs
        all_cityblocks[size] = subj_cityblocks
        all_chebyshevs[size] = subj_chebyshevs

        mean_rs.append(np.mean(subj_rs) if len(subj_rs) > 0 else np.nan)
        mean_cityblocks.append(np.mean(subj_cityblocks) if len(subj_cityblocks) > 0 else np.nan)
        mean_chebyshevs.append(np.mean(subj_chebyshevs) if len(subj_chebyshevs) > 0 else np.nan)

    results[mod] = {
        "mean_rs": mean_rs,
        "mean_cityblocks": mean_cityblocks,
        "mean_chebyshevs": mean_chebyshevs,
        "cityblock_similarity": distance_to_similarity(mean_cityblocks),
        "chebyshev_similarity": distance_to_similarity(mean_chebyshevs),
        "all_rs": all_rs,
        "all_cityblocks": all_cityblocks,
        "all_chebyshevs": all_chebyshevs,
    }

# Plot 1: Pearson r
plt.figure(figsize=(7, 5))
for mod in mods:
    marker = '^' if mod == "dummy" else 'o'
    plt.plot(sizes, results[mod]["mean_rs"], marker=marker, label=mod)

plt.xticks(sizes)
plt.xlabel("Mu Factor (Mu Factor = 1 means auto mu)")
plt.ylabel("Average Pearson r")
plt.title("Average M1, M2 Pearson ")
plt.grid(True)
plt.legend()
plt.show()

# Plot 2: Cityblock similarity
plt.figure(figsize=(7, 5))
for mod in mods:
    marker = '^' if mod == "dummy" else 'o'
    plt.plot(sizes, results[mod]["cityblock_similarity"], marker=marker, label=mod)

plt.xticks(sizes)
plt.xlabel("Mu Factor (Mu Factor = 1 means auto mu)")
plt.ylabel("Average Cityblock Similarity")
plt.title("NCRF_EC ")
plt.grid(True)
plt.legend()
plt.show()

# Plot 3: Chebyshev similarity
plt.figure(figsize=(7, 5))
for mod in mods:
    marker = '^' if mod == "dummy" else 'o'
    plt.plot(sizes, results[mod]["chebyshev_similarity"], marker=marker, label=mod)

plt.xticks(sizes)
plt.xlabel("Mu Factor (Mu Factor = 1 means auto mu)")
plt.ylabel("Average Chebyshev Similarity")
plt.title(" NCRF-EC")
plt.grid(True)
plt.legend()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

sizes = [0.25, 0.5, 1, 2.0, 4.0]
subjects = [f"sub-{i:02d}" for i in range(10, 31)]

def distance_to_similarity(distances):
    return 1 / (1 + np.array(distances, dtype=float))

mean_rs = []
mean_cityblocks = []
mean_chebyshevs = []

all_rs = {}
all_cityblocks = {}
all_chebyshevs = {}

for size in sizes:
    if size == 1:   # Auto mu
        model_dir = "models/samesize/effect"
        file_size = 2   # in this folder, size 2 means 400 trials
    else:
        model_dir = "models/newmu/effect"
        file_size = size

    subj_rs = []
    subj_cityblocks = []
    subj_chebyshevs = []

    for subject in subjects:
        try:
            inan1, anim1 = load.unpickle(f"{model_dir}/M1-{file_size}-{subject}.pickle")
            inan2, anim2 = load.unpickle(f"{model_dir}/M2-{file_size}-{subject}.pickle")

            d1 = inan1 if size == 1 else anim1
            d2 = inan2 if size == 1 else anim2

            d1_flat = d1.get_data().ravel()
            d2_flat = d2.get_data().ravel()

            r_contrast = np.corrcoef(d1_flat, d2_flat)[0, 1]
            cityblock_dist = distance.cityblock(d1_flat, d2_flat)
            chebyshev_dist = distance.chebyshev(d1_flat, d2_flat)

            if not np.isnan(r_contrast):
                subj_rs.append(r_contrast)

            if not np.isnan(cityblock_dist):
                subj_cityblocks.append(cityblock_dist)

            if not np.isnan(chebyshev_dist):
                subj_chebyshevs.append(chebyshev_dist)

        except Exception as e:
            print(f"Error for subject={subject}, size={size}: {e}")

    all_rs[size] = subj_rs
    all_cityblocks[size] = subj_cityblocks
    all_chebyshevs[size] = subj_chebyshevs

    mean_rs.append(np.mean(subj_rs) if len(subj_rs) > 0 else np.nan)
    mean_cityblocks.append(np.mean(subj_cityblocks) if len(subj_cityblocks) > 0 else np.nan)
    mean_chebyshevs.append(np.mean(subj_chebyshevs) if len(subj_chebyshevs) > 0 else np.nan)

cityblock_similarity = distance_to_similarity(mean_cityblocks)
chebyshev_similarity = distance_to_similarity(mean_chebyshevs)


plt.figure(figsize=(7, 5))
plt.plot(sizes, mean_rs, marker='o')
plt.xticks(sizes)
plt.xlabel("Mu Factor (Mu Factor = 1 means auto mu)")
plt.ylabel("Average Pearson r")
plt.title("(Effect)")
plt.grid(True)
plt.show()


plt.figure(figsize=(7, 5))
plt.plot(sizes, cityblock_similarity, marker='s')
plt.xticks(sizes)
plt.xlabel("Mu Factor (Mu Factor = 1 means auto mu)")
plt.ylabel("Average Cityblock Similarity")
plt.title("(Effect)")
plt.grid(True)
plt.show()


plt.figure(figsize=(7, 5))
plt.plot(sizes, chebyshev_similarity, marker='^')
plt.xticks(sizes)
plt.xlabel("Mu Factor (Mu Factor = 1 means auto mu)")
plt.ylabel("Average Chebyshev Similarity")
plt.title(" avg Pearson r (Effect)")
plt.grid(True)
plt.show()

# %%
