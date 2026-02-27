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
mu_factors = [0.25, 0.5, 2.0, 4.0]    #factors to be multiplied by auto mu

sizes = [2]
subjects = range(10, 31)

mods = ["dummy", "effect"]
for mod in mods:
    model_dir = f"models/newmu/{mod}" 

    ordered_runs = {1: ["02", "05"], 2: ["03", "01"]}
    auto_mu = {}
    for _, row in df_mu.iterrows():
        subj = row["subject"]
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
    
                print(f"\n{subject} M{model} auto_mu={mu_auto:.2e} -> {[f'{fac}x' for fac,_ in subj_mu_list]}")
    
                for fac, mu in subj_mu_list:
                    
                    modelfile = f"{model_dir}/{model}-{fac}-{subject}.pickle"
                    
    
                    if os.path.exists(modelfile):
                        print(f"  Exists: {subject} M{model} factor={fac}")
                        continue
    
                    try:
                        print(f"  Fitting: {subject} M{model} factor={fac} (mu={mu:.2e}) runs={run_list}")
    
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

# %%
