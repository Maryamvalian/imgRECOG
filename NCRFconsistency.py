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
import eelbrain as eel
from mne import *
from ncrf import fit_ncrf
import matplotlib.pyplot as plt
import seaborn as sns
   

# %% [markdown]
# # Estimate NCRF Models M1, M2 for all subjects

# %%
root = Path("~/Data/ds005810")
empty_room=root/"sub-emptyroom/ses-20211114/meg/sub-emptyroom_ses-20211114_task-noise_meg.fif"
raw_er = mne.io.read_raw_fif(empty_room, preload=True, verbose=False).pick('meg')
raw_er.filter(1., 40., phase="zero-double", verbose=False)
raw_er.resample(100, npad="auto", verbose=False)
noise_cov = mne.compute_raw_covariance(raw_er, method='shrunk', rank=None,verbose=False)

# %%
mod = "dummy"     
model_dir = f"models/samesize/1session/{mod}"       


sizes = [ 0.25, 0.5, 1, 2] 
for size in sizes:
    print(f">>> Size={size}")
    ordered_runs = {
                1: ["02", "05"],  
                2: ["03", "01"] 
                }
    order_cut= size if size>=1 else 1
    
    for i in range (1,31):  
        
        # Dataset 1, and Dataset 2 both in the same Session
        if i>9:
            session = "ImageNet01"
        elif i==7:
            session = "ImageNet04"   
        else:
            session = "ImageNet03" 
        subject = f"sub-{i:02d}"   
          
        
        for model in range (1,3):           #M1, M2

            run_list = ordered_runs[model][:int(order_cut)]   
            
            clean_fif = root / f"derivatives/preprocessed/raw/{subject}_ses-{session}_task-ImageNet_run-01_clean_meg.fif"
            clean = mne.io.read_raw_fif(clean_fif, preload=False,verbose=False)
            info= clean.info           
            meg_ndvar = load.fiff.raw_ndvar(clean)
            sensor=meg_ndvar.sensor

            
            
            modelfile = f"{model_dir}/{model}-{size}-{subject}.pickle"
            if not os.path.exists(modelfile):
                try:        
                    print(f"  computing fwd for {subject}-{session}... ")
                    fwd = compute_fwd_ndvar(subject, session,subjects_dir,info,sensor)
                       
                    meg_all = []
                    stim_all = []
                    for run in run_list:
                        
                        print(f"  Loading session-{session}, run-{run} MEG ...")
                        meg= load_meg_ndvar(subject, session, run)
                        event_table= make_event_table(subject, session, run)
                        
                        #trim MEG if size<1
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
        
                        meg_all.append(meg)     
        
            
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
# # Morphing

# %%
fwd_dir=Path("/Users/maryamvalian/Data/ds005810/derivatives/eelbrain/cache/raw")
subjects_dir = str(Path('~/Data/ds005810/derivatives/freesurfer/subjects').expanduser())
sizes = [0.25, 0.5, 1, 2] 

for mod in ["dummy", "effect"]:
    model_dir = f"models/samesize/1session/{mod}"      # All 30 subjects -same session chunk m1,m2
    
    for size in sizes:
        for i in range (1,31):
            subject = f"sub-{i:02d}"
                       
            for model in range (1,3):
                morphed_file = f"{model_dir}/M{model}-{size}-{subject}.pickle"   # M stands for Morphed 
                if not os.path.exists(morphed_file):
                    try:
                            
                        print(f"Morphing {mod}-{model}-{size}-{subject}...")
                        modelfile = f"{model_dir}/{model}-{size}-{subject}.pickle"
                        model= load.unpickle(modelfile)
                        hlist = model.h
                        inanim,anim = hlist[0],hlist[1]       # if mod is effect : anim: contrast, inanim: general
                        
                        # Read forward solution from cash
                        if i>9:
                            session = "ImageNet01"
                        elif i==7:
                            session = "ImageNet04"   
                        else:
                            session = "ImageNet03"    
                        fwd_file=fwd_dir / f"{subject}_ses-{session}/{subject}-fwd.fif"
                        fwd = mne.read_forward_solution(str(fwd_file), verbose=False)
                        
                        #morphing 
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
                               
                        an_L_fs, an_R_fs, anim_fs = morph_hemi(
                            stc_vec_anim,
                            subject=subject,
                            subject_to="fsaverage2",
                            subjects_dir=subjects_dir,
                            src_tag="vol-7",
                        )
                               
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
                       
                    except Exception as e:
                        print(f"\n Error processing {subject}: {e}\n") 

# %% [markdown]
# # Make dataset of similrities 

# %%
trials_per_subset = 200
dirs = {
    "NCRF-DC": "models/samesize/dc",
    "MNE": "models/samesize/MNE",
    "NCRF-EC": "models/samesize/effect",
}

def compute_results(method_name, model_dir):
    print(f"Processing {method_name} results...")
    summary_avg = []
    contrast_results, condition_results, summary_tmaps = [], [], []
    rows_raw = []

    for size in sizes:
        cases = []
        contrast_rs, anim_rs, inan_rs = [], [], []

        subset_n_trials = int(size * trials_per_subset)

        for i in range(1, 31):
            subject = f"sub-{i:02d}"
            try:
                inan1, anim1 = load_model_subset(model_dir, 1, subject, size)
                inan2, anim2 = load_model_subset(model_dir, 2, subject, size)

                if method_name == "NCRF-EC":
                    d1, d2 = anim1, anim2  # contrast saved in anim
                else:
                    d1, d2 = anim1 - inan1, anim2 - inan2

                cases.append([subject, "M1", "contrast", d1])
                cases.append([subject, "M2", "contrast", d2])

                i1 = inan1.get_data().ravel()
                i2 = inan2.get_data().ravel()
                a1 = anim1.get_data().ravel()
                a2 = anim2.get_data().ravel()
                d1_flat = d1.get_data().ravel()
                d2_flat = d2.get_data().ravel()

                if method_name != "NCRF-EC":
                    if np.std(a1) == 0 or np.std(a2) == 0:
                        r_anim = np.nan
                    else:
                        r_anim = np.corrcoef(a1, a2)[0, 1]

                    if np.std(i1) == 0 or np.std(i2) == 0:
                        r_inan = np.nan
                    else:
                        r_inan = np.corrcoef(i1, i2)[0, 1]

                    anim_rs.append(r_anim)
                    inan_rs.append(r_inan)

                    
                    rows_raw.append({
                        "method": method_name,
                        "subject": subject,
                        "Subset Size": subset_n_trials,
                        "kind": "anim",
                        "r": r_anim
                    })
                    rows_raw.append({
                        "method": method_name,
                        "subject": subject,
                        "Subset Size": subset_n_trials,
                        "kind": "inan",
                        "r": r_inan
                    })

                # Contrast
                if np.std(d1_flat) == 0 or np.std(d2_flat) == 0 or np.any(np.isnan(d1_flat)) or np.any(np.isnan(d2_flat)):
                    r_contrast = np.nan
                else:
                    r_contrast = np.corrcoef(d1_flat, d2_flat)[0, 1]

                contrast_rs.append(r_contrast)

                # per subject similarities for contrast
                rows_raw.append({
                    "method": method_name,
                    "subject": subject,
                    "Subset Size": subset_n_trials,
                    "kind": "contrast",
                    "r": r_contrast
                })

            except Exception as e:
                print(f"Skipping {subject} at size {size}: {e}")
                continue

        # Averaged model
        if cases:
            data = Dataset.from_caselist(["subject", "model", "animacy", "ncrf"], cases)
            data_avg = data.aggregate("model", drop_bad=True)
            m1 = data_avg["ncrf"][data_avg["model"] == "M1"][0].get_data().ravel()
            m2 = data_avg["ncrf"][data_avg["model"] == "M2"][0].get_data().ravel()
            r_avg = np.corrcoef(m1, m2)[0, 1]
            summary_avg.append({"Subset Size": subset_n_trials, "pearson_r": r_avg})

        # within-subject summaries
        if contrast_rs:
            contrast_results.append({
                "Subset Size": subset_n_trials,
                "Mean Pearson r": np.nanmean(contrast_rs),
                "N Subjects": int(np.sum(~np.isnan(contrast_rs)))
            })

        if method_name != "NCRF-EC":
            if anim_rs and inan_rs:
                condition_results.append({
                    "Subset Size": subset_n_trials,
                    "Anim": np.nanmean(anim_rs),
                    "Inan": np.nanmean(inan_rs),
                    "N Subjects": int(np.sum(~np.isnan(anim_rs)))
                })

        # t-map ,T2-Map
        if cases:
            data['ncrf_norm'] = [nd.norm('space') for nd in data['ncrf']]
            res_m1 = testnd.TTestOneSample(data.sub('model =="M1"')['ncrf_norm'], samples=0)
            res_m2 = testnd.TTestOneSample(data.sub('model == "M2"')['ncrf_norm'], samples=0)
            r_tmaps = np.corrcoef(res_m1.t.x.ravel(), res_m2.t.x.ravel())[0, 1]

            res_m1_t2 = testnd.Vector(data.sub('model =="M1"')['ncrf'], samples=0)
            res_m2_t2 = testnd.Vector(data.sub('model == "M2"')['ncrf'], samples=0)
            r_t2 = np.corrcoef(res_m1_t2.t2.x.ravel(), res_m2_t2.t2.x.ravel())[0, 1]

            summary_tmaps.append({
                "Subset Size": subset_n_trials,
                "Tmap_Corr": r_tmaps,
                "T2map_Corr": r_t2
            })

    results = {
        "avg": pd.DataFrame(summary_avg),
        "contrast": pd.DataFrame(contrast_results),
        "condition": pd.DataFrame(condition_results) if method_name != "NCRF-EC" else None,
        "tmap": pd.DataFrame(summary_tmaps),
        "raw": pd.DataFrame(rows_raw),
    }
    return results


results_ncrf_dc = compute_results("NCRF-DC", dirs["NCRF-DC"])
results_mne = compute_results("MNE", dirs["MNE"])
results_ncrf_ec = compute_results("NCRF-EC", dirs["NCRF-EC"])

print(f"MNE:\n{results_mne["contrast"]}")    
print(f"NCRF-DC:\n{results_ncrf_dc["contrast"]}") 
print(f"NCRF-EC:\n{results_ncrf_ec["contrast"]}") 

# %%
print(f"NCRF-DC:\n{results_ncrf_dc["raw"]}") 

# %%
#df = results_ncrf_dc["raw"].query('kind == "contrast"')[["subject", "Subset Size", "r"]]
#print(df.to_string(index=False))

# %% [markdown]
# # plot

# %% [markdown]
# ## Trial Size plots 

# %%
raw_all = pd.concat([
    results_mne["raw"],
    results_ncrf_dc["raw"],
    results_ncrf_ec["raw"],
    
    
    
], ignore_index=True)

df_contrast = raw_all.query('kind == "contrast"').copy()
df_contrast["Subset Size"] = pd.Categorical(
    df_contrast["Subset Size"],
    categories=sorted(df_contrast["Subset Size"].unique()),
    ordered=True
)

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 5))
ax = sns.violinplot(
    data=df_contrast,
    x="Subset Size",
    y="r",
    hue="method",      # remove this line if you want one method at a time
    inner="box",
    cut=0
)
ax.set_xlabel("Trial size (trials)")
ax.set_ylabel("Pearson r (M1 vs M2)")
ax.set_ylim(-0.3, 1)
plt.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

# %%
raw_all = pd.concat([
    
    results_ncrf_ec["raw"],
    results_ncrf_dc["raw"],
], ignore_index=True)

df_contrast = raw_all.query('kind == "contrast" and method in ["NCRF-DC", "NCRF-EC"]').copy()

df_contrast["Subset Size"] = pd.Categorical(
    df_contrast["Subset Size"],
    categories=sorted(df_contrast["Subset Size"].unique()),
    ordered=True
)

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 5))
ax = sns.violinplot(
    data=df_contrast,
    x="Subset Size",
    y="r",
    hue="method",
    split=True,     # left/right halves
    inner="quartile",
    cut=0
)

ax.set_xlabel("Trial size (trials)")
ax.set_ylabel("Pearson r (M1 vs M2)")
ax.set_ylim(-0.2, 1)



ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()
plt.show()


# %%
df_dc = results_ncrf_dc["avg"].copy()
df_dc["Method"] = "NCRF-DC"
df_mne = results_mne["avg"].copy()
df_mne["Method"] = "MNE"
df_ec = results_ncrf_ec["avg"].copy()
df_ec["Method"] = "NCRF-EC"
df_all = pd.concat([df_dc, df_mne, df_ec], ignore_index=True)

palette_custom = {"NCRF-DC": "#E68632", "MNE": "#3777B0", "NCRF-EC": "#228B22"}
plt.figure(figsize=(7, 4))
sns.lineplot(
    data=df_all,
    x="Subset Size",
    y="pearson_r",
    hue="Method",
    #style="Method",
    marker='o',
    markers=True,
    dashes=True,
    linewidth=2.2,
    palette=palette_custom,
)

plt.ylim(0, 1)
plt.ylabel("Pearson r", fontsize=11)
plt.xlabel("Number of Trials", fontsize=11)
plt.title("Averaged Model Correlations ", fontsize=13)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(title="", loc="upper left", frameon=False)
plt.tight_layout()
plt.show()

#Within subject (avg of corelation for each subject)

df_dc_contrast = results_ncrf_dc["contrast"].copy()
df_dc_contrast.rename(columns={"Mean Pearson r": "pearson_r"}, inplace=True)
df_dc_contrast["Method"] = "NCRF-DC"
df_mne_contrast = results_mne["contrast"].copy()
df_mne_contrast.rename(columns={"Mean Pearson r": "pearson_r"}, inplace=True)
df_mne_contrast["Method"] = "MNE"
df_ec_contrast = results_ncrf_ec["contrast"].copy()
df_ec_contrast.rename(columns={"Mean Pearson r": "pearson_r"}, inplace=True)
df_ec_contrast["Method"] = "NCRF-EC"
df_all = pd.concat([df_dc_contrast, df_mne_contrast, df_ec_contrast], ignore_index=True)

palette_custom = {"NCRF-DC": "#E68632", "MNE": "#3777B0", "NCRF-EC": "#228B22"}
plt.figure(figsize=(7, 4))
sns.lineplot(
    data=df_all,
    x="Subset Size",
    y="pearson_r",
    hue="Method",
    marker='o',
    markers=True,
    dashes=True,
    linewidth=2.2,
    palette=palette_custom,
)

plt.ylim(0, 1)
plt.ylabel("Mean Pearson r", fontsize=11)
plt.xlabel("Number of Trials", fontsize=11)
plt.title("Within-subject Contrast Correlations ", fontsize=13)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(title="", loc="upper left", frameon=False)
plt.tight_layout()
plt.show()

# %%
#PLT T-MAP and T2-map
df_dc = results_ncrf_dc["tmap"].copy()
df_dc["Method"] = "NCRF-DC"
df_mne = results_mne["tmap"].copy()
df_mne["Method"] = "MNE"
df_ec = results_ncrf_ec["tmap"].copy()
df_ec["Method"] = "NCRF-EC"
df_all = pd.concat([df_dc, df_mne, df_ec], ignore_index=True)


df_long = df_all.melt(
    id_vars=["Subset Size", "Method"],
    value_vars=["Tmap_Corr", "T2map_Corr"],
    var_name="Map Type",
    value_name="Correlation"
)

df_tmap = df_long[df_long["Map Type"] == "Tmap_Corr"]
df_t2map = df_long[df_long["Map Type"] == "T2map_Corr"]

palette = {
    "NCRF-DC": "#E68632",  
    "MNE": "#3777B0",      
    "NCRF-EC": "#228B22"   
}

# t-map
plt.figure(figsize=(7, 4))
sns.lineplot(
    data=df_tmap,
    x="Subset Size",
    y="Correlation",
    hue="Method",
    marker="o",
    linewidth=2,
    palette=palette
)
plt.ylim(0, 1)
plt.ylabel("Correlation (r)", fontsize=11)
plt.xlabel("Number of Trials", fontsize=11)
plt.title("T-map Correlations ", fontsize=13)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(title="", loc="upper left", frameon=False)
plt.tight_layout()
plt.show()

# t2
plt.figure(figsize=(7, 4))
sns.lineplot(
    data=df_t2map,
    x="Subset Size",
    y="Correlation",
    hue="Method",
    marker="o",
    linewidth=2,
    palette=palette
)
plt.ylim(0, 1)
plt.ylabel("Correlation (r)", fontsize=11)
plt.xlabel("Number of Trials", fontsize=11)
plt.title("T²-map Correlations ", fontsize=13)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(title="", loc="upper left", frameon=False)
plt.tight_layout()
plt.show()


# %%

# %%

# %%
