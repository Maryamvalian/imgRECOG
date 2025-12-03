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
# # 21 subjects

# %%
sizes = [0.25, 0.5, 1, 2, 4, 6, 8] #for i<10
#sizes = [0.25, 0.5, 1, 2]
trials_per_subset = 200
dirs = {
    "NCRF-DC": "models/samesize/dc",
    "MNE": "models/samesize/2sesmne",
    "NCRF-EC": "models/samesize/effect",
}
#---------------------------------
def load_model_subset(model_dir, m, subject, size):
   
   
    if "2sesmne" in model_dir.lower() or "mne" in model_dir.lower():
        file_path = f"{model_dir}/{m}-{size}-{subject}.pickle"
    else:
        file_path = f"{model_dir}/M{m}-{size}-{subject}.pickle"
    
    inan, anim = load.unpickle(file_path)
    return inan, anim
#-------------------------------------
def compute_results(method_name, model_dir):
    print(f"Processing {method_name} results...")
    summary_avg = []
    contrast_results, condition_results, summary_tmaps = [], [], []

    for size in sizes:
        cases = []
        contrast_rs, anim_rs, inan_rs = [], [], []

        for i in range(1, 10):
            
            if i in [4, 6, 7]: continue
            subject = f"sub-{i:02d}"
            try:
                inan1, anim1 = load_model_subset(model_dir, 1, subject, size)
                inan2, anim2 = load_model_subset(model_dir, 2, subject, size)
                if method_name=="NCRF-EC":
                    d1, d2 = inan1, inan2  #contrast is the first one in morphed file
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
    
                if method_name!="NCRF-EC":

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
                    
                
                if np.std(d1_flat) == 0 or np.std(d2_flat) == 0 or np.any(np.isnan(d1_flat)) or np.any(np.isnan(d2_flat)):
                    
                    r_contrast = np.nan
                else:
                    r_contrast = np.corrcoef(d1_flat, d2_flat)[0, 1]
        
                contrast_rs.append(r_contrast)

            except Exception as e:
                print(f"Skipping {subject} at size {size}: {e}")
                continue

        #Averaged model
        if cases:
            data = Dataset.from_caselist(["subject", "model", "animacy", "ncrf"], cases)
            data_avg = data.aggregate("model", drop_bad=True)
            m1 = data_avg["ncrf"][data_avg["model"] == "M1"][0].get_data().ravel()
            m2 = data_avg["ncrf"][data_avg["model"] == "M2"][0].get_data().ravel()
            r_avg = np.corrcoef(m1, m2)[0, 1]
            summary_avg.append({"Subset Size": int(size * trials_per_subset), "pearson_r": r_avg})

        #Mean R
        if contrast_rs:
            contrast_results.append({
                "Subset Size": int(size * trials_per_subset),
                "Mean Pearson r": np.nanmean(contrast_rs),
                "N Subjects": len(contrast_rs)
            })
        
        if method_name!="NCRF-EC":
            if anim_rs and inan_rs:
                condition_results.append({
                    "Subset Size": int(size * trials_per_subset),
                    "Anim": np.nanmean(anim_rs),
                    "Inan": np.nanmean(inan_rs),
                    "N Subjects": len(anim_rs)
                })
            
        # T-map
        if cases:
            data['ncrf_norm'] = [nd.norm('space') for nd in data['ncrf']]
            res_m1 = testnd.TTestOneSample(data.sub('model =="M1"')['ncrf_norm'], samples=0)
            res_m2 = testnd.TTestOneSample(data.sub('model == "M2"')['ncrf_norm'], samples=0)
            r_tmaps = np.corrcoef(res_m1.t.x.ravel(), res_m2.t.x.ravel())[0, 1]

            res_m1_t2 = testnd.Vector(data.sub('model =="M1"')['ncrf'], samples=0)
            res_m2_t2 = testnd.Vector(data.sub('model == "M2"')['ncrf'], samples=0)
            r_t2 = np.corrcoef(res_m1_t2.t2.x.ravel(), res_m2_t2.t2.x.ravel())[0, 1]

            summary_tmaps.append({
                "Subset Size": int(size * trials_per_subset),
                "Tmap_Corr": r_tmaps,
                "T2map_Corr": r_t2
            })

    
    results = {
        "avg": pd.DataFrame(summary_avg),
        "contrast": pd.DataFrame(contrast_results),
        "condition": pd.DataFrame(condition_results) if method_name!="NCRF-EC" else None,
        "tmap": pd.DataFrame(summary_tmaps),
    }
    return results


# ----------------------------------
results_ncrf_dc = compute_results("NCRF-DC", dirs["NCRF-DC"])
results_mne = compute_results("MNE", dirs["MNE"])
results_ncrf_ec = compute_results("NCRF-EC", dirs["NCRF-EC"])
print("All Results are ready.")
print(f"MNE:\n{results_mne["condition"]}")

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
    #style="Method",
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
df_mne_cond = results_mne["condition"].copy()
df_dc_cond = results_ncrf_dc["condition"].copy()
palette_cond = {"Animate": "#66BB66", "Inanimate": "#3399FF" }
nsub_mne = int(df_mne_cond["N Subjects"].iloc[0]) if not df_mne_cond.empty else 0
nsub_dc = int(df_dc_cond["N Subjects"].iloc[0]) if not df_dc_cond.empty else 0

#mne
plt.figure(figsize=(6, 4))
x = np.arange(len(df_mne_cond["Subset Size"]))  # positions for bars
width = 0.25                                    # bar width

plt.bar(x - width/2, df_mne_cond["Anim"], width, label="Animate", color="#66BB66", edgecolor="black", alpha=0.9)
plt.bar(x + width/2, df_mne_cond["Inan"], width, label="Inanimate", color="#3399FF", edgecolor="black", alpha=0.9)

plt.xticks(x, df_mne_cond["Subset Size"].astype(int))
plt.ylim(0, 1)
plt.xlabel("Subset Size", fontsize=11)
plt.ylabel("Mean Pearson r", fontsize=11)
plt.title(f"within subject MNE ", fontsize=13) #({nsub_mne})
plt.legend(title="", frameon=False)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

#NCRF---------------------------
plt.figure(figsize=(6, 4))
x = np.arange(len(df_dc_cond["Subset Size"]))
plt.bar(x - width/2, df_dc_cond["Anim"], width, label="Animate", color="#66BB66", edgecolor="black", alpha=0.9)
plt.bar(x + width/2, df_dc_cond["Inan"], width, label="Inanimate", color="#3399FF", edgecolor="black", alpha=0.9)

plt.xticks(x, df_dc_cond["Subset Size"].astype(int))
plt.ylim(0, 1)
plt.xlabel("Number of Trials", fontsize=11)
plt.ylabel("Mean Pearson r", fontsize=11)
plt.title(f"within subject NCRF-DC ", fontsize=13) #({nsub_dc})
plt.legend(title="", frameon=False)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()



# %% [markdown]
# # Explained var by Trial size

# %%

subjects = [f"sub-{i:02d}" for i in range(1, 10) if i not in [4, 6, 7]]
sizes = [  4 , 6 ,8 ]

subjects = [f"sub-{i:02d}" for i in range(10, 31) if i not in [4, 6, 7]]
sizes = [0.25, 0.5, 1, 2]

trials_per_subset = 200
trial_sizes = [s * trials_per_subset for s in sizes]

model_dirs = {
    "NCRF-DC": "models/samesize/dc",
    "NCRF-EC": "models/samesize/effect"
}
def compute_ev_for_model(model_dir):
    print(model_dir)
    ev_means = []
    for size in sizes:
        ev_list = []
        print(f"Processing size = {size}...")
        for subject in subjects:
            try:
                m1_file = f"{model_dir}/1-{size}-{subject}.pickle"
                m2_file = f"{model_dir}/2-{size}-{subject}.pickle"
                m1 = eel.load.unpickle(m1_file)
                m2 = eel.load.unpickle(m2_file)

                data2 = m2._data
                ev = m1.compute_explained_variance(data2)
                ev_list.append(float(ev))
            except Exception as e:
                print(f"{subject}: skipped ({e})")
        mean_ev = np.nanmean(ev_list) if ev_list else np.nan
        ev_means.append(mean_ev)
    return ev_means
#-------------------
ev_dc = compute_ev_for_model(model_dirs["NCRF-DC"])
ev_ec = compute_ev_for_model(model_dirs["NCRF-EC"])

#---------------------
plt.figure(figsize=(6,4))
plt.plot(trial_sizes, ev_dc, marker='o', linewidth=2, label='NCRF-DC')
plt.plot(trial_sizes, ev_ec, marker='s', linewidth=2, label='NCRF-EC')

plt.xticks(trial_sizes, fontsize=11)
plt.yticks(fontsize=11)
plt.xlabel("Number of Trials",fontsize=11)
plt.ylabel("Mean EV",fontsize=11)
plt.title("Cross-prediction EV (7 subject)",fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11, loc='best')
plt.tight_layout()
plt.show()


# %% [markdown]
# # MU experiment

# %%
mod="effect"
model_dir = "models/mu/ec"
mu_values = [1e-8, 1e-7, 1e-6, 1e-5, 4e-5, 1e-4, 4e-4, 1e-3]
thr=5e-12
size = 2                     
trials_per_subset = 200
subjects = [i for i in range(10, 31) if i not in [4, 6, 7]]

# ---------------------------------------
def load_model_mu(model_dir, m, subject, mu):
    
    file_path = f"{model_dir}/M{m}-{mu}-{subject}.pickle"
    inan, anim = load.unpickle(file_path)
    return inan, anim

# ---------------------------------------
def compute_results_mu(model_dir_base, mu_values):
    
    all_results = {}

    for mu in mu_values:
        print(f"Processing Mu= {mu}... ")
        summary_avg = []
        contrast_results = []
        summary_tmaps = []
        contrast_rs_all = []

        for i in subjects:
            subject = f"sub-{i:02d}"
            try:
    
                inan1, anim1 = load_model_mu(model_dir, 1, subject, mu)
                inan2, anim2 = load_model_mu(model_dir, 2, subject, mu)
                if mod == "dummy":
                    d1, d2 = anim1 - inan1, anim2 - inan2
                elif mod == "effect":
                    d1, d2 = inan1, inan2
                contrast_rs_all.append((subject, d1, d2))

            except Exception as e:
                print(f"Skipping {subject} MU={mu}: {e}")
                continue

        cases = []
        contrast_rs = []

        for subject, d1, d2 in contrast_rs_all:
            cases.append([subject, "M1", "contrast", d1])
            cases.append([subject, "M2", "contrast", d2])

            d1_flat = d1.get_data().ravel()
            d2_flat = d2.get_data().ravel()
            if np.std(d1_flat) == 0 or np.std(d2_flat) == 0 or np.any(np.isnan(d1_flat)) or np.any(np.isnan(d2_flat)):
                r_contrast = np.nan
            else:
                r_contrast = np.corrcoef(d1_flat, d2_flat)[0, 1]
            contrast_rs.append(r_contrast)

        #pearson-R(avg)
        if cases:
            data = Dataset.from_caselist(["subject", "model", "animacy", "ncrf"], cases)
            data_avg = data.aggregate("model", drop_bad=True)
            m1 = data_avg["ncrf"][data_avg["model"] == "M1"][0].get_data().ravel()
            m2 = data_avg["ncrf"][data_avg["model"] == "M2"][0].get_data().ravel()
            r_avg = np.corrcoef(m1, m2)[0, 1]
            #=================awcosine
            cos_aw_t, n_used, times = ndvar_AWcosine(data_avg["ncrf"][data_avg["model"] == "M1"][0], data_avg["ncrf"][data_avg["model"] == "M2"][0], thr=thr, mode="or")
            if cos_aw_t is None or len(cos_aw_t) == 0 or np.all(np.isnan(cos_aw_t)):
                awcos_mean = 0.0
            else:
                awcos_mean = np.nanmean(cos_aw_t)
            #================
                

            summary_avg.append({
                "Mu": mu,
                "Subset Size": int(size * trials_per_subset),
                "pearson_r": r_avg,
                "AW-Cosine": awcos_mean     #======= 
            })
            
           
            

        #Mean Pearson_R (within subject)
        
        # ================= AW-Cosine 
        if contrast_rs_all:
            awcos_subjects = []
            for subject, d1, d2 in contrast_rs_all:
                
                cos_aw_t, n_used, times = ndvar_AWcosine(d1, d2, thr=thr, mode="or")
                if cos_aw_t is None or len(cos_aw_t) == 0 or np.all(np.isnan(cos_aw_t)):
                    awcos = 0.0   
                else:
                    awcos = np.nanmean(cos_aw_t)
                
                awcos_subjects.append(awcos)
        
            mean_awcos = np.nanmean(awcos_subjects)
        #=====================================


    
        if contrast_rs:
            contrast_results.append({
                "Mu": mu,
                "Subset Size": int(size * trials_per_subset),
                "Mean Pearson r": np.nanmean(contrast_rs),
                "Mean AW-Cosine": mean_awcos, 
                "N Subjects": len(contrast_rs)
            })

        #T-map
        if cases:
            data['ncrf_norm'] = [nd.norm('space') for nd in data['ncrf']]
            res_m1 = testnd.TTestOneSample(data.sub('model == "M1"')['ncrf_norm'], samples=0)
            res_m2 = testnd.TTestOneSample(data.sub('model == "M2"')['ncrf_norm'], samples=0)
            r_tmaps = np.corrcoef(res_m1.t.x.ravel(), res_m2.t.x.ravel())[0, 1]

            res_m1_t2 = testnd.Vector(data.sub('model == "M1"')['ncrf'], samples=0)
            res_m2_t2 = testnd.Vector(data.sub('model == "M2"')['ncrf'], samples=0)
            r_t2 = np.corrcoef(res_m1_t2.t2.x.ravel(), res_m2_t2.t2.x.ravel())[0, 1]

            summary_tmaps.append({
                "Mu": mu,
                "Subset Size": int(size * trials_per_subset),
                "Tmap_Corr": r_tmaps,
                "T2map_Corr": r_t2
            })
        all_results[mu] = {
            "avg": pd.DataFrame(summary_avg),
            "contrast": pd.DataFrame(contrast_results),
            "tmap": pd.DataFrame(summary_tmaps),
        }

    return all_results
# -------------------------
results_mu = compute_results_mu(model_dir, mu_values)


df_avg_all = pd.concat([results_mu[mu]["avg"] for mu in mu_values], ignore_index=True)
df_contrast_all = pd.concat([results_mu[mu]["contrast"] for mu in mu_values], ignore_index=True)
df_tmap_all = pd.concat([results_mu[mu]["tmap"] for mu in mu_values], ignore_index=True)
print("\nAverage correlations:")
print(df_avg_all)
print("\nContrast correlations:")
print(df_contrast_all)
print("\nT-map correlations:")
print(df_tmap_all)



# %%
#mean_r
sns.set(style="whitegrid", context="talk")

plt.figure(figsize=(7, 4))

sns.lineplot(
    data=df_contrast_all,
    x="Mu",              
    y="Mean Pearson r",
    marker="o",
    linewidth=2.4,
    color="#E68632"
)

plt.xscale("log")
plt.xlabel("Regularization", fontsize=11)
plt.ylabel("Mean Pearson r ", fontsize=11)
plt.title("Within subject correlation(NCRF-DC)", fontsize=13)
plt.ylim(0, 1)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
#avg
sns.set(style="whitegrid", context="talk")

plt.figure(figsize=(7, 4))

sns.lineplot(
    data=df_avg_all,
    x="Mu",
    y="pearson_r",
    marker="o",
    linewidth=2.4,
    color="#E68632"
)

plt.xscale("log")
plt.xlabel("Regularization", fontsize=11)
plt.ylabel("pearson-R ", fontsize=11)
plt.title("Average Model Correlation(NCRF-DC)", fontsize=13)
plt.ylim(0, 1)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

sns.set(style="whitegrid", context="talk")

plt.figure(figsize=(7, 4))

#t-map
sns.lineplot(
    data=df_tmap_all,
    x="Mu",
    y="Tmap_Corr",
    marker="o",
    linewidth=2.4,
    color="#E68632",
    label="T-map"
)

#t2map
sns.lineplot(
    data=df_tmap_all,
    x="Mu",
    y="T2map_Corr",
    marker="s",
    linewidth=2.4,
    color="#3777B0",
    label="T2-map"
)

plt.xscale("log")
plt.xlabel("Regularization", fontsize=11)
plt.ylabel("Pearson-R", fontsize=11)
plt.title(f"T-map Correlations ({mod})", fontsize=13)
plt.ylim(-0.01, 1)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(title="", frameon=False)
plt.tight_layout()
plt.show()


# %%
sns.set(style="whitegrid", context="talk")

# -
plt.figure(figsize=(7, 4))

# Pearson r line
sns.lineplot(
    data=df_contrast_all,
    x="Mu",
    y="Mean Pearson r",
    marker="o",
    linewidth=2.4,
    color="#3777B0",
    label="Pearson r"
)

# AW-Cosine line
sns.lineplot(
    data=df_contrast_all,
    x="Mu",
    y="Mean AW-Cosine",
    marker="s",
    linewidth=2.4,
    color="#E68632",
    label="AW-Cosine"
)

plt.xscale("log")
plt.xlabel("Mu Values", fontsize=11)
plt.ylabel("Within-Subject Similarity", fontsize=11)
plt.title(f"Within-Subject  ({mod}),AW Threshold={thr})", fontsize=13)
plt.ylim(0, 1)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(title="", frameon=False)
plt.tight_layout()
plt.show()


#
plt.figure(figsize=(7, 4))

# Pearson r line
sns.lineplot(
    data=df_avg_all,
    x="Mu",
    y="pearson_r",
    marker="o",
    linewidth=2.4,
    color="#3777B0",
    label="Pearson r"
)

# AW-Cosine line
sns.lineplot(
    data=df_avg_all,
    x="Mu",
    y="AW-Cosine",
    marker="s",
    linewidth=2.4,
    color="#E68632",
    label="AW-Cosine"
)

plt.xscale("log")
plt.xlabel("Mu Values ", fontsize=11)
plt.ylabel("Similarity M1,M2", fontsize=11)
plt.title(f"Average Model ({mod})(AW Threshold={thr})", fontsize=13)
plt.ylim(-0.02, 1.01)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(title="", frameon=False)
plt.tight_layout()
plt.show()


# %%
plt.figure(figsize=(6, 4))

# --- NCRF-DC (blue) ---
plt.errorbar(
    mu_values,
    ev_dc,
    yerr=sem_dc,
    fmt='o',
    capsize=4,
    elinewidth=1.2,
    markersize=6,
    color='C0',
    ecolor='C0',
    label='NCRF-DC'
)
plt.plot(
    mu_values,
    ev_dc,
    color='C0',
    linewidth=1.8
)


plt.errorbar(
    mu_values,
    ev_ec,
    yerr=sem_ec,
    fmt='s',
    capsize=4,
    elinewidth=1.2,
    markersize=6,
    color='C1',
    ecolor='C1',
    label='NCRF-EC'
)
plt.plot(
    mu_values,
    ev_ec,
    color='C1',
    linewidth=1.8
)

plt.semilogx()
plt.xlabel("Mu Values", fontsize=9)
plt.ylabel("Mean EV", fontsize=9)
plt.yticks(fontsize=11)
plt.xticks(fontsize=11)
plt.title("Cross-prediction EV (Data2 by M1) - Fixed trial size 400", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11, loc='best')
plt.tight_layout()
plt.show()



# %%
subjects = [f"sub-{i:02d}" for i in range(10, 31)]
mu_values = [1e-8, 1e-7, 1e-6, 1e-5, 4e-5, 1e-4, 4e-4, 1e-3]

model_dirs = {
    "NCRF-DC": "models/mu",
    "NCRF-EC": "models/mu/ec"
}


def compute_ev_mu(model_dir):
    ev_means = []
    for mu in mu_values:
        ev_list = []
        print(f"Processing mu = {mu}...")
        for subject in subjects:
            try:
                m1_file = f"{model_dir}/1-{mu}-{subject}.pickle"
                m2_file = f"{model_dir}/2-{mu}-{subject}.pickle"
                m1 = eel.load.unpickle(m1_file)
                m2 = eel.load.unpickle(m2_file)

                data2 = m2._data
                ev = m1.compute_explained_variance(data2)
                ev_list.append(float(ev))

            except Exception as e:
                print(f"{subject}: skipped ({e})")

        mu_mean = np.nanmean(ev_list) if ev_list else np.nan
        ev_means.append(mu_mean)
    return ev_means
# -----
print("======NCRF-DC:")
ev_dc = compute_ev_mu(model_dirs["NCRF-DC"])
print("======NCRF-EC:")
ev_ec = compute_ev_mu(model_dirs["NCRF-EC"])
#----------
plt.figure(figsize=(6, 4))
plt.semilogx(mu_values, ev_dc, marker='o', linewidth=2, label='NCRF-DC')
plt.semilogx(mu_values, ev_ec, marker='s', linewidth=2, label='NCRF-EC')
plt.xlabel("Mu Values", fontsize=9)
plt.ylabel("Mean EV", fontsize=9)
plt.yticks(fontsize=11)
plt.xticks(fontsize=11)
plt.title("Cross-prediction EV (Data2 by M1)- Fixed trial size 400", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11,loc='best')
plt.tight_layout()
plt.show()


print("\nMarkdown Table:\n")
print("| Mu Value | Mean EV (NCRF-DC) | Mean EV (NCRF-EC) |")
print("|----------|-------------------|-------------------|")

for mu, dc, ec in zip(mu_values, ev_dc, ev_ec):
    print(f"| {mu} | {dc:.4f} | {ec:.4f} |")



# %%
plt.figure(figsize=(6, 4))
plt.semilogx(mu_values, ev_dc, marker='o', linewidth=2, label='NCRF-DC')
plt.semilogx(mu_values, ev_ec, marker='s', linewidth=2, label='NCRF-EC')

plt.xlabel("Mu Values", fontsize=9)
plt.ylabel("Mean EV", fontsize=9)
plt.yticks(fontsize=11)
plt.xticks(fontsize=11)
plt.title("Cross-prediction EV (Data2 by M1)- Fixed trial size 400", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)

# ===== Add shaded cross-validation range =====
mu_low = 1e-6      # minimum mu observed
mu_high = 3.41e-3  # maximum mu observed
#mu_low = 1e-5   #25%
#mu_high = 3.3e-5  #75%


#plt.axvspan(mu_low, mu_high, color='lightblue', alpha=0.25, label='CV range')
# ====== Add Cross-validated μ ranges ======

# Actual observed ranges based on your summary
mu_min = 1e-6        # minimum observed
mu_max = 3.412e-3    # maximum observed

mu_q25 = 1e-5        # 25th percentile
mu_q75 = 3.3e-5      # 75th percentile

# Shade full min–max range (light blue)
plt.axvspan(mu_min, mu_max, color='lightblue', alpha=0.20, label='CV range (min-max)')

# Shade IQR range (darker blue)
plt.axvspan(mu_q25, mu_q75, color='royalblue', alpha=0.25, label='CV range (25–75%)')







plt.legend(fontsize=11, loc='best')
plt.tight_layout()
plt.show()


# %%
model_dir = "models/mu"
subjects = [f"sub-{i:02d}" for i in range(10, 31)]
mu_values = [1e-8, 1e-7, 1e-6, 1e-5, 4e-5, 1e-4, 4e-4, 1e-3]
ev_m1_all = []
ev_m2_all = []
ev_cross_all = []

for mu in mu_values:
    ev_m1_list = []
    ev_m2_list = []
    ev_cross_list = []

    print(f"Processing mu= {mu}...")
    for subject in subjects:
        try:
           
            m1_file = f"{model_dir}/1-{mu}-{subject}.pickle"
            m2_file = f"{model_dir}/2-{mu}-{subject}.pickle"
            m1 = eel.load.unpickle(m1_file)
            m2 = eel.load.unpickle(m2_file)

            ev_m1 = float(m1.explained_var)       
            ev_m2 = float(m2.explained_var)       
            ev_cross = float(m1.compute_explained_variance(m2._data))  

            ev_m1_list.append(ev_m1)
            ev_m2_list.append(ev_m2)
            ev_cross_list.append(ev_cross)

            

        except Exception as e:
            print(f" {subject}: skipped ({e})")

    
    ev_m1_all.append(np.nanmean(ev_m1_list))
    ev_m2_all.append(np.nanmean(ev_m2_list))
    ev_cross_all.append(np.nanmean(ev_cross_list))

    


plt.figure(figsize=(7,4))
plt.semilogx(mu_values, ev_m1_all, marker='o', label='M1 EV (self)')
plt.semilogx(mu_values, ev_m2_all, marker='s', label='M2 EV (self)')
plt.semilogx(mu_values, ev_cross_all, marker='^', label='Cross EV (M1 predict Data2)')
plt.xlabel("Regularization ")
plt.ylabel("Average EV")
plt.title("Explained var NCRF-dc ")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()


# %%

ev_m1_all = []
ev_m2_all = []
ev_cross_avg_all = [] 

for mu in mu_values:
    ev_m1_list = []
    ev_m2_list = []
    ev_cross_avg_list = []

    print(f"Processing mu = {mu}...")
    for subject in subjects:
        try:
          
            m1_file = f"{model_dir}/1-{mu}-{subject}.pickle"
            m2_file = f"{model_dir}/2-{mu}-{subject}.pickle"
            m1 = eel.load.unpickle(m1_file)
            m2 = eel.load.unpickle(m2_file)

            
            ev_m1 = float(m1.explained_var)       
            ev_m2 = float(m2.explained_var)        
            ev_m1_to_data2 = float(m1.compute_explained_variance(m2._data))
            ev_m2_to_data1 = float(m2.compute_explained_variance(m1._data))

            ev_cross_avg = (ev_m1_to_data2 + ev_m2_to_data1) / 2

            ev_m1_list.append(ev_m1)
            ev_m2_list.append(ev_m2)
            ev_cross_avg_list.append(ev_cross_avg)

        except Exception as e:
            print(f" {subject}: skipped ({e})")

    ev_m1_all.append(np.nanmean(ev_m1_list))
    ev_m2_all.append(np.nanmean(ev_m2_list))
    ev_cross_avg_all.append(np.nanmean(ev_cross_avg_list))


plt.figure(figsize=(7,4))
plt.semilogx(mu_values, ev_m1_all, marker='o', label='M1 EV (self)')
plt.semilogx(mu_values, ev_m2_all, marker='s', label='M2 EV (self)')
plt.semilogx(mu_values, ev_cross_avg_all, marker='^', label='Mean Cross EV (Bidirected) ')

plt.xlabel("Regularization ")
plt.ylabel("AverageEV")
plt.title("Cross-dataset Expained variance")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()


# %%
