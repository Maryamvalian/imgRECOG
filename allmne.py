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
from Beyond import morph_hemi          

# %% [markdown]
# # Create STC ans save model

# %%
root = Path("~/Data/ds005810")
subjects_dir = str(Path('~/Data/ds005810/derivatives/freesurfer/subjects').expanduser())
empty_room=root/"sub-emptyroom/ses-20211114/meg/sub-emptyroom_ses-20211114_task-noise_meg.fif"
#rewrite=False         #force rewrite fwd

#Noise
raw_er = mne.io.read_raw_fif(empty_room, preload=True, verbose=False).pick('meg')
raw_er.filter(1., 40., phase="zero-double", verbose=False)
raw_er.resample(100, npad="auto", verbose=False)
noise_cov = mne.compute_raw_covariance(raw_er, method='shrunk', rank=None,verbose=False)

# %% [markdown]
# Only for subject 3 the stim_is_animate is defined as sting - use "True" instead of True.

# %%
root_epochs = Path("/Users/maryamvalian/Data/ds005810/derivatives/preprocessed/epochs")
fwd_dir=Path("/Users/maryamvalian/Data/ds005810/derivatives/eelbrain/cache/raw")
fwd_dir.mkdir(parents=True, exist_ok=True)

for i in range(1, 31):
    if (i<10):
        session="ImageNet04"            #"ImageNet03,04" 
        balanced_size=800                #all runs=8*200=1600 trail each cond 800
    else:
        session="ImageNet01"
        balanced_size=500
    
    subject = f"sub-{i:02d}"
    
    epo_file = root_epochs / f"{subject}_meg_epo.fif"
    clean_fif = root / f"derivatives/preprocessed/raw/{subject}_ses-{session}_task-ImageNet_run-01_clean_meg.fif"
    #fwd_file=fwd_dir / f"{subject}_ses-{session}/{subject}-fwd.fif"
    #fwd_file.parent.mkdir(parents=True, exist_ok=True)
    modelfile = f"models/all_runs/mne/{subject}-{session}-allruns-mne.pickle"

    if os.path.exists(modelfile):
        print(f"{subject} loaded from file.")
        continue
    try:
        

        if not epo_file.exists():
            print(f" Skipping {subject}: no epo file at {epo_file}")
            continue
    
        print(f"Loading {epo_file} ...")
        epochs = mne.read_epochs(str(epo_file), preload=True, verbose=False)
        clean = mne.io.read_raw_fif(clean_fif, preload=True, verbose=False)
    
        
        epochs_resamp = epochs.copy().resample(100, npad="auto", verbose=False)
        meta = epochs_resamp.metadata

        #sub-03 meta data is not boolean convert to boolean 
        if i==3:
            meta["stim_is_animate"] = meta["stim_is_animate"].apply(lambda x: True if str(x).lower() == "true" else False)
    
            
        mask_in = (
            (meta["subject"] == i) &
            (meta["session"] == session) &                  # (meta["run"] == int(run))&
            (meta["stim_is_animate"] ==False)
        )
    
        epochs_in = epochs_resamp[mask_in]
        print(f"#Inanim={len(epochs_in)}")
    
    
        mask_an = (
            (meta["subject"]== i) &
            (meta["session"]== session) &             #(meta["run"]== int(run))& 
            (meta["stim_is_animate"] ==True)
        )
    
        #epochs_an = epochs_resamp[mask_an]
        print(f"#Animate={len(epochs_an)}")

        print(f"Balancing trials ")
        epochs_an = resize_epochs(epochs_resamp[mask_an], balanced_size) 
        epochs_in = resize_epochs(epochs_resamp[mask_in], balanced_size)

        print(f"Blanced #Animate={len(epochs_an)}")
        print(f"Balanced #Inanimate={len(epochs_in)}")



        evoked_anim= epochs_an.average()
        evoked_inanim = epochs_in.average()
        

               
        #src_file = f"{subjects_dir}/{subject}/bem/{subject}-vol-7-src.fif"          #merged L,R same as ncrf
        src_file = f"{subjects_dir}/{subject}/bem/{subject}-vol-7-lr-src.fif"        #not merged l,R
        src = mne.read_source_spaces(str(src_file),verbose=False)
    
        bem_sol_fif=f"{subjects_dir}/{subject}/bem/{subject}-bem-sol.fif"
        bem_sol = mne.read_bem_solution(bem_sol_fif,verbose=False)
        
        trans_fif= f"{root}/derivatives/trans/{subject}-{session}-trans.fif"
        trans=mne.read_trans(trans_fif)

        print("   Computing FWD")
        fwd = mne.make_forward_solution(
                clean.info, trans, src, bem_sol,
                meg=True, eeg=False, mindist=0, verbose=False
            )
    
             
        print("   Inverse...")
        inv = mne.minimum_norm.make_inverse_operator(
            info=clean.info,
            forward=fwd,
            noise_cov=noise_cov,
            loose=1.0,    
            depth=0.8,
            fixed=False,
            verbose=False,
        )
        
        snr = 3.0
        lambda2 = 1.0 / snr**2
        
        
        stc_anim_vec = mne.minimum_norm.apply_inverse(
            evoked_anim, inv, lambda2=lambda2, method='MNE', pick_ori='vector', verbose=False)
        
        stc_inan_vec = mne.minimum_norm.apply_inverse(
            evoked_inanim, inv, lambda2=lambda2, method='MNE', pick_ori='vector', verbose=False)

        
        
        
        print(f"Morphing ...")
        print("   morphing 1/2")
        R,L,anim = morph_hemi(stc_anim_vec, subject=subject, subject_to="fsaverage2",
                              subjects_dir=subjects_dir, src_tag="vol-7")
        print("   morphing 2/2")
        R_in,L_in,inan = morph_hemi(stc_inan_vec, subject=subject, subject_to="fsaverage2",
                              subjects_dir=subjects_dir, src_tag="vol-7")
        
        save.pickle((inan,anim), modelfile)
                  
        
                         
        print(f"=====>{subject } done!")
    except Exception as e:
        print(f"Error processing {subject}: {e}")   
        

# %% [markdown]
# # Group Analysis

# %%
sessions= ["ImageNet03","ImageNet04"] if i<10 else ["ImageNet01"]
cases = []
for i in range(1, 31):
    sessions= ["ImageNet03","ImageNet04"] if i<10 else ["ImageNet01"]
    subject = f"sub-{i:02d}"
    for ses in sessions:
        
        modelfile = Path(f"models/all_runs/mne/{subject}-{ses}-allruns-mne.pickle")    
    
        if modelfile.exists():       
        
            inanim, anim = load.unpickle(modelfile)    
            
            cases.append([subject, 'inanimate', inanim])
            cases.append([subject, 'animate', anim])
        else:
            print(f"{subject} Skipped")
    
data = Dataset.from_caselist(['subject', 'animacy', 'stc'], cases)
data.head()

# %%
#average ImageNet 03 , ImageNet 04
data_agg=data
data_avg=data_agg.aggregate(('animacy% subject'), drop_bad=True)
data_avg

# %% [markdown]
# ## Paired Test

# %%
res = testnd.VectorDifferenceRelated(
    'stc',             
    'animacy',           
    'inanimate',     
    'animate',   
    match='subject',     
    data=data_avg,     
    tfce=True,           
    tstart=0.1,
    tstop=0.7,
    samples=1000
)
save.pickle(res, "Tests/mne/all_runs_mne_PT.pickle")

# %%
diff= res.masked_difference()
p = plot.Butterfly(diff.norm('space'), color='k',title='anim VS inanim')
times = [0.1,0.15,0.21,0.34,0.45,0.5,0.6]

for t in times:
    p.add_vline(t)
for t in times:
    f = plot.GlassBrain(diff.sub(time=t),title=f"anim vs inan(MNE), {t*1000}s")  

# %% [markdown]
# # One sample test
#

# %%
data_inan = data_avg.sub("animacy == 'inanimate'")
result_inan = testnd.Vector('stc', match='subject', data=data_inan, tfce=True, tstart=0.1, tstop=0.6,samples=1000)

data_an = data_avg.sub("animacy == 'animate'")
result_an = testnd.Vector('stc', match='subject', data=data_an, tfce=True, tstart=0.1, tstop=0.6,samples=1000)

save.pickle((result_an,result_inan), "Tests/mne/all_runs-mne_1ST.pickle")

# %%
p = plot.Butterfly(result_inan.masked_difference().norm('space'), color='k')
times = [0.13,0.25,0.4]
for t in times:
    p.add_vline(t)
for t in times:
    f = plot.GlassBrain(result_inan.masked_difference().sub(time=t),title=f"Inanimate, {t}s")  

p = plot.Butterfly(result_an.masked_difference().norm('space'), color='k')
times = [0.13,0.25,0.4]
for t in times:
    p.add_vline(t)
for t in times:
    f = plot.GlassBrain(result_an.masked_difference().sub(time=t),title=f"animate, {t}s")     

# %% [markdown]
# # COMMON RESPONSE

# %%
"""
mod=="common"
cases = []
for i in range(1, 31):
    
    subject = f"sub-{i:02d}"
    modelfile = Path(f"models/mne/{mod}-{subject}-mne.pickle")    

    if modelfile.exists():       
    
        common = load.unpickle(modelfile)    
        
       
        cases.append([subject, common])
    else:
        print(f"{subject} Skipped")
    
data_common= Dataset.from_caselist(['subject', 'stc'], cases)
data_common.head()
"""

# %%
#result_common = testnd.Vector('stc', match='subject', data=data_common, tfce=True, tstart=0.1, tstop=0.6,samples=1000)


# %%
"""p = plot.Butterfly(result_common.masked_difference().norm('space'), color='k')
times = [0.13,0.25,0.35,0.45]
for t in times:
    p.add_vline(t)
for t in times:
    f = plot.GlassBrain(result_common.masked_difference().sub(time=t),title=f"common MNE, {t}s") 
"""

# %% [markdown]
# # CONSISTENCY MAKE MODELS FOR DIFFERENT SIZE MEG 
# ## BALANCED DATA

# %%
root_epochs = Path("/Users/maryamvalian/Data/ds005810/derivatives/preprocessed/epochs")
#-------------------------------------------
#sizes = [0.25, 0.5, 1, 2,  4, 6 ,8]      #for i<10
#------------------------------------------
per_run=200

#----------
def resize_epochs(epochs, n):
    #for balancing the condition trial#
    
    if len(epochs) >= n:
        return epochs[:n]
    else:
        idx = np.random.choice(len(epochs), n, replace=True)
        return epochs[idx]
#--------------------
for size in sizes:
    
    print(f"========= size= {size}====== ")
    cut_per_cond=int(size * per_run/2)
    for i in range(1, 10):
        """
        if (i<10):
            session="ImageNet03"             
        else:
            session="ImageNet01"
        """
        
        subject = f"sub-{i:02d}"
        
        epo_file = root_epochs / f"{subject}_meg_epo.fif"
        
    
        ordered_runs = ["02", "01", "03", "06", "04", "05", "07", "08"]
        cut= size if size>=1 else 1
        subset = ordered_runs[:cut]
        
            
        for model in range (1,3):       #M1,M2
    
            session="ImageNet03" if model==1 else "ImageNet04"            
            run_list = [int(r) for r in subset]
            
            modelfile = f"models/samesize/2sesmne/{model}-{size}-{subject}.pickle"
        
            if os.path.exists(modelfile):
                print(f"{model}-{size}-{subject} loaded from file.")
                continue
            try:
                print(f"{subject}, Run= {run_list}, Session={session}")
                epochs = mne.read_epochs(str(epo_file), preload=True, verbose=False)
                clean_fif = root / f"derivatives/preprocessed/raw/{subject}_ses-{session}_task-ImageNet_run-01_clean_meg.fif"
                clean = mne.io.read_raw_fif(clean_fif, preload=True, verbose=False)
                
                epochs_resamp = epochs.copy().resample(100, npad="auto", verbose=False) 
                    
                meta = epochs_resamp.metadata
                #sub-03 meta data is not boolean convert to boolean 
                if i==3:
                    meta["stim_is_animate"] = meta["stim_is_animate"].apply(lambda x: True if str(x).lower() == "true" else False)
            
                    
                mask_in = (
                    (meta["subject"] == i) &
                    (meta["session"] == session) &   
                    (meta["run"].isin(run_list)) &           #(meta["run"] == int(run))&
                    (meta["stim_is_animate"] ==False)
                )
    
                mask_an = (
                    (meta["subject"]== i) &
                    (meta["session"]== session) & 
                    (meta["run"].isin(run_list)) & 
                    (meta["stim_is_animate"] ==True)
                )
            
                epochs_in = epochs_resamp[mask_in] # Not balancing
                epochs_an = epochs_resamp[mask_an]
                print(f"Befor Balancing: Animate={len(epochs_an)},Inanim={len(epochs_in)} ")
               
                
                
                epochs_an = resize_epochs(epochs_resamp[mask_an], cut_per_cond)
                epochs_in = resize_epochs(epochs_resamp[mask_in], cut_per_cond)        
                print(f"BALANCED : Animate={len(epochs_an)}, Inanim={len(epochs_in)}")
                
        
        
                evoked_anim= epochs_an.average()
                evoked_inanim= epochs_in.average()
                
                src_file = f"{subjects_dir}/{subject}/bem/{subject}-vol-7-lr-src.fif"        #not merged l,R
                src = mne.read_source_spaces(str(src_file),verbose=False)
            
                bem_sol_fif=f"{subjects_dir}/{subject}/bem/{subject}-bem-sol.fif"
                bem_sol = mne.read_bem_solution(bem_sol_fif,verbose=False)
                
                trans_fif= f"{root}/derivatives/trans/{subject}-{session}-trans.fif"
                trans=mne.read_trans(trans_fif)
        
                print("   Computing FWD")
                fwd = mne.make_forward_solution(
                        clean.info, trans, src, bem_sol,
                        meg=True, eeg=False, mindist=0, verbose=False
                    )
            
                          
                print("   Inverse...")
                inv = mne.minimum_norm.make_inverse_operator(
                    info=clean.info,
                    forward=fwd,
                    noise_cov=noise_cov,
                    loose=1.0,    
                    depth=0.8,
                    fixed=False,
                    verbose=False,
                )
                
                snr = 3.0
                lambda2 = 1.0 / snr**2
                
                
                stc_anim_vec = mne.minimum_norm.apply_inverse(
                    evoked_anim, inv, lambda2=lambda2, method='MNE', pick_ori='vector', verbose=False)
                
                stc_inan_vec = mne.minimum_norm.apply_inverse(
                    evoked_inanim, inv, lambda2=lambda2, method='MNE', pick_ori='vector', verbose=False)
        
                
                print(f"Morphing ...")
                print("   morphing 1/2")
                R,L,anim = morph_hemi(stc_anim_vec, subject=subject, subject_to="fsaverage2",
                                      subjects_dir=subjects_dir, src_tag="vol-7")
                print("   morphing 2/2")
                R_in,L_in,inan = morph_hemi(stc_inan_vec, subject=subject, subject_to="fsaverage2",
                                      subjects_dir=subjects_dir, src_tag="vol-7")
                
                save.pickle((inan,anim), modelfile)
                          
                
                                 
                print(f"=====>M{model}-{subject } done!")
            except Exception as e:
                print(f"Error processing {subject}: {e}")   
            

# %%
#ONE size plot among subjects
"""
model_dir="models/samesize/2sesmne"
n_subject=9                  #Loop sub-01,sub-02,...,sub-n_subjects
size=6

def load_model_subset(subset,subject,session,size):
    
    morphed_file = f"{model_dir}/{subset}-{size}-{subject}.pickle"
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
        #session = "ImageNet04" if i == 7 else "ImageNet03" - renamed model to imagenet03 even 04
        session="ImageNet03"

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
plt.title(f'MNE\n({int(size*200)} Trials, Non-overlapping Balanced Data)')
plt.legend(bbox_to_anchor=(1.02, 1),loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
"""

# %%
model_dir = "models/samesize/2sesmne"
n_subject = 9
sizes = [0.25, 0.5, 1, 2,3, 4, 6, 8]   
trials_per_subset = 200
#------------------------
model_dir="models/samesize/2sesmne"
n_subject=9                  #Loop sub-01,sub-02,...,sub-n_subjects
size=6

def load_model_subset(subset,subject,session,size):
    
    morphed_file = f"{model_dir}/{subset}-{size}-{subject}.pickle"
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
        #session = "ImageNet04" if i == 7 else "ImageNet03" - renamed model to imagenet03 even 04
        session="ImageNet03"

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
#--------------

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

df_animacy = pd.DataFrame(results)
print("\nSummary:")
print(df_animacy)



# ----------------------------------------
# Melt for seaborn plotting
df_melted = df_animacy.melt(id_vars="Subset Size", var_name="Animacy", value_name="Mean r")
#plot
plt.figure(figsize=(7, 4))
sns.barplot(
    data=df_melted,
    x="Subset Size", y="Mean r",
    hue="Animacy",
    palette={"Animate": "#66BB66", "Inanimate": "#3399FF"},
    edgecolor="black", width=0.5
)

plt.ylim(0, 1.0)
plt.ylabel("Mean Pearson r")
plt.title("MNE (9subjects) ")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(7, 4))
plt.plot(df_animacy["Subset Size"], df_animacy["Animate"],
         marker='o', label="Animate", color="#66BB66")
plt.plot(df_animacy["Subset Size"], df_animacy["Inanimate"],
         marker='o', label="Inanimate", color="#3399FF")

plt.title("MNE (9 subjects)")
plt.xlabel("Subset Size (number of trials)")
plt.ylabel("Mean Pearson r (M1 vs M2)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.show()


# %%
def corr_diff(n_subject,size):
    
    R_data = np.full(n_subject, np.nan)

    for i in range(1, n_subject + 1):
        subject = f"sub-{i:02d}"
        session = "ImageNet03" if i == 7 else "ImageNet04"

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

def fisher_r_to_z(r):
    r = np.clip(r, -0.999999, 0.999999)
    return 0.5 * np.log((1 + r) / (1 - r))


#------------------------------------------
model_dir = "models/samesize/2sesmne"
n_subject = 9
sizes = [0.25, 0.5, 1, 2, 3, 4,6,8]   
trials_per_subset = 200


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
plt.title("MNE (9subject) averaged correlatins")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt

#df from mne results

#hardcoded for ncrf results from consistency.py
ncrf_data = pd.DataFrame({
    "Subset Size": [50, 100, 200, 400, 600, 800, 1600],
    "mean_r": [0.109234, 0.127496, 0.231609, 0.295263, 0.400948, 0.421631, 0.535664]
})

plt.figure(figsize=(7, 5))
plt.plot(df["Subset Size"], df["mean_r"], marker='o', linewidth=2, label="MNE")
plt.plot(ncrf_data["Subset Size"], ncrf_data["mean_r"], marker='s', linewidth=2, label="NCRF")

plt.xlabel("Number of Trials")
plt.ylabel("Mean R (contrast Anim-Inanim)")
plt.title("Consistency Comparison MNE ns NCRF ")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# %%
#hard coded for now
ncrf_df = pd.DataFrame({
    "Subset Size": [50, 100, 200, 400, 600, 800, 1600],
    "Animate": [0.574344, 0.650199, 0.692284, 0.762682, 0.852699, 0.855087, 0.853538],
    "Inanimate": [0.605400, 0.702857, 0.744667, 0.791608, 0.876339, 0.871014, 0.860871]
})


plt.figure(figsize=(8, 6))
plt.plot(df_animacy["Subset Size"], df_animacy["Animate"], color='#1f77b4', linestyle='-', marker='o', linewidth=2, label="MNE - Animate")
plt.plot(df_animacy["Subset Size"], df_animacy["Inanimate"], color='#1f77b4', linestyle='--', marker='s', linewidth=2, label="MNE - Inanimate")

# NCRF (orange)
plt.plot(ncrf_df["Subset Size"], ncrf_df["Animate"], color='#ff7f0e', linestyle='-', marker='o', linewidth=2, label="NCRF - Animate")
plt.plot(ncrf_df["Subset Size"], ncrf_df["Inanimate"], color='#ff7f0e', linestyle='--', marker='s', linewidth=2, label="NCRF - Inanimate")

# Labels and title
plt.xlabel("Number of Trials (Subset Size)")
plt.ylabel("Mean Correlation (R)")
plt.title("Consistency vs. Trial Size for Animate and Inanimate Conditions")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()


# %%
#First avg all subjects then corr( avg(M1),avg(M2))
def load_model_subset(m, subject,size):
    
    file_path = f"{model_dir}/{m}-{size}-{subject}.pickle"
    inan, anim = load.unpickle(file_path)
    return inan, anim


model_dir = "models/samesize/2sesmne"
sizes = [0.25, 0.5, 1, 2, 3, 4, 6, 8]
trials_per_subset = 200

summary = []

for size in sizes:
    cases = []

    for i in range(1, 10):
        if i in [6, 7]:
            continue

        subject = f"sub-{i:02d}"
        try:
            inan1, anim1 = load_model_subset(1, subject, size)
            inan2, anim2 = load_model_subset(2, subject,  size)
            d1 = anim1 - inan1
            d2 = anim2 - inan2
            cases.append([subject, "M1", "contrast", d1])
            cases.append([subject, "M2", "contrast", d2])

        except Exception as e:
            print(f"Skipping {subject} at size {size}: {e}")
            continue

    
    data = Dataset.from_caselist(['subject', 'model', 'animacy', 'mne'], cases)
    data_avg = data.aggregate('model', drop_bad=True)

    m1 = data_avg['mne'][data_avg['model'] == 'M1'][0].get_data().ravel()
    m2 = data_avg['mne'][data_avg['model'] == 'M2'][0].get_data().ravel()

    r = np.corrcoef(m1, m2)[0, 1]

    summary.append({
        "Subset Size": int(size * trials_per_subset),
        "mean_r": r
    })


# -----------------------------
df = pd.DataFrame(summary)
print("\nSummary:")
print(df)

plt.figure(figsize=(6, 4))
sns.barplot(
    data=df,
    x="Subset Size",
    y="mean_r",
    color="maroon",
    edgecolor="black",
    width=0.5
)
plt.ylim(0, 1)
plt.ylabel("Pearson r", fontsize=11)
plt.xlabel("Number of Trials", fontsize=11)
plt.title("Across-subjects Averaged Models (MNE)- Contrast", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


# %% [markdown]
# # AWcosine 

# %%
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


# %%
#First avg all subjects then corr( avg(M1),avg(M2))
def load_model_subset(m, subject,size):
    
    file_path = f"{model_dir}/{m}-{size}-{subject}.pickle"
    inan, anim = load.unpickle(file_path)
    return inan, anim


model_dir = "models/samesize/2sesmne"
sizes = [0.25, 0.5, 1, 2, 3, 4, 6, 8]
trials_per_subset = 200

summary = []
print("Awcosine:")
for size in sizes:
    cases = []

    for i in range(1, 10):
        if i in [6, 7]:
            continue

        subject = f"sub-{i:02d}"
        try:
            inan1, anim1 = load_model_subset(1, subject, size)
            inan2, anim2 = load_model_subset(2, subject,  size)
            d1 = anim1 - inan1
            d2 = anim2 - inan2
            cases.append([subject, "M1", "contrast", d1])
            cases.append([subject, "M2", "contrast", d2])

        except Exception as e:
            print(f"Skipping {subject} at size {size}: {e}")
            continue

    
    data = Dataset.from_caselist(['subject', 'model', 'animacy', 'mne'], cases)
    data_avg = data.aggregate('model', drop_bad=True)

    m1 = data_avg['mne'][data_avg['model'] == 'M1'][0].get_data().ravel()
    m2 = data_avg['mne'][data_avg['model'] == 'M2'][0].get_data().ravel()

    r = np.corrcoef(m1, m2)[0, 1]

    #----------------------save plots
    avg_m1 = data_avg['mne'][data_avg['model'] == 'M1'][0]
    avg_m2 = data_avg['mne'][data_avg['model'] == 'M2'][0]
    
    #save_mne_figures(avg_m1, avg_m2, size=size)
    
    #-------------------------------------aw cos
    cos_aw_t, n_used, times = ndvar_AWcosine( avg_m1 , avg_m2 , thr=1e-12, mode="or")
    print(f"size={size} ,  awcos={cos_aw_t.mean()}")

    
    #-------------------------------
    summary.append({
        "Subset Size": int(size * trials_per_subset),
        "mean_r": r
    })


# -----------------------------
df = pd.DataFrame(summary)
print(f"\nSummary:\n {df}")


plt.figure(figsize=(6, 4))
sns.barplot(
    data=df,
    x="Subset Size",
    y="mean_r",
    color="maroon",
    edgecolor="black",
    width=0.5
)
plt.ylim(0, 1)
plt.ylabel("Pearson r", fontsize=11)
plt.xlabel("Number of Trials", fontsize=11)
plt.title("Across-subjects Averaged Models (MNE)- Contrast", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# %% [markdown]
# # SAVE PLOT samesize

# %%
from eelbrain import plot

def save_mne_figures(avg_m1, avg_m2, size):
    
    times = [0.12, 0.25, 0.4, 0.5]

    p1 = plot.Butterfly(avg_m1.norm('space'), color='k')
    for t in times:
        p1.add_vline(t)
    p1.save(f"m1_size{size}_butterfly.png")
    p1.close()
    for t in times:
        f = plot.GlassBrain(avg_m1.sub(time=t), title=f", {t}s")
        f.save(f"m1_size{size}_glass_{t:.2f}s.png")
        f.close()

  
    p2 = plot.Butterfly(avg_m2.norm('space'), color='k')
    for t in times:
        p2.add_vline(t)
    p2.save(f"m2_size{size}_butterfly.png")
    p2.close()

    for t in times:
        f = plot.GlassBrain(avg_m2.sub(time=t), title=f", {t}s")
        f.save(f"m2_size{size}_glass_{t:.2f}s.png")
        f.close()

    print(f"Figures saved for size={size}")


# %%
# merge pics

# %%
from PIL import Image
import os

def merge_mne_figures(size):
    
   
    times = [0.12, 0.25, 0.40, 0.50]
    def merge_condition(prefix):
        files = [f"{prefix}_size{size}_butterfly.png"] + \
                [f"{prefix}_size{size}_glass_{t:.2f}s.png" for t in times]

        existing = [f for f in files if os.path.exists(f)]
        if not existing:
            print(f" No figures found for {prefix.upper()}. Skipping.")
            return

        images = [Image.open(f) for f in existing]

        widths, heights = zip(*(img.size for img in images))
        max_width = max(widths)
        total_height = sum(heights)

        merged = Image.new("RGB", (max_width, total_height), "white")

        # Paste vertically
        y_offset = 0
        for img in images:
            merged.paste(img, (0, y_offset))
            y_offset += img.size[1]

        # Save merged image
        out_name = f"NCRF-EC{prefix}_size{size}_merged.png"
        merged.save(out_name)
        print(f"Saved merged figure: {out_name}")

        # Close individual images
        for img in images:
            img.close()

    # Run for M1 (inanimate) and M2 (animate)
    merge_condition("m1")
    merge_condition("m2")


# %%
merge_mne_figures(6)

# %% [markdown]
# # MNE 
# ## SUB 10 to 30
#  

# %%
#save models
oot_epochs = Path("/Users/maryamvalian/Data/ds005810/derivatives/preprocessed/epochs")
#-------------------------------------------
sizes = [0.25, 0.5, 1, 2]
#------------------------------------------
per_run=200
session="ImageNet01"      #10 to 30 
#----------
def resize_epochs(epochs, n):
    #for balancing the condition trial#
    
    if len(epochs) >= n:
        return epochs[:n]
    else:
        idx = np.random.choice(len(epochs), n, replace=True)
        return epochs[idx]
#--------------------
for size in sizes:
    
    print(f"========= size= {size}=================================== ")
    cut_per_cond=int(size * per_run/2)
    for i in range(10, 31):
        
        subject = f"sub-{i:02d}"
        
        epo_file = root_epochs / f"{subject}_meg_epo.fif"
        
    
        ordered_runs = {
            1: ["02", "05"],  
            2: ["03", "01"] 
            }
        cut= size if size>=1 else 1
                
            
        for model in range (1,3):       #M1,M2

            run_subset = ordered_runs[model][:int(cut)]   
            run_list = [int(r) for r in run_subset]    
    
            
            modelfile = f"models/samesize/2sesmne/{model}-{size}-{subject}.pickle"
        
            if os.path.exists(modelfile):
                print(f"{model}-{size}-{subject} loaded from file.")
                continue
            try:
                print(f"{subject}, Run= {run_list}, Session={session}")
                epochs = mne.read_epochs(str(epo_file), preload=True, verbose=False)
                clean_fif = root / f"derivatives/preprocessed/raw/{subject}_ses-{session}_task-ImageNet_run-01_clean_meg.fif"
                clean = mne.io.read_raw_fif(clean_fif, preload=True, verbose=False)
                
                epochs_resamp = epochs.copy().resample(100, npad="auto", verbose=False) 
                    
                meta = epochs_resamp.metadata
                #sub-03 meta data is not boolean convert to boolean 
                if i==3:
                    meta["stim_is_animate"] = meta["stim_is_animate"].apply(lambda x: True if str(x).lower() == "true" else False)
            
                    
                mask_in = (
                    (meta["subject"] == i) &
                    (meta["session"] == session) &   
                    (meta["run"].isin(run_list)) &           #(meta["run"] == int(run))&
                    (meta["stim_is_animate"] ==False)
                )
    
                mask_an = (
                    (meta["subject"]== i) &
                    (meta["session"]== session) & 
                    (meta["run"].isin(run_list)) & 
                    (meta["stim_is_animate"] ==True)
                )
            
                epochs_in = epochs_resamp[mask_in] # Not balancing
                epochs_an = epochs_resamp[mask_an]
                print(f"Befor Balancing: Animate={len(epochs_an)},Inanim={len(epochs_in)} ")
               
                
                
                epochs_an = resize_epochs(epochs_resamp[mask_an], cut_per_cond)
                epochs_in = resize_epochs(epochs_resamp[mask_in], cut_per_cond)        
                print(f"BALANCED : Animate={len(epochs_an)}, Inanim={len(epochs_in)}")
                
        
        
                evoked_anim= epochs_an.average()
                evoked_inanim= epochs_in.average()
                
                src_file = f"{subjects_dir}/{subject}/bem/{subject}-vol-7-lr-src.fif"        #not merged l,R
                src = mne.read_source_spaces(str(src_file),verbose=False)
            
                bem_sol_fif=f"{subjects_dir}/{subject}/bem/{subject}-bem-sol.fif"
                bem_sol = mne.read_bem_solution(bem_sol_fif,verbose=False)
                
                trans_fif= f"{root}/derivatives/trans/{subject}-{session}-trans.fif"
                trans=mne.read_trans(trans_fif)
        
                print("   Computing FWD")
                fwd = mne.make_forward_solution(
                        clean.info, trans, src, bem_sol,
                        meg=True, eeg=False, mindist=0, verbose=False
                    )
            
                          
                print("   Inverse...")
                inv = mne.minimum_norm.make_inverse_operator(
                    info=clean.info,
                    forward=fwd,
                    noise_cov=noise_cov,
                    loose=1.0,    
                    depth=0.8,
                    fixed=False,
                    verbose=False,
                )
                
                snr = 3.0
                lambda2 = 1.0 / snr**2
                
                
                stc_anim_vec = mne.minimum_norm.apply_inverse(
                    evoked_anim, inv, lambda2=lambda2, method='MNE', pick_ori='vector', verbose=False)
                
                stc_inan_vec = mne.minimum_norm.apply_inverse(
                    evoked_inanim, inv, lambda2=lambda2, method='MNE', pick_ori='vector', verbose=False)
        
                
                print(f"Morphing ...")
                print("   morphing 1/2")
                R,L,anim = morph_hemi(stc_anim_vec, subject=subject, subject_to="fsaverage2",
                                      subjects_dir=subjects_dir, src_tag="vol-7")
                print("   morphing 2/2")
                R_in,L_in,inan = morph_hemi(stc_inan_vec, subject=subject, subject_to="fsaverage2",
                                      subjects_dir=subjects_dir, src_tag="vol-7")
                
                save.pickle((inan,anim), modelfile)
                          
                
                                 
                print(f"==>M{model}-{subject } done!")
            except Exception as e:
                print(f"Error processing {subject}: {e}")   
            

# %%
#read models and plot results
model_dir = "models/samesize/2sesmne"
sizes = [0.25, 0.5, 1, 2]
trials_per_subset = 200


def load_model_subset(m, subject, size):
    file_path = f"{model_dir}/{m}-{size}-{subject}.pickle"
    inan, anim = load.unpickle(file_path)
    return inan, anim


summary_avg = []

for size in sizes:
    cases = []

    for i in range(10, 31):
        subject = f"sub-{i:02d}"
        try:
            inan1, anim1 = load_model_subset(1, subject, size)
            inan2, anim2 = load_model_subset(2, subject, size)
            d1 = anim1 - inan1
            d2 = anim2 - inan2
            cases.append([subject, "M1", "contrast", d1])
            cases.append([subject, "M2", "contrast", d2])

        except Exception as e:
            print(f"Skipping {subject} at size {size}: {e}")
            continue

    data = Dataset.from_caselist(["subject", "model", "animacy", "mne"], cases)
    data_avg = data.aggregate("model", drop_bad=True)

    m1 = data_avg["mne"][data_avg["model"] == "M1"][0].get_data().ravel()
    m2 = data_avg["mne"][data_avg["model"] == "M2"][0].get_data().ravel()

    r = np.corrcoef(m1, m2)[0, 1]

    summary_avg.append({
        "Subset Size": int(size * trials_per_subset),
        "pearson_r": r
    })

df_avg = pd.DataFrame(summary_avg)


contrast_results = []
anim_results = []
inan_results = []

for size in sizes:
    contrast_rs, anim_rs, inan_rs = [], [], []

    for i in range(10, 31):
        subject = f"sub-{i:02d}"
        try:
            inan1, anim1 = load_model_subset(1, subject, size)
            inan2, anim2 = load_model_subset(2, subject, size)

            a1 = anim1.get_data().ravel()
            a2 = anim2.get_data().ravel()
            i1 = inan1.get_data().ravel()
            i2 = inan2.get_data().ravel()

            r_anim = np.corrcoef(a1, a2)[0, 1]
            r_inan = np.corrcoef(i1, i2)[0, 1]
            r_contrast = np.corrcoef(a1 - i1, a2 - i2)[0, 1]

            anim_rs.append(r_anim)
            inan_rs.append(r_inan)
            contrast_rs.append(r_contrast)

        except Exception as e:
            print(f"Skipping {subject} at size {size}: {e}")
            continue

    if contrast_rs:
        contrast_results.append({
            "Subset Size": int(size * trials_per_subset),
            "Mean Pearson r": np.mean(contrast_rs),
            "N Subjects": len(contrast_rs)
        })
    if anim_rs:
        anim_results.append({
            "Subset Size": int(size * trials_per_subset),
            "Mean Pearson r": np.mean(anim_rs),
            "N Subjects": len(anim_rs)
        })
    if inan_rs:
        inan_results.append({
            "Subset Size": int(size * trials_per_subset),
            "Mean Pearson r": np.mean(inan_rs),
            "N Subjects": len(inan_rs)
        })

df_contrast = pd.DataFrame(contrast_results)
df_anim = pd.DataFrame(anim_results)
df_inan = pd.DataFrame(inan_results)

#plot 1
plt.figure(figsize=(6, 4))
sns.barplot(
    data=df_avg,
    x="Subset Size",
    y="pearson_r",
    color="maroon",
    edgecolor="black",
    width=0.5
)
plt.ylim(0, 1)
plt.ylabel("Pearson r", fontsize=11)
plt.xlabel("Number of Trials", fontsize=11)
plt.title("MNE-Averaged Models – Contrast (21 sbjs)", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

print(df_avg)
#plot 2
plt.figure(figsize=(6, 4))
sns.barplot(
    data=df_contrast,
    x="Subset Size",
    y="Mean Pearson r",
    color="maroon",
    edgecolor="black",
    width=0.5
)
plt.ylim(0, 1)
plt.ylabel("Mean Pearson r", fontsize=11)
plt.xlabel("Number of Trials", fontsize=11)
plt.title("MNE-Average correlation-contrast(21 sbjs)", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

print(df_contrast)
#plot3
plt.figure(figsize=(6, 4))
df_anim["Condition"] = "Animate"
df_inan["Condition"] = "Inanimate"
df_combined = pd.concat([df_anim, df_inan], ignore_index=True)

sns.barplot(
    data=df_combined,
    x="Subset Size",
    y="Mean Pearson r",
    hue="Condition",
    edgecolor="black",
    palette={"Animate": "#66BB66", "Inanimate": "#3399FF"},
    width=0.5
)
plt.ylim(0, 1)
plt.ylabel("Mean Pearson r", fontsize=11)
plt.xlabel("Number of Trials", fontsize=11)
plt.title("MNE-Average Correlation (21 sbjs) ", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend(title="", loc="upper left", frameon=False)
plt.tight_layout()
plt.show()

print(df_combined)

# %%

# %%

# %%

# %%

# %%
