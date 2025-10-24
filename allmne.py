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
sizes = [0.25, 0.5, 1, 2, 3, 4,] 
per_run=200
cut_per_cond=int(size * per_run/2)
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
    for i in range(1, 10):
        """
        if (i<10):
            session="ImageNet03"             
        else:
            session="ImageNet01"
        """
        
        subject = f"sub-{i:02d}"
        
        epo_file = root_epochs / f"{subject}_meg_epo.fif"
        
    
        if size== 8:
            subsets=[["08", "06", "05", "02", "01", "03", "04", "07"],["08", "06", "05", "02", "01", "03", "04", "07"]]
        elif size==6: 
            subsets= [["08", "06", "05", "02", "01", "03"],["08", "06", "05", "02", "01", "03"]]
        elif size==4:
            subsets=[["02", "01", "03", "05"],["03", "07", "01", "04"]]
        elif size==3:
            subsets=[["02", "01", "03"],["07", "04", "06"]]
        elif size==2:
            subsets=[["02", "01"],["08", "04"]]
        elif size==1:
            subsets=[["02"], ["08"]]
        else:
            subsets=[["08"],["03"]]    #size 0.5, 0.25
            
        for model in range (1,3):       #M1,M2
    
            session="ImageNet03" if model==1 else "ImageNet04"             #-------added to keep consistency between sizes all data from different session
            
            #subset=subsets[model-1]
            subset=subsets[0]         # define same runs but diff sessions
    
            
            run_list = [int(r) for r in subset]
            
            modelfile = f"models/samesize/2sesmne/{model}-{size}-{subject}.pickle"
        
            if os.path.exists(modelfile):
                print(f"{model}-{size}-{subject} loaded from file.")
                continue
            try:
                print(f"load epochs for {subject}")
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
                print("Befor Balancing:")
                print(f"#Animate={len(epochs_an)}")
                print(f"#Inanim={len(epochs_in)}")
                
                print("Balancing trials ")
                epochs_an = resize_epochs(epochs_resamp[mask_an], cut_per_cond)
                epochs_in = resize_epochs(epochs_resamp[mask_in], cut_per_cond)
    
                
                
                print(f"#Animate={len(epochs_an)}")
                print(f"#Inanim={len(epochs_in)}")
        
        
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
model_dir="models/samesize/mne"
n_subject=9                  #Loop sub-01,sub-02,...,sub-n_subjects
size=8

def load_model_subset(subset,subject,session,size):
    
    morphed_file = f"{model_dir}/{subset}-{size}-{subject}-{session}-mne.pickle"
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

# %%

# %%
model_dir = "models/samesize/mne"
n_subject = 9
sizes = [0.25, 0.5, 1, 2,  3, 4, 6, 8]   
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
plt.title("MNE Model Consistency ")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


# %%
def corr_diff(n_subject,size):
    
    R_data = np.full(n_subject, np.nan)

    for i in range(1, n_subject + 1):
        subject = f"sub-{i:02d}"
        session = "ImageNet03" if i == 7 else "ImageNet03"

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
model_dir = "models/samesize/mne"
n_subject = 9
sizes = [0.25, 0.5, 1, 2, 3, 4,6,  8]   
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
