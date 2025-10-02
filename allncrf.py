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


# %% [markdown]
# Noise Covariance:

# %%
empty_room=root/"sub-emptyroom/ses-20211114/meg/sub-emptyroom_ses-20211114_task-noise_meg.fif"
raw_er = mne.io.read_raw_fif(empty_room, preload=True, verbose=False).pick('meg')
raw_er.filter(1., 40., phase="zero-double", verbose=False)
raw_er.resample(100, npad="auto", verbose=False)
noise_cov = mne.compute_raw_covariance(raw_er, method='shrunk', rank=None,verbose=False)


# %% [markdown]
# Functions definition

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
    


# %% [markdown]
# # Fitting NCRF Models:
# <hr><br><br>
# <bold>All Sessions and Runs</bold> 

# %%
for i in range(1, 31):                                      
    if (i>9):
        sessions=["ImageNet01"]
        lastruns = [5]                     #run01,run02,..,run05      
        
    else:
        sessions=["ImageNet01", "ImageNet02","ImageNet03","ImageNet04"]
        lastruns=[2,2,8,8]
    
    subject = f"sub-{i:02d}"
    for idx, session in enumerate(sessions):
                
        modelfile = f"models/all_runs/{subject}-{session}-ncrf.pickle"
        if os.path.exists(modelfile):
            print(f"{subject}-{session} model file exists.")
            continue
        try:
            
            print(f"computing fwd for {subject}-{session}... ")
            
            clean_fif = root / f"derivatives/preprocessed/raw/{subject}_ses-{session}_task-ImageNet_run-01_clean_meg.fif"
            clean = mne.io.read_raw_fif(clean_fif, preload=False,verbose=False)
            info= clean.info           
            meg_ndvar = load.fiff.raw_ndvar(clean)
            sensor=meg_ndvar.sensor
            fwd = compute_fwd_ndvar(subject, session,subjects_dir,info,sensor)

            
            lastrun= lastruns[idx]
            if (i==4) and ((session=="ImageNet04")):   #check: make_event :raw file doesnt provided by openneuro
                lastrun=5
            runs = [f"{i:02d}" for i in range(1, lastrun+1)] 
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

# %% [markdown]
# # FWD COMP SEPERATE:
# If fwd file not founded during morphing:

# %%
subject="sub-11"
session="ImageNet01"
clean_fif = root / f"derivatives/preprocessed/raw/{subject}_ses-{session}_task-ImageNet_run-01_clean_meg.fif"
clean = mne.io.read_raw_fif(clean_fif, preload=False,verbose=False)
info= clean.info           
meg_ndvar = load.fiff.raw_ndvar(clean)
sensor=meg_ndvar.sensor
fwd = compute_fwd_ndvar(subject, session,subjects_dir,info,sensor)

# %% [markdown]
#
#
# WB works well <br>
# Cortex -> src should be merged! otherwise can not convert fwd to ndvar to feed fit_ncef<br>
# morph_nd drops right  brain when src is cortex<br>
# src is cortex-merged but not pruned.<br>
# **when change src type (cortex, merged, WB,..) make sure fwd is overwrited**<br>

# %% jupyter={"source_hidden": true}
"""
raw_er = mne.io.read_raw_fif(empty_room, preload=True, verbose=False).pick('meg')
raw_er.filter(1., 40., phase="zero-double", verbose=False)
raw_er.resample(100, npad="auto", verbose=False)
noise_cov = mne.compute_raw_covariance(raw_er, method='shrunk', rank=None,verbose=False)

for i in range(1, 31):
    if (i<10):
        run="01"
        session="ImageNet02"
    elif i==22:
        run="04"
        session="ImageNet01" 
    elif (i==11) or (i==30):
        run="01"
        session="ImageNet01"
        
    else:
        run="05"
        session="ImageNet01" 
    
    subject = f"sub-{i:02d}"
    modelfile = f"models/ncrf/{mod}-{subject}.pickle"
    raw_fif=root/f"{subject}/ses-{session}/meg/{subject}_ses-{session}_task-ImageNet_run-{run}_meg.fif"
    clean_fif = root / f"derivatives/preprocessed/raw/{subject}_ses-{session}_task-ImageNet_run-{run}_clean_meg.fif"
    fwd_file=fwd_dir / f"{subject}_{session}/{subject}-fwd.fif"
    
    if os.path.exists(modelfile):
        print(f"{subject} loaded from file.")
        continue
    try:
        print(f"extracting {subject}...")
        raw = mne.io.read_raw_fif(raw_fif, preload=True,verbose=False)
        clean = mne.io.read_raw_fif(clean_fif, preload=True,verbose=False)
        events = find_events(raw, stim_channel="UPPT001")
        stim = events[events[:, 2] == 2]   #ID2 : stim_on
        stim_samples = stim[:, 0]
        
        stim_times = stim_samples / raw.info["sfreq"]
        meta = pd.read_csv(f"/Users/maryamvalian/Data/ds005810/derivatives/detailed_events/{subject}_events.csv")
        meta_sub = meta[(meta['session'] == session) & (meta['run'] ==int(run))].reset_index(drop=True)
        n=len(meta_sub)
        anim_flags = meta_sub['stim_is_animate'].astype(str).str.lower().eq("true").to_numpy()
        event_table = pd.DataFrame({  "time": stim_times,  "animate": anim_flags })
        print(event_table)
        
        meg=clean
        meg.filter(1., 40., phase="zero-double", verbose=False)
        meg.resample(100, npad="auto", verbose=False)
        meg_ndvar = load.fiff.raw_ndvar(meg)

        
        
        sfreq = meg.info['sfreq']     #100       
        n_times = meg.n_times         #32000     
        
        

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
        #p=plot.LineStack(combine([stim1, stim2]), ylabels=["inanimate", "animate"], offset=1.5)
        
        src_file = f"{subjects_dir}/{subject}/bem/{subject}-vol-7-src.fif"
        src = mne.read_source_spaces(str(src_file),verbose=False)
        
        bem_sol_fif=f"{subjects_dir}/{subject}/bem/{subject}-bem-sol.fif"
        bem_sol = mne.read_bem_solution(bem_sol_fif,verbose=False)
        
        
        trans_fif= f"{root}/derivatives/trans/{subject}-{session}-trans.fif"
        trans=mne.read_trans(trans_fif)

        
        
        fwd_file=fwd_dir / f"{subject}_ses-{session}/{subject}-fwd.fif"
        fwd_file.parent.mkdir(parents=True, exist_ok=True)
        
        if fwd_file.exists() and not (rewrite):
            print(" Loading FWD ")
            fwd = mne.read_forward_solution(str(fwd_file), verbose=False)
        else:
            print(" Computing FWD...")
            fwd = mne.make_forward_solution(
                clean.info, trans, src, bem_sol,
                meg=True, eeg=False, mindist=0, verbose=False
            )
            mne.write_forward_solution(str(fwd_file), fwd, overwrite=True, verbose=False)
            print(f"   Saved FWD to {fwd_file}")
        
        
        #convert fwd to ndvar
        lf = load.mne.forward_operator(fwd,src='vol-7',
                                       subjects_dir=subjects_dir,
                                       adjacency=False,parc='aparc+aseg') 
        lf  = lf.sub(sensor=meg_ndvar.sensor)  

        print(len(src[0]['vertno']))           #check sanity
        print(len(fwd['src'][0]['vertno']))  

        print(f"Fitting NCRF")
        if mod=="common":
            stim=stim1
        else:
            stim=[stim1,stim2]
        args = (
                    meg_ndvar,
                    stim,
                    lf,
                    noise_cov,
                    0,
                    0.8,
                )
        kwargs = {'normalize': 'l1',
                          'in_place': False,
                          'mu':"auto",
                          'verbose': True, 
                          'n_iter': 10,
                          'n_iterc': 10,
                          'n_iterf': 100} 
        model = fit_ncrf(*args, **kwargs)       
        save.pickle(model, modelfile)
        
        print(f"==================>{subject} model saved!")
        
    except Exception as e:
         print(f"Error processing {subject}: {e}")                 
"""

# %% [markdown]
# # Morph and save
# <hr><br><br>
#
# ## Dataset from cases

# %%
cases = []
for i in range(10, 12):                                      
    if (i>9):
        sessions=["ImageNet01"]
        lastruns = [5]                     #run01,run02,..,run05   
    else:
        sessions=["ImageNet01", "ImageNet02","ImageNet03","ImageNet04"]
        lastruns=[2,2,8,8]
    
    subject = f"sub-{i:02d}"
    for idx, session in enumerate(sessions):
                
        morphed_file = f"models/all_runs/morphed/M{subject}{session}-ncrf.pickle"  #M stands for Morphed
        if os.path.exists(morphed_file):
            print(f"Loading {subject} from file.")
            inanim, anim = load.unpickle(morphed_file)
        else:
            print(f"Morphing {subject}-{session}...")
            modelfile = f"models/all_runs/{subject}-{session}-ncrf.pickle"
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
            #anim_fs= morph_nd(subject, 'fsaverage2', subjects_dir, anim, 'vol-7')                    #wholeBran : single grid
            #inanim_fs= morph_nd(subject, 'fsaverage2', subjects_dir, inanim, 'vol-7')
        
            anim= anim_fs.smooth('source', 0.01, 'gaussian')
            inanim= inanim_fs.smooth('source', 0.01, 'gaussian')
        
            save.pickle((inanim, anim), morphed_file)
            print(f"{subject}-{session}-Morphed Saved  \n")
    
        cases.append([subject, session, 'inanimate', inanim])
        cases.append([subject, session, 'animate', anim])
    
data = Dataset.from_caselist(['subject', 'session', 'animacy', 'ncrf'], cases)
data.head()

# %% [markdown]
# ## Average sessions

# %%
data_agg=data
data_avg=data_agg.aggregate(('animacy% subject'), drop_bad=True)
data_avg

# %% [markdown]
# # Group Analysis
# ## Paired Test

# %%
res = testnd.VectorDifferenceRelated(
    'ncrf',             
    'animacy',           
    'inanimate',     
    'animate',   
    match='subject',     
    data=data_avg,     
    tfce=True,           
    tstart=10,                           #ms not second, out put of ncrf 
    tstop=70,
    samples=1000
)
save.pickle(res, f"Tests/all_runs/all_{mod}-ncrf_PT.pickle")

# %%
diff= res.masked_difference()
p = plot.Butterfly(diff.norm('space'), color='k',title='anim VS inanim')
times = [15,21,34,45,50,60]
for t in times:
    p.add_vline(t)
for t in times:
    f = plot.GlassBrain(diff.sub(time=t),title=f"anim vs inan, {t}s")  

# %% [markdown]
# ## ONE Sample test

# %%
data_inan = data.sub("animacy == 'inanimate'")
result_inan = testnd.Vector('ncrf', match='subject', data=data_inan, tfce=True, tstart=1, tstop=600,samples=1000)

# %%
p = plot.Butterfly(result_inan.masked_difference().norm('space'), color='k')
times = [13,25,40]
for t in times:
    p.add_vline(t)
for t in times:
    f = plot.GlassBrain(result_inan.masked_difference().sub(time=t),title=f"Inanimate, {t}s")  

# %%
data_an = data.sub("animacy == 'animate'")
result_an = testnd.Vector('ncrf', match='subject', data=data_an, tfce=True, tstart=1, tstop=60,samples=1000)
result_an
save.pickle((result_an,result_inan), f"Tests/ncrf/{mod}-ncrf_1samptest.pickle")

# %%
p = plot.Butterfly(result_an.masked_difference().norm('space'), color='k')
times = [13,25,40]
for t in times:
    p.add_vline(t)
for t in times:
    f = plot.GlassBrain(result_an.masked_difference().sub(time=t),title=f"animate, {t}ms")  

# %% [markdown]
# ## Average

# %%
agg=data.aggregate('animacy', drop_bad=True)
agg_in = agg.sub("animacy == 'inanimate'")
agg_an = agg.sub("animacy == 'animate'")

diff=agg_an['ncrf']- agg_in['ncrf']

p = plot.Butterfly(diff.norm('space'), color='k')
times = [0.12,0.17,0.28,0.45]
for t in times:
    p.add_vline(t)
for t in times:
    f = plot.GlassBrain(diff.sub(time=t),title=f"diff: animate-Inanimate, {t}s") 

# %% [markdown]
# # COMMON GROUP analysis

# %%
mod="common"

# %%
cases = []
for i in range(1, 31):
    if (i<10):
        run="01"
        session="ImageNet02"
    elif i==22:
        run="04"
        session="ImageNet01" 
    elif (i==11) or (i==30):
        run="01"
        session="ImageNet01"
    elif i == 28:           
        continue        
    else:
        run="05"
        session="ImageNet01" 
    
    subject = f"sub-{i:02d}"
    
    morphed_file = f"models/ncrf/{mod}-{subject}-morphed.pickle"
    
    if os.path.exists(morphed_file):
        print(f"Loading {subject} morphed model from file.")
        common = load.unpickle(morphed_file)
    else:
                  
        print(f"Morphing {subject}...")
        model_file = f"models/ncrf/{mod}-{subject}.pickle"
        model= load.unpickle(model_file)
        hlist = model.h
        
        root = Path("~/Data/ds005810")
        clean_fif = root / f"derivatives/preprocessed/raw/{subject}_ses-{session}_task-ImageNet_run-{run}_clean_meg.fif"
        clean = mne.io.read_raw_fif(clean_fif, preload=True,verbose=False)
        
        src_file = f"{subjects_dir}/{subject}/bem/{subject}-vol-7-src.fif" #merged
        src = mne.read_source_spaces(str(src_file),verbose=False)
        
        bem_sol_fif=f"{subjects_dir}/{subject}/bem/{subject}-bem-sol.fif"
        bem_sol = mne.read_bem_solution(bem_sol_fif,verbose=False)
        
        
        trans_fif= f"{root}/derivatives/trans/{subject}-{session}-trans.fif"
        trans=mne.read_trans(trans_fif)
        
        fwd = mne.make_forward_solution(
                clean.info, trans, src, bem_sol,
                meg=True, eeg=False, mindist=0, verbose=False
            )

        
        stc_common = ndvar_merged_to_stc_lr(
        
        ndvar=hlist,
        fwd=fwd,
        subject=subject,
        subjects_dir=subjects_dir,
        src_tag="vol-7")

        _, _, common_fs = morph_hemi(
        stc_common,
        subject=subject,
        subject_to="fsaverage2",
        subjects_dir=subjects_dir,
        src_tag="vol-7")

        common= common_fs.smooth('source', 0.01, 'gaussian')
        save.pickle(common, morphed_file)
        
    cases.append([subject, common])

data_common = Dataset.from_caselist(['subject', 'ncrf'], cases)
data_common.tail()        

# %%
result_common = testnd.Vector('ncrf', match='subject', data=data_common, tfce=True, tstart=10, tstop=60,samples=1000)  #NCRF time

# %%
p = plot.Butterfly(result_common.masked_difference().norm('space'), color='k')
times = [13,25,35,45]
for t in times:
    p.add_vline(t)
for t in times:
    f = plot.GlassBrain(result_common.masked_difference().sub(time=t),title=f"common NCRF, {t/100}s") 

# %%
