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
from Beyond import morph_nd

# %%
root = Path("~/Data/ds005810")
subjects_dir = str(Path('~/Data/ds005810/derivatives/freesurfer/subjects').expanduser())

empty_room=root/"sub-emptyroom/ses-20211114/meg/sub-emptyroom_ses-20211114_task-noise_meg.fif"

# %% [markdown]
# # Creat Models

# %%

for i in range(1, 10):
    if (i<10):
        run="01"
        session="ImageNet02" 
    else:
        run="04"
        session="ImageNet01" 
    
    subject = f"sub-{i:02d}"
    modelfile = f"models/ncrf/{subject}.pickle"
    raw_fif=root/f"{subject}/ses-{session}/meg/{subject}_ses-{session}_task-ImageNet_run-{run}_meg.fif"
    clean_fif = root / f"derivatives/preprocessed/raw/{subject}_ses-{session}_task-ImageNet_run-{run}_meg_clean.fif"
    
    if os.path.exists(modelfile):
        print(f"{subject} loaded from file.")
        continue
    try:
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
                if is_anim:
                    stim2[idx] = 1.0
                else:
                    stim1[idx] = 1.0
        stim1 = NDVar(stim1, time)
        stim2 = NDVar(stim2, time)
        #p=plot.LineStack(combine([stim1, stim2]), ylabels=["inanimate", "animate"], offset=1.5)

        raw_er = mne.io.read_raw_fif(empty_room, preload=True, verbose=False).pick('meg')
        raw_er.filter(1., 40., phase="zero-double", verbose=False)
        raw_er.resample(100, npad="auto", verbose=False)
        noise_cov = mne.compute_raw_covariance(raw_er, method='shrunk', rank=None,verbose=False)

        
        src_file = f"{subjects_dir}/{subject}/bem/{subject}-vol-7-src.fif"
        src = mne.read_source_spaces(str(src_file),verbose=False)
        
        bem_sol_fif=f"{subjects_dir}/{subject}/bem/{subject}-bem-sol.fif"
        bem_sol = mne.read_bem_solution(bem_sol_fif,verbose=False)
        
        
        trans_fif= f"{root}/derivatives/trans/{subject}-{session}-trans.fif"
        trans=mne.read_trans(trans_fif)
        
        fwd = mne.make_forward_solution(meg.info,trans ,
                                        src, bem_sol,
                                        meg=True, eeg=False,
                                        mindist=0,               #======same size with src
                                        verbose=False
                                       )
        
        #convert fwd to ndvar
        lf = load.mne.forward_operator(fwd,src='vol-7',
                                       subjects_dir=subjects_dir,
                                       adjacency=False,parc='aparc+aseg') 
        lf  = lf.sub(sensor=meg_ndvar.sensor)  

        print(len(src[0]['vertno']))
        print(len(fwd['src'][0]['vertno']))  

        print(f"Fitting NCRF")
        args = (
                    meg_ndvar,
                    [stim1,stim2],
                    lf,
                    noise_cov,
                    0,
                    0.8,
                )
        kwargs = {'normalize': 'l1',
                          'in_place': False,
                          'mu':'auto',
                          'verbose': True, 
                          'n_iter': 10,
                          'n_iterc': 10,
                          'n_iterf': 100} 
        model = fit_ncrf(*args, **kwargs)  
        
        #---------------
        
        save.pickle(model, modelfile)
        
        hlist=model.h
        print(f"==================>{subject} done!")
        
    except Exception as e:
         print(f"Error processing {subject}: {e}")                 


# %% [markdown]
# # Morph and save
# ## Dataset from cases

# %%
cases = []
for i in range(1, 31):
    if i == 28:            #corrupted MEGs in all runs
        continue
    subject = f"sub-{i:02d}"
    morphed_file = f"models/ncrf/{subject}-morphed.pickle"
    
    if os.path.exists(morphed_file):
        print(f"Loading {subject} from file.")
        inanim, anim = load.unpickle(morphed_file)
    else:
                  
        print(f"Morphing {subject}...")
        model_file = f"models/ncrf/{subject}.pickle"
        model = load.unpickle(model_file)
        hlist = model.h
        inanim = hlist[0]
        anim = hlist[1]
               
        anim_fs = morph_nd(subject, 'fsaverage2', subjects_dir, anim, 'vol-7')
        inanim_fs = morph_nd(subject, 'fsaverage2', subjects_dir, inanim, 'vol-7')
        
        anim = anim_fs.smooth('source', 0.01, 'gaussian')
        inanim = inanim_fs.smooth('source', 0.01, 'gaussian')
        
        save.pickle((inanim, anim), morphed_file)
    
    cases.append([subject, 'inanimate', inanim])
    cases.append([subject, 'animate', anim])
    
data = Dataset.from_caselist(['subject', 'animacy', 'ncrf'], cases)
data.head()

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
    data=data,     
    tfce=True,           
    tstart=0.1,
    tstop=0.7,
    samples=1000
)
save.pickle(res, "Tests/ncrf_paired_test.pickle")

# %%
diff= res.masked_difference()
p = plot.Butterfly(diff.norm('space'), color='k',title='anim VS inanim')
times = [0.15,0.24,0.3,0.35,0.5,0.6]
for t in times:
    p.add_vline(t)
for t in times:
    f = plot.GlassBrain(diff.sub(time=t),title=f"anim vs inan, {t}s")  

# %% [markdown]
# ## ONE Sample test

# %%
data_inan = data.sub("animacy == 'inanimate'")
result_inan = testnd.Vector('ncrf', match='subject', data=data_inan, tfce=True, tstart=0.1, tstop=0.6,samples=1000)

# %%
p = plot.Butterfly(result_inan.masked_difference().norm('space'), color='k')
times = [0.13,0.25,0.4]
for t in times:
    p.add_vline(t)
for t in times:
    f = plot.GlassBrain(result_inan.masked_difference().sub(time=t),title=f"Inanimate, {t}s")  

# %%
data_an = data.sub("animacy == 'animate'")
result_an = testnd.Vector('ncrf', match='subject', data=data_an, tfce=True, tstart=0.1, tstop=0.6,samples=1000)
result_an

# %%
p = plot.Butterfly(result_an.masked_difference().norm('space'), color='k')
times = [0.13,0.22,0.27,0.4]
for t in times:
    p.add_vline(t)
for t in times:
    f = plot.GlassBrain(result_an.masked_difference().sub(time=t),title=f"animate, {t}s")  

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

# %%
