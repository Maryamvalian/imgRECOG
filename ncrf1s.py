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

from morph_nd import morph_nd

# %%
subject="sub-02"
run="01"
session="ImageNet02" 

root = Path("~/Data/ds005810")
subjects_dir = str(Path('~/Data/ds005810/derivatives/freesurfer/subjects').expanduser())

clean_fif = root / f"derivatives/preprocessed/raw/{subject}_ses-{session}_task-ImageNet_run-{run}_meg_clean.fif"
raw_fif=root/f"{subject}/ses-{session}/meg/{subject}_ses-{session}_task-ImageNet_run-{run}_meg.fif"
empty_room=root/"sub-emptyroom/ses-20211114/meg/sub-emptyroom_ses-20211114_task-noise_meg.fif"

# %%
raw = mne.io.read_raw_fif(raw_fif, preload=True,verbose=False)
clean= mne.io.read_raw_fif(clean_fif, preload=True,verbose=False)
events = find_events(raw, stim_channel="UPPT001")
#raw.info["bads"]  
#raw.n_times
fig = mne.viz.plot_events(events, sfreq=raw.info["sfreq"], first_samp=raw.first_samp)

# %%
stim = events[events[:, 2] == 2]   #ID2 : stim_on

stim_samples = stim[:, 0]

stim_times = stim_samples / raw.info["sfreq"]
len(stim) 

# %%
meta = pd.read_csv(f"/Users/maryamvalian/Data/ds005810/derivatives/detailed_events/{subject}_events.csv")
meta_sub = meta[(meta['session'] == session) & (meta['run'] ==int(run))].reset_index(drop=True)
meta_sub

# %%
n=len(meta_sub)
n

# %%
anim_flags = meta_sub['stim_is_animate'].astype(str).str.lower().eq("true").to_numpy()
anim_flags

# %%
event_table = pd.DataFrame({
           
    "time": stim_times, 
    
    "animate": anim_flags           
})

# %%
event_table

# %%
meg=raw.pick("meg")
meg.filter(1., 40., phase="zero-double", verbose=False)
meg.resample(100, npad="auto", verbose=False)

sfreq = meg.info['sfreq']     #100       
n_times = meg.n_times         #32000     

meg_ndvar = load.fiff.raw_ndvar(meg)
print(f"MEG: {meg_ndvar}")
     


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


p=plot.LineStack(combine([stim1, stim2]), ylabels=["inanimate", "animate"], offset=1.5)

# %%
#print(stim1.x)
#print(stim1.x[738])

# %% [markdown]
# ## NOISE COv

# %%
raw_er = mne.io.read_raw_fif(empty_room, preload=True, verbose=False).pick('meg')
raw_er.filter(1., 40., phase="zero-double", verbose=False)
raw_er.resample(100, npad="auto", verbose=False)

noise_cov = mne.compute_raw_covariance(raw_er, method='shrunk', rank=None,verbose=False)
noise_cov

# %% [markdown]
# # Lead FIeld

# %%
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

# %%
print(len(src[0]['vertno']))
print(len(fwd['src'][0]['vertno']))  

# %%
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
save_dir = Path('base_ncrf')
save_dir.mkdir(exist_ok=True)

filename = save_dir / f"{subject}.pickle"
save.pickle(model, filename)

hlist=model.h
hlist

# %%
"""
anim= morph_source_space(
    anim,
    subject_to='fsaverage2',
    copy=True,
    
    
    )
"""    

# %%
hlist=model.h
inanim=hlist[0]
anim=hlist[1]

anim_fs = morph_nd(subject, 'fsaverage2', subjects_dir, anim, 'vol-7')
inanim_fs = morph_nd(subject, 'fsaverage2', subjects_dir, inanim, 'vol-7')

anim_fs = anim_fs.smooth('source', 0.01, 'gaussian')
inanim_fs = inanim_fs.smooth('source', 0.01, 'gaussian')

# %%
p = plot.Butterfly(anim_fs.norm('space'), color='k')
times = [0.12,0.17,0.24,0.37,0.48,0.55]
for t in times:
    p.add_vline(t)
for t in times:
    f = plot.GlassBrain(anim_fs.sub(time=t),title=f"Animate {subject}, {t}s")  

p = plot.Butterfly(inanim_fs.norm('space'), color='k')
times = [0.12,0.17,0.24,0.37,0.48,0.55]
for t in times:
    p.add_vline(t)
for t in times:
    f = plot.GlassBrain(inanim_fs.sub(time=t),title=f"Inanimate {subject}, {t}s")      

# %% [markdown]
# ## MNE inverse

# %%
raw = mne.io.read_raw_fif(raw_fif, preload=True,verbose=False)
events = find_events(raw, stim_channel="UPPT001") #origin

raw2=raw
raw2.filter(1., 40., phase="zero-double", verbose=False)


rs_raw , rs_events = raw2.resample(100, npad="auto", events=events, stim_picks="UPPT001", verbose=False)

rs_raw_ndvar = load.fiff.raw_ndvar(rs_raw)

# %%
epochs =mne.Epochs(
    rs_raw, rs_events,          #_____
    tmin=-0.1,tmax=0.8,
    
    event_id={'stim_on': 2},
    verbose=False,
)

evoked = epochs.average()

inv= mne.minimum_norm.make_inverse_operator(
    info=rs_raw.info, forward=fwd, noise_cov=noise_cov,   #_____
    loose=1.0,    
    depth=0.8,
    fixed=False,
    verbose=False
)

snr= 3.0
lambda2 = 1.0/ snr**2
stc_vec = mne.minimum_norm.apply_inverse(
    evoked, inv,
    lambda2=lambda2,
    method="MNE",        
    pick_ori="vector", 
    verbose=False,
    
)

# %%
src_fs2 = mne.read_source_spaces(f"{subjects_dir}/fsaverage2/bem/fsaverage2-vol-7-src.fif",verbose=False)
len(src_fs2[0]['vertno'])

# %%
morph = mne.compute_source_morph(
    src=src,
    subject_from=subject,
    subject_to="fsaverage2",     
    subjects_dir=subjects_dir,
    spacing=7.0, 
    src_to=src_fs2,                                     #========force to use destination src size
    precompute=True,
    verbose=True,
)
stc_vec_fs = morph.apply(stc_vec)

# %%
stc_nd_fs = load.mne.stc_ndvar(
    stc_vec_fs,
    src='vol-7',                  # matches fsaverage2-vol-7-src.fif 
    subjects_dir=subjects_dir,
    subject='fsaverage2',
)
stc_nd_fs

# %%
p = plot.Butterfly(stc_nd_fs.norm('space'), color='k')
times = [0.12,0.17,0.24,0.37,0.48,0.55]
for t in times:
    p.add_vline(t)
for t in times:
    f = plot.GlassBrain(stc_nd_fs.sub(time=t),title=f"{subject}, {t}s") 

# %%
