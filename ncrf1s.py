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

# %%
subject="sub-01"
root = Path("~/Data/ds005810")
#subjects_dir = root / "derivatives" / "freesurfer" / "subjects"
#clean_fif = root / f"derivatives/preprocessed/raw/{subject}_ses-ImageNet01_task-ImageNet_run-01_meg_clean.fif"
raw_fif=root/f"{subject}/ses-ImageNet01/meg/{subject}_ses-ImageNet01_task-ImageNet_run-01_meg.fif"
empty_room=root/"sub-emptyroom/ses-20211114/meg/sub-emptyroom_ses-20211114_task-noise_meg.fif"

#mne_out = Path("/Users/maryamvalian/Data/ds005810/derivatives/mne")

subjects_dir = str(Path('~/Data/ds005810/derivatives/freesurfer/subjects').expanduser())

# %%
raw = mne.io.read_raw_fif(raw_fif, preload=True,verbose=False)

events = find_events(raw, stim_channel="UPPT001")
#raw.info["bads"]  
raw
#raw.n_times

# %%
stim = events[events[:, 2] == 2]   #ID2 : stim_on

stim_samples = stim[:, 0]

stim_times = stim_samples / raw.info["sfreq"]
len(stim) 

# %%
meta = pd.read_csv(f"/Users/maryamvalian/Data/ds005810/derivatives/detailed_events/{subject}_events.csv")
meta_r1 = meta[(meta['session'] == 'ImageNet01') & (meta['run'] == 1)].reset_index(drop=True)

# %%
n=len(meta_r1)
n

# %%
anim_flags = meta_r1.loc[:n-1, 'stim_is_animate'].to_numpy().astype(bool)

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
src_file = subjects_dir / subject / f"bem/{subject}-vol-7-src.fif"
src = mne.read_source_spaces(str(src_file),verbose=False)
src

# %%
mindist=0 #same src size for all subjects
bem_model = mne.make_bem_model(subject, ico=4, conductivity=(0.3,),verbose=False)
bem_sol   = mne.make_bem_solution(bem_model,verbose=False)

trans_fif= f"{root}/derivatives/trans/{subject}-trans.fif"
trans=mne.read_trans(trans_fif)

fwd = mne.make_forward_solution(meg.info,trans ,
                                src, bem_sol,
                                meg=True, eeg=False,
                                mindist=mindist,
                                verbose=False
                               )
print(fwd)

meg_ndvar = load.fiff.raw_ndvar(meg)
print(f"MEG: {meg_ndvar}")

#convert mne fwd to ndvar
lf = load.mne.forward_operator(fwd,src='vol-7',
                               subjects_dir=subjects_dir,
                               connectivity=False,parc='aparc+aseg') 
lf  = lf.sub(sensor=meg_ndvar.sensor)
lf

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
h = hlist[0]
                      

h= morph_source_space(
    h,
    subject_to='fsaverage',
    copy=True,
    
    )



h = h.smooth('source', 0.01, 'gaussian')
p = plot.Butterfly(h.norm('space'), color='k')
times = [0.12,0.17,0.25,0.41,0.55]
for t in times:
    p.add_vline(t)
for t in times:
    f = plot.GlassBrain(h.sub(time=t),title=f"ALL words, {t}s")  

# %% [markdown]
# ## MNE inverse

# %%
epochs =mne.Epochs(
    meg, events,
    tmin=-0.1,tmax=0.8,
    
    event_id={'stim_on': 2},
    verbose=False,
)

evoked = epochs.average()

inv= mne.minimum_norm.make_inverse_operator(
    info=raw.info, forward=fwd, noise_cov=noise_cov,
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

stc_nd = load.mne.stc_ndvar(
    stc_vec,
    src='vol-7',                         
    subjects_dir=SUBJECTS_DIR,
    subject=subject,
)

stc_nd

# %%
morphed_stc = morph_source_space(                   
    stc_nd,
    subject_to='fsaverage',
    copy=True,
    
    )

# %%
p = plot.Butterfly(morphed_stc.norm('space'), color='k')
times = [0.12,0.17,0.25,0.41,0.55,0.63]
for t in times:
    p.add_vline(t)
for t in times:
    f = plot.GlassBrain(morphed_stc.sub(time=t),title=f"stim_on morphed, {t}s") 

# %%
inv['src']

# %%
stc_vec

# %%
from mne import compute_source_morph

fs_src = mne.read_source_spaces(f"{subjects_dir}/fsaverage/bem/fsaverage-vol-7-src.fif")

morph = compute_source_morph(
    src=inv['src'],                 
    subject_from=subject,          
    subject_to='fsaverage',
    subjects_dir=SUBJECTS_DIR,
    verbose=False,
    #src_to=fs_src
)


stc_vec_fs = morph.apply(stc_vec, mri_resolution=True)       #


stc_ndvar = load.mne.stc_ndvar(
    stc_vec_fs,                    
    src='vol-7' ,         
    subjects_dir=SUBJECTS_DIR,
    subject='fsaverage',             #  <=======
)

# %%
inv['src']

# %%
fs_src

# %%
stc_vec

# %%
stc_vec_fs

# %%
p = plot.Butterfly(stc_ndvar.norm('space'), color='k')
times = [0.12,0.17,0.25,0.41,0.55,0.63]
for t in times:
    p.add_vline(t)
for t in times:
    f = plot.GlassBrain(stc_ndvar.sub(time=t),title=f"stim_on morphed, {t}s") 

# %%
"""
fs_src = mne.setup_volume_source_space(
    subject='fsaverage', subjects_dir=SUBJECTS_DIR,
    pos=7.0, mri='aseg.mgz', verbose=False
)
mne.write_source_spaces(
    f"{SUBJECTS_DIR}/fsaverage/bem/M-aseg-fsaverage-vol-7-src.fif",
    fs_src, overwrite=True
)
"""

# %%

# %%
fs = mne.read_source_spaces(
    f"{subjects_dir}/fsaverage/bem/fsaverage-vol-7-src.fif"
)
fs

# %%
fs2 = mne.read_source_spaces(
    f"{subjects_dir}/fsaverage/bem/1_fsaverage-vol-7-src.fif"
)
fs2
