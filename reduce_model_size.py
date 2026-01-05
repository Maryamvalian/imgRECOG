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
import eelbrain as eb
import seaborn as sns
import random

# %%
mod="dummy"  #common, effect, dummy, ortho (effect with unbalanced trial counts)
rewrite=True
root = Path("~/Data/ds005810")
subjects_dir = str(Path('~/Data/ds005810/derivatives/freesurfer/subjects').expanduser())
fwd_dir=Path("/Users/maryamvalian/Data/ds005810/derivatives/eelbrain/cache/raw")
model_dir = "models/consist"

# noise
empty_room=root/"sub-emptyroom/ses-20211114/meg/sub-emptyroom_ses-20211114_task-noise_meg.fif"
raw_er = mne.io.read_raw_fif(empty_room, preload=True, verbose=False).pick('meg')
raw_er.filter(1., 40., phase="zero-double", verbose=False)
raw_er.resample(100, npad="auto", verbose=False)
noise_cov = mne.compute_raw_covariance(raw_er, method='shrunk', rank=None,verbose=False)

#functions
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
    


# %%
for i in range (1,2):                # one subject
    session="ImageNet03"
    subject = f"sub-{i:02d}"
    subset=["02", "05" ,"03"]
    
    print(f"computing fwd for {subject}-{session}... ")
    clean_fif = root / f"derivatives/preprocessed/raw/{subject}_ses-{session}_task-ImageNet_run-01_clean_meg.fif"
    clean = mne.io.read_raw_fif(clean_fif, preload=False,verbose=False)
    info= clean.info           
    meg_ndvar = load.fiff.raw_ndvar(clean)
    sensor=meg_ndvar.sensor
    fwd = compute_fwd_ndvar(subject, session,subjects_dir,info,sensor)
    
    
    
    modelfile = f"models/reduce/{subject}_ncrf.pickle"
    if os.path.exists(modelfile):
        print(f"{subject} model file exists.")
        continue
    try:        
        
        meg_all = []
        stim_all = []
        for run in subset:
            
            print(f"Loading run-{run} MEG ...")
            meg= load_meg_ndvar(subject, session, run)
            meg_all.append(meg)
            event_table= make_event_table(subject, session, run)
            stim1,stim2= make_predictors_for_run(meg, event_table,mod=mod)
            predictors=[stim1,stim2]
            stim_all.append(predictors)  
            
        args = (meg_all , stim_all, fwd , noise_cov , 0,0.7)
        kwargs = {'normalize': 'l1','in_place': False,'mu':1e-3,
                  'verbose': True,'n_iter': 5,'n_iterc':5,'n_iterf': 10}        
        model = fit_ncrf(*args, **kwargs)  
        save.pickle(model, modelfile)
        print(f"\nModel saved to {modelfile}\n")  
        
    except Exception as e:
     print(f"\n----------- Error processing {subject}: {e}\n")   

# %%
model.mu

# %%
model._data

# %%
type(model._data)

# %%
import pickle
for attr in model._PICKLE_ATTRS:
    s = pickle.dumps(getattr(model, attr))
    print(len(s), ' ', attr)

# %%
model._data.__dict__.keys()


# %%
type(model._data._bE)     #b^T E   : B time-lagged stimulus matrix, E whitened MEG for each run

# %%
len(model._data._bE)      # be tedade run-ha hast subset = 2,5,3

# %%
len(model._data._EtE) , model._data._EtE[0].shape

# %%
len(model._data._bbt) , model._data._bbt[0].shape

# %%
x = model._data._bE[0]             #First run
x.shape ,x.dtype

# %% [markdown]
# # It is not necessary to store BE
# 1. Build RegressionData(meg_all, stim_all, ...)
# 2. That constructor recomputes:
#    - covariates
#    - whitened MEG
#    - BᵀE → _bE
#    - BᵀB → _bbt
#    - EᵀE → _EtE

# %%
what=model._data.tstart
what

# %%
len(model._data.meg)

# %%
len(model._data.covariates)

# %%
#Start of the TRF in seconds. Can define multiple tstarts for more than 1 predictor.
#tstop : float | list[float]
        #Stop of the TRF in seconds. Can define multiple tstops for more than 1 predictor.
len(model._data.tstart) , len(model._data.tstop)   

# %%
model._data.tstep

# %%
len(model._data._bE)

# %%
len(model._data._stim_names) , model._data._stim_names[0] ,model._data._stim_names[0]

# %%
# Decides the density of Gabor atoms. Bigger nlevel -> less dense basis
#By default it is set to `1`. `nlevesl > 2` should be used with caution
model._data.nlevel 

# %%
len(model._data.s_baseline)

# %%
#stores the value that was removed from each predictor so the stimulus is mean-centered before regression
# mean for predictor 0
model._data.s_baseline[0]                   

# %%
#divisor for normalization based on normalize otion : l1 ,..
len(model._data.s_scaling)

# %%
model._data.s_scaling[0]

# %%
len(model._data.s_normalization )   #for each run

# %%
model._data.s_normalization[0]   # for run0, for each predictor l1 normwt is the differe

# %%
model.tstop

# %%
model.tstart

# %%
model.tstep

# %%
len(model._data.meg)

# %%
model._data.meg[0].shape             #channels* timepoints

# %%
model._data.meg[0] is meg_all[0].x

# %%
(meg_all[0].x.shape)

# %%
type(model._data.meg[0])

# %%
model._data._prewhitened

# %%
model._data.filter_length

# %%
model._data.start, model._data.stop

# %%
model.gaussian_fwhm

# %%
model._data.gaussian_fwhm

# %%
len(model._data.basis)

# %%
model._data.basis[0].shape             #number of lag * number of basis functions

# %%
model._basis[0].shape

# %%
model.nlevel

# %%
# Full, self-contained script to:
# 1) Reconstruct RegressionData heavy cached arrays from meg_all, stim_all, noise_cov, and model metadata
# 2) Compare reconstructed arrays with model._data (meg, covariates, _bbt, _bE, _EtE)

import numpy as np
from scipy import linalg

# Adjust this import if your package path differs
# It must point to the RegressionData class you pasted from _model.py
from ncrf._model import RegressionData


def inv_sqrtm_whitener_from_cov(noise_cov: np.ndarray) -> np.ndarray:
    """
    Compute a whitening matrix W such that W @ cov @ W.T ≈ I
    This is only used if you do NOT pass model._whitening_filter.
    For an exact match with cached data, pass whitening_filter=model._whitening_filter.
    """
    e, v = linalg.eigh(noise_cov)
    e = np.real(e)
    tol = np.finfo(np.float64).eps * 1e2 * e.max()
    keep = e > tol
    inv_sqrt = np.zeros_like(e)
    inv_sqrt[keep] = 1.0 / np.sqrt(e[keep])
    # W has shape (n_sensors, n_sensors)
    W = (inv_sqrt[:, None] * v.T)
    return W


def reconstruct_heavy_cached_arrays(
    meg_all,
    stim_all,
    noise_cov,
    *,
    tstart,
    tstop,
    nlevel=1,
    baseline=None,
    scaling=None,
    stim_is_single=None,
    gaussian_fwhm=20.0,
    whitening_filter=None,
):
    """
    Rebuild the heavy cached arrays exactly the same way RegressionData does:
      - data.add_data(...) -> builds data.meg and data.covariates
      - data._prewhiten(W) -> applies whitening to data.meg
      - data._precompute() -> builds _bbt, _bE, _EtE

    Returns a dict with meg, covariates, _bbt, _bE, _EtE, and the RegressionData instance.
    """
    if len(meg_all) != len(stim_all):
        raise ValueError(f"len(meg_all)={len(meg_all)} must match len(stim_all)={len(stim_all)}")

    if whitening_filter is None:
        whitening_filter = inv_sqrtm_whitener_from_cov(noise_cov)

    data = RegressionData(
        tstart=tstart,
        tstop=tstop,
        nlevel=nlevel,
        baseline=baseline,
        scaling=scaling,
        stim_is_single=stim_is_single,
        gaussian_fwhm=gaussian_fwhm,
    )

    for meg, stim in zip(meg_all, stim_all):
        data.add_data(meg, stim)

    # Apply whitening and compute caches
    data._prewhiten(whitening_filter)
    data._precompute()

    return {
        "data": data,
        "meg": data.meg,
        "covariates": data.covariates,
        "_bbt": data._bbt,
        "_bE": data._bE,
        "_EtE": data._EtE,
    }


def compare_list_of_arrays(name, recon_list, cached_list, rtol=1e-10, atol=1e-12):
    """
    Compare two lists of numpy arrays.
    Raises AssertionError with useful debugging info if mismatch is found.
    """
    if len(recon_list) != len(cached_list):
        raise AssertionError(f"{name}: length mismatch {len(recon_list)} vs {len(cached_list)}")

    for i, (a, b) in enumerate(zip(recon_list, cached_list)):
        if a.shape != b.shape:
            raise AssertionError(f"{name}[{i}]: shape mismatch {a.shape} vs {b.shape}")

        if not np.allclose(a, b, rtol=rtol, atol=atol):
            diff = np.max(np.abs(a - b))
            where = np.unravel_index(np.argmax(np.abs(a - b)), a.shape)
            raise AssertionError(
                f"{name}[{i}]: values differ\n"
                f"  max abs diff: {diff}\n"
                f"  at index: {where}\n"
                f"  recon value: {a[where]}\n"
                f"  cached value: {b[where]}\n"
            )


def reconstruct_and_compare_with_model(model, meg_all, stim_all, noise_cov, rtol=1e-10, atol=1e-12):
    """
    Reconstruct and compare against model._data for:
      - meg
      - covariates
      - _bbt
      - _bE
      - _EtE

    Uses model._whitening_filter for exact match, and uses nlevel from model._data (since it is only there).
    """
    if model._data is None:
        raise ValueError("model._data is None. You need the cached data present to compare.")

    recon = reconstruct_heavy_cached_arrays(
        meg_all=meg_all,
        stim_all=stim_all,
        noise_cov=noise_cov,
        tstart=model.tstart,
        tstop=model.tstop,
        nlevel=model._data.nlevel,
        baseline=model._stim_baseline,
        scaling=model._stim_scaling,
        stim_is_single=model._stim_is_single,
        gaussian_fwhm=model.gaussian_fwhm,
        whitening_filter=model._whitening_filter,  # key line for exact match
    )

    d = model._data

    compare_list_of_arrays("meg", recon["meg"], d.meg, rtol=rtol, atol=atol)
    compare_list_of_arrays("covariates", recon["covariates"], d.covariates, rtol=rtol, atol=atol)
    compare_list_of_arrays("_bbt", recon["_bbt"], d._bbt, rtol=rtol, atol=atol)
    compare_list_of_arrays("_bE", recon["_bE"], d._bE, rtol=rtol, atol=atol)
    compare_list_of_arrays("_EtE", recon["_EtE"], d._EtE, rtol=rtol, atol=atol)

    print("OK: reconstructed meg, covariates, _bbt, _bE, _EtE match model._data within tolerance.")
    return recon





 # %%
 recon = reconstruct_and_compare_with_model(
     model=model,
     meg_all=meg_all,
     stim_all=stim_all,
     noise_cov=noise_cov,
     rtol=1e-10,
     atol=1e-5,              #absolute tolerance |a-b|<=atol+rtol*|b| #handle error near zero
 )

# %%
model.explained_var

# %%
origin_data=model._data
ev = model.compute_explained_variance(origin_data)
ev

# %%
# Test : compute_explained_variance

origin_data = model._data
ev_orig = model.compute_explained_variance(origin_data)
print("EV original:", ev_orig)

recon = reconstruct_heavy_cached_arrays(
    meg_all=meg_all,
    stim_all=stim_all,
    noise_cov=noise_cov,
    tstart=model.tstart,
    tstop=model.tstop,
    nlevel=model._data.nlevel,
    baseline=model._stim_baseline,
    scaling=model._stim_scaling,
    stim_is_single=model._stim_is_single,
    gaussian_fwhm=model.gaussian_fwhm,
    whitening_filter=model._whitening_filter,
    in_place=False,
    do_post_normalization=True,
)
recon_data = recon["data"]
ev_recon = model.compute_explained_variance(recon_data)
print("EV reconstructed:", ev_recon)


diff = float(abs(ev_recon - ev_orig))
print("diff:", diff)

print("EV allclose:", np.allclose(ev_recon, ev_orig, rtol=1e-10, atol=1e-8))


# %%
model.nlevel = model._data.nlevel
model._data = None


# %%
model= load.unpickle("models/reduce/sub-01_ncrf")

# %%
#model.nlevel      #this way it is not saved by ncrf class

# %%
model._stim_normalization = {
    "s_normalization": model._stim_normalization,
    "nlevel": model._data.nlevel,
}
model._data = None
reduced_modelfile = "models/reduce/sub-01_ncrf_reduced.pickle"
save.pickle(model, reduced_modelfile)

# %%
m2 = load.unpickle(reduced_modelfile)
recon = reconstruct_heavy_cached_arrays(
    meg_all=meg_all,
    stim_all=stim_all,
    noise_cov=noise_cov,
    tstart=m2.tstart,
    tstop=m2.tstop,
    nlevel=m2._stim_normalization["nlevel"],
    baseline=m2._stim_baseline,
    scaling=m2._stim_scaling,
    stim_is_single=m2._stim_is_single,
    gaussian_fwhm=m2.gaussian_fwhm,
    whitening_filter=m2._whitening_filter,
    in_place=False,
    do_post_normalization=True,
)

ev = m2.compute_explained_variance(recon["data"])
print("EV (reloaded reduced model):", ev)


# %%
def attach_recon_data(model, recon_data, *, overwrite=False):
    
    if model._data is not None and not overwrite:
        raise ValueError(
            "model._data already exists. "
            "Use overwrite=True if you really want to replace it."
        )

    
    model._data = recon_data

    
    if model.tstart != recon_data.tstart:
        raise ValueError("tstart mismatch between model and reconstructed data")

    if model.tstop != recon_data.tstop:
        raise ValueError("tstop mismatch between model and reconstructed data")

    if model.gaussian_fwhm != recon_data.gaussian_fwhm:
        raise ValueError("gaussian_fwhm mismatch")

    return model


# %%
attach_recon_data(m2, recon["data"])

# %%
ev = m2.compute_explained_variance(m2._data)
ev

# %% [markdown]
# m2.h

# %%
m2.h

# %%
