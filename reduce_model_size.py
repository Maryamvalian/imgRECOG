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
from eelbrain import *
import numpy as np
from pathlib import Path
import os
from Beyond import *
from scipy import linalg
from ncrf._model import RegressionData

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
# create megall[] , stimall[] used to estimate model
# model is already stimated and saved
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
    """
    if os.path.exists(modelfile):
        print(f"{subject} model file exists.")
        continue
    """    
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
            
        """
        args = (meg_all , stim_all, fwd , noise_cov , 0,0.7)
        kwargs = {'normalize': 'l1','in_place': False,'mu':1e-3,
                  'verbose': True,'n_iter': 5,'n_iterc':5,'n_iterf': 10}        
        model = fit_ncrf(*args, **kwargs)  
        save.pickle(model, modelfile)
        print(f"\nModel saved to {modelfile}\n")  
        """
    except Exception as e:
     print(f"\n----------- Error processing {subject}: {e}\n")   
    print("Meg, Stimulus ready!")  

# %%
model= load.unpickle("models/reduce/sub-01_ncrf")
model.explained_var, model.compute_explained_variance(model._data)

# %%
"""
def inv_sqrtm_whitener_from_cov(noise_cov: np.ndarray) -> np.ndarray:
    
    #Used only if you do NOT pass whitening_filter explicitly.
   
    e, v = linalg.eigh(noise_cov)
    e = np.real(e)
    tol = np.finfo(np.float64).eps * 1e2 * e.max()
    keep = e > tol

    inv_sqrt = np.zeros_like(e)
    inv_sqrt[keep] = 1.0 / np.sqrt(e[keep])

    # shape: (n_sensors, n_sensors)
    return (inv_sqrt[:, None] * v.T)
"""

def reconstruct_data(
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
    in_place=False,
    do_post_normalization=True,
):
    
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
        
        if not in_place:
            if isinstance(stim, (list, tuple)):
                stim = [s.copy() for s in stim]
            else:
                stim = stim.copy()
        data.add_data(meg, stim)

    if do_post_normalization:
        data.post_normalization()

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


def compare_list_of_arrays(name, recon_list, cached_list, rtol=1e-10, atol=1e-5):
    
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
                f"  recon value: {a[where]}\n"
                f"  cached value: {b[where]}\n"
            )


def reconstruct_and_compare_with_model(model, meg_all, stim_all, noise_cov, rtol=1e-10, atol=1e-5):
   
    if getattr(model, "_data", None) is None:
        raise ValueError("model._data is None. Load the full model (with _data) to compare.")

    recon = reconstruct_data(
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

    d = model._data
    compare_list_of_arrays("meg", recon["meg"], d.meg, rtol=rtol, atol=atol)
    compare_list_of_arrays("covariates", recon["covariates"], d.covariates, rtol=rtol, atol=atol)
    compare_list_of_arrays("_bbt", recon["_bbt"], d._bbt, rtol=rtol, atol=atol)
    compare_list_of_arrays("_bE", recon["_bE"], d._bE, rtol=rtol, atol=atol)
    compare_list_of_arrays("_EtE", recon["_EtE"], d._EtE, rtol=rtol, atol=atol)

    print("OK: reconstructed meg, covariates, _bbt, _bE, _EtE match model._data within tolerance.")
    return recon



# %%
# Test : compute_explained_variance

origin_data = model._data
ev_orig = model.compute_explained_variance(origin_data)
print("EV original:", ev_orig)

recon = reconstruct_data(
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
recon = reconstruct_and_compare_with_model(
    model=model,
    meg_all=meg_all,
    stim_all=stim_all,
    noise_cov=noise_cov,
    rtol=1e-12,
    atol=1e-10,
)

# %%
#Reduced And Save
reduced_model=model
reduced_model._stim_normalization = {
    "s_normalization": reduced_model._stim_normalization,
    "nlevel": reduced_model._data.nlevel,
}
reduced_model._data = None
reduced_modelfile = "models/reduce/sub-01_reduced.pickle"
save.pickle(reduced_model, reduced_modelfile)

# %%
#reconstruct
m2 = load.unpickle(reduced_modelfile)
recon = reconstruct_data(
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
#Attach _data again to the model
def attach_recon_data(model, recon_data, *, overwrite=False):
    
    if model._data is not None and not overwrite:
        raise ValueError(
            "model._data already exists. "
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
ev = m2.compute_explained_variance(m2._data)

# %%
ev

# %%
#compare pickle size
full = Path("models/reduce/sub-01_ncrf.pickle")
reduced = Path("models/reduce/sub-01_ncrf_reduced.pickle")

for label, path in [("Full", full), ("Reduced", reduced)]:
    size_mb = path.stat().st_size / (1024 ** 2)
    print(f"{label} model: {size_mb:.2f} MB")


# %%
ev

# %%
