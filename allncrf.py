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
from matplotlib import pyplot as plt

# %%
mod="effect"  #common, effect, dummy, ortho (effect with unbalanced trial counts)
rewrite=True
root = Path("~/Data/ds005810")
subjects_dir = str(Path('~/Data/ds005810/derivatives/freesurfer/subjects').expanduser())
fwd_dir=Path("/Users/maryamvalian/Data/ds005810/derivatives/eelbrain/cache/raw")


# %%
#Global plot setting from https://eelbrain.readthedocs.io/en/stable/recipes.html
# Configure the matplotlib figure style
FONT = 'Arial'
FONT_SIZE = 8
RC = {
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.transparent': True,
    # Font
    'font.family': 'sans-serif',
    'font.sans-serif': FONT,
    'font.size': FONT_SIZE,
    'figure.labelsize': FONT_SIZE,
    'figure.titlesize': FONT_SIZE,
    'axes.labelsize': FONT_SIZE,
    'axes.titlesize': FONT_SIZE,
    'xtick.labelsize': FONT_SIZE,
    'ytick.labelsize': FONT_SIZE,    
    'legend.fontsize': FONT_SIZE,
}
plt.rcParams.update(RC)

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
                
        modelfile = f"models/all_runs/ncrf-ec/{subject}-{session}-ncrf.pickle"
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
"""
subject="sub-11"
session="ImageNet01"
clean_fif = root / f"derivatives/preprocessed/raw/{subject}_ses-{session}_task-ImageNet_run-01_clean_meg.fif"
clean = mne.io.read_raw_fif(clean_fif, preload=False,verbose=False)
info= clean.info           
meg_ndvar = load.fiff.raw_ndvar(clean)
sensor=meg_ndvar.sensor
fwd = compute_fwd_ndvar(subject, session,subjects_dir,info,sensor)
"""

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
# <hr><br>
#
# ## Dataset from cases

# %%
cases = []
for i in range(1, 31):                                      
    if (i>9):
        sessions=["ImageNet01"]
        lastruns = [5]                     #run01,run02,..,run05   
    else:
        #sessions=["ImageNet01", "ImageNet02","ImageNet03","ImageNet04"]
        #lastruns=[2,2,8,8]
        #Only session 3,4 we can not average diff size ncrf so we keep 3,4 : 
        sessions=["ImageNet03","ImageNet04"]
        lastruns=[8,8]
    
    subject = f"sub-{i:02d}"
    for idx, session in enumerate(sessions):
                
        morphed_file = f"models/all_runs/morphed/M{subject}-{session}-ncrf.pickle"  #M stands for Morphed
        if os.path.exists(morphed_file):
            print(f"Loading {subject}-{session} from file.")
            inanim, anim = load.unpickle(morphed_file)
        else:
            try:
                
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
                               
                
                anim= anim_fs.smooth('source', 0.01, 'gaussian')
                inanim= inanim_fs.smooth('source', 0.01, 'gaussian')
            
                save.pickle((inanim, anim), morphed_file)
                print(f"\n{subject}-{session}-Morphed Saved \n ")
                
            except Exception as e:
                print(f"\n----------- Error processing {subject}: {e}\n")   
            
    
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
    tstart=100,                           #ms not second, out put of ncrf 
    tstop=700,
    samples=1000
)
#save.pickle(res, f"Tests/all_runs/allruns-ncrf_PT.pickle")

# %%
diff = res.masked_difference()

p = plot.Butterfly(diff.norm('space'), color='k', title='anim VS inanim')
times = [110, 250, 350, 500]

for t in times:
    p.add_vline(t)

for t in times:
    f = plot.GlassBrain(diff.sub(time=t), title=f"anim vs inanim, {t} ms")
    #f.plot_colorbar()


# %%

#---------------------------------
y = diff.norm('space').max('source')

fig = plt.figure(figsize=(6.5, 3.5))
plt.plot(y.time.times, y.x, )    #color='k'

for t in [118, 244, 359, 492]:
    plt.axvline(t, color='k', linewidth=0.5)

plt.title('Contrast: Animate-Inanimate')
plt.xlabel('Time [ms]')
plt.ylabel('maximum difference')
plt.tight_layout()

fig.figure.savefig("figures/contrast_BF_max.pdf",
    bbox_inches="tight",
    transparent=True
)

plt.show()

# %%

# %%
from matplotlib import pyplot as plt
from eelbrain import plot
from PIL import Image
import os

# =========================
# Global plot settings
# =========================

FONT = 'Arial'
FONT_SIZE = 8
LINEWIDTH = 0.5

plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.transparent': True,

    'font.family': 'sans-serif',
    'font.sans-serif': FONT,
    'font.size': FONT_SIZE,
    'axes.labelsize': FONT_SIZE,
    'axes.titlesize': FONT_SIZE,
    'xtick.labelsize': FONT_SIZE,
    'ytick.labelsize': FONT_SIZE,
    'legend.fontsize': FONT_SIZE,

    'axes.linewidth': LINEWIDTH,
    'grid.linewidth': LINEWIDTH,
    'lines.linewidth': LINEWIDTH,
    'patch.linewidth': LINEWIDTH,
    'xtick.major.width': LINEWIDTH,
    'xtick.minor.width': LINEWIDTH,
    'ytick.major.width': LINEWIDTH,
    'ytick.minor.width': LINEWIDTH,
})

os.makedirs("figures/tmp", exist_ok=True)

# =========================
# Data
# =========================

diff = res.masked_difference()

times = [110, 250, 350, 500]

# =========================
# Butterfly plot
# =========================

p = plot.Butterfly(
    diff.norm('space'),
    color='k',
    title=None
)

for t in times:
    p.add_vline(t)

p.figure.suptitle(
    "anim VS inanim",
    fontsize=FONT_SIZE,
    fontfamily=FONT,
    y=0.98
)

p.figure.savefig(
    "figures/contrast_butterfly.pdf",
    bbox_inches="tight",
    pad_inches=0.02,
    transparent=True
)

plt.close(p.figure)

# =========================
# GlassBrain plots
# =========================

image_paths = []

for t in times:

    f = plot.GlassBrain(
        diff.sub(time=t),
        title=None,
        w=5,
        h=1.8
    )

    f.figure.suptitle(
        f"anim vs inanim, {t} ms",
        fontsize=FONT_SIZE,
        fontfamily=FONT,
        y=0.98
    )

    out = f"figures/tmp/contrast_gb_{t}.png"

    f.figure.savefig(
        out,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.01,
        transparent=False
    )

    plt.close(f.figure)

    image_paths.append(out)

# =========================
# Stack GlassBrains vertically
# =========================

imgs = [Image.open(p).convert("RGB") for p in image_paths]

width = max(img.width for img in imgs)
height = sum(img.height for img in imgs)

combined = Image.new("RGB", (width, height), "white")

y = 0
for img in imgs:
    x = (width - img.width) // 2
    combined.paste(img, (x, y))
    y += img.height

combined.save(
    "figures/contrast_GBs.pdf",
    "PDF",
    resolution=300.0
)

print("Saved:")
print("figures/contrast_butterfly.pdf")
print("figures/contrast_GBs.pdf")

# %%
diff= res.masked_difference()
p = plot.Butterfly(diff.norm('space'), color='k',title='anim VS inanim')
times = [110,250,350,500]
for t in times:
    p.add_vline(t)
for t in times:
    f = plot.GlassBrain(diff.sub(time=t),title=f"anim vs inan, {t}s")  

# %%
#ROI_mask

# %%
diff_250=diff.sub(time=250)


d = diff_250.x  
magnitude = np.linalg.norm(d, axis=1)

threshold = np.percentile(magnitude, 95)
ROI_mask = magnitude >= threshold


roi_value = (diff_250 * ROI_mask).mean('space')


# %% [markdown]
# ## ONE Sample test

# %%

# %%
data_inan = data_avg.sub("animacy == 'inanimate'")
result_inan = testnd.Vector('ncrf', match='subject', data=data_inan, tfce=True, tstart=1, tstop=600,samples=1000)

# %%
y = result_inan.masked_difference().norm('space').max('source')

fig, ax = plt.subplots(figsize=(6.5, 3.5))

ax.plot(y.time.times, y.x, ) #color='k'

for t in [130, 240]:
    ax.axvline(t, color='k', linewidth=0.5)

ax.set_title('Inanimate')
ax.set_xlabel('Time [ms]')
ax.set_ylabel('Maximum NCRF')

fig.tight_layout()


fig.figure.savefig("figures/Inanim_BF_max.pdf",
    bbox_inches="tight",
    transparent=True
)
plt.show()

# %%
p = plot.Butterfly(result_inan.masked_difference().norm('space'), color='k')
times = [130,250]
for t in times:
    p.add_vline(t)
for t in times:
    f = plot.GlassBrain(result_inan.masked_difference().sub(time=t),title=f"Inanimate, {t}s")  

# %%
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from eelbrain import plot

times = [130, 250]

# =========================
# Save Butterfly separately
# =========================

p = plot.Butterfly(
    result_inan.masked_difference().norm('space'),
    color='k'
)

for t in times:
    p.add_vline(t)

p.figure.savefig(
    "figures/inanim_butterfly.pdf",
    bbox_inches="tight",
    transparent=True
)

plt.close(p.figure)

# =========================
# Save all GlassBrains in one PDF
# =========================

with PdfPages("figures/inanim_glassbrains.pdf") as pdf:

    for t in times:

        f = plot.GlassBrain(
            result_inan.masked_difference().sub(time=t),
            title=f"Inanimate, {t} ms",
            w=4,
            h=3
        )

        pdf.savefig(
            f.figure,
            bbox_inches="tight",
            transparent=True
        )

        plt.close(f.figure)

print("Saved:")
print("figures/inanim_butterfly.pdf")
print("figures/inanim_glassbrains.pdf")

# %%
from matplotlib import pyplot as plt
from eelbrain import plot
from PIL import Image
import os

os.makedirs("figures/tmp", exist_ok=True)

times = [130, 250]
image_paths = []

# Save each GlassBrain as tightly cropped PNG first
for t in times:
    f = plot.GlassBrain(
        result_inan.masked_difference().sub(time=t),
        title=f"Inanimate, {t} ms",
        w=5,
        h=2.0
    )

    out = f"figures/tmp/glassbrain_{t}.png"

    f.figure.savefig(
        out,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
        transparent=False
    )

    plt.close(f.figure)
    image_paths.append(out)

# Load cropped images
imgs = [Image.open(p).convert("RGB") for p in image_paths]

# Stack them vertically
width = max(img.width for img in imgs)
height = sum(img.height for img in imgs)

combined = Image.new("RGB", (width, height), "white")

y = 0
for img in imgs:
    x = (width - img.width) // 2
    combined.paste(img, (x, y))
    y += img.height

# Save as one PDF
combined.save(
    "figures/inanim_glassbrains.pdf",
    "PDF",
    resolution=300.0
)

print("Saved: figures/inanim_glassbrains.pdf")

# %%

# %%

# %%
