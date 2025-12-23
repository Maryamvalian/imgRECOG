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
# # Load, Morph if needed, make Dataset

# %%
cases = []
for i in range(1, 31): 
    if i==6 or i==7: continue
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


# %% [markdown]
# ## ONE Sample test (Inan)

# %%
data_inan = data_avg.sub("animacy == 'inanimate'")
data_an = data_avg.sub("animacy == 'animate'")

#One sample ttest
result_inan = testnd.Vector('ncrf', match='subject', data=data_inan, tfce=True, tstart=1, tstop=600,samples=1000)
result_an = testnd.Vector('ncrf', match='subject', data=data_an, tfce=True, tstart=1, tstop=600,samples=1000)

# %%
p = plot.Butterfly(result_inan.masked_difference().norm('space'), color='k')
times = [130,240,400]
for t in times:
    p.add_vline(t)
for t in times:
    f = plot.GlassBrain(result_inan.masked_difference().sub(time=t),title=f"Inanimate, {t}s")  

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
# ## Aggregate and difference

# %%
agg=data.aggregate('animacy', drop_bad=True)
agg_in = agg.sub("animacy == 'inanimate'")
agg_an = agg.sub("animacy == 'animate'")

diff=agg_an['ncrf']- agg_in['ncrf']

p = plot.Butterfly(diff.norm('space'), color='k')
times = [120,250,500]
for t in times:
    p.add_vline(t)
for t in times:
    f = plot.GlassBrain(diff.sub(time=t),title=f"diff: animate-Inanimate, {t}s") 

# %%

# %%








# %%

# %%

# %%

# %% [markdown]
# ### Contrast then testnd.vector

# %%
ds_an = data_avg.sub("animacy == 'animate'")
ds_in = data_avg.sub("animacy == 'inanimate'")

ds_an.sort('subject')  
ds_in.sort('subject')   

ds_diff = ds_an.copy()
ds_diff['ncrf'] = ds_an['ncrf'] - ds_in['ncrf']

result_diff = testnd.Vector(
    'ncrf', match='subject', data=ds_diff,
    tfce=True, tstart=1, tstop=600, samples=5000
)



# %%
print(result_diff)


# %%
stat = result_diff.t2  
#collaps time
stat_static = stat.sub(time=(100, 500)).mean('time')
f = plot.GlassBrain(
    stat_static,
    cmap='cold_hot',
    title='Animate – Inanimate (Hotelling T², 150–250 ms)'
)
f.plot_colorbar()

# %%
ds_an['ncrf_mag'] = ds_an['ncrf'].norm('space')
ds_in['ncrf_mag'] = ds_in['ncrf'].norm('space')

ds_diff['ncrf_mag'] = ds_an['ncrf_mag'] - ds_in['ncrf_mag']

result = testnd.TTestOneSample(
    'ncrf_mag', match='subject', data=ds_diff,
    tfce=True, tstart=1, tstop=600,  samples=1000
)


# %%
t_map = result.t                      
t_static = t_map.sub(time=(100, 150)).mean('time')

f = plot.GlassBrain(
    t_static,
    #cmap='cold_hot',
    title='t value'
)
f.plot_colorbar()

# %%
print(t_static.x.min(), t_static.x.max())

# %%

# %%

# %%

# %%
t_map = result.t

p = plot.Butterfly(t_map, color='k')
times = [120, 250, 500]
for tt in times:
    p.add_vline(tt)


for tt in times:
    t_snap = t_map.sub(time=(tt - 1, tt + 1)).mean('time') 
    v = abs(t_snap.x).max()  

    f = plot.GlassBrain(
        t_snap,
        cmap='cold_hot',
        vmin=-v, vmax=v,
        title=f"t-map (animate - inanimate), ~{tt} ms"
    )
    f.plot_colorbar()


# %%

# %%
ds_an['ncrf_mag'] = ds_an['ncrf'].norm('space')
ds_in['ncrf_mag'] = ds_in['ncrf'].norm('space')

ds_diff['ncrf_mag'] = ds_an['ncrf_mag'] - ds_in['ncrf_mag']

result = testnd.TTestOneSample(
    'ncrf_mag', match='subject', data=ds_diff,
    tfce=True, tstart=1, tstop=600,  samples=1000
)

# %%
p_static = result.p.sub(time=(100, 120)).min('time')

# mask: significant
mask =  (p_static.x < 0.05)   #(t_static.x < 0) &

t_pos_sig = t_static.copy()
t_pos_sig.x[~mask] = 0

f = plot.GlassBrain(
    t_pos_sig,
    cmap='cold_hot',
    vmin=t_pos_sig.x.min(),
    vmax=t_pos_sig.x.max(),
    title='Negetive and significant (anim < inanim)',
    symmetric_cbar=False
)
f.plot_colorbar()


p_static = result.p.sub(time=(220, 230)).min('time')

# mask: significant
mask =  (p_static.x < 0.05)   #(t_static.x < 0) &

t_pos_sig = t_static.copy()
t_pos_sig.x[~mask] = 0

f = plot.GlassBrain(
    t_pos_sig,
    cmap='cold_hot',
    vmin=t_pos_sig.x.min(),
    vmax=t_pos_sig.x.max(),
    title='Negetive and significant (anim < inanim)',
    symmetric_cbar=False
)
f.plot_colorbar()

# %%

# %%

# %%
# positive
t_pos = t_static.copy()
t_pos.x[t_pos.x <= 0] = 0

# negative
t_neg = t_static.copy()
t_neg.x[t_neg.x >= 0] = 0
t_neg.x = -t_neg.x


plot.GlassBrain(t_pos, cmap='hot', title='Animate > Inanimate')
plot.GlassBrain(t_neg, cmap='hot', title='Inanimate > Animate')


# %%

# %%

# %%
stat = res.t2         
pmap = res.p          


times = [110, 250, 350, 500]

for t in times:
    stat_snap = stat.sub(time=(t-1, t+1)).mean('time')
    p_snap = pmap.sub(time=(t-1, t+1)).min('time')  

    #significance
    masked = stat_snap * (p_snap < 0.05)

    
    f = plot.GlassBrain(
        masked,
        cmap='cold_hot',          
        vmin=0,
        vmax=masked.x.max() if masked.x.max() > 0 else 1,
        symmetric_cbar=False,
        title=f"TFCE significant T² (p<0.05), ~{t} ms"
    )
    f.plot_colorbar()


# %%
