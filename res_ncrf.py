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
from eelbrain import *
from pathlib import Path
from morph_nd import morph_nd
import os

subjects_dir = str(Path('~/Data/ds005810/derivatives/freesurfer/subjects').expanduser())

# %%
cases = []
for i in range(1, 8):
    subject = f"sub-{i:02d}"
    morphed_file = f"base_ncrf/{subject}-morphed.pickle"
    
    if os.path.exists(morphed_file):
        print(f"Morphed {subject} loaded from file.")
        inanim, anim = load.unpickle(morphed_file)
    else:
        print(f"Computing morphed for {subject}...")
        model_file = f"base_ncrf/{subject}.pickle"
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
    
data = Dataset.from_caselist(['subject', 'type', 'ncrf'], cases)
data.head()

# %%
res = testnd.VectorDifferenceRelated(
    'ncrf',             
    'type',           
    'inanimate',     
    'animate',   
    match='subject',     
    data=data,     
    tfce=True,           
    tstart=0.1,
    tstop=0.6,
    samples=1000
)
