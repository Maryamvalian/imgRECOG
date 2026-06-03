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
from ncrf import fit_ncrf
from eelbrain import NDVar, UTS
from eelbrain import plot, combine
from eelbrain import *
from eelbrain._data_obj import VolumeSourceSpace
import os
from pathlib import Path
from matplotlib import pyplot as plt
import nibabel as nib            #for loading atlas 
from scipy.spatial import cKDTree    #a fast nearest-neighbor search algorithm for reassign label to unlabeled sources
import copy # for deep copy

# %%
mod="dummy"

cases = []

for i in range(1, 31):

    if i > 9:
        sessions = ["ImageNet01"]
        lastruns = [5]
    else:
        sessions = ["ImageNet03", "ImageNet04"]
        lastruns = [8, 8]

    subject = f"sub-{i:02d}"

    for idx, session in enumerate(sessions):

        morphed_file = f"models/all_runs/morphed/M{subject}-{session}-ncrf.pickle"

        try:
            if os.path.exists(morphed_file):
                print(f"Loading {subject}-{session} from file.")
                inanim, anim = load.unpickle(morphed_file)

            else:
                print(f"Morphing {subject}-{session}...")

                modelfile = f"models/all_runs/{subject}-{session}-ncrf.pickle"
                model = load.unpickle(modelfile)
                hlist = model.h

                if mod == "dummy":
                    inanim = hlist[0]
                    anim = hlist[1]

                elif mod == "effect":
                    h_mean, h_contrast = hlist[0], hlist[1]
                    anim = h_mean + h_contrast
                    inanim = h_mean - h_contrast

                else:
                    raise ValueError(f"Unknown mod: {mod}")

                fwd_file = fwd_dir / f"{subject}_ses-{session}/{subject}-fwd.fif"
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
                _, _, anim_fs = morph_hemi(
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

                anim = anim_fs.smooth('source', 0.01, 'gaussian')
                inanim = inanim_fs.smooth('source', 0.01, 'gaussian')

                save.pickle((inanim, anim), morphed_file)
                print(f"\n{subject}-{session}-Morphed Saved\n")

            cases.append([subject, session, 'inanimate', inanim])
            cases.append([subject, session, 'animate', anim])

        except Exception as e:
            print(f"\n----------- Error processing {subject}-{session}: {e}\n")
            continue

data = Dataset.from_caselist(
    ['subject', 'session', 'animacy', 'ncrf'],
    cases
)

data.head()

data_agg=data
data_avg=data_agg.aggregate(('animacy% subject'), drop_bad=True)
data_avg

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

# %%
diff = res.masked_difference()
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
trials_per_subset = 200
dirs = {
    "NCRF-DC": "models/samesize/1session/dummy",
    "MNE": "models/samesize/1session/MNE",
    "NCRF-EC": "models/samesize/1session/effect",
    
}
def load_model_subset(model_dir, m, subject, size):
   
   
    file_path = f"{model_dir}/M{m}-{size}-{subject}.pickle"
    #if "2sesmne" in model_dir.lower() or "mne" in model_dir.lower(): file_path = f"{model_dir}/{m}-{size}-{subject}.pickle" #for 2sesmne
    inan, anim = load.unpickle(file_path)
    
    return inan, anim

# %%
"""
size = 1
method_name = "NCRF-DC"
model_dir = dirs[method_name]

cases = []

for i in range(1, 31):
    subject = f"sub-{i:02d}"

    try:
        inan1, anim1 = load_model_subset(
            model_dir=model_dir,
            m=1,
            subject=subject,
            size=size
        )

        cases.append([subject, "inanimate", inan1])
        cases.append([subject, "animate", anim1])

    except Exception as e:
        print(f"Skipping {subject}: {e}")
        continue

data_025 = Dataset.from_caselist(
    ["subject", "animacy", "ncrf"],
    cases
)

res_025 = testnd.VectorDifferenceRelated(
    "ncrf",
    "animacy",
    "inanimate",
    "animate",
    match="subject",
    data=data_025,
    tfce=True,
    tstart=100,
    tstop=700,
    samples=1000
)

diff_025 = res_025.masked_difference()

y_025 = diff_025.norm("space").max("source")

fig, ax = plt.subplots(figsize=(6.5, 3.5))

ax.plot(
    y_025.time.times,
    y_025.x,
    color="k"
)

for t in [118, 244, 359, 492]:
    ax.axvline(
        t,
        color="k",
        linewidth=0.5
    )

ax.set_title(f"trialsize={size*trials_per_subset}")
ax.set_xlabel("Time [ms]")
ax.set_ylabel("Maximum difference")

fig.tight_layout()

fig.savefig(
    "figures/ncrf_dc_025_contrast_max_butterfly.pdf",
    bbox_inches="tight",
    transparent=True
)

plt.show()
"""

# %%
"""
sizes_to_plot = [0.25, 0.5, 1, 2]

method_name = "NCRF-DC"
model_dir = dirs[method_name]

all_curves = {}

#reduced sizes less than one session : 50,100,200,400 trials

for size in sizes_to_plot:

    print(f"\nProcessing size={size}")

    cases = []

    for i in range(1, 31):
        subject = f"sub-{i:02d}"

        try:
            inan1, anim1 = load_model_subset(
                model_dir=model_dir,
                m=1,
                subject=subject,
                size=size
            )

            cases.append([subject, "inanimate", inan1])
            cases.append([subject, "animate", anim1])

        except Exception as e:
            print(f"Skipping {subject}: {e}")
            continue

    data_size = Dataset.from_caselist(
        ["subject", "animacy", "ncrf"],
        cases
    )

    res_size = testnd.VectorDifferenceRelated(
        "ncrf",
        "animacy",
        "inanimate",
        "animate",
        match="subject",
        data=data_size,
        tfce=True,
        tstart=100,
        tstop=700,
        samples=1000
    )

    diff_size = res_size.masked_difference()
    y_size = diff_size.norm("space").max("source")

    all_curves[size] = y_size


#one session size= 5 or 1000 trail

cases = []

one_session_dir = "models/all_runs/morphed"

for i in range(1, 31):

    subject = f"sub-{i:02d}"

    if i > 9:
        session = "ImageNet01"
    else:
        session = "ImageNet03"

    morphed_file = f"{one_session_dir}/M{subject}-{session}-ncrf.pickle"

    try:
        inanim, anim = load.unpickle(morphed_file)

        cases.append([subject, "inanimate", inanim])
        cases.append([subject, "animate", anim])

    except Exception as e:
        print(f"Skipping one-session {subject}-{session}: {e}")
        continue

data_onesession = Dataset.from_caselist(
    ["subject", "animacy", "ncrf"],
    cases
)

res_onesession = testnd.VectorDifferenceRelated(
    "ncrf",
    "animacy",
    "inanimate",
    "animate",
    match="subject",
    data=data_onesession,
    tfce=True,
    tstart=100,
    tstop=700,
    samples=1000
)

diff_onesession = res_onesession.masked_difference()
y_onesession = diff_onesession.norm("space").mean("source")


#all runs

cases = []

all_runs_dir = "models/all_runs/morphed"

for i in range(1, 31):

    if i > 9:
        sessions = ["ImageNet01"]
    else:
        sessions = ["ImageNet03", "ImageNet04"]

    subject = f"sub-{i:02d}"

    for session in sessions:

        morphed_file = f"{all_runs_dir}/M{subject}-{session}-ncrf.pickle"

        try:
            inanim, anim = load.unpickle(morphed_file)

            cases.append([subject, session, "inanimate", inanim])
            cases.append([subject, session, "animate", anim])

        except Exception as e:
            print(f"Skipping all-runs {subject}-{session}: {e}")
            continue

data_all = Dataset.from_caselist(
    ["subject", "session", "animacy", "ncrf"],
    cases
)

data_all_avg = data_all.aggregate(
    "animacy % subject",
    drop_bad=True
)

res_all = testnd.VectorDifferenceRelated(
    "ncrf",
    "animacy",
    "inanimate",
    "animate",
    match="subject",
    data=data_all_avg,
    tfce=True,
    tstart=100,
    tstop=700,
    samples=1000
)

diff_all = res_all.masked_difference()
y_all = diff_all.norm("space").max("source")


#plot all

fig, ax = plt.subplots(figsize=(7, 4))

colors = {
    0.25: "tab:blue",
    0.5: "tab:orange",
    1: "tab:green",
    2: "tab:red",
}

for size, y in all_curves.items():

    n_trials = int(size * trials_per_subset)

    ax.plot(
        y.time.times,
        y.x,
        label=f"{n_trials} trials",
        color=colors[size]
    )

ax.plot(
    y_onesession.time.times,
    y_onesession.x,
    label="1000 trials",
    color="tab:purple",
    linewidth=1.0
)

ax.plot(
    y_all.time.times,
    y_all.x,
    label="All runs",
    color="k",
    linewidth=1.2
)

for t in [118, 244, 359, 492]:
    ax.axvline(
        t,
        color="gray",
        linewidth=0.5
    )

ax.set_title("NCRF-DC contrast across trial sizes")
ax.set_xlabel("Time [ms]")
ax.set_ylabel("Maximum difference")

ax.legend(frameon=False)

fig.tight_layout()

fig.savefig(
    "figures/ncrf_dc_trialsize_onesession_allruns.pdf",
    bbox_inches="tight",
    transparent=True
)

plt.show()
"""

# %% [markdown]
# ## CONTRAST 
# ### MEan of all sources

# %%
sizes_to_plot = [0.25, 0.5, 1, 2]

method_name = "NCRF-DC"
model_dir = dirs[method_name]

all_curves = {}

#reduced sizes : less than one session

for size in sizes_to_plot:

    print(f"\nProcessing size={size}")

    cases = []

    for i in range(1, 31):
        subject = f"sub-{i:02d}"

        try:
            inan1, anim1 = load_model_subset(
                model_dir=model_dir,
                m=1,
                subject=subject,
                size=size
            )

            cases.append([subject, "inanimate", inan1])
            cases.append([subject, "animate", anim1])

        except Exception as e:
            print(f"Skipping {subject}: {e}")
            continue

    data_size = Dataset.from_caselist(
        ["subject", "animacy", "ncrf"],
        cases
    )

    res_size = testnd.VectorDifferenceRelated(
        "ncrf",
        "animacy",
        "inanimate",
        "animate",
        match="subject",
        data=data_size,
        tfce=True,
        tstart=100,
        tstop=700,
        samples=1000
    )

    diff_size = res_size.masked_difference()

    # Mean across sources instead of max
    y_size = diff_size.norm("space").mean("source")

    all_curves[size] = y_size


#one session

cases = []

one_session_dir = "models/all_runs/morphed"

for i in range(1, 31):

    subject = f"sub-{i:02d}"

    if i > 9:
        session = "ImageNet01"
    else:
        session = "ImageNet03"

    morphed_file = f"{one_session_dir}/M{subject}-{session}-ncrf.pickle"

    try:
        inanim, anim = load.unpickle(morphed_file)

        cases.append([subject, "inanimate", inanim])
        cases.append([subject, "animate", anim])

    except Exception as e:
        print(f"Skipping one-session {subject}-{session}: {e}")
        continue

data_onesession = Dataset.from_caselist(
    ["subject", "animacy", "ncrf"],
    cases
)

res_onesession = testnd.VectorDifferenceRelated(
    "ncrf",
    "animacy",
    "inanimate",
    "animate",
    match="subject",
    data=data_onesession,
    tfce=True,
    tstart=100,
    tstop=700,
    samples=1000
)

diff_onesession = res_onesession.masked_difference()

# Mean across sources instead of max
y_onesession = diff_onesession.norm("space").mean("source")


#all runs

cases = []

all_runs_dir = "models/all_runs/morphed"

for i in range(10, 31):

    if i > 9:
        sessions = ["ImageNet01"]
    else:
        sessions = ["ImageNet03", "ImageNet04"]

    subject = f"sub-{i:02d}"

    for session in sessions:

        morphed_file = f"{all_runs_dir}/M{subject}-{session}-ncrf.pickle"

        try:
            inanim, anim = load.unpickle(morphed_file)

            cases.append([subject, session, "inanimate", inanim])
            cases.append([subject, session, "animate", anim])

        except Exception as e:
            print(f"Skipping all-runs {subject}-{session}: {e}")
            continue

data_all = Dataset.from_caselist(
    ["subject", "session", "animacy", "ncrf"],
    cases
)

data_all_avg = data_all.aggregate(
    "animacy % subject",
    drop_bad=True
)

res_all = testnd.VectorDifferenceRelated(
    "ncrf",
    "animacy",
    "inanimate",
    "animate",
    match="subject",
    data=data_all_avg,
    tfce=True,
    tstart=100,
    tstop=700,
    samples=1000
)

diff_all = res_all.masked_difference()

# Mean across sources instead of max
y_all = diff_all.norm("space").mean("source")



# plot all

fig, ax = plt.subplots(figsize=(7, 4))

colors = {
    0.25: "tab:blue",
    0.5: "#d8b38a",
    1: "tab:green",
    2: "tab:red",
}

for size, y in all_curves.items():

    n_trials = int(size * trials_per_subset)

    ax.plot(
        y.time.times,
        y.x,
        label=f"{n_trials} trials",
        color=colors[size]
    )

ax.plot(
    y_onesession.time.times,
    y_onesession.x,
    label="1000 trials",
    color="tab:purple",
    linewidth=1.0
)

ax.plot(
    y_all.time.times,
    y_all.x,
    label="All runs",
    color="k",
    linewidth=1.2
)

for t in [118, 244, 359, 492]:
    ax.axvline(
        t,
        color="gray",
        linewidth=0.5
    )

ax.set_title("NCRF-DC contrast across trial sizes")
ax.set_xlabel("Time [ms]")
ax.set_ylabel("Mean difference")

ax.legend(frameon=False)

fig.tight_layout()

fig.savefig(
    "figures/ncrf_dc_trialsize_onesession_allruns_mean.pdf",
    bbox_inches="tight",
    transparent=True
)

plt.show()

# %% [markdown]
# ## CONDITION : INANIMATE 
# ### MEAN of all sources
#
#
#

# %%
sizes_to_plot = [0.25, 0.5, 1, 2]

method_name = "NCRF-DC"
model_dir = dirs[method_name]

all_curves = {}

#same sizes

for size in sizes_to_plot:

    print(f"\nProcessing size={size}")

    cases = []

    for i in range(1, 31):
        subject = f"sub-{i:02d}"

        try:
            inan1, anim1 = load_model_subset(
                model_dir=model_dir,
                m=1,
                subject=subject,
                size=size
            )

            # Only inanimate
            cases.append([subject, "inanimate", inan1])

        except Exception as e:
            print(f"Skipping {subject}: {e}")
            continue

    data_size = Dataset.from_caselist(
        ["subject", "animacy", "ncrf"],
        cases
    )

    data_inan = data_size.sub("animacy == 'inanimate'")

    res_size = testnd.Vector(
        "ncrf",
        match="subject",
        data=data_inan,
        tfce=True,
        tstart=1,
        tstop=600,
        samples=1000
    )

    diff_size = res_size.masked_difference()
    y_size = diff_size.norm("space").mean("source") #max

    all_curves[size] = y_size


# 1 session= 1000 trails

cases = []

one_session_dir = "models/all_runs/morphed"

for i in range(1, 31):

    subject = f"sub-{i:02d}"

    if i > 9:
        session = "ImageNet01"
    else:
        session = "ImageNet03"

    morphed_file = f"{one_session_dir}/M{subject}-{session}-ncrf.pickle"

    try:
        inanim, anim = load.unpickle(morphed_file)

        # Only inanimate
        cases.append([subject, "inanimate", inanim])

    except Exception as e:
        print(f"Skipping one-session {subject}-{session}: {e}")
        continue

data_onesession = Dataset.from_caselist(
    ["subject", "animacy", "ncrf"],
    cases
)

data_onesession_inan = data_onesession.sub("animacy == 'inanimate'")

res_onesession = testnd.Vector(
    "ncrf",
    match="subject",
    data=data_onesession_inan,
    tfce=True,
    tstart=1,
    tstop=600,
    samples=1000
)

diff_onesession = res_onesession.masked_difference()
y_onesession = diff_onesession.norm("space").mean("source")   #max


# all runs
cases = []

all_runs_dir = "models/all_runs/morphed"

for i in range(1, 31):

    if i > 9:
        sessions = ["ImageNet01"]
    else:
        sessions = ["ImageNet03", "ImageNet04"]

    subject = f"sub-{i:02d}"

    for session in sessions:

        morphed_file = f"{all_runs_dir}/M{subject}-{session}-ncrf.pickle"

        try:
            inanim, anim = load.unpickle(morphed_file)

            # Only inanimate
            cases.append([subject, session, "inanimate", inanim])

        except Exception as e:
            print(f"Skipping all-runs {subject}-{session}: {e}")
            continue

data_all = Dataset.from_caselist(
    ["subject", "session", "animacy", "ncrf"],
    cases
)

data_all_avg = data_all.aggregate(
    "animacy % subject",
    drop_bad=True
)

data_all_inan = data_all_avg.sub("animacy == 'inanimate'")

res_all = testnd.Vector(
    "ncrf",
    match="subject",
    data=data_all_inan,
    tfce=True,
    tstart=1,
    tstop=600,
    samples=1000
)

diff_all = res_all.masked_difference()
y_all = diff_all.norm("space").mean("source")   #max


#plot

fig, ax = plt.subplots(figsize=(7, 4))

colors = {
    0.25: "tab:blue",
    0.5: "tab:orange",
    1: "tab:green",
    2: "tab:red",
}

for size, y in all_curves.items():

    n_trials = int(size * trials_per_subset)

    ax.plot(
        y.time.times,
        y.x,
        label=f"{n_trials} trials",
        color=colors[size]
    )

ax.plot(
    y_onesession.time.times,
    y_onesession.x,
    label="1000 trials",
    color="tab:purple",
    linewidth=1.0
)

ax.plot(
    y_all.time.times,
    y_all.x,
    label="All runs",
    color="k",
    linewidth=1.2
)

for t in [130, 240]:
    ax.axvline(
        t,
        color="gray",
        linewidth=0.5
    )

ax.set_title("NCRF-DC inanimate response across trial sizes")
ax.set_xlabel("Time [ms]")
ax.set_ylabel("Mean NCRF")

ax.legend(frameon=False)

fig.tight_layout()

fig.savefig(
    "figures/inanim_BF_mean.pdf",
    bbox_inches="tight",
    transparent=True
)

plt.show()

# %% [markdown]
# ## ROI VISUAL

# %%
root = Path("~/Data/ds005810")
subjects_dir = str(Path('~/Data/ds005810/derivatives/freesurfer/subjects').expanduser())

# %%
#  Choose an NDVar
diff = res.masked_difference()

# This has source + time dimensions
d = diff.norm("space")

# Get source coordinates 
coords = d.source.coordinates

# Eelbrain VolumeSourceSpace usually has coordinates in meters : Convert to mm if needed
#FreeSurfer atlases use millimeters
if np.nanmax(np.abs(coords)) < 10:
    coords_mm = coords * 1000
else:
    coords_mm = coords


# Load FreeSurfer aparc+aseg atlas

atlas_file = os.path.join(
    subjects_dir,
    "fsaverage2",
    "mri",
    "aparc+aseg.mgz"
)

atlas = nib.load(atlas_file)
atlas_data = atlas.get_fdata()

# Convert source RAS coordinates to voxel coordinates
#The affine matrix converts voxel->RAS so inverse affine converts RAS->voxel
ras_to_vox = np.linalg.inv(atlas.affine)

coords_h = np.c_[coords_mm, np.ones(len(coords_mm))]   # add 4th cordinate :[x y z 1] needed affine transforms use 4*4 matrices
vox = coords_h @ ras_to_vox.T
vox = np.round(vox[:, :3]).astype(int)       #Rounds them to nearest voxel.

# Keep only valid voxel indices :Checks whether voxel coordinates are inside the MRI volume.
valid = (
    (vox[:, 0] >= 0) & (vox[:, 0] < atlas_data.shape[0]) &
    (vox[:, 1] >= 0) & (vox[:, 1] < atlas_data.shape[1]) &
    (vox[:, 2] >= 0) & (vox[:, 2] < atlas_data.shape[2])
)

source_labels = np.full(len(coords_mm), np.nan)  #Creates an array for storing labels initially all Nan

# 1.find corresponding MRI voxel  2. read atlas label 3. assign label to the source index
#Example : source 37 → voxel [81,92,77]
#            atlas_data[81,92,77] = 1007
#         so src 37 is in fusiform
source_labels[valid] = atlas_data[
    vox[valid, 0],
    vox[valid, 1],
    vox[valid, 2]
]


# how many source in each label id : (Remove Nan first befor counting how many source in eah label
unique_labels, counts = np.unique(
    source_labels[~np.isnan(source_labels)],
    return_counts=True
)
# 0 : unlabeled , 
# 2/41 : Left/Right cerebrl white matter
# 77 : (WM hypointensities) MRI artifact : uncertain tissue in MRI (non-meaningful)
for label_id, count in zip(unique_labels.astype(int), counts):
    print(label_id, count)

# %%
bad_label_ids = [0, 2, 41, 77]

good_labels = (
    ~np.isnan(source_labels) &
    ~np.isin(source_labels, bad_label_ids)
)

print("Total sources:", len(source_labels))
print("Good cortical labels:", np.sum(good_labels))
print("Bad/unlabeled:", np.sum(~good_labels))

# %%
"""
visual_label_ids = [
    1005, 2005,  # cuneus (left 1005, right 2005)
    1011, 2011,  # lateral occipital
    1013, 2013,  # lingual                      Brodman : 17,18,19
    1021, 2021,  # pericalcarine (V1 region)
    1007, 2007,  # fusiform
]

visual_mask = np.isin(source_labels, visual_label_ids)
visual_source_idx = np.where(visual_mask)[0]

print("Number of visual ROI sources:", len(visual_source_idx))
print("Visual source indices:", visual_source_idx)

d_visual = diff.sub(source=visual_source_idx)
y_visual = d_visual.norm("space").mean("source")
"""

# %%

"""
fig, ax = plt.subplots(figsize=(6.5, 3.5))

ax.plot(
    y_visual.time.times,
    y_visual.x,
    color="k"
)

for t in [118, 244, 359, 492]:
    ax.axvline(t, color="gray", linewidth=0.5)

ax.set_title("Visual ROI mean")
ax.set_xlabel("Time [ms]")
ax.set_ylabel("Mean NCRF in visual ROI")

fig.tight_layout()

fig.savefig(
    "figures/visual_roi_mean.pdf",
    bbox_inches="tight",
    transparent=True
)

plt.show()
"""

# %%

sizes_to_plot = [0.25, 0.5, 1, 2]

method_name = "NCRF-DC"
model_dir = dirs[method_name]

all_curves = {}

#define ROI
# V1 : Pericalcarine : broadman area 17 (Striarte cortex)
#Extrastriate cortex :
#  V2 
#  V4 : lingual 
#  V4a :fusiform
#  V5: middle temporal visual area (MT)  : motion perception _ ? not related to our experiment

visual_label_ids = [
    1005, 2005,  # cuneus : V2/V3 dorsal
   # 1011, 2011,  # lateral occipital
    1013, 2013,  # lingual : v2/V3/V4 ventral
    1021, 2021,  # pericalcarine   : V1
    #1007, 2007,  # fusiform
    #1009,2009,  # IT : inferior temporal
]

visual_mask = np.isin(source_labels, visual_label_ids)
visual_source_idx = np.where(visual_mask)[0]

print("Number of visual ROI sources:", len(visual_source_idx))
print("Visual source indices:", visual_source_idx)


#less than one session

for size in sizes_to_plot:

    print(f"\nProcessing size={size}")

    cases = []

    for i in range(1, 31):
        subject = f"sub-{i:02d}"

        try:
            inan1, anim1 = load_model_subset(
                model_dir=model_dir,
                m=1,
                subject=subject,
                size=size
            )

            cases.append([subject, "inanimate", inan1])
            cases.append([subject, "animate", anim1])

        except Exception as e:
            print(f"Skipping {subject}: {e}")
            continue

    data_size = Dataset.from_caselist(
        ["subject", "animacy", "ncrf"],
        cases
    )

    res_size = testnd.VectorDifferenceRelated(
        "ncrf",
        "animacy",
        "inanimate",
        "animate",
        match="subject",
        data=data_size,
        tfce=True,
        tstart=100,
        tstop=700,
        samples=1000
    )

    diff_size = res_size.masked_difference()

    # Visual ROI mean
    d_visual_size = diff_size.sub(source=visual_source_idx)
    y_size = d_visual_size.norm("space").mean("source")

    all_curves[size] = y_size


#one sessoion

cases = []

one_session_dir = "models/all_runs/morphed"

for i in range(1, 31):

    subject = f"sub-{i:02d}"

    if i > 9:
        session = "ImageNet01"
    else:
        session = "ImageNet03"

    morphed_file = f"{one_session_dir}/M{subject}-{session}-ncrf.pickle"

    try:
        inanim, anim = load.unpickle(morphed_file)

        cases.append([subject, "inanimate", inanim])
        cases.append([subject, "animate", anim])

    except Exception as e:
        print(f"Skipping one-session {subject}-{session}: {e}")
        continue

data_onesession = Dataset.from_caselist(
    ["subject", "animacy", "ncrf"],
    cases
)

res_onesession = testnd.VectorDifferenceRelated(
    "ncrf",
    "animacy",
    "inanimate",
    "animate",
    match="subject",
    data=data_onesession,
    tfce=True,
    tstart=100,
    tstop=700,
    samples=1000
)

diff_onesession = res_onesession.masked_difference()

# Visual ROI mean
d_visual_onesession = diff_onesession.sub(source=visual_source_idx)
y_onesession = d_visual_onesession.norm("space").mean("source")


# all runs

cases = []

all_runs_dir = "models/all_runs/morphed"

for i in range(1, 31):

    if i > 9:
        sessions = ["ImageNet01"]
    else:
        sessions = ["ImageNet03", "ImageNet04"]

    subject = f"sub-{i:02d}"

    for session in sessions:

        morphed_file = f"{all_runs_dir}/M{subject}-{session}-ncrf.pickle"

        try:
            inanim, anim = load.unpickle(morphed_file)

            cases.append([subject, session, "inanimate", inanim])
            cases.append([subject, session, "animate", anim])

        except Exception as e:
            print(f"Skipping all-runs {subject}-{session}: {e}")
            continue

data_all = Dataset.from_caselist(
    ["subject", "session", "animacy", "ncrf"],
    cases
)

data_all_avg = data_all.aggregate(
    "animacy % subject",
    drop_bad=True
)

res_all = testnd.VectorDifferenceRelated(
    "ncrf",
    "animacy",
    "inanimate",
    "animate",
    match="subject",
    data=data_all_avg,
    tfce=True,
    tstart=100,
    tstop=700,
    samples=1000
)

diff_all = res_all.masked_difference()

# Visual ROI mean
d_visual_all = diff_all.sub(source=visual_source_idx)
y_all = d_visual_all.norm("space").mean("source")


#plot all

fig, ax = plt.subplots(figsize=(7, 4))

colors = {
    0.25: "tab:blue",
    0.5: "#e69f00",
    1: "tab:green",
    2: "tab:red",
}

for size, y in all_curves.items():

    n_trials = int(size * trials_per_subset)

    ax.plot(
        y.time.times,
        y.x,
        label=f"{n_trials} trials",
        color=colors[size],
        linewidth=1.0
    )

ax.plot(
    y_onesession.time.times,
    y_onesession.x,
    label="1000 trials",
    color="tab:purple",
    linewidth=1.0
)

ax.plot(
    y_all.time.times,
    y_all.x,
    label="All runs",
    color="k",
    linewidth=1.2
)

for t in [118, 244, 359, 492]:
    ax.axvline(
        t,
        color="gray",
        linewidth=0.5
    )

ax.set_title("Contrast ROI : Pericalcarine-Cuneuse-Lingual")
ax.set_xlabel("Time [ms]")
ax.set_ylabel("Mean difference")

ax.legend(frameon=False)

fig.tight_layout()

fig.savefig(
    "figures/ROI/visual.pdf",
    bbox_inches="tight",
    transparent=True
)

plt.show()

# %% [markdown]
# # NCRF_EC : one sample test on contrast

# %%
"""
sizes_to_plot = [0.25, 0.5, 1, 2]

method_name = "NCRF-EC"
model_dir = dirs[method_name]

all_curves = {}

# =========================
# Visual ROI
# =========================

visual_label_ids = [
    1005, 2005,  # cuneus
    1011, 2011,  # lateral occipital
    1013, 2013,  # lingual
    1021, 2021,  # pericalcarine
    1007, 2007,  # fusiform
    1009, 2009,  # inferior temporal
]

visual_mask = np.isin(source_labels, visual_label_ids)
visual_source_idx = np.where(visual_mask)[0]

print("Number of visual ROI sources:", len(visual_source_idx))


# =========================
# Same-size subset models
# =========================

for size in sizes_to_plot:

    print(f"\nProcessing size={size}")

    cases = []

    for i in range(1, 31):
        subject = f"sub-{i:02d}"

        try:
            inan1, anim1 = load_model_subset(
                model_dir=model_dir,
                m=1,
                subject=subject,
                size=size
            )

            # For NCRF-EC, contrast is saved in anim1
            contrast = anim1

            cases.append([subject, "contrast", contrast])

        except Exception as e:
            print(f"Skipping {subject}: {e}")
            continue

    data_size = Dataset.from_caselist(
        ["subject", "animacy", "ncrf"],
        cases
    )

    res_size = testnd.Vector(
        "ncrf",
        match="subject",
        data=data_size,
        tfce=True,
        tstart=100,
        tstop=700,
        samples=1000
    )

    diff_size = res_size.masked_difference()

    d_visual_size = diff_size.sub(source=visual_source_idx)
    y_size = d_visual_size.norm("space").mean("source")

    all_curves[size] = y_size


# =========================
# One full session
# =========================

cases = []

one_session_dir = "models/all_runs/effect/morphed"

for i in range(1, 31):

    subject = f"sub-{i:02d}"

    if i > 9:
        session = "ImageNet01"
    else:
        session = "ImageNet03"

    morphed_file = f"{one_session_dir}/M{subject}-{session}-ncrf.pickle"

    try:
        inanim, anim = load.unpickle(morphed_file)

        # For NCRF-EC, contrast is saved in anim
        cases.append([subject, "contrast", anim])

    except Exception as e:
        print(f"Skipping one-session {subject}-{session}: {e}")
        continue

data_onesession = Dataset.from_caselist(
    ["subject", "animacy", "ncrf"],
    cases
)

res_onesession = testnd.Vector(
    "ncrf",
    match="subject",
    data=data_onesession,
    tfce=True,
    tstart=100,
    tstop=700,
    samples=1000
)

diff_onesession = res_onesession.masked_difference()

d_visual_onesession = diff_onesession.sub(source=visual_source_idx)
y_onesession = d_visual_onesession.norm("space").mean("source")


# =========================
# All runs
# =========================

cases = []

all_runs_dir = "models/all_runs/effect/morphed"

for i in range(1, 31):

    if i > 9:
        sessions = ["ImageNet01"]
    else:
        sessions = ["ImageNet03", "ImageNet04"]

    subject = f"sub-{i:02d}"

    for session in sessions:

        morphed_file = f"{all_runs_dir}/M{subject}-{session}-ncrf.pickle"

        try:
            inanim, anim = load.unpickle(morphed_file)

            # For NCRF-EC, contrast is saved in anim
            cases.append([subject, session, "contrast", anim])

        except Exception as e:
            print(f"Skipping all-runs {subject}-{session}: {e}")
            continue

data_all = Dataset.from_caselist(
    ["subject", "session", "animacy", "ncrf"],
    cases
)

data_all_avg = data_all.aggregate(
    "animacy % subject",
    drop_bad=True
)

res_all = testnd.Vector(
    "ncrf",
    match="subject",
    data=data_all_avg,
    tfce=True,
    tstart=100,
    tstop=700,
    samples=1000
)

diff_all = res_all.masked_difference()

d_visual_all = diff_all.sub(source=visual_source_idx)
y_all = d_visual_all.norm("space").mean("source")


# =========================
# Plot
# =========================

fig, ax = plt.subplots(figsize=(7, 4))

colors = {
    0.25: "tab:blue",
    0.5: "#e69f00",
    1: "tab:green",
    2: "tab:red",
}

for size, y in all_curves.items():

    n_trials = int(size * trials_per_subset)

    ax.plot(
        y.time.times,
        y.x,
        label=f"{n_trials} trials",
        color=colors[size],
        linewidth=1.0
    )

ax.plot(
    y_onesession.time.times,
    y_onesession.x,
    label="1000 trials",
    color="tab:purple",
    linewidth=1.0
)

ax.plot(
    y_all.time.times,
    y_all.x,
    label="All runs",
    color="k",
    linewidth=1.2
)

for t in [118, 244, 359, 492]:
    ax.axvline(
        t,
        color="gray",
        linewidth=0.5
    )

ax.set_title("NCRF-EC visual ROI contrast across trial sizes")
ax.set_xlabel("Time [ms]")
ax.set_ylabel("Mean contrast in visual ROI")

ax.legend(frameon=False)

fig.tight_layout()

fig.savefig(
    "figures/ROI/visual_ncrf_ec.pdf",
    bbox_inches="tight",
    transparent=True
)

plt.show()
"""

# %% [markdown]
# ### Which labeled source is closest to this unlabeled source?
#
# instead of checking distances one-by-one manually use k-nearest neighberhood by cKDTree algorithm from scipy

# %%
# Coordinates of good and bad sources
bad_label_ids = [0, 2, 41, 77]

good_mask = (
    ~np.isnan(source_labels) &
    ~np.isin(source_labels, bad_label_ids)
)

bad_mask = ~good_mask
#-------------------------------
good_coords = coords_mm[good_mask]
bad_coords = coords_mm[bad_mask]

good_labels = source_labels[good_mask]

# Build nearest-neighbor tree from good labeled sources
tree = cKDTree(good_coords)

# For each bad source, find nearest good source
distances, nearest_idx = tree.query(bad_coords, k=1)

# Threshold in mm
max_dist_mm = 10

reassigned_labels = source_labels.copy()

bad_indices = np.where(bad_mask)[0]

for i, src_idx in enumerate(bad_indices):
    if distances[i] <= max_dist_mm:
        reassigned_labels[src_idx] = good_labels[nearest_idx[i]]
    else:
        reassigned_labels[src_idx] = np.nan


new_labels, new_counts = np.unique(
    reassigned_labels[~np.isnan(reassigned_labels)],
    return_counts=True
)

for label_id, count in zip(new_labels.astype(int), new_counts):
    print(label_id, count)        

# %%
# OLD counts from original source_labels
old_labels, old_counts = np.unique(
    source_labels[~np.isnan(source_labels)],
    return_counts=True
)

old_dict = dict(zip(old_labels.astype(int), old_counts))


# NEW counts from reassigned_labels
new_labels, new_counts = np.unique(
    reassigned_labels[~np.isnan(reassigned_labels)],
    return_counts=True
)

new_dict = dict(zip(new_labels.astype(int), new_counts))


# Compare
all_labels = sorted(set(old_dict.keys()) | set(new_dict.keys()))

print("Label | Old | New | Difference")
print("--------------------------------")

for label in all_labels:
    old_n = old_dict.get(label, 0)
    new_n = new_dict.get(label, 0)
    diff = new_n - old_n

    print(f"{label:4d} | {old_n:3d} | {new_n:3d} | {diff:+d}")

# %%
old_valid = np.sum(
    ~np.isnan(source_labels) &
    ~np.isin(source_labels, bad_label_ids)
)


new_valid = np.sum(~np.isnan(reassigned_labels)) #valid


newly_labeled = new_valid - old_valid

print("Previously labeled cortical sources:", old_valid)
print("After reassignment:", new_valid)
print("Newly labeled sources:", newly_labeled)

# %%
# Find Threshold
print("Min distance:", distances.min())
print("Max distance:", distances.max())
print("Mean distance:", distances.mean())

for t in [3, 5, 7, 10]:
    n_pass = np.sum(distances <= t)
    print(f"Threshold {t} mm: {n_pass} / {len(distances)} reassigned")

# %%
print("Bad before:", np.sum(bad_mask))
print("Reassigned:", np.sum(distances <= max_dist_mm))
print("Still NaN after:", np.sum(np.isnan(reassigned_labels)))

# %%
#check ROI's sources
visual_label_ids = [
    1005, 2005,
    1011, 2011,
    1013, 2013,
    1021, 2021,
    1007, 2007,
    1009, 2009,
]

print("Label | Old | New | Difference")
print("--------------------------------")

for label in visual_label_ids:
    old_n = old_dict.get(label, 0)
    new_n = new_dict.get(label, 0)
    diff = new_n - old_n
    print(f"{label:4d} | {old_n:3d} | {new_n:3d} | {diff:+d}")

# %%



src = d.source.get_source_space()

#bad_label_ids = [0, 2, 41, 77]        #unlabeled, LH white, RH white, MRI artifact
bad_label_ids = [0]

bad_mask = (
    np.isnan(source_labels) |
    np.isin(source_labels, bad_label_ids)
)

bad_src = copy.deepcopy(src)

# IMPORTANT: map bad_mask onto the real MNE vertex numbers
all_used_vertices = src[0]["vertno"]          # length 1751
bad_vertices = all_used_vertices[bad_mask]    # actual MNE vertex IDs

bad_src[0]["vertno"] = bad_vertices
bad_src[0]["nuse"] = len(bad_vertices)

bad_src[0]["inuse"][:] = 0
bad_src[0]["inuse"][bad_vertices] = 1

print("Bad sources:", len(bad_vertices))

fig = mne.viz.plot_bem(
    subject=subject,
    subjects_dir=subjects_dir,
    brain_surfaces="white",
    orientation="sagittal",
    slices=[70, 80, 90, 100, 110],
    src=bad_src,
)

plt.show()

# %%
# create folder and save plot
out_dir = "figures/bad"
os.makedirs(out_dir, exist_ok=True)

out_file = os.path.join(out_dir, "unlabeled_sources_sagittal.pdf")

fig.savefig(out_file, bbox_inches="tight")

print("Saved:", out_file)


# %% [markdown]
# # Find sources  to add to left cuneus Based on MRI VOXELS
#

# %%
target_label = 1005   # left cuneus

# all atlas voxels labeled 1005
target_vox = np.array(np.where(atlas_data == target_label)).T
print("Atlas voxels with label 1005:", len(target_vox))

# Convert atlas voxel coordinates to RAS mm coordinates
target_vox_h = np.c_[target_vox, np.ones(len(target_vox))]
target_ras = target_vox_h @ atlas.affine.T
target_ras = target_ras[:, :3]

# Choose only unlabeled sources
unlabeled_mask = source_labels == 0
unlabeled_coords = coords_mm[unlabeled_mask]
unlabeled_indices = np.where(unlabeled_mask)[0]
print("Unlabeled sources:", len(unlabeled_coords))

#Build KDTree from atlas cuneus voxels
tree_1005 = cKDTree(target_ras)
dist_1005, nearest_1005_idx = tree_1005.query(unlabeled_coords, k=1)

threshold_mm = 7.0

candidate_mask = dist_1005 <= threshold_mm
candidate_source_indices = unlabeled_indices[candidate_mask]
candidate_distances = dist_1005[candidate_mask]

print(f"Unlabeled sources within {threshold_mm} mm of label 1005:", len(candidate_source_indices))

for src_idx, dist in zip(candidate_source_indices, candidate_distances):
    print(f"source {src_idx}: distance to 1005 = {dist:.2f} mm")

# %%
#256^3=~16.8 million MRI voxelsvbut we have 1751 voxel in source
print(atlas_data.shape)

# %%
# Make a copy so you don't overwrite the original labels accidentally
source_labels_fixed = source_labels.copy()

# Re-assign candidate unlabeled sources to left cuneus
source_labels_fixed[candidate_source_indices] = target_label

print("Reassigned sources:", len(candidate_source_indices))
print("New number of sources with label 1005:",
      np.sum(source_labels_fixed == target_label))


# %%
total_reassigned = 0

visual_label_ids = [
    1005, 2005,  # cuneus
    1011, 2011,  # lateral occipital
    1013, 2013,  # lingual
    1021, 2021,  # pericalcarine
    1007, 2007,  # fusiform
    1009, 2009,  # inferior temporal
]

threshold_mm = 7.0


source_labels_fixed = source_labels.copy()


unlabeled_mask = source_labels == 0
unlabeled_coords = coords_mm[unlabeled_mask]
unlabeled_indices = np.where(unlabeled_mask)[0]

print("Initial unlabeled sources:", len(unlabeled_indices))

for target_label in visual_label_ids:

    print(f"\nProcessing label {target_label}")

    # Atlas voxels belonging to this ROI
    target_vox = np.array(np.where(atlas_data == target_label)).T

    if len(target_vox) == 0:
        print("No atlas voxels found.")
        continue

    # voxel -> RAS mm
    target_vox_h = np.c_[target_vox, np.ones(len(target_vox))]
    target_ras = target_vox_h @ atlas.affine.T
    target_ras = target_ras[:, :3]

    
    tree = cKDTree(target_ras)
    dist, _ = tree.query(unlabeled_coords, k=1)

    candidate_mask = dist <= threshold_mm

    candidate_source_indices = unlabeled_indices[candidate_mask]
    n_found = len(candidate_source_indices)
    total_reassigned += n_found

    print(
        f"Found {n_found} sources "
        f"within {threshold_mm} mm"
    )

    # assign label
    source_labels_fixed[candidate_source_indices] = target_label

print("Total reassigned sources:", total_reassigned)

# %%
#STill UnlaBeled after fixing plot 

# %%

# Plot unlabeled sources AFTER fixing/re-assigning

src = d.source.get_source_space()

bad_label_ids = [0]

bad_mask_after = (
    np.isnan(source_labels_fixed) |
    np.isin(source_labels_fixed, bad_label_ids)
)

bad_src_after = copy.deepcopy(src)

# Map mask onto real MNE vertex numbers
all_used_vertices = src[0]["vertno"]
bad_vertices_after = all_used_vertices[bad_mask_after]

bad_src_after[0]["vertno"] = bad_vertices_after
bad_src_after[0]["nuse"] = len(bad_vertices_after)

bad_src_after[0]["inuse"][:] = 0
bad_src_after[0]["inuse"][bad_vertices_after] = 1

print("Unlabeled sources after fixing:", len(bad_vertices_after))
print("Unlabeled sources before fixing:", np.sum(source_labels == 0))
print("Reassigned unlabeled sources:", np.sum((source_labels == 0) & (source_labels_fixed != 0)))

fig = mne.viz.plot_bem(
    subject=subject,
    subjects_dir=subjects_dir,
    brain_surfaces="white",
    orientation="sagittal",
    slices=[70, 80, 90, 100, 110],
    src=bad_src_after,
)

plt.show()


out_dir = "figures/bad"
os.makedirs(out_dir, exist_ok=True)

out_file = os.path.join(out_dir, "After_fixing.pdf")

fig.savefig(out_file, bbox_inches="tight")

print("Saved:", out_file)

# %% [markdown]
# # plot ROI AFTER fixing

# %%

sizes_to_plot = [0.25, 0.5, 1, 2]

trials_per_subset = 200

method_name = "NCRF-DC"
model_dir = dirs[method_name]

all_curves = {}


visual_label_ids = [
    1005, 2005,  # cuneus
    #1011, 2011,  # lateral occipital
    1013, 2013,  # lingual
    1021, 2021,  # pericalcarine / V1
    #1007, 2007,  # fusiform
    #1009, 2009,  # inferior temporal
]

visual_mask_fixed = np.isin(source_labels_fixed, visual_label_ids)
visual_source_idx_fixed = np.where(visual_mask_fixed)[0]

n_before = np.sum(np.isin(source_labels, visual_label_ids))
n_after = len(visual_source_idx_fixed)

print("Number of visual ROI sources before fixing:", n_before)
print("Number of visual ROI sources after fixing:", n_after)
print("Newly added ROI sources:", n_after - n_before)
print("Visual source indices after fixing:")
print(visual_source_idx_fixed)

# Safety check
if len(visual_source_idx_fixed) == 0:
    raise RuntimeError("No visual ROI sources found after fixing labels.")


# less than 1 session

for size in sizes_to_plot:

    print(f"\nProcessing size={size}")

    cases = []

    for i in range(1, 31):

        subject = f"sub-{i:02d}"

        try:
            inan1, anim1 = load_model_subset(
                model_dir=model_dir,
                m=1,
                subject=subject,
                size=size
            )

            cases.append([subject, "inanimate", inan1])
            cases.append([subject, "animate", anim1])

        except Exception as e:
            print(f"Skipping {subject}: {e}")
            continue

    data_size = Dataset.from_caselist(
        ["subject", "animacy", "ncrf"],
        cases
    )

    res_size = testnd.VectorDifferenceRelated(
        "ncrf",
        "animacy",
        "inanimate",
        "animate",
        match="subject",
        data=data_size,
        tfce=True,
        tstart=100,
        tstop=700,
        samples=1000
    )

    diff_size = res_size.masked_difference()

    # ROI mean using fixed labels
    d_visual_size = diff_size.sub(source=visual_source_idx_fixed)
    y_size = d_visual_size.norm("space").mean("source")

    all_curves[size] = y_size


#1 session

cases = []

one_session_dir = "models/all_runs/morphed"

for i in range(1, 31):

    subject = f"sub-{i:02d}"

    if i > 9:
        session = "ImageNet01"
    else:
        session = "ImageNet03"

    morphed_file = f"{one_session_dir}/M{subject}-{session}-ncrf.pickle"

    try:
        inanim, anim = load.unpickle(morphed_file)

        cases.append([subject, "inanimate", inanim])
        cases.append([subject, "animate", anim])

    except Exception as e:
        print(f"Skipping one-session {subject}-{session}: {e}")
        continue

data_onesession = Dataset.from_caselist(
    ["subject", "animacy", "ncrf"],
    cases
)

res_onesession = testnd.VectorDifferenceRelated(
    "ncrf",
    "animacy",
    "inanimate",
    "animate",
    match="subject",
    data=data_onesession,
    tfce=True,
    tstart=100,
    tstop=700,
    samples=1000
)

diff_onesession = res_onesession.masked_difference()

d_visual_onesession = diff_onesession.sub(source=visual_source_idx_fixed)
y_onesession = d_visual_onesession.norm("space").mean("source")


#all runs

cases = []

all_runs_dir = "models/all_runs/morphed"

for i in range(1, 31):

    subject = f"sub-{i:02d}"

    if i > 9:
        sessions = ["ImageNet01"]
    else:
        sessions = ["ImageNet03", "ImageNet04"]

    for session in sessions:

        morphed_file = f"{all_runs_dir}/M{subject}-{session}-ncrf.pickle"

        try:
            inanim, anim = load.unpickle(morphed_file)

            cases.append([subject, session, "inanimate", inanim])
            cases.append([subject, session, "animate", anim])

        except Exception as e:
            print(f"Skipping all-runs {subject}-{session}: {e}")
            continue

data_all = Dataset.from_caselist(
    ["subject", "session", "animacy", "ncrf"],
    cases
)

data_all_avg = data_all.aggregate(
    "animacy % subject",
    drop_bad=True
)

res_all = testnd.VectorDifferenceRelated(
    "ncrf",
    "animacy",
    "inanimate",
    "animate",
    match="subject",
    data=data_all_avg,
    tfce=True,
    tstart=100,
    tstop=700,
    samples=1000
)

diff_all = res_all.masked_difference()

d_visual_all = diff_all.sub(source=visual_source_idx_fixed)
y_all = d_visual_all.norm("space").mean("source")


#plot

fig, ax = plt.subplots(figsize=(7, 4))

colors = {
    0.25: "tab:blue",
    0.5: "#e69f00",
    1: "tab:green",
    2: "tab:red",
}

for size, y in all_curves.items():

    n_trials = int(size * trials_per_subset)

    ax.plot(
        y.time.times,
        y.x,
        label=f"{n_trials} trials",
        color=colors[size],
        linewidth=1.0
    )

ax.plot(
    y_onesession.time.times,
    y_onesession.x,
    label="1000 trials",
    color="tab:purple",
    linewidth=1.0
)

ax.plot(
    y_all.time.times,
    y_all.x,
    label="All runs",
    color="k",
    linewidth=1.2
)

for t in [118, 244, 359, 492]:
    ax.axvline(
        t,
        color="gray",
        linewidth=0.5
    )

ax.set_title("NCRF-DC  ROI pericalcarine_cuneuese-lingual_after_fixing")
ax.set_xlabel("Time [ms]")
ax.set_ylabel("Mean difference in visual ROI")

ax.legend(frameon=False)

fig.tight_layout()

os.makedirs("figures/ROI", exist_ok=True)

fig.savefig(
    "figures/ROI/pericalcarine_cuneuese-lingual_after_fixing.pdf",
    bbox_inches="tight",
    transparent=True
)

plt.show()

# %%

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from eelbrain import Dataset, testnd, load

# -----------------------------
# Define ROI using fixed labels
# -----------------------------

visual_label_ids = [
    #1005, 2005,  # cuneus
    #1011, 2011,  # lateral occipital
    #1013, 2013,  # lingual
    #1021, 2021,  # pericalcarine / V1
    #1007, 2007,  # fusiform
    #1009, 2009,  # inferior temporal
   1008, 2008,   #"inferior_parietal"
]

visual_mask_fixed = np.isin(source_labels_fixed, visual_label_ids)
visual_source_idx_fixed = np.where(visual_mask_fixed)[0]

print("Number of visual ROI sources:", len(visual_source_idx_fixed))

if len(visual_source_idx_fixed) == 0:
    raise RuntimeError("No visual ROI sources found.")

#tr=1000

cases = []

one_session_dir = "models/all_runs/morphed"

for i in range(1, 31):

    subject = f"sub-{i:02d}"

    if i > 9:
        session = "ImageNet01"
    else:
        session = "ImageNet03"

    morphed_file = f"{one_session_dir}/M{subject}-{session}-ncrf.pickle"

    try:
        inanim, anim = load.unpickle(morphed_file)

        cases.append([subject, "inanimate", inanim])
        cases.append([subject, "animate", anim])

    except Exception as e:
        print(f"Skipping one-session {subject}-{session}: {e}")
        continue

data_onesession = Dataset.from_caselist(
    ["subject", "animacy", "ncrf"],
    cases
)

print("Loaded cases:", len(cases))

res_onesession = testnd.VectorDifferenceRelated(
    "ncrf",
    "animacy",
    "inanimate",
    "animate",
    match="subject",
    data=data_onesession,
    tfce=True,
    tstart=100,
    tstop=700,
    samples=1000
)

diff_onesession = res_onesession.masked_difference()


#ROI

d_visual = diff_onesession.sub(source=visual_source_idx_fixed)

# Convert vector response to magnitude
d_visual_norm = d_visual.norm("space")



# Shape should be: source x time
x = d_visual_norm.x

# Mean across ROI sources
mean_roi = np.nanmean(x, axis=0)

# SEM across ROI sources
sem_roi = np.nanstd(x, axis=0, ddof=1) / np.sqrt(x.shape[0])

# Time axis
times = d_visual_norm.time.times


#plot

fig, ax = plt.subplots(figsize=(7, 4))
#main_color = "#E69F00" # orange
main_color = "#0072B2" #blue
ax.plot(
    times,
    mean_roi,
    color=main_color,
    linewidth=1.5,
    label="Mean across ROI sources"
)

ax.fill_between(
    times,
    mean_roi - sem_roi,
    mean_roi + sem_roi,
    color=main_color,
    alpha=0.18,
    linewidth=0,
    label="SEM across ROI sources"
)

for t in [118, 244, 359, 492]:
    ax.axvline(
        t,
        color="gray",
        linewidth=0.5
    )

ax.set_title("ROI lateral occipital: 1000 trials")
ax.set_xlabel("Time [ms]")
ax.set_ylabel("Mean difference in visual ROI")

ax.legend(frameon=False)

fig.tight_layout()

os.makedirs("figures/ROI", exist_ok=True)

fig.savefig(
    "figures/ROI/1000_lateral_occi.pdf",
    bbox_inches="tight",
    transparent=True
)

plt.show()
"""

# %% [markdown]
# # ROI Only for 1000 trial

# %%
# ROI definitions

ROI_LABELS = {
    "Lateral_Occipital": [1011, 2011],

    "medial_visual": [
        1005, 2005,  # cuneus
        1013, 2013,  # lingual
        1021, 2021,  # pericalcarine / V1
    ],

    "fusiform": [1007, 2007],

    "inferior_temporal": [1009, 2009],
}


ROI_COLORS = {
    "Lateral_Occipital": "#0072B2",   # blue
    "medial_visual": "#009E73",       # green
    "fusiform": "#CC79A7",            # purple/pink
    "inferior_temporal": "#E69F00",   # orange
}


ROI_TITLES = {
    "Lateral_Occipital": "Lateral occipital",
    "medial_visual": "Medial visual: cuneus, lingual, pericalcarine",
    "fusiform": "Fusiform",
    "inferior_temporal": "Inferior temporal",
}



def load_one_session_data():
    cases = []
    one_session_dir = "models/all_runs/morphed"

    for i in range(1, 31):

        subject = f"sub-{i:02d}"

        if i > 9:
            session = "ImageNet01"
        else:
            session = "ImageNet03"

        morphed_file = f"{one_session_dir}/M{subject}-{session}-ncrf.pickle"

        try:
            inanim, anim = load.unpickle(morphed_file)

            cases.append([subject, "inanimate", inanim])
            cases.append([subject, "animate", anim])

        except Exception as e:
            print(f"Skipping {subject}-{session}: {e}")
            continue

    data = Dataset.from_caselist(
        ["subject", "animacy", "ncrf"],
        cases
    )

    print("Loaded cases:", len(cases))

    return data


data_onesession = load_one_session_data()

res_onesession = testnd.VectorDifferenceRelated(
    "ncrf",
    "animacy",
    "inanimate",
    "animate",
    match="subject",
    data=data_onesession,
    tfce=True,
    tstart=100,
    tstop=700,
    samples=1000
)

diff_onesession = res_onesession.masked_difference()

#----------------------------------


def plot_ROI(roi_name, error="sem", save=True):
    

    if roi_name not in ROI_LABELS:
        raise ValueError(
            f"Unknown ROI: {roi_name}. "
            f"Choose from {list(ROI_LABELS.keys())}"
        )

    label_ids = ROI_LABELS[roi_name]

    roi_mask = np.isin(source_labels_fixed, label_ids)
    roi_source_idx = np.where(roi_mask)[0]

    print(f"\nROI: {roi_name}")
    print("Label IDs:", label_ids)
    print("Number of ROI sources:", len(roi_source_idx))

    if len(roi_source_idx) == 0:
        raise RuntimeError(f"No sources found for ROI: {roi_name}")

    # Restrict contrast to ROI
    d_roi = diff_onesession.sub(source=roi_source_idx)

    # Vector magnitude
    d_roi_norm = d_roi.norm("space")

    # shape: source x time
    x = d_roi_norm.x

    mean_roi = np.nanmean(x, axis=0)

    if error == "sem":
        err_roi = np.nanstd(x, axis=0, ddof=1) / np.sqrt(x.shape[0])
        err_label = "SEM across ROI sources"
    elif error == "sd":
        err_roi = np.nanstd(x, axis=0, ddof=1)
        err_label = "SD across ROI sources"
    else:
        raise ValueError("error must be 'sem' or 'sd'")

    times = d_roi_norm.time.times

    main_color = ROI_COLORS.get(roi_name, "#0072B2")

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(
        times,
        mean_roi,
        color=main_color,
        linewidth=2.0,
        label="Mean across ROI sources"
    )

    ax.fill_between(
        times,
        mean_roi - err_roi,
        mean_roi + err_roi,
        color=main_color,
        alpha=0.20,
        linewidth=0,
        label=err_label
    )

    for t in [118, 244, 359, 492]:
        ax.axvline(
            t,
            color="gray",
            linewidth=0.5,
            linestyle="--",
            alpha=0.7
        )

    ax.set_title(f"{ROI_TITLES[roi_name]}: 1000 trials")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Mean difference in ROI")
    ax.legend(frameon=False)

    fig.tight_layout()

    if save:
        os.makedirs("figures/ROI", exist_ok=True)

        filename = f"figures/ROI/1000_{roi_name}_{error}.pdf"

        fig.savefig(
            filename,
            bbox_inches="tight",
            transparent=True
        )

        print("Saved:", filename)

    plt.show()

    return fig, ax

#--------------------------


for roi in [
    "Lateral_Occipital",
    "medial_visual",
    "fusiform",
    "inferior_temporal",
]:
    plot_ROI(roi, error="sem")

# %%
