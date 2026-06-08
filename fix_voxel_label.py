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
import copy # for creating a copy of sources


# %%

def reassign_unlabeled_sources(
    source_labels,
    coords_mm,
    atlas_data,
    atlas_affine,
    threshold_mm=7.0,
    include_white_matter=True,
):
    
    source_labels_fixed = source_labels.copy()

    # Find unlabeled sources
    unlabeled_mask = source_labels == 0
    unlabeled_coords = coords_mm[unlabeled_mask]
    unlabeled_indices = np.where(unlabeled_mask)[0]

    print("Initial unlabeled sources:", len(unlabeled_indices))

    if len(unlabeled_indices) == 0:
        print("No unlabeled sources found.")
        return source_labels_fixed

    # Atlas voxels with real labels
    valid_atlas_mask = atlas_data != 0

    # FreeSurfer aseg white matter labels
    # 2  = Left-Cerebral-White-Matter
    # 41 = Right-Cerebral-White-Matter
    white_matter_labels = [2, 41]

    if not include_white_matter:
        valid_atlas_mask &= ~np.isin(atlas_data, white_matter_labels)

    atlas_labeled_vox = np.array(np.where(valid_atlas_mask)).T
    atlas_labeled_labels = atlas_data[valid_atlas_mask]

    #print("Atlas voxels used:", len(atlas_labeled_vox))

    if len(atlas_labeled_vox) == 0:
        raise ValueError("No valid atlas voxels found.")

    # Convert atlas voxel coordinates to RAS mm
    atlas_labeled_vox_h = np.c_[
        atlas_labeled_vox,
        np.ones(len(atlas_labeled_vox))
    ]

    atlas_labeled_ras = atlas_labeled_vox_h @ atlas_affine.T
    atlas_labeled_ras = atlas_labeled_ras[:, :3]

    # Build KDTree from all valid labeled atlas voxels
    tree = cKDTree(atlas_labeled_ras)

    # For each unlabeled source, find nearest atlas voxel
    dist, nearest_idx = tree.query(unlabeled_coords, k=1)

    # Only assign sources close enough
    assign_mask = dist <= threshold_mm

    candidate_source_indices = unlabeled_indices[assign_mask]
    nearest_labels = atlas_labeled_labels[nearest_idx[assign_mask]]

    # Assign nearest atlas labels
    source_labels_fixed[candidate_source_indices] = nearest_labels

    print("Reassigned sources:", len(candidate_source_indices))
    print("Still unlabeled:", np.sum(source_labels_fixed == 0))

    return source_labels_fixed


# %%
root = Path("~/Data/ds005810")
subjects_dir = str(Path('~/Data/ds005810/derivatives/freesurfer/subjects').expanduser())

# %%
subject="sub-11"
session="ImageNet01"
all_runs_dir = "models/all_runs/morphed"
morphed_file = f"{all_runs_dir}/M{subject}-{session}-ncrf.pickle"
inanim, anim = load.unpickle(morphed_file)


norm = anim.norm("space")
# Get source coordinates 
coords = norm.source.coordinates
if np.nanmax(np.abs(coords)) < 10:
    coords_mm = coords * 1000
else:
    coords_mm = coords

# %%
threshold_mm = 3.0

atlas_file = os.path.join(
    subjects_dir,
    "fsaverage2",
    "mri",
    "aparc+aseg.mgz"
)

atlas = nib.load(atlas_file)       #read neuroimage(MRI brain image but not raw: atlas segments) atlas file
atlas_data = atlas.get_fdata()

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

#------------------------------
source_labels_fixed = reassign_unlabeled_sources(
    source_labels=source_labels,
    coords_mm=coords_mm,
    atlas_data=atlas_data,
    atlas_affine=atlas.affine,
    threshold_mm=threshold_mm,
    include_white_matter=True,            #can assigned labels to white matter
)


# %%
def plot_unlabeled_sources_on_mri(
    labels,
    d,
    subject,
    subjects_dir,
    out_file,
    bad_label_ids=[0],
    slices=[70, 80, 90, 100, 110],
    orientation="sagittal",
):
    
    src = d.source.get_source_space()

    bad_mask = (
        np.isnan(labels) |
        np.isin(labels, bad_label_ids)
    )

    bad_src = copy.deepcopy(src)

    all_used_vertices = src[0]["vertno"]

    if len(bad_mask) != len(all_used_vertices):
        raise ValueError(
            f"Mask length {len(bad_mask)} does not match "
            f"source vertices length {len(all_used_vertices)}"
        )

    bad_vertices = all_used_vertices[bad_mask]

    bad_src[0]["vertno"] = bad_vertices
    bad_src[0]["nuse"] = len(bad_vertices)

    bad_src[0]["inuse"][:] = 0
    bad_src[0]["inuse"][bad_vertices] = 1

    print("Bad sources:", len(bad_vertices))
    print("Saving to:", out_file)

    fig = mne.viz.plot_bem(
        subject=subject,
        subjects_dir=subjects_dir,
        brain_surfaces="white",
        orientation=orientation,
        slices=slices,
        src=bad_src,
    )

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight")

    plt.show()

    return fig


# %%
fig_before = plot_unlabeled_sources_on_mri(
    labels=source_labels,
    d=norm,
    subject=subject,
    subjects_dir=subjects_dir,
    out_file="figures/bad/unlabeled_sources_before_fixing_sagittal.pdf",
    bad_label_ids=[0],
)

fig_after = plot_unlabeled_sources_on_mri(
    labels=source_labels_fixed,
    d=norm,
    subject=subject,
    subjects_dir=subjects_dir,
    out_file="figures/bad/unlabeled_sources_after_fixing_sagittal.pdf",
    bad_label_ids=[0],
)

# %%

# %%

# %%
