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
import numpy as np
from eelbrain import *
from eelbrain._data_obj import VolumeSourceSpace
import os    
from pathlib import Path
from matplotlib import pyplot as plt
import nibabel as nib            #for loading atlas 
from scipy.spatial import cKDTree    #a fast nearest-neighbor search algorithm for reassign label to unlabeled sources
import copy # for creating a copy of sources
from eelbrain._utils import mne_utils            #for converting numeric labels to string labels


# %%
def reassign_unlabeled_sources(
    source_labels,
    coords_mm,                #RAS coords of source points
    atlas_data,
    atlas_affine,             #a-fine : 4*4 transform matrxi : voxel = > RAS (mm)
    threshold_mm=5.0,
):
    source_labels_fixed = source_labels.copy()
    unlabeled_mask = (source_labels == 0) | np.isnan(source_labels)
    unlabeled_coords = coords_mm[unlabeled_mask]
    unlabeled_indices = np.where(unlabeled_mask)[0]             #source index between 0 to 1750 ( Not vertex ID!)
    #print(f"unlabeled indice = {unlabeled_indices}")

    if len(unlabeled_indices) == 0:
        print("No unlabeled sources found.")
        return source_labels_fixed

    valid_atlas_mask = atlas_data != 0

    atlas_labeled_vox = np.array(np.where(valid_atlas_mask)).T
    atlas_labeled_labels = atlas_data[valid_atlas_mask]

    # Convert atlas voxel coordinates to RAS mm
    atlas_labeled_vox_h = np.c_[                     #add 4th cooordinatte 1 for affine is 4*4
        atlas_labeled_vox,
        np.ones(len(atlas_labeled_vox))
    ]
    atlas_labeled_ras = atlas_labeled_vox_h @ atlas_affine.T
    atlas_labeled_ras = atlas_labeled_ras[:, :3]

    # Find approximate nearest neighbour
    tree = cKDTree(atlas_labeled_ras)
    dist, nearest_idx = tree.query(unlabeled_coords, k=1)

    assign_mask = dist <= threshold_mm

    candidate_source_indices = unlabeled_indices[assign_mask]
    nearest_labels = atlas_labeled_labels[nearest_idx[assign_mask]]

    source_labels_fixed[candidate_source_indices] = nearest_labels

    print("Reassigned sources:", len(candidate_source_indices))
    print("Still unlabeled:", np.sum(source_labels_fixed == 0))
    

    return source_labels_fixed

# %% [markdown]
# # src coordinates RAS mm

# %%
root = Path("~/Data/ds005810")
subjects_dir = str(Path('~/Data/ds005810/derivatives/freesurfer/subjects').expanduser())
subject="sub-11"
session="ImageNet01"
all_runs_dir = "models/all_runs/morphed"
morphed_file = f"{all_runs_dir}/M{subject}-{session}-ncrf.pickle"
inanim, anim = load.unpickle(morphed_file)

# Get source coordinates 
coords = anim.source.coordinates
if np.nanmax(np.abs(coords)) < 10:
    coords_mm = coords * 1000
else:
    coords_mm = coords
coords_mm

# %%
threshold_mm = 5.0
print(f"Threshold= {threshold_mm} mm")

atlas_file = os.path.join(
    subjects_dir,
    "fsaverage2",
    "mri",
    "aparc+aseg.mgz"
)

atlas = nib.load(atlas_file)       #read neuroimage(MRI brain image but not raw: atlas segments) atlas file
atlas_data = atlas.get_fdata()

ras_to_vox = np.linalg.inv(atlas.affine)          #inverse matrix : we need RAS=>VOX( coords_mm is RAS) but affine is for Vox =>RAS

coords_h = np.c_[coords_mm, np.ones(len(coords_mm))]   # homogeneous RAS coordinates: add 4th cordinate :[x y z 1] needed affine transforms use 4*4 matrices
vox = coords_h @ ras_to_vox.T
vox = np.round(vox[:, :3]).astype(int)       #voxels are integer so round them to nearest. after product we dont need 4th coordinate


source_labels = []                             
for x, y, z in vox:
    label = atlas_data[x, y, z]                     #example: atlas_data[12,54,13] returns 1005
    source_labels.append(label)

source_labels = np.array(source_labels)              #for each of 1751 source points we have read a label from MRI atlas
print("Number of sources with label 0:", np.sum(source_labels == 0))     #atlas_data for some vox is unlabeled (0)
#------------------------------

source_labels_fixed = reassign_unlabeled_sources(
    source_labels=source_labels,
    coords_mm=coords_mm,                      #coordination RASmm of each source point
    atlas_data=atlas_data,
    atlas_affine=atlas.affine,
    threshold_mm=threshold_mm,
    
)

# %%
atlas.shape


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
    """
    Only plots unlabeled sources. makes a new src that only includes unlabeled sources, then plots.
    """
    src = d.source.get_source_space()

    bad_mask = (
        np.isnan(labels) |
        np.isin(labels, bad_label_ids)
    )

    bad_src = copy.deepcopy(src)

    all_used_vertices = src[0]["vertno"]          #MNE vertex ID of all 1751  

    bad_vertices = all_used_vertices[bad_mask]

    bad_src[0]["vertno"] = bad_vertices
    bad_src[0]["nuse"] = len(bad_vertices)

    bad_src[0]["inuse"][:] = 0
    bad_src[0]["inuse"][bad_vertices] = 1

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
    d=anim,
    subject=subject,
    subjects_dir=subjects_dir,
    out_file="figures/bad/unlabeled_sources_before_fixing_sagittal.pdf",
    bad_label_ids=[0],
)

fig_after = plot_unlabeled_sources_on_mri(
    labels=source_labels_fixed,
    d=anim,
    subject=subject,
    subjects_dir=subjects_dir,
    out_file="figures/bad/unlabeled_sources_after_fixing_sagittal.pdf",
    bad_label_ids=[0],
)

# %%
src=anim.source.get_source_space()

# %%
src[0].keys()

# %%
src[0]["vertno"]  #vertex numbers:MNE's IDs for the active source locations

# %%
src[0]["nuse"]   #Number of active (valid) source location. ( 1751 grid points survived the filtering)

# %%
src[0]["inuse"]   #A binary array telling whether each possible source location is active.
#out of 16675 only 1751 is active. In binary array there are 1751 '1' and the rest is zero
#16675 : Candidate Grid points ---> only acceptable candidate  (inside brain volume, not too close to skull, In regions's excluded by source space construction

# %%
for source_idx, vertex_id in enumerate(src[0]["vertno"][:5]):
    print(
        f"source index = {source_idx:4d}   "
        f"vertex id = {vertex_id:5d}"
    )


# %% [markdown]
# # ROI after fixed labels

# %% [markdown]
# ## Load Dataset and run ttest

# %%
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

# %%
# Convert numeric labels to FreeSurfer string labels
label_names = mne_utils.get_volume_source_space_labels()

source_label_names_fixed = np.array([
    label_names[int(label)]
    if not np.isnan(label)
    else "NaN"
    for label in source_labels_fixed
])


ROI_LABELS = {
    "Lateral_Occipital": [
        "ctx-lh-lateraloccipital",
        "ctx-rh-lateraloccipital",
    ],

    "medial_visual": [
        "ctx-lh-cuneus",
        "ctx-rh-cuneus",
        "ctx-lh-lingual",
        "ctx-rh-lingual",
        "ctx-lh-pericalcarine",
        "ctx-rh-pericalcarine",
    ],

    "fusiform": [
        "ctx-lh-fusiform",
        "ctx-rh-fusiform",
    ],

    "inferior_temporal": [
        "ctx-lh-inferiortemporal",
        "ctx-rh-inferiortemporal",
    ],
}


ROI_COLORS = {
    "Lateral_Occipital": "#0072B2",
    "medial_visual": "#009E73",
    "fusiform": "#CC79A7",
    "inferior_temporal": "#E69F00",
}


ROI_TITLES = {
    "Lateral_Occipital": "Lateral occipital",
    "medial_visual": "Medial visual: cuneus, lingual, pericalcarine",
    "fusiform": "Fusiform",
    "inferior_temporal": "Inferior temporal",
}


def plot_ROI(roi_name, diff, source_label_names, error="sem", save=True):

    if roi_name not in ROI_LABELS:
        raise ValueError(f"Unknown ROI: {roi_name}")

    roi_label_names = ROI_LABELS[roi_name]

    roi_mask = np.isin(source_label_names, roi_label_names)

    print(f"\nROI: {roi_name}")
    print("ROI label names:", roi_label_names)
    print("Number of ROI sources:", np.sum(roi_mask))

    if np.sum(roi_mask) == 0:
        raise RuntimeError(f"No sources found for ROI: {roi_name}")

    d_roi = diff.sub(source=roi_mask)

    d_roi_norm = d_roi.norm("space")
    x = d_roi_norm.x

    mean_roi = np.nanmean(x, axis=0)

    if error == "sem":
        err_roi = np.nanstd(x, axis=0, ddof=1) / np.sqrt(x.shape[0])
        err_label = "SEM"
    elif error == "sd":
        err_roi = np.nanstd(x, axis=0, ddof=1)
        err_label = "SD"
    else:
        raise ValueError("error must be 'sem' or 'sd'")

    times = d_roi_norm.time.times
    main_color = ROI_COLORS.get(roi_name, "#0072B2")

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(
        times,
        mean_roi,
        color=main_color,
        linewidth=2,
        label="Mean across ROI sources",
    )

    ax.fill_between(
        times,
        mean_roi - err_roi,
        mean_roi + err_roi,
        color=main_color,
        alpha=0.2,
        linewidth=0,
        label=err_label,
    )

    for t in [118, 244, 359, 492]:
        ax.axvline(
            t,
            color="gray",
            linewidth=0.5,
            linestyle="--",
            alpha=0.7,
        )

    ax.set_title(f"{ROI_TITLES[roi_name]}: 1000 trials")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Mean difference in ROI")
    ax.legend(frameon=False)

    fig.tight_layout()

    if save:
        os.makedirs("figures/ROI", exist_ok=True)
        filename = f"figures/ROI/1000_{roi_name}_{error}.pdf"
        fig.savefig(filename, bbox_inches="tight", transparent=True)
        print("Saved:", filename)

    plt.show()

    return fig, ax


for roi in ROI_LABELS:
    plot_ROI(
        roi_name=roi,
        diff=diff_onesession,
        source_label_names=source_label_names_fixed,
        error="sem",
        save=True,
    )

# %% [markdown]
# # Compare before/after fixing label ROI plots

# %%
source_label_names = np.array([
    label_names[int(label)]
    if not np.isnan(label)
    else "NaN"
    for label in source_labels
])
for roi_name, roi_labels in ROI_LABELS.items():

    n_before = np.sum(
        np.isin(source_label_names, roi_labels)
    )

    n_after = np.sum(
        np.isin(source_label_names_fixed, roi_labels)
    )

    print(
        f"{roi_name:20s} "
        f"before={n_before:4d} "
        f"after={n_after:4d}"
    )

# %%
for roi in ROI_LABELS:
    plot_ROI(
        roi_name=roi,
        diff=diff_onesession,
        source_label_names=source_label_names,
        error="sem",
        save=True
    )

# %% [markdown]
# # Plot ROI BRAIN
#

# %%
from matplotlib.colors import ListedColormap
from nilearn import plotting
from nilearn.plotting import find_xyz_cut_coords



subject = "fsaverage2"   

atlas_file = os.path.join(
    subjects_dir,
    subject,
    "mri",
    "aparc+aseg.mgz"
)

brain_file = os.path.join(              #background MRI of ROI
    subjects_dir,
    subject,
    "mri",
    "brain.mgz"
)

atlas = nib.load(atlas_file)
brain_img = nib.load(brain_file)

#--------------------------
def make_roi_volume(atlas_img, roi_name):
    roi_label_names = ROI_LABELS[roi_name]

    atlas_data = atlas_img.get_fdata()
    roi_vol = np.zeros(atlas_data.shape, dtype=np.int16)

    label_names = mne_utils.get_volume_source_space_labels()

    roi_value = 1

    for label_value, label_name in label_names.items():
        if label_name in roi_label_names:
            roi_vol[atlas_data == int(label_value)] = roi_value
            roi_value += 1

    roi_img = nib.Nifti1Image(
        roi_vol,
        atlas_img.affine,
        atlas_img.header
    )

    print("\nROI:", roi_name)
    print("ROI labels:", roi_label_names)
    print("Unique values:", np.unique(roi_img.get_fdata(), return_counts=True))
    print("Number of ROI voxels:", np.sum(roi_img.get_fdata() > 0))

    return roi_img


#------------------------
def plot_roi_nilearn(roi_name, atlas_img, bg_img, save=True):
    plt.close("all")

    roi_img = make_roi_volume(
        atlas_img=atlas_img,
        roi_name=roi_name,
    )

    roi_data = roi_img.get_fdata()

    if np.sum(roi_data > 0) == 0:
        raise RuntimeError(f"No voxels found for ROI: {roi_name}")

    cut_coords = find_xyz_cut_coords(roi_img)

    color = ROI_COLORS[roi_name]

    cmap = ListedColormap([
        (0, 0, 0, 0),  # background transparent
        color,
        color,
        color,
        color,
        color,
        color,
    ])

    display = plotting.plot_roi(
        roi_img,
        bg_img=bg_img,
        display_mode="ortho",
        cut_coords=cut_coords,
        cmap=cmap,
        alpha=0.85,
        title=ROI_TITLES[roi_name],
        black_bg=False,
        draw_cross=False,
    )

    if save:
        os.makedirs("figures/ROI_nilearn", exist_ok=True)
        filename = f"figures/ROI_nilearn/{roi_name}_nilearn.pdf"
        display.savefig(filename)
        print("Saved:", filename)

    plotting.show()

    return display


#-------------
for roi in ROI_LABELS:
    plot_roi_nilearn(
        roi_name=roi,
        atlas_img=atlas,
        bg_img=brain_img,
        save=True,
    )

# %%
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting


# --------------------------
# Load MRI background
# --------------------------

subject = "fsaverage2"

brain_file = os.path.join(
    subjects_dir,
    subject,
    "mri",
    "brain.mgz"
)

brain_img = nib.load(brain_file)


# --------------------------
# Fixed labels: numeric -> string
# --------------------------

label_names = mne_utils.get_volume_source_space_labels()

source_label_names_fixed = np.array([
    label_names[int(label)]
    if not np.isnan(label)
    else "NaN"
    for label in source_labels_fixed
])


# --------------------------
# Plot fixed ROI source points
# --------------------------

def plot_fixed_sources_nilearn(
    roi_name,
    source_label_names_fixed,
    coords,
    bg_img,
    save=True,
    n_slices=8,
    marker_size=45,
):
    plt.close("all")

    roi_label_names = ROI_LABELS[roi_name]

    roi_mask = np.isin(
        source_label_names_fixed,
        roi_label_names
    )

    print("\nROI:", roi_name)
    print("ROI labels:", roi_label_names)
    print("Number of fixed source points:", np.sum(roi_mask))

    if np.sum(roi_mask) == 0:
        raise RuntimeError(f"No fixed sources found for ROI: {roi_name}")

    roi_coords = coords[roi_mask]

    # MNE source coordinates are often in meters.
    # Nilearn uses millimeters.
    if np.nanmax(np.abs(roi_coords)) < 1:
        roi_coords = roi_coords * 1000

    color = ROI_COLORS[roi_name]

    z_min = roi_coords[:, 2].min()
    z_max = roi_coords[:, 2].max()

    if z_min == z_max:
        cut_coords = [z_min]
    else:
        cut_coords = np.linspace(z_min, z_max, n_slices)

    display = plotting.plot_anat(
        bg_img,
        display_mode="x",
        cut_coords=cut_coords,
        title=ROI_TITLES[roi_name] + " fixed sources",
        black_bg=False,
        draw_cross=False,
    )

    display.add_markers(
        marker_coords=roi_coords,
        marker_color=color,
        marker_size=marker_size,
    )

    if save:
        os.makedirs("figures/ROI_nilearn_fixed_sources", exist_ok=True)
        filename = f"figures/ROI_nilearn_fixed_sources/{roi_name}_fixed_sources.pdf"
        display.savefig(filename)
        print("Saved:", filename)

    plotting.show()

    return display


# --------------------------
# Plot all fixed ROIs
# --------------------------

for roi in ROI_LABELS:
    plot_fixed_sources_nilearn(
        roi_name=roi,
        source_label_names_fixed=source_label_names_fixed,
        coords=coords,
        bg_img=brain_img,
        save=True,
        n_slices=8,
        marker_size=45,
    )

# %%
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from nilearn import plotting


subject = "fsaverage2"

atlas_file = os.path.join(subjects_dir, subject, "mri", "aparc+aseg.mgz")
brain_file = os.path.join(subjects_dir, subject, "mri", "brain.mgz")

atlas = nib.load(atlas_file)
brain_img = nib.load(brain_file)


label_names = mne_utils.get_volume_source_space_labels()

source_label_names_fixed = np.array([
    label_names[int(label)]
    if not np.isnan(label)
    else "NaN"
    for label in source_labels_fixed
])


def make_atlas_roi_volume(atlas_img, roi_name):
    roi_label_names = ROI_LABELS[roi_name]

    atlas_data = atlas_img.get_fdata()
    roi_vol = np.zeros(atlas_data.shape, dtype=np.int16)

    roi_value = 1

    for label_value, label_name in label_names.items():
        if label_name in roi_label_names:
            roi_vol[atlas_data == int(label_value)] = roi_value
            roi_value += 1

    return nib.Nifti1Image(
        roi_vol,
        atlas_img.affine,
        atlas_img.header,
    )


def best_xz_cut_coords_from_fixed_sources(source_coords):
    """
    For display_mode='xz'.
    Returns [best_x, best_z].
    """

    left = source_coords[source_coords[:, 0] < 0]
    right = source_coords[source_coords[:, 0] > 0]

    if len(left) > 0 and len(right) > 0:
        # Bilateral ROI: choose the side with more sources
        if len(right) >= len(left):
            best_x = np.median(right[:, 0])
        else:
            best_x = np.median(left[:, 0])
    else:
        # One-sided or mostly centered ROI
        best_x = np.median(source_coords[:, 0])

    best_z = np.median(source_coords[:, 2])

    return [float(best_x), float(best_z)]


def plot_parcel_with_fixed_sources_best_xz(
    roi_name,
    atlas_img,
    bg_img,
    source_label_names_fixed,
    coords,
    save=True,
):
    plt.close("all")

    roi_label_names = ROI_LABELS[roi_name]

    # Atlas anatomical parcel
    roi_img = make_atlas_roi_volume(
        atlas_img=atlas_img,
        roi_name=roi_name,
    )

    if np.sum(roi_img.get_fdata() > 0) == 0:
        raise RuntimeError(f"No atlas ROI voxels found for {roi_name}")

    # Fixed source points
    source_mask = np.isin(
        source_label_names_fixed,
        roi_label_names,
    )

    if np.sum(source_mask) == 0:
        raise RuntimeError(f"No fixed source points found for {roi_name}")

    source_coords = coords[source_mask]

    # MNE coords are usually in meters; Nilearn uses millimeters
    if np.nanmax(np.abs(source_coords)) < 1:
        source_coords = source_coords * 1000

    cut_coords = best_xz_cut_coords_from_fixed_sources(source_coords)

    print("\nROI:", roi_name)
    print("ROI labels:", roi_label_names)
    print("Number of fixed source points:", np.sum(source_mask))
    print("Number of atlas ROI voxels:", np.sum(roi_img.get_fdata() > 0))
    print("Best cut coords [x, z]:", cut_coords)

    color = ROI_COLORS[roi_name]

    cmap = ListedColormap([
        (0, 0, 0, 0),
        color,
        color,
        color,
        color,
        color,
        color,
    ])

    display = plotting.plot_roi(
        roi_img,
        bg_img=bg_img,
        display_mode="xz",      # sagittal + axial only
        cut_coords=cut_coords,  # [x, z]
        cmap=cmap,
        alpha=0.40,
        title=f"{ROI_TITLES[roi_name]} ",
        black_bg=False,
        draw_cross=False,
    )

    display.add_markers(
        marker_coords=source_coords,
        marker_color=color,
        marker_size=60,
    )

    if save:
        os.makedirs("figures/ROI_parcel_plus_fixed_sources", exist_ok=True)
        filename = (
            f"figures/ROI_parcel_plus_fixed_sources/"
            f"{roi_name}_parcel_plus_fixed_sources_xz.pdf"
        )
        display.savefig(filename)
        print("Saved:", filename)

    plotting.show()

    return display


for roi in ROI_LABELS:
    plot_parcel_with_fixed_sources_best_xz(
        roi_name=roi,
        atlas_img=atlas,
        bg_img=brain_img,
        source_label_names_fixed=source_label_names_fixed,
        coords=coords,
        save=True,
    )

# %%
from matplotlib.colors import ListedColormap

roi_img = make_roi_volume(
    atlas_img=atlas,
    roi_name="Lateral_Occipital",
)
cmap_lh_rh = ListedColormap([
    (0, 0, 0, 0),  # 0 background
    "#0072B2",     # 1 LH
    "#D55E00",     # 2 RH
])

display = plotting.plot_roi(
    roi_img,
    bg_img=brain_img,
    display_mode="x",
    cut_coords=[-40, -30, -20, -10, 10, 20, 30, 40],
    cmap=cmap_lh_rh,
    vmin=0,
    vmax=2,
    alpha=0.8,
    black_bg=False,
    draw_cross=False,
     title=f"{roi_name} ",
)

plotting.show()

# %%
roi_mask = np.isin(
    
    source_label_names_fixed,
    ROI_LABELS["medial_visual"],
    
)

anim_roi_med = anim.copy()
anim_roi_med.x[~roi_mask, :, :] = 0

# %%
plot.GlassBrain(anim_roi_med.sub(time=110),title=f"Medial visual ROI {t} ms") 

# %%
plot.GlassBrain(anim.sub(time=110),title=f"Anim{t} ms") 

# %%
