from scipy.stats import t
import numpy as np
#for vol2src
import os
import nibabel as nib
import mne
import matplotlib.pyplot as plt
from nilearn import datasets, surface, plotting
from nilearn.image import smooth_img
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap



def loftus_masson(data, confidence=0.95):
    
    """
    Compute Loftus-Masson within-subject confidence intervals.
    Reference: "Using confidence intervals in within-subject designs, Loftus & Masson,1994)

    """
    data = np.asarray(data, dtype=float)

    if data.ndim != 2:
        raise ValueError(
            "data must be a 2D array with shape "
            "(n_subjects, n_conditions)"
        )

    if np.isnan(data).any():
        raise ValueError("data contains missing values")

    n_subjects, n_conditions = data.shape

    if n_subjects < 2:
        raise ValueError("At least two subjects are required")

    if n_conditions < 2:
        raise ValueError("At least two conditions are required")

    # Original condition means
    means = data.mean(axis=0)

    # Mean of each subject across conditions
    subject_means = data.mean(axis=1, keepdims=True)

    # Mean across all subjects and conditions
    grand_mean = data.mean()

    # Remove between-subject variability
    normalized_data = data - subject_means + grand_mean

    # Deviations around each condition mean after normalization
    residuals = normalized_data - normalized_data.mean(
        axis=0,
        keepdims=True,
    )

    # Subject × condition interaction sum of squares
    ss_subject_condition = np.sum(residuals**2)

    # Subject × condition degrees of freedom
    df_subject_condition = (
        (n_subjects - 1) * (n_conditions - 1)
    )

    # Subject × condition mean square
    ms_subject_condition = (
        ss_subject_condition / df_subject_condition
    )

    # Loftus-Masson within-subject SEM
    sem_within = np.sqrt(
        ms_subject_condition / n_subjects
    )

    # Critical t-value
    alpha = 1 - confidence
    t_critical = t.ppf(
        1 - alpha / 2,
        df_subject_condition,
    )

    ci = t_critical * sem_within

    return means, sem_within, ci


def vol_stc_to_img(
    x,                           
    subject="fsaverage2",
    subjects_dir="",
    scale=1e12,
):
    """
    Converts volume source to a Nifti Image. 
    x: a volume-based estimate with one value per source point 
    """
    x = np.asarray(x).ravel() * scale

    template_img = nib.load(f"{subjects_dir}/{subject}/mri/brain.mgz")
    meg_vol = np.zeros(template_img.shape)

    src_file = f"{subjects_dir}/{subject}/bem/{subject}-vol-7-src.fif"
    src = mne.read_source_spaces(src_file, verbose=False)

    coords = src[0]["rr"][src[0]["inuse"].astype(bool)]

    if np.abs(coords).max() < 10:
        coords = coords * 1000

    ras_to_vox = np.linalg.inv(template_img.affine)
    coords_h = np.c_[coords, np.ones(len(coords))]
    vox = np.round(coords_h @ ras_to_vox.T).astype(int)[:, :3]

    valid = (
        (vox[:, 0] >= 0) & (vox[:, 0] < meg_vol.shape[0]) &
        (vox[:, 1] >= 0) & (vox[:, 1] < meg_vol.shape[1]) &
        (vox[:, 2] >= 0) & (vox[:, 2] < meg_vol.shape[2])
    )

    meg_vol[vox[valid, 0], vox[valid, 1], vox[valid, 2]] = x[valid]

    meg_img = nib.Nifti1Image(meg_vol, template_img.affine)

    print("Active sources:", len(coords))
    print("Valid sources inside MRI:", valid.sum())
    print("Non-zero voxels:", np.count_nonzero(meg_vol))

    return meg_img


def volume_to_surface_textures(
    x,
    subject="fsaverage2",
    subjects_dir="",
    scale=1e12,
    smooth_fwhm=8,
    radius=7,
):
    """
    Convert a volume-based source estimate (x) into surface values that can be plotted on 
    the cortical surface (texture_left, texture_right).
    """
    meg_img = vol_stc_to_img(
        x,
        subject=subject,
        subjects_dir=subjects_dir,
        scale=scale,
    )

    meg_img_smooth = smooth_img(meg_img, fwhm=smooth_fwhm)

    fsaverage = datasets.fetch_surf_fsaverage("fsaverage5") # # Load the fsaverage5 cortical surface meshes
    
    # Brain surface is two separate meshes for L, R
    texture_left = surface.vol_to_surf(
        meg_img_smooth,
        surf_mesh=fsaverage["pial_left"],
        inner_mesh=fsaverage["white_left"],
        interpolation="linear",
        radius=radius,
    )

    texture_right = surface.vol_to_surf(
        meg_img_smooth,
        surf_mesh=fsaverage["pial_right"],
        inner_mesh=fsaverage["white_right"],
        interpolation="linear",
        radius=radius,
    )
    
    texture_left = np.nan_to_num(texture_left, nan=0.0)                 #sourface values[0.2, 0.4, ....] doesnt have mesh info only value
    texture_right = np.nan_to_num(texture_right, nan=0.0)

    return texture_left, texture_right, fsaverage


def get_surface_plot_params(
    texture_left,
    texture_right,
    threshold_ratio=0.10,
):
    xmin = min(texture_left.min(), texture_right.min())
    xmax = max(texture_left.max(), texture_right.max())

    eps = 1e-20

    if xmin >= 0:
        vmin = 0
        vmax = xmax
        threshold = threshold_ratio * vmax

        cmap = LinearSegmentedColormap.from_list(
            "positive_activity",
            [
                (1.0, 1.0, 1.0, 0.0),
                (1.0, 0.8, 0.8, 0.6),
                (1.0, 0.0, 0.0, 0.9),
                (0.35, 0.0, 0.0, 1.0),
            ],
            N=256,
        )

    elif xmax <= 0:
        vmin = xmin
        vmax = 0
        threshold = threshold_ratio * abs(vmin)

        cmap = LinearSegmentedColormap.from_list(
            "negative_activity",
            [
                (0.0, 0.0, 0.35, 1.0),
                (0.0, 0.2, 1.0, 0.9),
                (0.8, 0.9, 1.0, 0.6),
                (1.0, 1.0, 1.0, 0.0),
            ],
            N=256,
        )
    else:
        vmax = max(abs(xmin), abs(xmax))
        vmin = -vmax
        threshold = threshold_ratio * vmax
    
        cmap = plt.get_cmap("cold_hot").copy()
    
        colors = cmap(np.linspace(0, 1, 256))
    
        # location of ±threshold in the colormap
        center = 128
        half_width = int(128 * threshold / vmax)
    
        colors[
            center-half_width:center+half_width+1
        ] = [0.85, 0.85, 0.85, 1.0]   # light gray
    
        cmap = LinearSegmentedColormap.from_list(
            "cold_hot_threshold",
            colors
        )
    if abs(vmax - vmin) <= eps:
        raise ValueError("Values are too small after projection.")

    return vmin, vmax, threshold, cmap


def plot_surface_views(
    texture_left,
    texture_right,
    fsaverage,
    title="",
    views=("lateral", "ventral", "medial"),
    threshold_ratio=0.10,
    save=None,
):
    vmin, vmax, threshold, cmap = get_surface_plot_params(
        texture_left,
        texture_right,
        threshold_ratio=threshold_ratio,
    )

    fig, axes = plt.subplots(
        len(views),                          #row: number of view
        2,                                    #column= 2 (l,R)
        figsize=(6, 3.0 * (len(views))),
        subplot_kw={"projection": "3d"},
    )

    if len(views) == 1:
        axes = np.array([axes])

    for row, view in enumerate(views):
        plotting.plot_surf_stat_map(
            fsaverage["infl_left"],
            texture_left,
            hemi="left",
            view=view,
            bg_map=fsaverage["sulc_left"],
            cmap=cmap,
            threshold=threshold,
            vmin=vmin,
            vmax=vmax,
            colorbar=False,
            axes=axes[row, 0],
            title=f"L - {view}",
        )

        plotting.plot_surf_stat_map(
            fsaverage["infl_right"],
            texture_right,
            hemi="right",
            view=view,
            bg_map=fsaverage["sulc_right"],
            cmap=cmap,
            threshold=threshold,
            vmin=vmin,
            vmax=vmax,
            colorbar=False,
            axes=axes[row, 1],
            title=f"R - {view}",
        )

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    fig.subplots_adjust(
    wspace=-0.06,   # horizontal space
    hspace=-0.06    # vertical space
)

    #color_bar
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes([0.92, 0.12, 0.02, 0.20]) #colorbar location
    fig.colorbar(sm, cax=cax)
    
    
    if save is not None:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(f"{save}.png", dpi=300, bbox_inches="tight")
        fig.savefig(f"{save}.svg", bbox_inches="tight")

    plt.show()

    return fig


def Plot_vol2surf(
    x,
    title="",
    subject="fsaverage2",
    subjects_dir="",
    scale=1e12,
    smooth_fwhm=8,
    radius=7,
    threshold_ratio=0.10,
    views=("lateral", "ventral", "medial"),
    save=None,
    RC={},
):
    texture_left, texture_right, fsaverage = volume_to_surface_textures(
        x,
        subject=subject,
        subjects_dir=subjects_dir,
        scale=scale,
        smooth_fwhm=smooth_fwhm,
        radius=radius,
    )

    fig = plot_surface_views(
        texture_left,
        texture_right,
        fsaverage,
        title=title,
        views=views,
        threshold_ratio=threshold_ratio,
        save=save,
    )

    return fig    