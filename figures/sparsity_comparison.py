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
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from eelbrain import load

sys.path.append(str(Path.cwd().parent))

from ncrf_analysis import *
from utils import loftus_masson

# %%
# Data location 
model_dir = Path("../models/all_runs")
dc_dir = model_dir / "ncrf-dc"
ec_dir = model_dir / "ncrf-ec"

fig_dir = Path("figures")
fig_dir.mkdir(parents=True, exist_ok=True)

# Figure style
FONT = "Arial"
FONT_SIZE = 16

RC = {
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.transparent": True,
    "savefig.bbox": "tight",

    "font.family": "sans-serif",
    "font.sans-serif": FONT,
    "font.size": FONT_SIZE,

    "figure.labelsize": FONT_SIZE,
    "figure.titlesize": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE - 2,
}
plt.rcParams.update(RC)


def plot_l1(ax, times,mean_ec, sem_ec,  mean_dc, sem_dc):
    
    ax.plot(  times, mean_ec, linewidth=2,label="NCRF-EC")
    ax.fill_between(  times,  mean_ec - sem_ec, mean_ec + sem_ec,  alpha=0.25)
    ax.plot(  times,   mean_dc,linewidth=2,  label="NCRF-DC" )
    
    ax.fill_between( times, mean_dc - sem_dc,  mean_dc + sem_dc, alpha=0.25)
    ax.set(xlabel="Time (ms)", ylabel="Mean L1 norm")
    ax.legend(loc="upper right")


def plot_nonzero_sources(ax,times, mean_ec, sem_ec, mean_dc, sem_dc):
    
    ax.plot(  times,  mean_ec, linewidth=2,  label="NCRF-EC" )
    ax.fill_between(  times,   mean_ec - sem_ec,  mean_ec + sem_ec,  alpha=0.25)
    ax.plot( times, mean_dc,linewidth=2,label="NCRF-DC")

    ax.fill_between( times, mean_dc - sem_dc,mean_dc + sem_dc,  alpha=0.25 )
    ax.set(xlabel="Time (ms)",ylabel="Non-zero sources (%)")
    ax.legend(loc="upper right")

def loftus_masson_timecourse(values_ec, values_dc):
    
    values_ec = np.asarray(values_ec, dtype=float)
    values_dc = np.asarray(values_dc, dtype=float)

    if values_ec.ndim != 2 or values_dc.ndim != 2:
        raise ValueError(
            "EC and DC arrays must have shape "
            "(n_subjects, n_times)."
        )

    if values_ec.shape != values_dc.shape:
        raise ValueError(
            "EC and DC arrays must have the same shape. "
            f"Got {values_ec.shape} and {values_dc.shape}."
        )

    n_times = values_ec.shape[1]

    mean_ec = np.empty(n_times)
    mean_dc = np.empty(n_times)
    sem_within = np.empty(n_times)

    for time_index in range(n_times):

        # subjects × conditions
        data_at_time = np.column_stack([
            values_ec[:, time_index],
            values_dc[:, time_index],
        ])

        means, sem, _ = loftus_masson(data_at_time)

        mean_ec[time_index] = means[0]
        mean_dc[time_index] = means[1]
        sem_within[time_index] = sem

    # The same within-subject SEM applies to both conditions
    sem_ec = sem_within.copy()
    sem_dc = sem_within.copy()

    return mean_ec, mean_dc, sem_ec, sem_dc

# Compute non-zero counts from original models
def compute_nonzero_counts_from_models(
    model_dir,
    mod,
    atol=0.0,
):
    model_dir = Path(model_dir)

    if mod not in {"effect", "dummy"}:
        raise ValueError(
            "mod must be either 'effect' or 'dummy'."
        )

    model_files = sorted(
        model_dir.glob("sub-*-ncrf.pickle")
    )

    if not model_files:
        raise FileNotFoundError(
            f"No original NCRF model files found in:\n"
            f"{model_dir}"
        )

    times_by_subject = {}
    counts_by_subject = {}
    n_sources_by_subject = {}

    for model_file in model_files:

        subject = model_file.name.split("-ImageNet")[0]

        print(f"Loading {model_file.name}")

        model = load.unpickle(model_file)
        kernels = model.h

        if len(kernels) != 2:
            raise ValueError(
                f"{model_file.name} has {len(kernels)} kernels; "
                "expected exactly 2."
            )

        if mod == "effect":

            # Effect coding:
            # general, contrast
            _, contrast = kernels

            # No need to multiply by 2 here because scaling
            # does not change whether a source is exactly zero.

        else:

            # Dummy coding:
            # inanimate, animate
            inanimate, animate = kernels

            contrast = animate - inanimate

        # Norm over the three dipole components
        source_norm = contrast.norm("space")
        values = source_norm.x

        if values.ndim != 2:
            raise ValueError(
                f"Unexpected source-norm shape in "
                f"{model_file.name}: {values.shape}. "
                "Expected source × time."
            )

        subject_times = source_norm.time.times

        subject_counts = np.sum(
            values > atol,
            axis=0,
        )

        times_by_subject[subject] = subject_times
        counts_by_subject[subject] = subject_counts
        n_sources_by_subject[subject] = values.shape[0]

    return (
        times_by_subject,
        counts_by_subject,
        n_sources_by_subject,
    )



# Load NCRF datasets for L1 analysis
data_ec = create_ncrf_dataset( mod="effect", path=ec_dir)
data_dc = create_ncrf_dataset( mod="dummy", path=dc_dir)



# Scale only the EC contrast NCRFs for direct magnitude comparison
contrast_mask = data_ec["animacy"] == "contrast"
data_ec["ncrf"][contrast_mask] *= 2



# Panel A: L1 norm
times_ec, ec_by_subject = compute_l1(
    data_ec,
    mod="effect",
)

times_dc, dc_by_subject = compute_l1(
    data_dc,
    mod="dummy",
)

times_ec = np.asarray(times_ec)
times_dc = np.asarray(times_dc)

if not np.allclose(times_ec, times_dc):
    raise ValueError(
        "EC and DC L1 time axes do not match."
    )

subjects_l1 = sorted(
    set(ec_by_subject) & set(dc_by_subject)
)

if not subjects_l1:
    raise ValueError(
        "No matching subjects between EC and DC L1 results."
    )

# Actual subject-level arrays:
# subjects × time
l1_ec = np.stack([
    ec_by_subject[subject]
    for subject in subjects_l1
])

l1_dc = np.stack([
    dc_by_subject[subject]
    for subject in subjects_l1
])

(
    mean_ec,
    mean_dc,
    sem_ec,
    sem_dc,
) = loftus_masson_timecourse(
    l1_ec,
    l1_dc,
)



# Panel B

(
    times_ec_by_subject,
    ec_counts_by_subject,
    ec_n_sources,
) = compute_nonzero_counts_from_models(
    model_dir=ec_dir,
    mod="effect",
    atol=0.0,
)

(
    times_dc_by_subject,
    dc_counts_by_subject,
    dc_n_sources,
) = compute_nonzero_counts_from_models(
    model_dir=dc_dir,
    mod="dummy",
    atol=0.0,
)


subjects_nonzero = sorted(
    set(ec_counts_by_subject)
    & set(dc_counts_by_subject)
)

if not subjects_nonzero:
    raise ValueError(
        "No matching subjects between EC and DC "
        "non-zero results."
    )


reference_subject = subjects_nonzero[0]

reference_times = np.asarray(
    times_ec_by_subject[reference_subject]
)


for subject in subjects_nonzero:

    times_ec_subject = np.asarray(
        times_ec_by_subject[subject]
    )

    times_dc_subject = np.asarray(
        times_dc_by_subject[subject]
    )

    if not np.allclose(
        times_ec_subject,
        times_dc_subject,
    ):
        raise ValueError(
            f"EC and DC time axes do not match "
            f"for {subject}."
        )

    if not np.allclose(
        reference_times,
        times_ec_subject,
    ):
        raise ValueError(
            f"Time axis for {subject} differs from "
            f"{reference_subject}."
        )

    if ec_n_sources[subject] != dc_n_sources[subject]:
        raise ValueError(
            f"EC and DC have different source counts "
            f"for {subject}: "
            f"{ec_n_sources[subject]} versus "
            f"{dc_n_sources[subject]}."
        )


# Actual subject-level non-zero count arrays:
# subjects × time
nonzero_ec = np.stack([
    ec_counts_by_subject[subject]
    for subject in subjects_nonzero
])

nonzero_dc = np.stack([
    dc_counts_by_subject[subject]
    for subject in subjects_nonzero
])


# Convert counts to percentages separately for each subject
nonzero_pct_ec = np.stack([
    100
    * ec_counts_by_subject[subject]
    / ec_n_sources[subject]
    for subject in subjects_nonzero
])

nonzero_pct_dc = np.stack([
    100
    * dc_counts_by_subject[subject]
    / dc_n_sources[subject]
    for subject in subjects_nonzero
])


# Loftus–Masson mean and SEM for Panel B
(mean_nonzero_pct_ec,mean_nonzero_pct_dc,sem_nonzero_pct_ec,sem_nonzero_pct_dc) = loftus_masson_timecourse(
    nonzero_pct_ec,nonzero_pct_dc)


times_l1 = times_ec.copy()

if np.max(np.abs(times_l1)) < 10:
    times_l1 = times_l1 * 1000

times_nonzero = reference_times.copy()

if np.max(np.abs(times_nonzero)) < 10:
    times_nonzero = times_nonzero * 1000


if len(times_l1) != len(mean_ec):
    raise ValueError(
        "L1 time axis and L1 values have "
        "different lengths."
    )

if len(times_nonzero) != len(mean_nonzero_pct_ec):
    raise ValueError(
        "Non-zero time axis and non-zero values have "
        "different lengths."
    )

# Combined figure
fig, axes = plt.subplots(2,1, figsize=(8, 8), constrained_layout=True)

# Panel A: L1 norm
plot_l1( axes[0], times_l1, mean_ec, sem_ec,  mean_dc, sem_dc)

# Panel B: non-zero contrast sources
plot_nonzero_sources( axes[1], times_nonzero, mean_nonzero_pct_ec, sem_nonzero_pct_ec, 
                      mean_nonzero_pct_dc,sem_nonzero_pct_dc)


axes[0].set_title("A) Contrast Model Size",loc="center")
axes[1].set_title("B) Non-zero Contrast Sources", loc="center")

for ax in axes:
    ax.set_xlim(0, 700)
    for spine in ax.spines.values():
        spine.set_visible(True)
axes[1].set_ylim(bottom=0)
axes[1].legend(loc="lower right")

fig.savefig(fig_dir / "sparsity.pdf")
plt.show()


# %%

# %%
