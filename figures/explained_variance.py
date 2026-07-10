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
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from eelbrain import load
import pandas as pd
import seaborn as sns


# %%
# Data locations
base_dir = Path("/Users/maryamvalian/Code/imgRECOG/models/samesize/1session")

# Save figure
fig_dir = Path("figures")
fig_dir.mkdir(parents=True, exist_ok=True)
fig_file = fig_dir / "EV_paired_train_test.png"

# Configure the matplotlib figure style
FONT = 'Arial'
FONT_SIZE = 14
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

# Load EV values
rows = []

for i in range(1, 31):
    subject = f"sub-{i:02d}"

    ev = {}
    for mod in ("dummy", "effect"):
        m1, m2 = [
            load.unpickle(base_dir / mod / f"{chunk}-2-{subject}.pickle")
            for chunk in (1, 2)
        ]

        ev[mod] = {
            "train": m1.explained_var,
            "test": m1.compute_explained_variance(m2._data),
        }

    rows.append({
        "Subject": subject,
        "DC Training EV": ev["dummy"]["train"],
        "EC Training EV": ev["effect"]["train"],
        "DC Cross-validated EV": ev["dummy"]["test"],
        "EC Cross-validated EV": ev["effect"]["test"],
    })

df_wide = pd.DataFrame(rows)

# Arrays
DC_EV_train = df_wide["DC Training EV"].to_numpy()
EC_EV_train = df_wide["EC Training EV"].to_numpy()
DC_EV_test = df_wide["DC Cross-validated EV"].to_numpy()
EC_EV_test = df_wide["EC Cross-validated EV"].to_numpy()


# %%
from scipy.stats import t


def loftus_masson_ci(data, confidence=0.95):
    
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

# %%
# CI: Uncertainty of the mean after removing between-subject variability 


test_data = np.column_stack([
    DC_EV_test,
    EC_EV_test,
])

train_data = np.column_stack([
    DC_EV_train,
    EC_EV_train,
])

means_train, sem_within_train, ci_train = loftus_masson_ci(train_data)
means_test, sem_within_test, ci_test = loftus_masson_ci(test_data)


# %%
#plot 

fig, axes = plt.subplots(
    1,
    2,
    figsize=(8, 5),
    sharey=True,
)

# Find common limits across train and test
all_lower = [
    means_train[0] - ci_train,
    means_train[1] - ci_train,
    means_test[0] - ci_test,
    means_test[1] - ci_test,
]

all_upper = [
    means_train[0] + ci_train,
    means_train[1] + ci_train,
    means_test[0] + ci_test,
    means_test[1] + ci_test,
]

ymin = min(all_lower)
ymax = max(all_upper)

padding = 0.05 * (ymax - ymin)

# Training
axes[0].bar(
    ["DC", "EC"],
    means_train,
    yerr=[sem_within_train, sem_within_train],
    capsize=6,
    color=["#ff7f0e", "#1f77b4"],
    edgecolor="black",
)

axes[0].set_ylabel("Explained Variance")
axes[0].set_title("Training\nMean ± (within-subject SEM)")

# Test
axes[1].bar(
    ["DC", "EC"],
    means_test,
    yerr=[sem_within_test, sem_within_test],
    capsize=6,
    color=["#ff7f0e", "#1f77b4"],
    edgecolor="black",
)

axes[1].set_title("Test\nMean ± (within-subject SEM)")

# Same limits for both panels
axes[0].set_ylim(ymin - padding, ymax + padding)
axes[1].set_ylim(ymin - padding, ymax + padding)

plt.tight_layout()
plt.show()

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Compute paired differences
df_diff = pd.DataFrame({
    "Subject": df_wide["Subject"],
    "Training": df_wide["EC Training EV"] - df_wide["DC Training EV"],
    "Cross-validated": df_wide["EC Cross-validated EV"] - df_wide["DC Cross-validated EV"],
})

# Long format
df_plot = df_diff.melt(
    id_vars="Subject",
    var_name="Group",
    value_name="Difference",
)

plt.figure(figsize=(5, 6))

sns.swarmplot(
    data=df_plot,
    x="Group",
    y="Difference",
    hue="Group",
    palette={
        "Training": "#74C476",
        "Cross-validated": "#E76F51",
    },
    size=8,
    legend=False,
)

plt.axhline(0, color="k", linestyle="--", linewidth=1, alpha=0.5)
plt.ylabel("EV Difference (EC − DC)")
plt.xlabel("")
plt.title("Subject-wise EV Differences")

plt.tight_layout()
plt.savefig("figures/EV_difference_swarm.png", dpi=300, bbox_inches="tight")
plt.show()

# %% [markdown]
# # Contribution of Contrast : 
# <br><hr><br><br>

# %%
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Data locations
base_dir = Path("/Users/maryamvalian/Code/imgRECOG/models/samesize/1session")
effect_dir = base_dir / "effect"

# Where to save the figure
fig_dir = Path("figures")
fig_dir.mkdir(parents=True, exist_ok=True)
fig_file = fig_dir / "EC_contrast_RF_contribution_train400test400.png"

rows = []

for i in range(1, 31):
    subject = f"sub-{i:02d}"

    # chunk 1 model, chunk 2 test data/model
    m1, m2 = [
        load.unpickle(effect_dir / f"{chunk}-2-{subject}.pickle")
        for chunk in (1, 2)
    ]

    # ---------- Full EC EV ----------
    ev_full_train = m1.explained_var
    ev_full_test = m1.compute_explained_variance(m2._data)

    # ---------- General-only EC ----------
    m1_general_only = copy.deepcopy(m1)

    # theta columns are [general coefficients | contrast coefficients]
    n_cols = m1_general_only.theta.shape[1] // 2

    # set contrast RF coefficients to zero
    m1_general_only.theta[:, n_cols:] = 0

    ev_general_train = m1_general_only.compute_explained_variance(m1._data)
    ev_general_test = m1_general_only.compute_explained_variance(m2._data)

    rows.append({
        "Subject": subject,

        "Full EC Training EV": ev_full_train,
        "General-only Training EV": ev_general_train,
        "Contrast Contribution Training": ev_full_train - ev_general_train,

        "Full EC Cross-validated EV": ev_full_test,
        "General-only Cross-validated EV": ev_general_test,
        "Contrast Contribution Cross-validated": ev_full_test - ev_general_test,
    })


# DataFrame
df_ec = pd.DataFrame(rows)

display(df_ec)

# Long format for plot
df_plot = df_ec.melt(
    id_vars="Subject",
    value_vars=[
        "Contrast Contribution Training",
        "Contrast Contribution Cross-validated",
    ],
    var_name="Group",
    value_name="EV Contribution",
)

# Plot
plt.figure(figsize=(5, 6))

sns.swarmplot(
    data=df_plot,
    x="Group",
    y="EV Contribution",
    size=8,
    hue="Group",
    palette={
        "Contrast Contribution Training": "#74C476",
        "Contrast Contribution Cross-validated": "#E76F51",
    },
    legend=False,
)

plt.axhline(0, color="k", linestyle="--", alpha=0.5, linewidth=1)
plt.ylabel("EV_fullModel - EV_generalOnly")
plt.xlabel("")
plt.title("EC: Contribution of Animacy Contrast RF")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(fig_file, bbox_inches="tight")
plt.show()

print("Mean training contribution:",
      df_ec["Contrast Contribution Training"].mean())

print("Mean cross-validated contribution:",
      df_ec["Contrast Contribution Cross-validated"].mean())

# %%
df_general_plot = df_ec.melt(
    id_vars="Subject",
    value_vars=[
        "General-only Training EV",
        "General-only Cross-validated EV",
    ],
    var_name="Group",
    value_name="EV",
)

plt.figure(figsize=(5, 6))

sns.swarmplot(
    data=df_general_plot,
    x="Group",
    y="EV",
    size=8,
    hue="Group",
    palette={
        "General-only Training EV": "#74C476",
        "General-only Cross-validated EV": "#E76F51",
    },
    legend=False,
)

plt.axhline(0, color="k", linestyle="--", alpha=0.5, linewidth=1)
plt.ylabel("EV")
plt.xlabel("")
plt.title("EC: General Visual Response EV")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(fig_dir / "EC_general_only_EV_train_test.png", bbox_inches="tight")
plt.show()

print("Mean general-only training EV:",
      df_ec["General-only Training EV"].mean())

print("Mean general-only cross-validated EV:",
      df_ec["General-only Cross-validated EV"].mean())

# %%
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

base_dir = Path("/Users/maryamvalian/Code/imgRECOG/models/samesize/1session")
effect_dir = base_dir / "effect"

fig_dir = Path("figures")
fig_dir.mkdir(parents=True, exist_ok=True)

rows = []

for i in range(1, 31):
    subject = f"sub-{i:02d}"

    m1, m2 = [
        load.unpickle(effect_dir / f"{chunk}-2-{subject}.pickle")
        for chunk in (1, 2)
    ]

    n_cols = m1.theta.shape[1] // 2

    # Full model
    ev_full_train = m1.explained_var
    ev_full_test = m1.compute_explained_variance(m2._data)

    # General-only model: remove contrast RF
    m_general = copy.deepcopy(m1)
    m_general.theta[:, n_cols:] = 0

    ev_general_train = m_general.compute_explained_variance(m1._data)
    ev_general_test = m_general.compute_explained_variance(m2._data)

    # Contrast-only model: remove general RF
    m_contrast = copy.deepcopy(m1)
    m_contrast.theta[:, :n_cols] = 0

    ev_contrast_train = m_contrast.compute_explained_variance(m1._data)
    ev_contrast_test = m_contrast.compute_explained_variance(m2._data)

    rows.append({
        "Subject": subject,

        "Full Training EV": ev_full_train,
        "General-only Training EV": ev_general_train,
        "Contrast-only Training EV": ev_contrast_train,

        "Full Cross-validated EV": ev_full_test,
        "General-only Cross-validated EV": ev_general_test,
        "Contrast-only Cross-validated EV": ev_contrast_test,

        "Drop from removing contrast Training": ev_full_train - ev_general_train,
        "Drop from removing contrast Cross-validated": ev_full_test - ev_general_test,
    })

df_ec_parts = pd.DataFrame(rows)
display(df_ec_parts)

# %%
df_plot = df_ec_parts.melt(
    id_vars="Subject",
    value_vars=[
        "Full Training EV",
        "General-only Training EV",
        "Contrast-only Training EV",
        "Full Cross-validated EV",
        "General-only Cross-validated EV",
        "Contrast-only Cross-validated EV",
    ],
    var_name="Condition",
    value_name="EV",
)

plt.figure(figsize=(8, 14))

sns.swarmplot(
    data=df_plot,
    x="Condition",
    y="EV",
    hue="Condition",
    size=7,
    legend=False,
)

plt.axhline(0, color="k", linestyle="--", alpha=0.5, linewidth=1)
plt.ylabel("EV")
plt.xlabel("")
plt.title("EC: Full vs General-only vs Contrast-only EV")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()

plt.savefig(fig_dir / "EC_full_general_contrast_EV.png", bbox_inches="tight")
plt.show()

# %%

# %%

# %%

# %%
