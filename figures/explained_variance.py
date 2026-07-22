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
from eelbrain import load,Var
import pandas as pd
import seaborn as sns
from eelbrain.test import WilcoxonSignedRank

# %%
# Data locations
base_dir = Path("/Users/maryamvalian/Code/imgRECOG/models/samesize/1session")

# Save figure
fig_dir = Path("figures")
fig_dir.mkdir(parents=True, exist_ok=True)


# Configure the matplotlib figure style
FONT = 'Arial'
FONT_SIZE = 16
RC = {
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.transparent': True,
    'savefig.bbox': "tight",
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


test_data = np.column_stack([
    DC_EV_test,
    EC_EV_test,
])

train_data = np.column_stack([
    DC_EV_train,
    EC_EV_train,
])


if np.any(DC_EV_train <= 0) or np.any(EC_EV_train <= 0):
    raise ValueError("Training EV contains zero or negative values.")

if np.any(DC_EV_test <= 0) or np.any(EC_EV_test <= 0):
    raise ValueError("Cross-validated EV contains zero or negative values.")

train_ratio = EC_EV_train / DC_EV_train
test_ratio = EC_EV_test / DC_EV_test

train_log = np.log(train_ratio)
test_log = np.log(test_ratio)

# Non-parametric statistical test

ev_test_result = WilcoxonSignedRank( Var(EC_EV_test), Var(DC_EV_test), tail=0)
ev_train_result =  WilcoxonSignedRank( Var(EC_EV_train), Var(DC_EV_train), tail=0)
print(f" Test data: {ev_test_result},\n Train data {ev_train_result}")

# %%
# Plot
ev_plot_df = pd.DataFrame({
    "Subject": np.tile(df_wide["Subject"].to_numpy(), 4),

    "Dataset": (
        ["Training"] * len(df_wide)
        + ["Training"] * len(df_wide)
        + ["Cross-validated"] * len(df_wide)
        + ["Cross-validated"] * len(df_wide)
    ),

    "Coding": (
        ["DC"] * len(df_wide)
        + ["EC"] * len(df_wide)
        + ["DC"] * len(df_wide)
        + ["EC"] * len(df_wide)
    ),

    "EV": np.concatenate([
        DC_EV_train,
        EC_EV_train,
        DC_EV_test,
        EC_EV_test,
    ]),
})


plot_df = pd.DataFrame({
    "log_ratio": np.concatenate([
        train_log,
        test_log,
    ]),

    "Dataset": (
        ["Training"] * len(train_log)
        + ["Cross-validated"] * len(test_log)
    ),
})


coding_palette = {
    "DC": "#ff7f0e",
    "EC": "#1f77b4",
}

dataset_palette = {
    "Training": "#66C56C",
    "Cross-validated": "#E76F51",
}

datasets = [
    "Training",
    "Cross-validated",
]

# Combined figure
fig, axes = plt.subplots(
    1,
    2,
    figsize=(8, 6),
    gridspec_kw={
        "width_ratios": [1.4, 1],
        "wspace": 0.28,
    },
)

ax_ev = axes[0]
ax_ratio = axes[1]

# Panel A --------------

sns.violinplot(
    data=ev_plot_df,
    x="Dataset",
    y="EV",
    hue="Coding",
    order=datasets,
    hue_order=["DC", "EC"],
    palette=coding_palette,
    inner="box",
    cut=0.0,
    linewidth=0.7,
    saturation=0.8,
    split=True,
    ax=ax_ev,
    legend=True,
    gap=0.08,
    inner_kws=dict(
        box_width=6,      
        whis_width=1.5,      
        color="black",
    ),
)


offset = 0.2 # horizontal space between ec and dc subject points
rng = np.random.default_rng(42)

for group_position, dataset in enumerate(datasets):

    sub_df = ev_plot_df.loc[
        ev_plot_df["Dataset"] == dataset
    ]

    wide = sub_df.pivot(
        index="Subject",
        columns="Coding",
        values="EV",
    )

    x_dc = group_position - offset
    x_ec = group_position + offset

    
    # Paired subject lines
    for _, row in wide.iterrows():

        ax_ev.plot(
            [x_dc, x_ec],
            [row["DC"], row["EC"]],
            color="0.30",
            alpha=0.70,
            linewidth=0.8,
            zorder=3,
        )

    
    # subject points
    dc_jitter = rng.uniform(
        -0.018,
        0.018,
        len(wide),
    )

    ec_jitter = rng.uniform(
        -0.018,
        0.018,
        len(wide),
    )

    ax_ev.scatter(
        x_dc + dc_jitter,
        wide["DC"].to_numpy(),
        color=coding_palette["DC"],
        edgecolor="white",
        linewidth=0.4,
        s=30,
        zorder=4,
    )

    ax_ev.scatter(
        x_ec + ec_jitter,
        wide["EC"].to_numpy(),
        color=coding_palette["EC"],
        edgecolor="white",
        linewidth=0.4,
        s=30,
        zorder=4,
    )

ax_ev.set_xlabel("")
ax_ev.set_ylabel("Explained Variance")

ax_ev.set_xticks([0, 1])
ax_ev.set_xticklabels([
    "Training",
    "Cross-validated",
])

ax_ev.set_xlim(-0.55, 1.55)

# edit legend
handles, labels = ax_ev.get_legend_handles_labels()

ax_ev.legend(
    handles[:2],
    ["NCRF-DC", "NCRF-EC"],
    title=None,
    frameon=False,
    loc="upper right",
    fontsize=FONT_SIZE-3
)

# Panel B
sns.swarmplot(
    data=plot_df,
    x="Dataset",
    y="log_ratio",
    hue="Dataset",
    order=datasets,
    palette=dataset_palette,
    size=7,
    legend=False,
    ax=ax_ratio,
)


# Median lines
for i, dataset in enumerate(datasets):

    median_value = plot_df.loc[
        plot_df["Dataset"] == dataset,
        "log_ratio",
    ].median()

    ax_ratio.hlines(
        median_value,
        i - 0.30,
        i + 0.30,
        colors="black",
        linewidth=1.5,
        zorder=4,
    )


# Equal-EV reference
ax_ratio.axhline(
    0,
    linestyle="--",
    color="gray",
    linewidth=1,
    zorder=1,
)


# Panel B formatting
ax_ratio.set_ylabel(
    r"$\ln(\mathrm{EV}_{\mathrm{EC}}/"
    r"\mathrm{EV}_{\mathrm{DC}})$", labelpad=-11,fontsize= FONT_SIZE-1
)

ax_ratio.set_xlabel("")

ax_ratio.set_xticks([0, 1])

ax_ratio.set_xticklabels([
    "Training",
    "Cross-\nvalidated",
])


# Panel labels
ax_ev.text( 0, 1.07, "(A)", transform=ax_ev.transAxes, fontsize=FONT_SIZE,  va="top")
ax_ratio.text( 0, 1.07,  "(B)", transform=ax_ratio.transAxes, fontsize=FONT_SIZE, va="top")

fig.savefig(fig_dir / "EV.pdf")

plt.show()
