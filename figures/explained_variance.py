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
dummy_dir = base_dir / "dummy"
effect_dir = base_dir / "effect"

# Where to save the figure
fig_dir = Path("figures")
fig_dir.mkdir(parents=True, exist_ok=True)
fig_file = fig_dir / "EV_diff_train400test400.png"

# Configure figure style
FONT = "Arial"
FONT_SIZE = 14
RC = {
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.transparent": True,
    "font.family": "sans-serif",
    "font.sans-serif": FONT,
    "font.size": FONT_SIZE,
    "figure.labelsize": FONT_SIZE,
    "figure.titlesize": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
}
plt.rcParams.update(RC)


# Load EV values through all 30 subjects
rows= []
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
        "Training EV": ev["effect"]["train"] - ev["dummy"]["train"],
        "Cross-validated EV": ev["effect"]["test"] - ev["dummy"]["test"],
    })


# Make dataframe for plotting
df_wide = pd.DataFrame(rows)

df = df_wide.melt(
    id_vars="Subject",
    value_vars=["Training EV", "Cross-validated EV"],
    var_name="Group",
    value_name="Difference",
)


# Plot
plt.figure(figsize=(5, 6))

sns.swarmplot(
    data=df,
    x="Group",
    y="Difference",
    size=8,
    hue="Group",
    palette={
        "Training EV": "#74C476",
        "Cross-validated EV": "#E76F51",
    },
    legend=False,
)

plt.axhline(0, color="k", linestyle="--", alpha=0.5, linewidth=1)
plt.ylabel("EV Difference (EC - DC)")
plt.xlabel("")
plt.title("Subject-wise EV Differences")
plt.tight_layout()

plt.savefig(fig_file, bbox_inches="tight")
plt.show()

# %%
