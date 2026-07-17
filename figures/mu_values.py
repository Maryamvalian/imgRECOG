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
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from eelbrain import load
from eelbrain import Var
from eelbrain.test import WilcoxonSignedRank
import pandas as pd
import seaborn as sns

# %%
# Data locations
base_dir = Path("../models/all_runs")
dc_dir = base_dir / "ncrf-dc"
ec_dir = base_dir / "ncrf-ec" 

# Where to save the figure
fig_dir = Path("figures")
fig_dir.mkdir(parents=True, exist_ok=True)
fig_file = fig_dir / "mu.png"

# Configure figure style
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
    "legend.fontsize": FONT_SIZE,
}
plt.rcParams.update(RC)


def get_subject_mu(model_dir):
    mus = []

    for i in range(1, 31):
        subject = f"sub-{i:02d}"
        session = "ImageNet01" if i > 9 else "ImageNet03"
        model_file = model_dir / f"{subject}-{session}-ncrf.pickle"

        try:
            model = load.unpickle(model_file)
            mus.append(model.mu)
        except Exception as e:
            mus.append(np.nan)
            print(f"Could not load {model_file}: {e}")

    return np.array(mus)


# Load mu values
mu_dc = get_subject_mu(dc_dir)
mu_ec = get_subject_mu(ec_dir)

# Mu Ratio
mu_ratio = mu_ec / mu_dc

# Log trnasform of ratio
log_ratio = np.log(mu_ratio)

# %%
# Plot Mu values + log transformed Mu ratio
x = np.arange(1, 31)

# Sort subjects by EC mu values 
order = np.argsort(mu_ec)
mu_ec_sorted = mu_ec[order]
mu_dc_sorted = mu_dc[order]

df = pd.DataFrame({
    "Model": ["Subjects"] * len(log_ratio),
    "Ratio": log_ratio,
})

fig, axes = plt.subplots(1, 2, figsize=(8, 7), width_ratios=[1.3, 1])

# Plot Mu values
ax = axes[0]

ax.plot(mu_ec_sorted, x, "^", label="NCRF-EC")
ax.plot(mu_dc_sorted, x, "s", label="NCRF-DC")

ax.set_xlabel(r"$\mu$ Value")
ax.set_ylabel("Subjects")

ax.set_xscale("log")
ax.grid(True, alpha=0.4)

ax.legend(loc="upper right")

ax.set_yticks(np.arange(1, 31))
ax.set_yticklabels([])
ax.invert_yaxis()

ax.set_title("A)")

# Plot Mu ratio
ax = axes[1]

sns.swarmplot(
    data=df,
    x="Model",
    y="Ratio",
    size=8,
    ax=ax,
    color="green"
)

ax.axhline(0, color="red", linestyle="--", linewidth=1.5)

ax.set_ylabel(r"$\ln(\mu_{EC}/\mu_{DC})$")


ax.grid(axis="y", alpha=0.3)
ax.set_xlabel("Subjects")
ax.set_xticklabels([])

ax.set_title("B)")

plt.tight_layout()
plt.savefig(fig_dir / "mu.pdf")
plt.show()

# %% [markdown]
# # Statistical non-parametric test : Wilcoxon

# %%
# W = 0: There was not a single subject whose EC mu was lower than DC mu.
# eelbrain report smallest rank as W
# Null hypothesis: mu_ec= mu-dc : There is no systematic difference between paired mu values (ec,dc)


# Convert NumPy arrays to Eelbrain Var objects
mu_ec_var = Var(mu_ec, name="mu_ec")
mu_dc_var = Var(mu_dc, name="mu_dc")

mu_test = WilcoxonSignedRank(
    mu_ec_var,
    mu_dc_var,
    tail=0,
)

print(
    f"Wilcoxon signed-rank test comparing mu_EC and mu_DC: "
    f"W = {mu_test.w:.2f}, p = {mu_test.p:.3e}"
)

# %%

# %%
