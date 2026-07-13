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
from scipy.stats import ttest_1samp

import pandas as pd
import seaborn as sns


# %%

# Data locations
base_dir = Path("../models/all_runs")
dc_dir = base_dir / "ncrf-dc"
ec_dir = base_dir / "ncrf-ec" / "reduce"

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


# Plot
x = np.arange(1, 31)

plt.figure(figsize=(12, 5))

plt.plot(x, mu_ec, "^", label="NCRF-EC")
plt.plot(x, mu_dc, "s", label="NCRF-DC")

plt.xticks(x)
plt.xlabel("Subject")
plt.ylabel("Mu Values")

plt.yscale("log")
plt.grid(True, alpha=0.4)

plt.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
)

plt.tight_layout()
plt.savefig(fig_file)
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
# Mu Ratio
mu_ratio = mu_ec / mu_dc

# Log trnasform of ratio
log_ratio = np.log(mu_ratio)
#t-test on log-transformed ratio
t, p = ttest_1samp(log_ratio, popmean=0)           #muEC=muDC => mu ratio=1 log(mu ratio)=0
print(f"t = {t:.3f}, p = {p:.5f}")

# %%
# plot 
x = np.arange(1, 31)
plt.figure(figsize=(12, 6))

plt.plot(
    x,
    mu_ratio,
    "o",
    markersize=10,
    label=r"mu ratio"
)

plt.axhline(
    1,
    linestyle="--",
    color="black",
    label="EC = DC"
)

plt.xticks(x)
plt.xlabel("Subject")
plt.ylabel(r"$\mu_{EC}/\mu_{DC}$")

plt.grid(True, alpha=0.3)

plt.legend()
plt.tight_layout()

#plt.yscale("log")
plt.axhline(1, color="black", linestyle="--")
plt.show()

# %%
# Swarm plot 
df = pd.DataFrame({
    "Model": ["Subjects"] * len(mu_ratio),
    "Ratio": mu_ratio,
})

plt.figure(figsize=(4, 5))

sns.swarmplot(
    data=df,
    x="Model",
    y="Ratio",
    size=8,
)

plt.axhline(1, color="red", linestyle="--", linewidth=1.5)

plt.ylabel(r"Mu Ratio (EC/DC)")
plt.xlabel("")

plt.yscale("log")  

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# %%







# %%
