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

# %%
# Statistical test to reject null hypothesis
# t = 2.62703599152203, p = 0.0136183194390469 < 0.05
from scipy.stats import ttest_rel 
t, p = ttest_rel(mu_ec, mu_dc)                      
print(f"t = {t}, p = {p}")

# %%
