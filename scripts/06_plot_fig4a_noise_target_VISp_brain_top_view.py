""" 
Figure 4a: Top view of mouse brain with VISp and Liu et al. target locations

"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from iblatlas.atlas import AllenAtlas
from iblatlas.plots import plot_scalar_on_slice
import config as C

ba = AllenAtlas()

# VISp in Beryl mapping
acronyms = np.array(["VISp"])
values = np.array([0.6])

fig, ax = plt.subplots(figsize=(1.7, 2.3))
plot_scalar_on_slice(
    acronyms,
    values,
    slice="top",
    mapping="Beryl",
    hemisphere="both",
    background="boundary",
    cmap="Greys",
    brain_atlas=ba,
    clevels=[0, 1],
    show_cbar=False,
    ax=ax
)

# Liu et al. target: AP = -3.5 mm, ML = ±2.5 mm
ap_um = -3500
ml_left_um = -2500
ml_right_um = 2500

ax.scatter(
    [ml_left_um, ml_right_um],
    [ap_um, ap_um],
    s=8,
    c="black",
    marker="x",
    linewidths=0.7,
    zorder=5
)

# ax.set_title("Top view: VISp and Liu et al. V1 target", fontsize=6)
ax.set_xlabel("ML (μm)", fontsize=5)
ax.set_ylabel("AP (μm)", fontsize=5)
ax.tick_params(axis='both', labelsize=4)
for spine in ax.spines.values():
    spine.set_linewidth(0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
fig.savefig(C.FIGPATH / "VISp_top_view_with_Liu_targets.pdf", dpi=300)