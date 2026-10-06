"""
Figure 1b: Age distribution of mice in the dataset.
The distribution is shown for all mice (mouse-based) and for all sessions (session-based).

"""

#%%
# === Imports ===
import config as C
import numpy as np
import pandas as pd
import seaborn as sns
from one.api import ONE
from scripts.utils.plot_utils import figure_style
from scripts.utils.behavior_utils import compute_choice_history
from scripts.utils.data_utils import add_age_group
from scripts.utils.io import save_figure
import matplotlib.ticker as mticker
import logging

log = logging.getLogger(__name__)

one = ONE()
figure_style()

# === Load helper ===
def load_trial_table(filepath):
    """Load trials table and compute choice history."""
    try:
        trials_table = pd.read_csv(filepath)
        log.info(f"{len(set(trials_table['eid']))} sessions loaded")
        trials_table['trialnum'] = trials_table['trial_index']
        trials_table = compute_choice_history(trials_table)
        return trials_table
    except Exception as err:
        log.error(f'errored: {err}')
        return None


#== Panel b: Age distribution ===
def plot_age_distribution(trials_table, save_fig=True, session_based=False):
    """
    Plot age distribution (mouse-based or session-based).

    Parameters
    ----------
    trials_table : pd.DataFrame
        Trial data.
    save_fig : bool
        If True, save figure to FIGPATH.
    session_based : bool
        If True, plot sessions (mean age per eid); else plot mice.
    """

    if session_based:
        age_info = (trials_table.groupby("eid", as_index=False)["mouse_age"].min())
        age_info = add_age_group(age_info)
        xlab = "Age at recording (months)"
        ylab = "Number of sessions"
    else:
        age_info = (trials_table.groupby("mouse_name", as_index=False)["mouse_age"].min())
        age_info = add_age_group(age_info)

        xlab = "Age (months)"
        ylab = "Number of mice"

    # group-level stats for annotation
    stats_df = (age_info.groupby("age_group")["age_months"]
                        .agg(n="count", mean="mean")
                        .reset_index())
    stats_map = {r["age_group"]: r for _, r in stats_df.iterrows()}

    g = sns.displot(
        age_info, x="age_months", hue="age_group",
        hue_order=["young", "old"], binwidth=0.5,
        palette=C.PALETTE, legend=False, height=2.36, aspect=1,
        multiple="stack"
    )
    g.set_axis_labels(xlab, ylab)

    # annotate per-axes (FacetGrid returns a figure-like obj)
    for ax in g.axes.flat:
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        # helpful y ticks for dense hist
        try:
            ymax = max(ax.get_yticks()) if len(ax.get_yticks()) else 0
            step = max(1, int(round(ymax / 5))) if ymax else 1
            ax.set_yticks(np.arange(0, ymax + step, step))
        except Exception:
            pass

        # dynamic text blocks
        if "young" in stats_map:
            st = stats_map["young"]
            txt = f"Young, " + r"$n_{\mathrm{mice}}=" + f"{st['n']}" + r"$" + f"\nM(age)={st['mean']:.2f}"
            ax.text(0.40, 0.80, txt, transform=ax.transAxes, fontsize=7,
                    linespacing=0.8, color=C.PALETTE["young"], va="top")
        if "old" in stats_map:
            st = stats_map["old"]
            txt = f"Old, " + r"$n_{\mathrm{mice}}=" + f"{st['n']}" + r"$" + f"\nM(age)={st['mean']:.2f}"
            ax.text(0.60, 0.40, txt, transform=ax.transAxes, fontsize=7,
                    linespacing=0.8, color=C.PALETTE["old"], va="top")

        sns.despine(offset=2, trim=False, ax=ax)

    if save_fig:
        fname = "f1b_distribution_age_allmice_sessionbased_2025.pdf" if session_based \
                else "f1b_distribution_age_allmice_mousebased_2025.pdf"
        save_figure(g, C.FIGPATH / fname, add_timestamp=True)



trials_table_file = C.DATAPATH / 'ibl_included_eids_trials_table2025_full.csv'
trials_table = load_trial_table(trials_table_file)
trials_table = add_age_group(trials_table)

if trials_table is not None:
    # histplot: age distribution
    plot_age_distribution(trials_table,  save_fig=True, session_based=False)
