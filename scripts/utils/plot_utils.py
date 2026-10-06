"""
Plotting utilities.

Includes
--------
- figure_style / set_seaborn : Standardized plot styles (IBL-like, print-friendly).
- create_slice_org_axes      : Slice-organized brain region layout for multi-panel plots.
- break_xaxis                : Draw discontinuous axis markers.
- map_p_value                : Convert p-values to formatted string.
- add_window_label           : Draw labeled horizontal bars for analysis windows.
"""
import seaborn as sns
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ibl_style.utils import get_coords, MM_TO_INCH, double_column_fig
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.transforms as mtransforms


def figure_style():
    """
    Apply IBL-style plotting defaults for small scientific figures.
    - Uses Arial font, thin axes, small ticks.
    - Intended for multi-panel journal figures.
    """
    sns.set_theme(style="ticks", context="paper",
            rc={"font.size": 7,
                "axes.titlesize": 8,
                "axes.labelsize": 7,
                "axes.linewidth": 0.5,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "legend.title_fontsize": 7,
                "lines.linewidth": 1,
                "lines.markersize": 4,
                "xtick.labelsize": 6,
                "ytick.labelsize": 6,
                "savefig.transparent": False,
                "xtick.major.size": 2.5,
                "ytick.major.size": 2.5,
                "xtick.major.width": 0.5,
                "ytick.major.width": 0.5,
                "xtick.minor.size": 2,
                "ytick.minor.size": 2,
                "xtick.minor.width": 0.5,
                "ytick.minor.width": 0.5,
                "axes.labelcolor": "black",
                "text.color": "black",
                "xtick.color": "black",
                "ytick.color": "black",
                "axes.edgecolor": "black",
                })
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['font.family'] = 'Arial'


def create_slice_org_axes_17panels(fg, MM_TO_INCH, fig=None):
    """
    Create a slice-organized layout of brain regions (IBL atlas standard).

    Parameters
    ----------
    fg : module
        Figure grid utility (e.g. figrid).
    MM_TO_INCH : float
        Conversion factor from mm to inches.
    fig : matplotlib.Figure or None
        If None, create a new double-column figure.

    Returns
    -------
    fig : matplotlib.Figure
    axs : dict
        Mapping of region name → Axes object.
    """
    if fig is None:
        fig = double_column_fig()

    width, height = fig.get_size_inches() / MM_TO_INCH

    xspans = get_coords(width, ratios=[1, 1, 1, 1], space=20, pad=5, span=(0, 1))
    yspans = get_coords(height, ratios=[1, 1, 1, 1, 1], space=10, pad=5, span=(0, 0.6))

    layout = {
        'MOs': (0, 0), 'ACA': (0, 1), 'CP': (0, 2), 'LS': (0, 3), 'ACB': (0, 4),
        'mPFC': (1, 2), 'ORB': (1, 3), 'OLF': (1, 4),
        'VISp': (2, 1), 'VISpm': (2, 2), 'SCm': (2, 3), 'MBm': (2, 4),
        'PPC': (3, 0), 'CA1': (3, 1), 'DG': (3, 2), 'LP': (3, 3), 'PO': (3, 4),
    }

    axs = {
        region: fg.place_axes_on_grid(fig, xspan=xspans[xi], yspan=yspans[yi])
        for region, (xi, yi) in layout.items()
    }

    return fig, axs


def break_xaxis(y=0, **kwargs):
    """
    Draw visual markers for discontinuous x-axis (hacky overlay).
    Places small // markers near ±30.
    """
    # axisgate: show axis discontinuities with a quick hack
    # https://twitter.com/StevenDakin/status/1313744930246811653?s=19
    # first, white square for discontinuous axis
    plt.text(-30, y, '-', fontsize=14, fontweight='bold',
             horizontalalignment='center', verticalalignment='center',
             color='w')
    plt.text(30, y, '-', fontsize=14, fontweight='bold',
             horizontalalignment='center', verticalalignment='center',
             color='w')

    # put little dashes to cut axes
    plt.text(-30, y, '/ /', horizontalalignment='center',
             verticalalignment='center', fontsize=6, fontweight='bold')
    plt.text(30, y, '/ /', horizontalalignment='center',
             verticalalignment='center', fontsize=6, fontweight='bold')


def format_p_value(p_value):
    if p_value < 0.001:
        return r"< 0.001***"
    elif p_value < 0.005:
        return r"< 0.005**"
    elif p_value < 0.01:
        return r"< 0.01*"
    return f"= {p_value:.3f}"


def add_window_label(ax, x_start, x_end, label, *,
                     location='outside',               
                     line_pad=0.02,                    
                     text_pad=0.015,                   
                     lw=1, fontsize=7):
    """
    Draw a horizontal line above/below an interval and annotate it.

    Parameters
    ----------
    ax : matplotlib.Axes
    x_start, x_end : float
        Interval in data coordinates.
    label : str
        Window label.
    location : {'inside','outside'}
        Place inside or above axis.
    """
    # y：inside:1 - line_pad，outside:1 + line_pad
    if location == 'inside':
        y_line = 1.0 - line_pad
    else:
        y_line = 1.0 + line_pad

    # x -> data, y -> axes
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

    ax.plot([x_start, x_end], [y_line, y_line],
            transform=trans, color='k', lw=lw, clip_on=False)

    # txt
    # label_fmt = rf'$\it{{{label}}}$'
    label_fmt = label
    ax.text((x_start + x_end)/2, y_line + (text_pad if location=='outside' else -text_pad),
            label_fmt, transform=trans, ha='center',
            va=('bottom' if location=='outside' else 'top'),
            fontsize=fontsize, clip_on=False)
