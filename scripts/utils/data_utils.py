"""
Utility functions for data handling, shuffling, statistics, and age labeling.

Functions
---------
- load_filtered_recordings : Load list of sessions/probes after QC.
- normalize_units          : Min-max normalize neural data across units.
- add_age_group            : Add categorical age groups (young/old) and scaled age.
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import os
import pingouin as pg
import math
import config as C


def load_filtered_recordings(datapath=C.DATAPATH, filename = 'BWM_LL_release_afterQC_df.csv'):
    """
    Load a pre-filtered recordings table (after QC).

    Parameters
    ----------
    datapath : Path
        Base path to data folder (default: C.DATAPATH).
    filename : str
        CSV file with filtered sessions/probes.

    Returns
    -------
    recordings_filtered : pd.DataFrame
        Loaded DataFrame.
    """
    try:
        recordings_filtered = pd.read_csv( datapath / f"{filename}" )
    except Exception as err:
        print(f'errored: {err}')
        recordings_filtered =np.nan
    return recordings_filtered


def normalize_units(matrix):
    """
    Normalize neural activity per unit (min-max scaling).

    Parameters
    ----------
    matrix : np.ndarray
        Shape (trials, units, timepoints).

    Returns
    -------
    result : np.ndarray
        Normalized array with same shape, values in [0, 1].
    """
    trials, units, timepoints = matrix.shape
    result = np.zeros_like(matrix)
    
    for i in range(units):
        data = matrix[:, i, :].flatten()
        min_val, max_val = np.min(data), np.max(data)

        # max-min normalization if max_val> min_val
        if max_val > min_val:
            result[:, i, :] = (matrix[:, i, :] - min_val) / (max_val - min_val)

        else:
            result[:, i, :] = np.zeros_like(matrix[:, i, :]) 
    
    return result


def add_age_group(df):
    """
    Add age-related columns: categorical group + months + years.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'mouse_age' (days) or 'age_at_recording' (days).

    Returns
    -------
    pd.DataFrame
        Copy with new columns:
        - 'age_group' (young/old, by C.AGE_GROUP_THRESHOLD in days)
        - 'age_months'
        - 'age_years'
    """
    out = df.copy()

    if 'mouse_age' in out.columns:
        age = out['mouse_age']
    elif 'age_at_recording' in out.columns:
        age = out['age_at_recording']
    else:
        raise KeyError("Expected 'mouse_age' or 'mouse_Age_at_recording' in DataFrame.")

    out['age_group'] = (age > C.AGE_GROUP_THRESHOLD).map({True: 'old', False: 'young'})
    out['age_months'] = age / 30
    out['age_years'] = age / 365
    
    return out

