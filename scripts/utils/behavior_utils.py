"""
Utility functions for behavioral preprocessing and metrics.

Functions
---------
- create_trials_table : Extract trials for sessions, add metadata
- clean_rts           : Remove implausible RTs (too fast/slow)
- filter_trials       : Apply trial filters (events, RT cutoff, first 400)
"""
#%%
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import variation
from scipy import stats
from tqdm import tqdm


def create_trials_table(eids, one):
    """
    Extract all trials from the given sessions, add metadata, and return a DataFrame.

    Parameters
    ----------
    eids: Session eids to load from ONE
    one: ONE instance.

    Returns
    -------
    df_trials : pd.DataFrame
        Trial dataframe with subject/session metadata.
    err_list : list
        List of (eid, error) for sessions that failed to load.
    """
    all_trials = []
    err_list = []
    for i, eid in enumerate(tqdm(eids, desc="Processing eids")):
        print(f"[{i+1}/{len(eids)}] Loading {eid}")
        try:
            # Load trials object (prefer revision 2025-03-03 if available)
            try:
                trials_obj = one.load_object(eid, 'trials', revision='2025-03-03')
            except Exception:
                print(f"Revision '2025-03-03' not found for {eid}, loading default revision.")
                trials_obj = one.load_object(eid, 'trials')

            # Add signed contrast and aligned RTs
            trials_obj['signed_contrast'] = 100 * np.diff(np.nan_to_num(np.c_[trials_obj['contrastLeft'], 
                                                        trials_obj['contrastRight']]))
    
            trials_obj["response_times_from_stim"] = trials_obj["response_times"] - trials_obj["stimOn_times"]
            trials_obj["firstMovement_times_from_stim"] = trials_obj["firstMovement_times"] - trials_obj["stimOn_times"]
            
            # Convert to DataFrame
            trials = trials_obj.to_df() 
            trials['trial_index'] = trials.index # to keep track of choice history
            trials['response'] = trials['choice'].map({1: 0, 0: np.nan, -1: 1}) #-1: turning the wheel CCW; +1 turning CW
            trials['eid'] = eid

            # retrieve the mouse name, session etc
            ref_dict = one.eid2ref(eid)
            trials['mouse_name'] = ref_dict.subject
            # sess_date = ref_dict.date
            session_details = one.get_details(eid)
            sess_date = session_details['start_time'][:10]

            try:
                subj = one.alyx.rest('subjects', 'list', nickname=ref_dict.subject)
                subj_dob = subj[0]['birth_date']
                sex = subj[0]['sex']
                age_at_recording = (datetime.strptime(sess_date, '%Y-%m-%d') - datetime.strptime(subj_dob, '%Y-%m-%d')).days
            except Exception:
                age_at_recording, sex = np.nan, np.nan
            trials['mouse_age'] = age_at_recording
            trials['mouse_sex'] = sex
            trials['date'] = ref_dict.date

            all_trials.append(trials)
        except BaseException as e:
            print('Attention: ', eid, e)
            err_list.append((eid, e))

    df_trials = pd.concat(all_trials, ignore_index=True)   
    
    # keep only relevant columns
    df_trials = df_trials[['eid', 'mouse_name', 'mouse_age', 'mouse_sex', 'date', 'signed_contrast', 'probabilityLeft',
                           'goCue_times', 'stimOn_times', 'firstMovement_times', 'response', 'choice','response_times', 'feedback_times',
                           'response_times_from_stim', 'firstMovement_times_from_stim', 'feedbackType', 'trial_index']]

    return df_trials, err_list


def create_trials_table_prior(eids, one):
    """
    Extract all trials from the given sessions, add metadata, and return a DataFrame.

    Parameters
    ----------
    eids: Session eids to load from ONE
    one: ONE instance.

    Returns
    -------
    df_trials : pd.DataFrame
        Trial dataframe with subject/session metadata.
    err_list : list
        List of (eid, error) for sessions that failed to load.
    """
    all_trials = []
    err_list = []
    for i, eid in enumerate(tqdm(eids, desc="Processing eids")):
        print(f"[{i+1}/{len(eids)}] Loading {eid}")
        try:
            # Load trials object (prefer revision 2025-03-03 if available)
            try:
                trials_obj = one.load_object(eid, 'trials', revision='2025-03-03')
            except Exception:
                print(f"Revision '2025-03-03' not found for {eid}, loading default revision.")
                trials_obj = one.load_object(eid, 'trials')

            # Add signed contrast and aligned RTs
            trials_obj['signed_contrast'] = 100 * np.diff(np.nan_to_num(np.c_[trials_obj['contrastLeft'], 
                                                        trials_obj['contrastRight']]))
    
            trials_obj["response_times_from_stim"] = trials_obj["response_times"] - trials_obj["stimOn_times"]
            trials_obj["firstMovement_times_from_stim"] = trials_obj["firstMovement_times"] - trials_obj["stimOn_times"]
            
            # Convert to DataFrame
            trials = trials_obj.to_df() 
            trials['trial_index'] = trials.index # to keep track of choice history
            trials['response'] = trials['choice'].map({1: 0, 0: np.nan, -1: 1}) #-1: turning the wheel CCW; +1 turning CW
            trials['eid'] = eid

            # retrieve the mouse name, session etc
            ref_dict = one.eid2ref(eid)
            trials['mouse_name'] = ref_dict.subject
            # sess_date = ref_dict.date
            session_details = one.get_details(eid)
            sess_date = session_details['start_time'][:10]

            try:
                subj = one.alyx.rest('subjects', 'list', nickname=ref_dict.subject)
                subj_dob = subj[0]['birth_date']
                sex = subj[0]['sex']
                age_at_recording = (datetime.strptime(sess_date, '%Y-%m-%d') - datetime.strptime(subj_dob, '%Y-%m-%d')).days
            except Exception:
                age_at_recording, sex = np.nan, np.nan
            trials['mouse_age'] = age_at_recording
            trials['mouse_sex'] = sex
            trials['date'] = ref_dict.date

            all_trials.append(trials)
        except BaseException as e:
            print('Attention: ', eid, e)
            err_list.append((eid, e))

    df_trials = pd.concat(all_trials, ignore_index=True)   
    
    # keep only relevant columns
    df_trials = df_trials[['eid', 'mouse_name', 'mouse_age', 'mouse_sex', 'date','contrastLeft', 'contrastRight', 'signed_contrast', 'probabilityLeft',
                           'goCue_times', 'stimOn_times', 'firstMovement_times', 'response', 'choice','response_times', 'feedback_times',
                           'response_times_from_stim', 'firstMovement_times_from_stim', 'feedbackType', 'trial_index']]

    return df_trials, err_list


def clean_rts(rt, cutoff=[0.08, 2]):
    """
    Clean reaction times by removing outliers (too fast/slow).

    Parameters
    ----------
    rt : pd.Series or np.ndarray
        Raw RT values.
    cutoff : [low, high]
        Acceptable range in seconds.

    Returns
    -------
    rt_clean : np.ndarray
        Cleaned RT values (invalid set to NaN).
    """
    assert (0 < np.nanmedian(rt) < 3) # median RT should be within some reasonable bounds

    print('cleaning RTs...')
    # remove RTs below and above cutoff
    rt_clean = rt.copy()
    low, high = cutoff

    if low is not None:
        rt_clean[rt_clean < low] = np.nan

    if high is not None:
        rt_clean[rt_clean > high] = np.nan

    return rt_clean


def filter_trials(trials_df, exclude_nan_event_trials=True, trial_type ='first400', event_list=None, 
                  clean_rt=True, rt_variable=None,rt_cutoff=None):
    """
    Apply standard trial filters: drop missing events, subset trials, apply RT cutoff.

    Returns
    -------
    pd.DataFrame
        Filtered trial-level data with 'rt' and 'rt_raw' columns.
    """
    if exclude_nan_event_trials == True:
        trials_df['exclude_nan_event_mask'] = np.where(trials_df[event_list].notna().all(axis=1), 1, 0) # if 都非空，则1，否则0
        trials_df = trials_df.loc[trials_df['exclude_nan_event_mask']==1]
    if trial_type == 'first400':
        # groupby subject, pick only the first 400 trials 
        trials_df = trials_df.groupby('eid').head(400).reset_index(drop=True)
    elif trial_type == 'all':
        trials_df = trials_df.reset_index(drop=True)
    trials_df['rt_raw'] = trials_df[rt_variable].copy()
    trials_df['rt'] = clean_rts(trials_df[rt_variable], cutoff=rt_cutoff)    
    if clean_rt:# remove RTs that sit outside the cutoff window 
        trials_df = trials_df[~trials_df['rt'].isna()]
    return trials_df


def compute_choice_history(trials):
    """
    Add choice history columns (previous/next response, feedback, contrast).

    Parameters
    ----------
    trials : pd.DataFrame
        Must contain 'response', 'feedbackType', 'signed_contrast', 'trialnum'.

    Returns
    -------
    pd.DataFrame
        With extra columns for prev/next resp/fb/contrast.
    """
    print('adding choice history columns to database...')

    # append choice history 
    trials['prevresp'] = trials.response.shift(1)
    trials['prevfb'] = trials.feedbackType.shift(1)
    trials['prevcontrast'] = np.abs(trials.signed_contrast.shift(1))

    # also append choice future 
    trials['nextresp'] = trials.response.shift(-1)
    trials['nextfb'] = trials.feedbackType.shift(-1)
    trials['nextcontrast'] = np.abs(trials.signed_contrast.shift(-1))

    # remove when not consecutive based on trial_index
    trials_not_consecutive = (trials.trialnum - trials.trialnum.shift(1)) != 1.
    for col in ['prevresp', 'prevfb', 'prevcontrast']:
        trials.loc[trials_not_consecutive, col] = np.nan

    return trials
