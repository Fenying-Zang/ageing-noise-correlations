# config.py
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent

# 关键：让 scripts/ 成为可导入路径，这样旧的 "from scripts.utils..." 不会报错
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


PROJECT_ROOT = Path(__file__).resolve().parent
DATAPATH = PROJECT_ROOT / 'data'
FIGPATH   = PROJECT_ROOT / 'figures'

RESULTSPATH = PROJECT_ROOT / 'results'  # 可有可无，但以后方便

PALETTE = {'young': '#78c679', 'old': '#2c7fb8'}
PALETTE5 = ["#bfb4ca", "#9285a3", "#756388", "#5b496e", "#4B3169"]
PALETTE5_2GROUPS = {
    'old':   ['#c6dbef', '#9ecae1', '#6baed6', '#3182bd', '#08519c'],
    'young': ['#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45'],
}

COLORS_SWANSON = ['#78c679', 'silver', '#2c7fb8']
COLORS_SWANSON_INVERT = ['#2c7fb8', 'silver', '#78c679']

EVENT_LIST = [
    'stimOn_times',
    'choice',
    'feedback_times',
    'probabilityLeft',
    'firstMovement_times',
    'response_times',  # added to BWM setting
    'feedbackType',
]

ROIS = [
    'MOs', 'ACA', 'CP', 'LS', 'ACB', 'mPFC', 'ORB', 'OLF',
    'VISp+pm', 'SCm', 'MBm', 'PPC', 'CA1', 'DG', 'LP', 'PO'
]
ROIS_visp = [
    'MOs', 'ACA', 'CP', 'LS', 'ACB', 'mPFC', 'ORB', 'OLF',
    'VISp', 'SCm', 'MBm', 'PPC', 'CA1', 'DG', 'LP', 'PO'
]
ROIS_vis_seperate = [
    'MOs', 'ACA', 'CP', 'LS', 'ACB', 'mPFC', 'ORB', 'OLF','VISp',
    'VISpm', 'SCm', 'MBm', 'PPC', 'CA1', 'DG', 'LP', 'PO'
]

BERYL_NAMES = [
    'VISa', 'VISam', 'VISp', 'VISpm', 'CA1', 'DG', 'PO',
    'LP', 'APN', 'MRN', 'SCm', 'MOs', 'ACAv', 'ACAd',
    'PL', 'ILA', 'ORBm', 'ORBl', 'ORBvl', 'TTd', 'DP',
    'AON', 'ACB', 'CP', 'LSr', 'LSc', 'LSv'
]

ROIS_RS = ['PPC', 'CA1', 'DG', 'LP', 'PO']
BERYL_NAMES_RS = ['VISa', 'VISam', 'CA1', 'DG', 'LP','PO']

TRIAL_TYPE = 'first400'
RT_VARIABLE_NAME = 'response_times_from_stim'    # 或 'firstMovement_times_from_stim'
RT_CUTOFF = [0.08, 2.0]                           # 80ms, 2s

AGE2USE = 'age_years'
AGE_GROUP_THRESHOLD = 228.3

SAVE_RESULTS = True

BIN_SIZE = 0.1                      # FF 与 FR 目前共用窗口
ALIGN_EVENT = 'stim'                # 'stim' | 'move' | 'feedback'
# EVENT_EPOCH = [-0.4, 0.8]
EVENT_EPOCH = [-0.5, 0.8] #TODO: discuss the epoch range, necessary for noise correlation calculation?
SMOOTHING = 'sliding'
SLIDE_KWARGS = {'n_win': 5, 'causal': 1}

CLEAN_RT = True
FIRING_RATE_THRESHOLD = 1
PRESENCE_RATIO_THRESHOLD = 0.95
PIDS_WITHOUT_ILBLSORTOR = ['57edc590-a53d-403c-9aab-d58ee51b6a24', 'daadb3f1-bef2-474e-a659-72922f3fcc5b', '61bb2bcd-37b4-4bcc-8f40-8681009a511a', 'ee2ce090-696a-40f5-8f29-7107339bf08e']

PRE_TIME = 0.0
POST_TIME = 0.26
TOLERANCE = 1e-6
RANDOM_STATE =123

METRICS_WITHOUT_MEANSUB = [
    ('pre_fr', 'mean'),
    ('post_fr', 'mean'),
    ('fr_delta_modulation', 'mean'),
    ('pre_ff', 'mean'),
    ('post_ff', 'mean'),
    ('ff_quench', 'mean'),
    ('ff_quench_modulation', 'mean'),
]

METRICS_WITH_MEANSUB = [
    ('pre_ff', 'mean'),
    ('post_ff', 'mean'),
    ('ff_quench', 'mean'),
]

N_PERMUT_NEURAL_OMNIBUS = 1000#50
N_PERMUT_NEURAL_REGIONAL = 1000#50

# Noise correlation config
# NC_MODE = "pooled_zscore"   # "by_condition_avg" (paper-aligned) or "pooled_zscore"
NC_MODE = "pooled_zscore"   # main analysis: within-condition z-score, then pooled Pearson r


NC_MIN_TRIALS_PER_COND = 10
NC_MIN_TRIALS_TOTAL = 50
NC_OUTLIER_SD = None#3.0            # None to disable outlier trial removal

# Whether to also compute signal correlations
SC_ENABLE = True              # set True when you want signal corr
SC_MIN_CONDS = 4

NC_WINDOWS = {
    "pre": {
        "subfoldername": "500_pre",
        "nc_window": (-0.5, 0),
        "sc_window": (-0.5, 0),
    },
    "post": {
        "subfoldername": "500_post",
        "nc_window": (0, 0.5),
        "sc_window": (0, 0.5),
    },
}
