"""Central configuration for the quantile-omniprediction experiments.

Filesystem paths, geo groupings, the per-geo grid ``unit`` map, the per-geo
outlier-clipping table, the quantile levels, the learning-rate (eta) sweeps, and
a helper for building symmetric quantile subsets.

Import this module as ``import config as cfg`` and read values off it; nothing
here has side effects.
"""
from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Resolve paths relative to the repository root so the code works regardless of
# the current working directory (notebook vs. CLI).
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SRC_DIR, os.pardir))

DATA_DIR = os.path.join(REPO_ROOT, "data", "hospitalizations")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
FIG_DIR = os.path.join(REPO_ROOT, "figures")

# Preprocess artefact directories (produced by ``preprocess.py``).
STATES_DIR = os.path.join(DATA_DIR, "preprocess_states")
STATES_WEEKLY_DIR = os.path.join(DATA_DIR, "preprocess_states_weekly")
TRIMMED_DIR = os.path.join(REPO_ROOT, "data", "trimmed")
BASE_FORECASTER_DIR = os.path.join(RESULTS_DIR, "base_forecasters")

# Default output folders for the three algorithms (override on the CLI).
SAVE_DIR_OMNI = os.path.join(RESULTS_DIR, "omni")
SAVE_DIR_QL_PB = os.path.join(RESULTS_DIR, "hedge_ql")
SAVE_DIR_WQL = os.path.join(RESULTS_DIR, "omni_wql")
SENSITIVITY_DIR = os.path.join(RESULTS_DIR, "sensitivity_us")

# Compact summaries read by notebooks/paper_figures.ipynb (committed to git).
SUMMARY_DIR = os.path.join(RESULTS_DIR, "summary")
MAIN_SUMMARY_PATH = os.path.join(SUMMARY_DIR, "main_summary.pkl")
SENSITIVITY_SUMMARY_PATH = os.path.join(SUMMARY_DIR, "sensitivity_summary.pkl")

# ---------------------------------------------------------------------------
# Forecasters
# ---------------------------------------------------------------------------
# Full set of forecasters kept during preprocessing.
FORECASTER_LIST: List[str] = [
    "COVIDhub-4_week_ensemble", "COVIDhub-trained_ensemble", "JHUAPL-SLPHospEns",
    "CU-select", "GT-DeepCOVID", "COVIDhub-baseline", "Karlen-pypm",
    "JHU_IDD-CovidSP", "MOBS-GLEAM_COVID", "USC-SI_kJalpha", "JHUAPL-Bucky",
    "JHUAPL-Gecko", "MUNI-ARIMA", "CMU-TimeSeries", "UMass-trends_ensemble",
    "PSI-DICE", "IHME-CurveFit", "CUB_PopCouncil-SLSTM", "BPagano-RtDriven",
    "LANL-GrowthRate", "UVA-Ensemble",
]

# Ensemble models are treated separately (held out from the base-forecaster set
# the online algorithm competes against).
ENS_MODEL_NAMES: List[str] = [
    "COVIDhub-4_week_ensemble", "COVIDhub-trained_ensemble", "JHUAPL-SLPHospEns",
]

# ---------------------------------------------------------------------------
# Geographies
# ---------------------------------------------------------------------------
GEO_FULL_LIST: List[str] = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI",
    "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN",
    "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH",
    "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA",
    "WV", "WI", "WY", "US",
]

GEO_BIG: List[str] = ["CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI", "NJ", "VA", "WA"]
GEO_MED: List[str] = ["AZ", "TN", "MA", "IN", "MO", "MD", "CO", "WI", "MN", "SC", "AL", "LA", "KY", "OR", "OK", "CT"]
GEO_SMALL: List[str] = ["UT", "NV", "IA", "AR", "KS", "MS", "NM", "ID", "NE", "WV", "HI", "NH", "ME", "MT", "RI", "DE", "SD", "ND", "AK", "DC", "VT", "WY"]

# Order used for the main sweeps (big -> med -> small -> US).
GEO_LIST: List[str] = GEO_BIG + GEO_MED + GEO_SMALL + ["US"]

# ---------------------------------------------------------------------------
# Grid unit per geo
# ---------------------------------------------------------------------------
_GEO_UNIT_GROUPS: List[Tuple[int, Sequence[str]]] = [
    (500, ["US"]),
    (100, ["CA", "TX", "FL", "NY"]),
    (50, ["PA", "IL", "OH", "GA", "NC", "MI", "NJ"]),
    (20, ["VA", "WA", "AZ", "TN", "MA", "IN", "MO", "MD", "CO", "WI", "MN", "SC", "AL", "LA", "KY", "OK", "CT", "AR"]),
    (10, ["OR", "UT", "NV", "IA", "KS", "MS", "MT", "DC"]),
    (5, ["NM", "ID", "NE", "WV", "HI", "NH", "ME", "DE", "SD"]),
    (2, ["RI", "ND", "AK", "VT", "WY"]),
]

GEO_UNIT: Dict[str, int] = {geo: unit for unit, geos in _GEO_UNIT_GROUPS for geo in geos}


def unit_for_geo(geo: str) -> int:
    """Grid spacing (in raw hospitalization counts) used to discretize a geo."""
    if geo not in GEO_UNIT:
        raise KeyError(f"No grid unit configured for geo '{geo}'.")
    return GEO_UNIT[geo]


# ---------------------------------------------------------------------------
# Outlier clipping table
# ---------------------------------------------------------------------------
# Each geo maps to a list of (alpha_cut, cuts) specs applied in order. A spec
# clips the upper tail of every non-ensemble forecaster: for the quantile level
# ``alpha_cut`` and every higher level, the forecast is clipped to the matching
# entry of ``cuts`` (the first entry also clips all levels below alpha_cut up to
# alpha_cut itself, matching the original preprocessing).
OUTLIER_CUTS: Dict[str, List[Tuple[float, List[float]]]] = {
    "CA": [(0.9, [110000, 120000, 125000, 130000])],
    "FL": [(0.9, [80000, 90000, 100000, 121000])],
    "TX": [(0.9, [90000, 100000, 120000, 135000])],
    "NY": [(0.975, [80000, 85000])],
    "PA": [(0.9, [35000, 40000, 50000, 55000])],
    "IL": [(0.9, [25000, 30000, 31000, 35000])],
    "OH": [(0.9, [35000, 40000, 45000, 55000])],
    "GA": [(0.9, [43000, 45000, 49000, 50000])],
    "MI": [(0.9, [30000, 35000, 45000, 50000])],
    "NJ": [(0.975, [45000, 50000])],
    "VA": [(0.99, [25000])],
    "WA": [(0.95, [16000, 18000, 20000])],
    "TN": [(0.95, [25000, 30000, 35000])],
    "MA": [(0.95, [18000, 19000, 20000])],
    "IN": [(0.975, [25000, 32000])],
    "MO": [(0.95, [30000, 33000, 36000])],
    "MD": [(0.9, [16000, 18000, 20000, 21000])],
    "CO": [(0.9, [17000, 20000, 22000, 25000])],
    "WI": [(0.9, [10000, 12000, 13000, 15000])],
    "MN": [(0.9, [8000, 11000, 13000, 14000])],
    "SC": [(0.9, [20000, 22000, 27000, 32000])],
    "AL": [(0.975, [20000, 30000])],
    "LA": [(0.975, [35000, 40000])],
    "OR": [(0.9, [9000, 10000, 15000, 16000])],
    "OK": [(0.99, [25000])],
    "CT": [(0.99, [17000])],
    "UT": [(0.975, [12000, 13000])],
    "NV": [(0.99, [16000])],
    "IA": [(0.9, [6000, 7000, 11000, 15000])],
    "AR": [(0.9, [8000, 9000, 12500, 15000])],
    "KS": [(0.975, [12000, 15000])],
    # NE is clipped twice in the original code; both are applied in sequence.
    "NE": [(0.975, [9000, 10000]), (0.95, [7000, 8500, 9000])],
    "NM": [(0.9, [2500, 3000, 4000, 5000])],
    "ID": [(0.975, [4500, 5500])],
    "WV": [(0.975, [8000, 10000])],
    "HI": [(0.99, [6000])],
    "ME": [(0.9, [2700, 3800, 4200, 5000])],
    "MT": [(0.9, [5000, 6000, 7000, 7500])],
    "DE": [(0.9, [3000, 3500, 4000, 4500])],
    "SD": [(0.975, [3000, 4000])],
    "ND": [(0.9, [1500, 1700, 2000, 2500])],
    "AK": [(0.9, [1500, 1700, 2000, 2200])],
    "DC": [(0.9, [4500, 5000, 6000, 6500])],
    "VT": [(0.9, [800, 900, 1000, 1100])],
    "WY": [(0.975, [1300, 1500])],
    "US": [(0.95, [900000, 910000, 920000])],
}

# ---------------------------------------------------------------------------
# Quantile levels
# ---------------------------------------------------------------------------
ALPHA_LIST: np.ndarray = np.array([
    0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99,
])
N_LEVELS = len(ALPHA_LIST)  # 23

# ---------------------------------------------------------------------------
# Dates / dimensions
# ---------------------------------------------------------------------------
DATES_LIST_PD = pd.date_range(start="2020-12-29", end="2023-05-30", freq="7D")
DATES_LIST = DATES_LIST_PD.strftime("%Y-%m-%d")
T = 127

# ---------------------------------------------------------------------------
# Learning-rate (eta) sweeps
# ---------------------------------------------------------------------------
ETA_LIST_OMNI: np.ndarray = np.round(
    np.power(10, np.concatenate([[-3.0, -2.0], np.arange(-1.5, 1.6, 0.25)])), 4
)
ETA_LIST_QL: np.ndarray = np.sort(np.round(
    np.power(10, np.concatenate([[-3.0, -2.0], np.arange(-1, 4.1, 0.5), np.arange(-0.25, 2.1, 0.5)])), 4
))

# ---------------------------------------------------------------------------
# Sensitivity-experiment configuration (US only, w=4, eta_multiplier=1.0)
# ---------------------------------------------------------------------------
SENSITIVITY_GEO = "US"
SENSITIVITY_WEEK = 4
SENSITIVITY_ETA_MULTIPLIER = 1.0

# Grid sizes to sweep for the US timing + error experiments.
SENSITIVITY_UNITS: List[int] = [1000, 500, 250, 100, 50, 25]

# Number of quantile levels to sweep (odd sizes so 0.5 is always included).
SENSITIVITY_N_QUANTILES: List[int] = [23, 21, 15, 9, 5]

# Fractions of base forecasters to sweep (selected by data availability).
SENSITIVITY_FORECASTER_FRACS: List[float] = [0.4, 0.6, 0.8, 1.0]


def symmetric_quantile_subset(k: int, alpha_list: np.ndarray = ALPHA_LIST) -> np.ndarray:
    """Return ``k`` quantile levels symmetric about 0.5.

    ``k`` must be odd (so the median 0.5 is included). Levels are chosen as
    evenly spaced as possible from the center out to both extremes, then
    mirrored, guaranteeing symmetry about the central level.
    """
    alpha_list = np.asarray(alpha_list)
    n = len(alpha_list)
    if k == n:
        return alpha_list.copy()
    if k % 2 == 0:
        raise ValueError(f"k must be odd so 0.5 is included, got {k}.")
    if k < 1 or k > n:
        raise ValueError(f"k={k} out of range 1..{n}.")
    center = n // 2  # index of 0.5 in the (odd-length, symmetric) full list
    num_side = (k + 1) // 2  # indices from center..end inclusive
    upper_idx = np.round(np.linspace(center, n - 1, num_side)).astype(int)
    upper_idx = np.unique(upper_idx)
    lower_idx = 2 * center - upper_idx  # mirror about the center
    idx = np.unique(np.concatenate([lower_idx, upper_idx]))
    if len(idx) != k:
        raise ValueError(f"Could not build a symmetric subset of size {k} (got {len(idx)}).")
    return alpha_list[idx]
