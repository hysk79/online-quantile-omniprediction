"""Data loading and array assembly.

The old code fed ``forecasts_dict`` (a nested dict of pandas Series) straight
into the online algorithm, which then rebuilt a ``(T, N, F)`` array with a
triple-nested Python comprehension **and** re-rounded by ``unit`` on *every*
run. Here we assemble that ``(T, N, F)`` array **once** in original count scale;
callers slice it (levels / forecasters) and round it (``round_to_grid``) cheaply
and vectorized. This removes the redundant work the algorithm used to repeat.
"""
from __future__ import annotations

import gzip
import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config as cfg


def assemble_preds_array(forecasts_dict_w: dict, forecaster_names: List[str],
                         alpha_list, dates_list) -> np.ndarray:
    """Assemble a ``(T, N, F)`` prediction array in original (count) scale.

    Axis order matches the original code: axis 0 = date, axis 1 = quantile
    level (``alpha_list`` order), axis 2 = forecaster (``forecaster_names``
    order). Each series is reindexed to ``dates_list`` exactly once.
    """
    alpha_list = list(alpha_list)
    arr = np.stack([
        np.stack([
            forecasts_dict_w[f][a].reindex(dates_list).to_numpy(dtype=np.float64)
            for f in forecaster_names
        ], axis=1)  # (T, F) for this level
        for a in alpha_list
    ], axis=1)  # (T, N, F)
    T = len(dates_list)
    assert arr.shape == (T, len(alpha_list), len(forecaster_names)), arr.shape
    return arr


def round_to_grid(arr: np.ndarray, unit: int) -> np.ndarray:
    """Divide by ``unit`` and round to the integer grid (vectorized)."""
    return np.round(np.asarray(arr, dtype=np.float64) / unit)


def _trimmed_path(geo: str, raw: bool, trimmed_dir: Optional[str] = None) -> str:
    trimmed_dir = trimmed_dir or cfg.TRIMMED_DIR
    return os.path.join(trimmed_dir, f"{geo}{'_raw' if raw else ''}.pkl.gz")


def load_trimmed(geo: str, raw: bool = False, trimmed_dir: Optional[str] = None) -> dict:
    """Load the gzipped trimmed pickle of one geo (see ``preprocess.build_trimmed``)."""
    with gzip.open(_trimmed_path(geo, raw, trimmed_dir), "rb") as fh:
        return pickle.load(fh)


def load_geo_arrays(geo: str, week: int, raw: bool = False,
                    trimmed_dir: Optional[str] = None,
                    forecaster_names: Optional[List[str]] = None) -> Dict:
    """Load a trimmed per-geo pickle and return ready-to-use numpy arrays.

    Returns a dict with:
      - ``Y`` (pd.Series) and ``y_arr`` (np.ndarray), original count scale
      - ``preds_TNF`` (T, N, F) base forecasters, original scale
      - ``ens_TNF`` (T, N, F_ens) ensemble models, original scale (if present)
      - ``forecaster_names``, ``ens_model_names``, ``alpha_list``,
        ``dates_list``, ``unit``, ``T``, ``N``, ``F``.

    ``forecaster_names`` may be passed to select/reorder a subset of base
    forecasters (used by the forecaster-count sensitivity experiment).
    """
    d = load_trimmed(geo, raw, trimmed_dir)

    alpha_list = np.asarray(d["alpha_list"], dtype=np.float64)
    dates_list = d["dates_list"]
    unit = int(d["unit"])
    Y = d["Y"][dates_list]
    y_arr = np.asarray(Y.values, dtype=np.float64)

    forecasts_dict_w = d["forecasts_dict"][week]
    if forecaster_names is None:
        forecaster_names = list(forecasts_dict_w.keys())
    preds_TNF = assemble_preds_array(forecasts_dict_w, forecaster_names, alpha_list, dates_list)

    out = {
        "geo": geo,
        "week": week,
        "unit": unit,
        "Y": Y,
        "y_arr": y_arr,
        "alpha_list": alpha_list,
        "dates_list": dates_list,
        "forecaster_names": forecaster_names,
        "preds_TNF": preds_TNF,
        "T": preds_TNF.shape[0],
        "N": preds_TNF.shape[1],
        "F": preds_TNF.shape[2],
    }

    ens_dict = d.get("forecasts_ens_dict", {}).get(week)
    if ens_dict is not None:
        ens_names = list(ens_dict.keys())
        out["ens_model_names"] = ens_names
        out["ens_TNF"] = assemble_preds_array(ens_dict, ens_names, alpha_list, dates_list)
    return out


def forecaster_availability_ranking(geo: str, week: int,
                                    trimmed_dir: Optional[str] = None) -> List[str]:
    """Rank base forecasters by data availability (most non-NaN first).

    Uses the *raw* (pre-imputation) trimmed pickle, because the imputed data has
    every forecaster at 100% availability, which would make the ranking
    meaningless.
    """
    d = load_trimmed(geo, raw=True, trimmed_dir=trimmed_dir)
    alpha_list = list(d["alpha_list"])
    dates_list = d["dates_list"]
    fc = d["forecasts_dict"][week]

    scored = []
    for f, per_alpha in fc.items():
        non_na = sum(int(per_alpha[a].reindex(dates_list).notna().sum()) for a in alpha_list)
        scored.append((f, non_na))
    # Stable sort by descending availability (ties keep dict/original order).
    scored.sort(key=lambda r: -r[1])
    return [f for f, _ in scored]
