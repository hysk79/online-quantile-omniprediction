"""CLI entry point for the main omniprediction experiments.

Runs one of the three algorithms over a grid of (geo, week, eta_multiplier),
parallelized with joblib, and pickles each result. Data assembly happens once
per (geo, week); the eta sweep reuses the assembled array (no per-run dict
rebuild).

Examples
--------
Run omni-v2 for all geos, weeks 1-4, the default eta sweep::

    python run_experiment.py --algo omni_v2 --geos all

Run WQL for just US and CA, week 4, a single eta, into a custom folder::

    python run_experiment.py --algo wql --geos US CA --weeks 4 \
        --etas 1.0 --output-dir ../results/my_run

The thin ``notebooks/run_experiments.ipynb`` calls :func:`run_sweep` directly
so you only set ``output_dir`` and the sweep lists there.
"""
from __future__ import annotations

import argparse
import os
import pickle
from typing import List, Optional, Sequence

import numpy as np
from joblib import Parallel, delayed

import config as cfg
import data_io
import utils
from algorithms import run_omni_v2, run_omni_wql, run_ql_pb_opt

_ALGOS = {
    "omni_v2": (run_omni_v2, cfg.SAVE_DIR_OMNI, cfg.ETA_LIST_OMNI, utils.exp_name_omni_v2),
    "wql": (run_omni_wql, cfg.SAVE_DIR_WQL, cfg.ETA_LIST_OMNI, utils.exp_name_omni_v2),
    "ql_pb": (run_ql_pb_opt, cfg.SAVE_DIR_QL_PB, cfg.ETA_LIST_QL, utils.exp_name_ql_pb),
}


def _resolve_geos(geos: Sequence[str]) -> List[str]:
    if len(geos) == 1 and geos[0].lower() == "all":
        return cfg.GEO_LIST
    return list(geos)


def run_sweep(algo: str, geos: Sequence[str], weeks: Sequence[int],
              etas: Optional[Sequence[float]] = None,
              output_dir: Optional[str] = None, suffix: str = "",
              n_jobs: int = -1, skip_existing: bool = True, verbose: int = 10):
    """Run an algorithm over (geo, week, eta) and pickle each result.

    Data is assembled once per (geo, week) and shared across the eta sweep.
    """
    if algo not in _ALGOS:
        raise ValueError(f"Unknown algo '{algo}'. Choose from {list(_ALGOS)}.")
    run_fn, default_dir, default_etas, name_fn = _ALGOS[algo]
    output_dir = output_dir or default_dir
    etas = list(default_etas if etas is None else etas)
    geos = _resolve_geos(geos)
    os.makedirs(output_dir, exist_ok=True)

    # Assemble arrays once per (geo, week).
    loaded_cache = {}
    for geo in geos:
        for w in weeks:
            loaded_cache[(geo, w)] = data_io.load_geo_arrays(geo, w)

    def _name(geo, w, eta):
        if algo == "ql_pb":
            return name_fn(w, geo, eta, True, suffix=suffix)
        return name_fn(w, geo, eta, suffix=suffix)

    def _one(geo, w, eta):
        try:
            loaded = loaded_cache[(geo, w)]
            res = run_fn(preds_TNF=loaded["preds_TNF"], y_arr=loaded["y_arr"],
                         unit=loaded["unit"], alpha_list=loaded["alpha_list"],
                         eta_multiplier=eta)
            path = os.path.join(output_dir, f"results_{_name(geo, w, eta)}.pkl")
            pickle.dump(res, open(path, "wb"))
            return {"ok": True, "geo": geo, "w": w, "eta": eta}
        except Exception as e:  # noqa: BLE001 - report and continue the sweep
            print(f"{geo} w{w} eta{eta} FAILED: {e}")
            return {"ok": False, "geo": geo, "w": w, "eta": eta, "error": str(e)}

    jobs = [(geo, w, eta) for geo in geos for w in weeks for eta in etas
            if not (skip_existing and os.path.exists(
                os.path.join(output_dir, f"results_{_name(geo, w, eta)}.pkl")))]
    print(f"[{algo}] output_dir={output_dir}  total jobs: {len(jobs)}")

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=verbose)(
        delayed(_one)(geo, w, eta) for geo, w, eta in jobs)
    n_ok = sum(r["ok"] for r in results)
    print(f"[{algo}] done: {n_ok}/{len(results)} succeeded")
    return results


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run omniprediction experiments.")
    p.add_argument("--algo", choices=list(_ALGOS), default="omni_v2")
    p.add_argument("--geos", nargs="+", default=["all"],
                   help="Geo codes (e.g. US CA) or 'all'.")
    p.add_argument("--weeks", nargs="+", type=int, default=[1, 2, 3, 4])
    p.add_argument("--etas", nargs="+", type=float, default=None,
                   help="eta_multiplier values; default is the algo's sweep in config.")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--suffix", default="")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--no-skip-existing", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    a = _parse_args(argv)
    run_sweep(algo=a.algo, geos=a.geos, weeks=a.weeks, etas=a.etas,
              output_dir=a.output_dir, suffix=a.suffix, n_jobs=a.n_jobs,
              skip_existing=not a.no_skip_existing)


if __name__ == "__main__":
    main()
