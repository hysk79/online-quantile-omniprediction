# Online Quantile Omniprediction for Proper Losses

Code, data, and figures for the paper *Online Quantile Omniprediction for Proper Losses*.
The experiments combine the quantile forecasts of 18 COVID-19 Forecast Hub models for weekly
COVID-19 hospitalizations (51 states + DC and the US aggregate, 1–4-week-ahead horizons,
23 quantile levels, 127 weeks from 2020-12-29).

## Setup

```bash
conda env create -f environment.yaml
conda activate online-quantile-omniprediction
jupyter lab
```

## Reproducing the figures

Open `notebooks/paper_figures.ipynb` and run all cells (takes seconds). Each figure has its own
`Figure N` section and is written to `figures/`. The notebook only reads the compact summaries in
`results/summary/`, so it doesn't need any experiment to be rerun.

| Figure | File(s) |
| --- | --- |
| 2 | `Omni_error_over_time.pdf` |
| 3 | `Error_diff_density.pdf`, `Difficulty_plot.pdf` |
| 4 | `Sensitivity.pdf` |
| 5 | `runtime_combined.pdf` |
| 6 | `Quantile_estimates_US_1wk.pdf`, `Quantile_estimates_US_4wk.pdf` |
| 7 | `HQL_Sensitivity.pdf` |
| 8 | `Sensitivity_grid_QL.pdf` |
| 9 | `QL_std_rank.pdf` |

## Rerunning the experiments

`notebooks/run_experiments.ipynb` runs the whole pipeline: base-forecaster metrics, learning-rate
sweeps of every algorithm over all locations and horizons, the US sensitivity runs, and finally
the summaries. The main sweeps can also be run from the command line:

```bash
cd src
python run_experiment.py --algo omni_v2 --geos all   # our algorithm (Algorithm 1)
python run_experiment.py --algo ql_pb   --geos all   # Hedge-QL baseline
python run_experiment.py --algo wql     --geos all   # two-player WQL omnipredictor (appendix)
```

Full results take several GB and are git-ignored.

## Repository layout

```
src/
  algorithms.py        our algorithm (run_omni_v2), WQL omnipredictor (run_omni_wql), Hedge-QL (run_ql_pb_opt)
  solver_omni_v2.py    per-step minimax subroutine of our algorithm
  solver_wql.py        per-step subroutine of the WQL omnipredictor
  metrics.py           pinball loss, elementary scores, omni-error traces
  config.py            paths, locations, grid unit per location, quantile levels, learning-rate sweeps
  data_io.py           loads a location's data as (T, N, B) arrays
  preprocess.py        raw CSVs -> data/trimmed, base-forecaster metrics
  run_experiment.py    sweeps + command-line interface
  sensitivity.py       US runtime / grid-unit experiments
  summary.py           compacts results into results/summary/
  paper_plots.py       plotting helpers for paper_figures.ipynb
notebooks/
  run_experiments.ipynb
  paper_figures.ipynb
data/trimmed/          preprocessed forecasts and targets (tracked)
results/summary/       compact results behind every figure (tracked)
figures/               paper figures (PDF)
```

## Data

- **`data/trimmed/{GEO}.pkl.gz`**: per-location weekly targets and quantile forecasts for the four
  horizons, after outlier clipping. The 18 base forecasters and the ensembles (COVIDhub-Ens,
  JHUAPL-SLPEns) are stored separately. Missing forecasts are filled with the median of the other
  forecasters. Load one with `data_io.load_geo_arrays(geo, week)`.
- **`data/trimmed/{GEO}_raw.pkl.gz`**: the same, without imputation. It is only used to rank the
  raw base forecasters in Figure 9 and to order forecasters by availability.
- **Raw source:** the Forecast Hub evaluation data,
  [hospitalizations.zip](https://forecast-eval.s3.us-east-2.amazonaws.com/hospitalizations.zip).
  To rebuild `data/trimmed/` from it, unzip `hospitalizations_{1,2,3,4}wk.csv` into
  `data/hospitalizations/` and run step 0 of `run_experiments.ipynb`.

## Results

- **`results/summary/main_summary.pkl`** (about 27 MB) holds, for every (horizon, location), the
  following, all at every learning rate:
  - the target;
  - the grid size;
  - per-step omni-error and pinball-loss traces of our algorithm, Hedge-QL, the base forecasters
    and the ensembles.

  For US-agg it also holds every run's quantile predictions. It is used by Figures 2, 3, 4, 6, 7
  and 9.
- **`results/summary/sensitivity_summary.pkl`** holds the US runtimes and the grid-unit sweep
  (Figures 5 and 8).

These summaries are rebuilt by `summary.build_main_summary()` and
`summary.build_sensitivity_summary()` (step 4 of `run_experiments.ipynb`). All other
`results/` subfolders are produced by the experiments and are not tracked.
