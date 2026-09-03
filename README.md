# DeepRAB

Deep-learning ranking for **subgroup identification and predictive biomarker
discovery** in randomised trials.

DeepRAB fits a concrete-autoencoder feature selector against an A-learning loss,
so it learns which covariates drive *treatment benefit* — not which covariates
predict outcome. Two things come out of a run:

1. a **selection probability** for every candidate biomarker, and
2. a **predicted benefit subgroup** for every subject.

Nothing in the fitting or model-selection path uses a known responder label.
That is deliberate: real trial data does not have one. Model selection is driven
by held-out A-learning loss, which is a criterion you can actually compute on a
real study.

This repository contains two things: an interactive **Shiny app** for analysing
one trial, and the **simulation code** from the paper.

---

## 1. The Shiny app

A single-page app for uploading a trial dataset, tuning a small hyperparameter
grid, and reading off the feature ranking and predicted subgroups.

### Install

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # macOS / Linux
```

Python 3.13 (see `.python-version`).

### Run

```bash
TF_USE_LEGACY_KERAS=1 python -m shiny run --port 8000 app.py
```

Then open <http://127.0.0.1:8000>. On Windows you can double-click
`run_app.bat`, which sets the environment variables and opens the browser for
you (edit `PYEXE` at the top to point at your interpreter).

> **`TF_USE_LEGACY_KERAS=1` is not optional.** TensorFlow ≥ 2.16 ships Keras 3,
> which removed `keras.backend.in_train_phase` — the call that switches the
> concrete selection layer between its stochastic training branch and its
> deterministic argmax inference branch. `tf-keras` restores the Keras 2 API.
> `app.py` and `deeprab_engine.py` both `os.environ.setdefault(...)` above any
> TensorFlow import, so this is handled for the normal entry points. See
> [DEPLOY.md](DEPLOY.md) if you add a new one.

### Input format

One CSV, one row per subject. You name the columns in the sidebar, so they can
be called anything:

| Role | Notes |
|---|---|
| Treatment | binary, 0 / 1 |
| Outcome | continuous, or binary 0 / 1 |
| Follow-up time + event indicator | time-to-event instead of a single outcome |
| Covariates | the candidate biomarkers to rank |
| Known label | *optional*. If your file has a true responder flag (e.g. from a simulation), name it and the app reports an AUC against it as a check. It is never used to choose a model. |

Numeric covariates pass through; low-cardinality non-numeric columns are one-hot
encoded (first level dropped), with each resulting column ranked separately.

`synthetic_clinical_data.csv` in this repo is a ready-to-upload example
(1000 subjects, 13 covariates, continuous outcome, `sigpos` as the known label).

### Endpoints

| Endpoint | Loss | Subgroup effect reported |
|---|---|---|
| Continuous | squared A-learning loss | IPW mean difference |
| Binary (0/1) | logistic A-learning loss | IPW risk difference |
| Time-to-event | Cox partial likelihood | treatment log hazard ratio |

### Tuning effort

Grid presets vary K × learning rate; random presets also sample architecture,
epoch budget and final temperature. Configurations are ranked by *mean*
validation loss across restarts and the winner is the rank-1 configuration — not
the single best individual fit, which is a badly biased statistic on one
validation split.

| Preset | Search | Fits |
|---|---|---|
| Fast | grid, 3 K × 2 lr | 6 |
| Balanced | grid, 4 K × 2 lr × 2 restarts | 16 |
| Thorough | grid, 5 K × 2 lr × 3 restarts | 30 |
| Wide | random, 50 configs × 2 restarts | ~100 |
| Exhaustive | random, 100 configs × 2 restarts | ~200 |

Measured on 1000 subjects × 13 covariates, single CPU core: Fast is ~12 s
continuous / ~20 s time-to-event; Thorough is ~80 s / ~150 s. Add ~10 s once per
process for the first TensorFlow import. Cox is slower because its loss is not
separable across rows, so it trains full-batch — one epoch is one gradient step.

The **Advanced** accordion exposes the raw search space, restarts, ensemble size
(top-M), split fractions and seed.

### Tabs

- **Data** — column summary, parsing notes, preview
- **Tuning** — validation loss by configuration, winning hyperparameters, per-fit table
- **Feature selection** — selection probability per covariate, plot and table
- **Predicted subgroups** — subject counts and estimated treatment effect within each predicted subgroup, with bootstrap CIs, plus a score-threshold slider
- **Download** — per-subject predictions, feature probabilities, tuning table (CSV) and a run summary (TXT)
- **How to read this** — in-app interpretation guide

### Deployment

[DEPLOY.md](DEPLOY.md) covers deploying to Posit Connect: which files go in the
bundle, runtime and threading settings, expected runtime, and a smoke test.

---

## 2. Using the engine without the app

`deeprab_engine.py` has no Shiny import, so it can be scripted, unit-tested, or
called from a notebook:

```python
import pandas as pd
from deeprab_engine import AnalysisSpec, run_analysis

df = pd.read_csv("synthetic_clinical_data.csv")

spec = AnalysisSpec(
    outcome="continuous",
    y_col="y",
    trt_col="treatment",
    x_cols=("age", "sex", "bmi", *[f"X{i}" for i in range(1, 11)]),
    truth_col="sigpos",        # optional, for the AUC check only
    pi_mode="known", pi_known=0.5,
    preset="fast",
    seed=1,
)

res = run_analysis(df, spec)
print(res.best)                # winning hyperparameters
print(res.features)            # selection probability per covariate
print(res.subjects.head())     # per-subject contrast + subgroup
```

`run_analysis` also accepts a `progress(frac, msg)` callback. The returned
`AnalysisResult` carries `runs`, `configs`, `features`, `subjects`, `best`,
`info` and `messages`.

Relative to the paper's simulation code the app engine makes three deliberate
changes, documented at the top of `deeprab_engine.py`: one fit/validation/test
split rather than ten resamples (a clinician analysing one trial wants one
answer); `tryout_limit=1`, because the ranking of the concrete logits settles
long before the softmax sharpens; and the ensemble averages the raw contrast `f`
rather than its ranks, which preserves the `f = 0` threshold that defines the
subgroup.

---

## 3. Simulation code from the paper

The demo covers Simulation Scenario I for continuous outcomes with the
prognostic effect set to 0 and the predictive effect to 0.1. Both the A-learning
loss used in the paper and a weight-learning loss are implemented.

| File | Purpose |
|---|---|
| `Deep_learning_subgroup_v2.py` | the model: concrete autoencoder, A-learning / weight-learning losses, Cox and IPW helpers |
| `demo_code_Simple_tunning_v2.py` | the paper's demo — tuning over one scenario |
| `run_synthetic_worker.py` | one replicate of the benchmark, writes to `results_synthetic/` |
| `analyze_synthetic.py` | aggregates replicates into summary tables |
| `_launch.sh` | runs 10 replicates in parallel |
| `datapath/` | simulation inputs for the binary, continuous and time-to-event scenarios |
| `make_slides.py` | builds the summary deck |

`Deep_learning_subgroup_v2.py` is also a runtime dependency of the app —
`deeprab_engine.py` imports `cox_fit_ridge`, `ipw_effect` and the selector from
it.

`_launch.sh` and `run_app.bat` contain machine-specific interpreter paths; edit
them before use.

You are encouraged to change the parameters and apply the method to other
datasets with different predictive and prognostic effects.

---

## Data handling

The app reads uploaded files into memory and writes nothing back to disk.
Confirm this satisfies your data-governance rules before pointing it at patient
data, and set your hosting platform's access controls accordingly. All CSVs in
this repository are simulated.

## Repository layout

```
app.py                          Shiny UI + server
deeprab_engine.py               analysis engine (no Shiny import)
Deep_learning_subgroup_v2.py    model + losses (needed by both)
requirements.txt                pinned dependencies
DEPLOY.md                       Posit Connect deployment guide
run_app.bat                     local launcher (Windows)
synthetic_clinical_data.csv     example upload
datapath/                       simulation inputs
demo_code_Simple_tunning_v2.py  paper demo
run_synthetic_worker.py         benchmark worker
analyze_synthetic.py            benchmark aggregation
```
