# Deploying the DeepRAB subgroup explorer to Posit Connect

## Files the app needs

```
app.py                        Shiny UI + server
deeprab_engine.py             analysis engine (no Shiny import)
Deep_learning_subgroup_v2.py  model: concrete autoencoder + A-learning losses
requirements.txt              pinned dependencies
```

Everything else in this folder (the demo scripts, `datapath/`, `results_synthetic/`,
`logs/`) is for local benchmarking and should **not** be deployed.

## Deploy

```bash
pip install rsconnect-python
rsconnect add --server https://<your-connect>/ --name work --api-key <KEY>

rsconnect deploy shiny . \
  --entrypoint app:app \
  --name work \
  --title "DeepRAB subgroup explorer" \
  --exclude 'datapath/*' \
  --exclude 'results_synthetic/*' \
  --exclude 'logs/*' \
  --exclude '.venv/*' \
  --exclude '__pycache__/*' \
  --exclude 'demo_code_*' \
  --exclude 'run_synthetic_worker.py' \
  --exclude 'analyze_synthetic.py' \
  --exclude '*.csv'
```

To review the bundle before it leaves your machine, write the manifest first:

```bash
rsconnect write-manifest shiny . --entrypoint app:app --overwrite
```

## The one thing that will break if you change it

**`TF_USE_LEGACY_KERAS=1` must be set before TensorFlow is imported.**

`app.py` and `deeprab_engine.py` both call `os.environ.setdefault(...)` at the top
of the module, above any TF import, so this is handled. But if you refactor
imports, add a wrapper module, or import TensorFlow from a new entry point, set
the variable first or `ConcreteSelect.call` dies with:

```
AttributeError: module 'keras._tf_keras.keras.backend' has no attribute 'in_train_phase'
```

TensorFlow ≥ 2.16 bundles Keras 3, which dropped `in_train_phase` — the call that
switches the concrete layer between its stochastic training branch and its
deterministic argmax inference branch. `tf-keras` restores the Keras 2 API.

Belt and braces: set it as an environment variable on the Connect content item too
(Content → Vars → `TF_USE_LEGACY_KERAS = 1`).

## Runtime settings on Connect

| Setting | Suggested | Why |
|---|---|---|
| Max processes | 2–4 | TensorFlow resident memory is ~400 MB per process |
| Max connections per process | 3–5 | fitting is CPU-bound; more users just queue |
| Min processes | 1 | keeps TF warm — a cold first import costs ~10 s |
| Idle timeout | ≥ 900 s | a Thorough run plus exploration exceeds the default |
| CPU request | ≥ 2 cores | see below |

**Threading.** TensorFlow defaults to grabbing every visible core, which on a
shared Connect node means several sessions fighting each other. If Connect reports
more cores than the content item is actually allotted, pin it:

```
TF_NUM_INTRAOP_THREADS = 2
TF_NUM_INTEROP_THREADS = 1
OMP_NUM_THREADS = 2
```

**Why the app stays responsive.** Fitting runs in
`asyncio.to_thread` inside a `reactive.extended_task`. That matters on Connect
specifically: a blocking call in a Shiny for Python server function stalls the
event loop for *every* session in that process, not just the one that started it.

## Expected runtime

Measured on 1000 subjects × 13 covariates, single CPU core, `tryout_limit=1`:

| Effort | Fits | Continuous / binary | Time-to-event |
|---|---|---|---|
| Fast | 6 | ~12 s | ~20 s |
| Balanced | 16 | ~40 s | ~70 s |
| Thorough | 30 | ~80 s | ~150 s |

Add ~5 s for nuisance models, and ~10 s once per process for the first
TensorFlow import. Cox is slower because its loss is not separable across rows,
so it trains full-batch: one epoch is one gradient step.

## Smoke test after deploying

1. Open the app; the sidebar should render with no file loaded.
2. Upload a CSV. The outcome / treatment / covariate selectors appear and guess
   sensible defaults from the column names.
3. Switch the endpoint to **Time-to-event** — the outcome select is replaced by
   follow-up time and event indicator.
4. Press **Run analysis**. A progress bar appears and reports each fit.
5. All six tabs render, three plots draw, and the four downloads produce files.

## Data handling

Uploaded files land in the Connect process's temp directory and are read into
memory; nothing is written back to disk by the app. Confirm that this satisfies
your data-governance rules before pointing it at patient data, and set the
content item's access controls accordingly.
