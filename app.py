"""
DeepRAB subgroup explorer -- Shiny for Python app.

Upload a trial dataset, pick the endpoint type and columns, tune a small
hyperparameter grid, and read off (a) the selection probability of every
candidate biomarker and (b) the predicted benefit subgroup for every subject.

Nothing in the fitting or model-selection path uses a known responder label --
that is the point, since real trial data does not have one.  If your file DOES
contain one (e.g. from a simulation), name it in "Known label" and the app will
report an AUC against it as a check; it is never used to choose a model.

Deploy to Posit Connect:
    rsconnect deploy shiny . --entrypoint app:app --name <server>
See DEPLOY.md for the full checklist.
"""

from __future__ import annotations

import asyncio
import io
import os
import threading
import traceback

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")   # must precede any TF import
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shiny import App, reactive, render, ui

from deeprab_engine import (
    ARCHITECTURES,
    ENDPOINTS,
    PRESETS,
    SEARCH_SPACE,
    AnalysisSpec,
    arch_from_label,
    arch_label,
    run_analysis,
    subgroup_effect,
)

NONE = "(none)"
ACCENT = "#2b6cb0"
GREY = "#94a3b8"

# Labels for the searched dimensions, used by the Advanced controls and by the
# tuning plot's panel titles.
DIM_LABELS = {
    "K": "K (features selected)",
    "lr": "learning rate",
    "hidden": "hidden layers",
    "epochs": "epochs",
    "min_temp": "min_temp",
    "dropout": "dropout",
    "l2": "L2 penalty",
}


def _fig(w=7.2, h=4.0):
    fig, ax = plt.subplots(figsize=(w, h), dpi=110)
    _style(ax)
    return fig, ax


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e2e8f0", lw=0.8)
    ax.set_axisbelow(True)
    return ax


# --------------------------------------------------------------------------- #
# Tuning-panel renderers.
#
# These take an AnalysisResult and return a figure / DataFrame / string, with no
# reactive or UI dependency, so they can be exercised directly from a script --
# the same reason deeprab_engine.py keeps Shiny out of the engine.  The @render
# functions below are thin wrappers over them.
# --------------------------------------------------------------------------- #
def hp_columns(r) -> list:
    """
    Hyperparameter columns worth showing: the ones that varied, plus K and lr
    always.  Leaving two all-zero dropout / L2 columns in an already wide table
    is just noise in the default setup.
    """
    varied = set(r.info.get("searched", []))
    return [c for c in ("K", "lr", "hidden", "epochs", "min_temp", "dropout", "l2")
            if c in {"K", "lr"} or c in varied]


def tuning_figure(r):
    """
    One panel per searched dimension, every configuration a dot.

    A single line-per-lr across K described the old two-dimensional grid fine,
    but it cannot show a five-dimensional random search.  Small multiples answer
    the question the search is actually run to answer -- which knob moves the
    validation loss and which is noise -- because a dimension that matters shows
    a level shift between its columns while one that does not shows the same
    spread in every column.
    """
    c = r.configs
    dims = [d for d in r.info.get("searched", []) if d in c.columns] or ["K"]
    fig, axes = plt.subplots(1, len(dims), dpi=110, sharey=True,
                             figsize=(max(7.2, 2.5 * len(dims)), 4.0))
    axes = np.atleast_1d(axes)
    rng = np.random.RandomState(0)      # fixed jitter: the plot must not flicker
    win = c.iloc[0]

    # Architectures sort by capacity, not alphabetically: a plain string sort
    # puts "128-64-32" before "16", which reads as nonsense on an axis.
    arch_order = {arch_label(h): i for i, h in enumerate(ARCHITECTURES)}

    def _level_key(dim, v):
        if dim == "hidden":
            return arch_order.get(v, len(arch_order))
        return v

    for ax, dim in zip(axes, dims):
        _style(ax)
        levels = sorted(c[dim].unique(), key=lambda v: _level_key(dim, v))
        pos = {v: i for i, v in enumerate(levels)}
        xs = np.array([pos[v] for v in c[dim]], dtype=float)
        # Jitter only when levels hold several configs each, otherwise the dots
        # stack into one blob per column and hide the spread.
        if len(c) > len(levels):
            xs = xs + rng.uniform(-0.16, 0.16, len(xs))
        ax.scatter(xs, c.val_loss_mean, s=26, color=GREY, alpha=0.75,
                   edgecolors="none", zorder=2)
        for v in levels:               # level medians, as a reference for the eye
            m = float(c.val_loss_mean[c[dim] == v].median())
            ax.plot([pos[v] - 0.3, pos[v] + 0.3], [m, m],
                    color="#475569", lw=1.8, zorder=3)
        ax.scatter([pos[win[dim]]], [win.val_loss_mean], s=170, facecolors="none",
                   edgecolors=ACCENT, lw=2.4, zorder=5)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([f"{v:g}" if isinstance(v, (int, float, np.number))
                            else str(v) for v in levels],
                           fontsize=8, rotation=45, ha="right")
        ax.set_xlim(-0.6, len(levels) - 0.4)
        ax.set_title(DIM_LABELS.get(dim, dim), fontsize=9)

    axes[0].set_ylabel("validation loss  (lower = better)")
    # The winner's ring is drawn at the minimum, so it needs headroom below or
    # the marker is clipped by the axis.
    axes[0].margins(y=0.10)
    fig.suptitle("Each dot is one configuration (mean over its restarts); "
                 "ring = selected; bar = level median", fontsize=8.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def configs_table(r) -> pd.DataFrame:
    hp = hp_columns(r)
    d = r.configs[["rank"] + hp + ["val_loss_mean", "val_loss_sd", "val_loss_min",
                                   "mean_max", "n_selected", "n_runs"]].copy()
    d.columns = (["rank"] + [DIM_LABELS.get(c, c) for c in hp]
                 + ["val loss (mean)", "val loss (sd)", "val loss (best)",
                    "mean_max", "features used", "fits"])
    return d.round(4)


def runs_table(r) -> pd.DataFrame:
    hp = hp_columns(r)
    d = (r.runs.sort_values("val_loss").reset_index(drop=True)
         [hp + ["restart", "val_loss", "mean_max", "n_selected", "features"]].copy())
    d.columns = ([DIM_LABELS.get(c, c) for c in hp]
                 + ["restart", "val loss", "mean_max", "# features",
                    "selected features"])
    return d.round(4)


def near_ties_note(r) -> str:
    """
    How close is the runner-up?  If the gap to rank 2 is inside one
    restart-to-restart sd the search did not really separate them, and saying so
    is more useful than a confident-looking single winner.
    """
    c, b = r.configs, r.best
    sd = b.get("val_loss_sd", np.nan)
    if len(c) < 2 or not np.isfinite(sd):
        return ""
    if float(c.val_loss_mean.iloc[1] - c.val_loss_mean.iloc[0]) >= float(sd):
        return ""
    n_near = int((c.val_loss_mean <= c.val_loss_mean.iloc[0] + sd).sum())
    return (f" {n_near} configurations sit within one restart sd of the winner, "
            f"so treat them as indistinguishable rather than ranked.")


def report_text(r) -> str:
    b, i = r.best, r.info
    buf = io.StringIO()
    w = buf.write
    w("DeepRAB subgroup analysis\n=========================\n\n")
    w(f"Endpoint            : {i['outcome']}  (loss {i['loss_name']})\n")
    w(f"Rows analysed       : {i['n_rows']}  ({i['n_dropped']} dropped)\n")
    w(f"Design columns      : {i['n_features']}\n")
    w(f"Split fit/val/test  : {i['n_fit']}/{i['n_val']}/{i['n_test']}\n")
    w(f"Treated fraction    : {i['treated_frac']:.3f}\n")
    if i.get("event_rate") is not None:
        w(f"Event rate          : {i['event_rate']:.3f}\n")
    w(f"Fits completed      : {i['n_fits_done']}/{i['n_fits_planned']}\n")
    w(f"Search              : {i['search']}  "
      f"({i['n_configs_fitted']} configurations x {i['restarts']} restarts, "
      f"base epochs {i['epochs_base']})\n")
    w(f"Dimensions searched : {', '.join(i['searched']) or 'none'}\n\n")
    w("Selected hyperparameters (by MEAN held-out validation loss across the\n")
    w("configuration's restarts; label-free)\n")
    w(f"  K = {b['K']}, lr = {b['lr']:g}, hidden = {arch_label(b['hidden'])}, "
      f"epochs = {b['epochs']}\n")
    w(f"  min_temp = {b['min_temp']:g}, dropout = {b['dropout']:g}, "
      f"l2 = {b['l2']:g}\n")
    w(f"  validation loss = {b['val_loss']:.6f}"
      f"   (best single restart {b['val_loss_best_run']:.6f}"
      f", sd {b['val_loss_sd']:.6f})   mean_max = {b['mean_max']:.4f}\n")
    w(f"  ensemble of top {b['n_ensemble']} fits "
      f"from {b['n_configs_ensembled']} configuration(s)\n")
    w(f"  features in the winning fit: {b['features']}\n")
    w("  (restarts of one configuration can select different features; the\n")
    w("   ensemble averages the contrast f over all of them.)\n")
    tie = near_ties_note(r)
    if tie:
        w(f" {tie.strip()}\n")
    w("\n")
    w(f"Subgroup separation  validation = {i['val_gain']:+.4f}   "
      f"test = {i['test_gain']:+.4f}\n")
    for k in ("auc_all", "auc_test"):
        if k in i:
            w(f"{k} = {i[k]:.4f}\n")
    w("\nFeature selection probabilities\n")
    w(r.features.round(4).to_string(index=False))
    w("\n\nTuning table\n")
    w(configs_table(r).to_string(index=False))
    if r.messages:
        w("\n\nNotes\n")
        for m in r.messages:
            w(f"  - {m}\n")
    return buf.getvalue()


# =================================================================== the UI #
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_file("file", "1. Trial data (CSV)", accept=[".csv"], multiple=False),
        ui.input_radio_buttons("outcome", "2. Endpoint", ENDPOINTS, selected="continuous"),
        ui.output_ui("column_pickers"),
        ui.hr(),
        ui.input_radio_buttons(
            "pi_mode", "Treatment assignment",
            {"known": "Randomised, known ratio", "estimate": "Estimate from covariates"},
            selected="known",
        ),
        ui.panel_conditional(
            "input.pi_mode === 'known'",
            ui.input_numeric("pi_known", "P(treated)", value=0.5, min=0.05, max=0.95, step=0.05),
        ),
        ui.input_radio_buttons(
            "preset", "Tuning effort",
            {k: v["label"] for k, v in PRESETS.items()}, selected="fast",
        ),
        ui.output_ui("effort_note"),
        ui.accordion(
            ui.accordion_panel(
                "Advanced",
                ui.input_radio_buttons(
                    "search", "Search strategy",
                    {"grid": "Grid (K x learning rate only)",
                     "random": "Random (also architecture, epochs, min_temp)"},
                    selected="grid",
                ),
                ui.input_checkbox_group("K_values", "K (number of selected features)",
                                        {str(k): str(k) for k in SEARCH_SPACE["K"]},
                                        selected=["2", "3", "4"], inline=True),
                ui.input_checkbox_group("lr_values", "Learning rate",
                                        {f"{v:g}": f"{v:g}" for v in SEARCH_SPACE["lr"]},
                                        selected=["0.001", "0.005"], inline=True),
                # Random-only dimensions.  Hidden in grid mode, where they are
                # pinned to the single legacy default and would just be
                # misleading controls that do nothing.
                ui.panel_conditional(
                    "input.search === 'random'",
                    ui.input_numeric("n_configs", "Configurations to sample",
                                     50, min=1, max=500, step=10),
                    ui.input_checkbox_group(
                        "hidden_values", "Hidden layers (nodes per layer)",
                        {arch_label(h): arch_label(h) for h in ARCHITECTURES},
                        selected=[arch_label(h) for h in ARCHITECTURES], inline=True),
                    ui.input_checkbox_group(
                        "epochs_mult_values", "Epoch budget (x the preset base)",
                        {f"{m:g}": f"{m:g}x" for m in SEARCH_SPACE["epochs_mult"]},
                        selected=[f"{m:g}" for m in SEARCH_SPACE["epochs_mult"]],
                        inline=True),
                    ui.input_checkbox_group(
                        "min_temp_values", "Final temperature (min_temp)",
                        {f"{v:g}": f"{v:g}" for v in (0.01, 0.02, 0.05, 0.1)},
                        selected=["0.01", "0.02", "0.05", "0.1"], inline=True),
                    ui.input_checkbox_group(
                        "dropout_values", "Dropout rate (after each ReLU)",
                        {f"{v:g}": f"{v:g}" for v in (0.0, 0.1, 0.2, 0.3)},
                        selected=["0"], inline=True),
                    ui.input_checkbox_group(
                        "l2_values", "L2 penalty",
                        {f"{v:g}": f"{v:g}" for v in (0.0, 1e-5, 1e-4, 1e-3)},
                        selected=["0"], inline=True),
                    ui.help_text(
                        "Tick more than one dropout or L2 value to add that "
                        "dimension to the search; a single value pins it."
                    ),
                ),
                ui.input_numeric("restarts", "Restarts per configuration", 1, min=1, max=10),
                ui.input_numeric("epochs", "Base epochs (0 = preset default)", 0,
                                 min=0, max=5000, step=50),
                ui.input_numeric("top_m", "Ensemble size (top-M runs)", 3, min=1, max=20),
                ui.input_slider("val_frac", "Validation fraction (tuning)", 0.1, 0.4, 0.25, step=0.05),
                ui.input_slider("test_frac", "Test fraction (held out)", 0.0, 0.4, 0.20, step=0.05),
                ui.input_numeric("seed", "Random seed", 1, min=0, max=10**6),
            ),
            open=False,
        ),
        ui.hr(),
        ui.input_action_button("run", "Run analysis", class_="btn-primary w-100"),
        ui.output_ui("run_status"),
        width=340,
    ),
    ui.navset_tab(
        ui.nav_panel(
            "Data",
            ui.output_ui("data_summary"),
            ui.output_ui("messages_ui"),
            ui.h5("Preview"),
            ui.output_data_frame("preview_tbl"),
        ),
        ui.nav_panel(
            "Tuning",
            ui.output_ui("best_card"),
            ui.h5("Validation loss by hyperparameter configuration"),
            ui.p(ui.tags.small(
                "Lower is better. The loss is the held-out A-learning loss matching your "
                "endpoint (squared / logistic / Cox partial likelihood) computed on the "
                "validation split. It uses no responder label, so this is the criterion "
                "you can actually apply to real data."
            )),
            ui.output_plot("tuning_plot", height="380px"),
            ui.output_data_frame("configs_tbl"),
            ui.h5("Individual fits"),
            ui.output_data_frame("runs_tbl"),
        ),
        ui.nav_panel(
            "Feature selection",
            ui.output_ui("feature_note"),
            ui.output_plot("feature_plot", height="460px"),
            ui.output_data_frame("features_tbl"),
        ),
        ui.nav_panel(
            "Predicted subgroups",
            ui.output_ui("subgroup_controls"),
            ui.output_ui("subgroup_cards"),
            ui.h5("Estimated treatment effect within each predicted subgroup"),
            ui.output_data_frame("subgroup_tbl"),
            ui.output_plot("score_plot", height="380px"),
        ),
        ui.nav_panel(
            "Download",
            ui.p("Per-subject predictions, the feature ranking, and the tuning table."),
            ui.download_button("dl_subjects", "Subject-level predictions (CSV)",
                               class_="btn-outline-primary mb-2"),
            ui.br(),
            ui.download_button("dl_features", "Feature selection probabilities (CSV)",
                               class_="btn-outline-primary mb-2"),
            ui.br(),
            ui.download_button("dl_configs", "Tuning results (CSV)",
                               class_="btn-outline-primary mb-2"),
            ui.br(),
            ui.download_button("dl_report", "Run summary (TXT)",
                               class_="btn-outline-primary"),
        ),
        ui.nav_panel("How to read this", ui.output_ui("help_ui")),
    ),
    title="DeepRAB subgroup explorer",
    fillable=False,
)


# =============================================================== the server #
def server(input, output, session):
    raw = reactive.value(None)          # uploaded DataFrame
    result = reactive.value(None)       # AnalysisResult
    err = reactive.value("")
    prog = {"frac": 0.0, "msg": ""}
    lock = threading.Lock()

    def iget(name, default=None):
        """
        Read an input that may not exist yet.  The outcome-specific selects are
        created by a render.ui, so on the first pass y_col / time_col / event_col
        are simply absent; treating that as `default` is cleaner than guarding
        every call site.
        """
        try:
            return getattr(input, name)()
        except Exception:
            return default

    # ------------------------------------------------------------- ingestion #
    @reactive.effect
    @reactive.event(input.file)
    def _load():
        f = input.file()
        if not f:
            return
        try:
            df = pd.read_csv(f[0]["datapath"])
            df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
            if df.shape[1] < 3:
                raise ValueError("need at least 3 columns")
            raw.set(df)
            result.set(None)
            err.set("")
        except Exception as exc:
            raw.set(None)
            err.set(f"Could not read that file: {exc}")

    @render.ui
    def column_pickers():
        df = raw()
        if df is None:
            return ui.help_text("Upload a CSV to choose columns.")
        cols = list(df.columns)
        num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        oc = input.outcome()

        def guess(cands, pool):
            low = {c.lower(): c for c in pool}
            for c in cands:
                if c in low:
                    return low[c]
            return pool[0] if pool else None

        if oc == "tte":
            outcome_inputs = [
                ui.input_select("time_col", "Follow-up time", num,
                                selected=guess(["time", "t", "aval", "days"], num)),
                ui.input_select("event_col", "Event indicator (1 = event)", cols,
                                selected=guess(["event", "status", "cnsr", "died"], cols)),
            ]
        else:
            outcome_inputs = [
                ui.input_select("y_col", "Outcome", num,
                                selected=guess(["y", "resp", "response", "chg", "aval"], num)),
            ]

        reserved = {"treatment", "trt", "arm", "group", "y", "time", "t", "event",
                    "status", "cnsr", "sigpo", "sigpos", "id", "usubjid", "subjid"}
        default_x = [c for c in num if c.lower() not in reserved]

        return ui.div(
            *outcome_inputs,
            ui.input_select("trt_col", "Treatment (2 levels)", cols,
                            selected=guess(["treatment", "trt", "arm", "group"], cols)),
            ui.input_selectize("x_cols", "Candidate biomarkers / covariates", cols,
                               selected=default_x, multiple=True),
            ui.input_select("truth_col", "Known responder label (optional)",
                            [NONE] + cols, selected=NONE),
            ui.help_text("Non-numeric covariates with <= 12 levels are one-hot encoded."),
        )

    @reactive.effect
    @reactive.event(input.preset)
    def _sync_preset():
        """
        Push the preset into the Advanced controls.

        Without this the preset is cosmetic: _spec() reads K / lr / restarts from
        Advanced, so those values would silently override whatever preset was
        chosen. Writing the preset into the widgets keeps one source of truth and
        lets the user see exactly which grid they are about to run, then tweak it.
        """
        p = PRESETS[input.preset()]
        ui.update_radio_buttons("search", selected=p["search"])
        ui.update_numeric("restarts", value=int(p["restarts"]))
        if p["search"] == "grid":
            ui.update_checkbox_group("K_values", selected=[str(k) for k in p["K"]])
            ui.update_checkbox_group("lr_values", selected=[f"{v:g}" for v in p["lr"]])
        else:
            # The wide presets search the full space, so open every dimension
            # back up -- otherwise a user who narrows K under Fast and then
            # switches to Wide gets a "wide" search over three K values.
            ui.update_numeric("n_configs", value=int(p["n_configs"]))
            ui.update_checkbox_group("K_values",
                                     selected=[str(k) for k in SEARCH_SPACE["K"]])
            ui.update_checkbox_group("lr_values",
                                     selected=[f"{v:g}" for v in SEARCH_SPACE["lr"]])
            ui.update_checkbox_group(
                "hidden_values", selected=[arch_label(h) for h in ARCHITECTURES])
            ui.update_checkbox_group(
                "epochs_mult_values",
                selected=[f"{m:g}" for m in SEARCH_SPACE["epochs_mult"]])
            ui.update_checkbox_group("min_temp_values",
                                     selected=["0.01", "0.02", "0.05", "0.1"])

    @render.ui
    def effort_note():
        try:
            spec = _spec()
        except Exception:
            return ui.help_text("")
        n = spec.n_fits()
        per = 2.5 if spec.outcome != "tte" else 4.0
        # In random mode n is an upper bound: K is clamped to the number of
        # design columns and configs are deduped, both of which need the data.
        lead = "up to " if spec.search == "random" else ""
        note = (f"{lead}{n} model fits, roughly {int(n * per)}-{int(n * per * 2)} s "
                f"plus nuisance fitting.")
        if spec.search == "random":
            note += (f"  {spec.n_configs} configurations x {spec.restarts} restarts; "
                     f"configs are ranked by mean validation loss.")
        return ui.help_text(note)

    # ------------------------------------------------------------------ spec #
    def _spec() -> AnalysisSpec:
        oc = input.outcome()
        xs = tuple(iget("x_cols") or ())
        Ks = tuple(sorted(int(k) for k in (iget("K_values") or ["2", "3", "4"])))
        lrs = tuple(sorted(float(v) for v in (iget("lr_values") or ["0.001", "0.005"])))
        truth = iget("truth_col", NONE)
        search = iget("search", "grid") or "grid"

        def _floats(name, fallback):
            """Checkbox groups return strings, and an empty tick-list means
            'fall back to the pinned default' rather than 'search nothing'."""
            vals = iget(name) or ()
            return tuple(sorted(float(v) for v in vals)) or tuple(fallback)

        archs = tuple(arch_from_label(h) for h in (iget("hidden_values") or ()))
        return AnalysisSpec(
            outcome=oc,
            y_col=(iget("y_col") if oc != "tte" else None),
            time_col=(iget("time_col") if oc == "tte" else None),
            event_col=(iget("event_col") if oc == "tte" else None),
            trt_col=iget("trt_col", "") or "",
            x_cols=xs,
            truth_col=(truth if truth and truth != NONE else None),
            pi_mode=iget("pi_mode", "known"),
            pi_known=float(iget("pi_known", 0.5) or 0.5),
            preset=iget("preset", "fast"),
            search=search,
            n_configs=int(iget("n_configs", 50) or 50),
            K_values=Ks,
            lr_values=lrs,
            hidden_values=(archs or None),
            epochs_mult_values=_floats("epochs_mult_values", (1.0,)),
            min_temp_values=_floats("min_temp_values", (0.05,)),
            dropout_values=_floats("dropout_values", (0.0,)),
            l2_values=_floats("l2_values", (0.0,)),
            restarts=int(iget("restarts", 1) or 1),
            epochs=(int(iget("epochs", 0) or 0) or None),
            top_m=int(iget("top_m", 3) or 3),
            val_frac=float(iget("val_frac", 0.25)),
            test_frac=float(iget("test_frac", 0.20)),
            seed=int(iget("seed", 1) or 0),
        )

    # ------------------------------------------------------- the long-running fit #
    @reactive.extended_task
    async def fit_task(df: pd.DataFrame, spec: AnalysisSpec):
        def cb(frac, msg):
            with lock:
                prog["frac"], prog["msg"] = float(frac), str(msg)

        # to_thread, not a direct call: run_analysis is CPU-bound and blocking, so
        # awaiting it inline would freeze the event loop -- and on Connect that
        # loop is shared by every concurrent session, not just this one.
        return await asyncio.to_thread(run_analysis, df, spec, cb)

    @reactive.effect
    @reactive.event(input.run)
    def _go():
        df = raw()
        if df is None:
            err.set("Upload a CSV first.")
            return
        try:
            spec = _spec()
            if len(spec.x_cols) < 2:
                raise ValueError("select at least 2 candidate covariates")
        except Exception as exc:
            err.set(str(exc))
            return
        err.set("")
        result.set(None)
        with lock:
            prog["frac"], prog["msg"] = 0.0, "Starting (TensorFlow loads on first run)"
        fit_task(df, spec)

    @reactive.effect
    def _collect():
        st = fit_task.status()
        if st == "success":
            try:
                result.set(fit_task.result())
            except Exception as exc:
                err.set(f"{exc}")
        elif st == "error":
            try:
                fit_task.result()
            except Exception as exc:
                err.set("".join(traceback.format_exception_only(type(exc), exc)).strip())

    @render.ui
    def run_status():
        if err():
            return ui.div(ui.tags.small(err()), class_="text-danger mt-2")
        if fit_task.status() == "running":
            reactive.invalidate_later(0.4)
            with lock:
                f, m = prog["frac"], prog["msg"]
            return ui.div(
                ui.tags.progress(value=f"{f:.3f}", max="1", style="width:100%"),
                ui.tags.small(m), class_="mt-2",
            )
        if result() is not None:
            return ui.div(ui.tags.small("Analysis complete."),
                          class_="text-success mt-2")
        return None

    def R():
        r = result()
        if r is None:
            raise RuntimeError("no result")
        return r

    # ------------------------------------------------------------ Data panel #
    @render.ui
    def data_summary():
        df = raw()
        if df is None:
            return ui.div(
                ui.h5("Start here"),
                ui.tags.ol(
                    ui.tags.li("Upload a CSV with one row per subject."),
                    ui.tags.li("Choose the endpoint type: continuous, binary, or time-to-event."),
                    ui.tags.li("Map the outcome, treatment, and candidate biomarker columns."),
                    ui.tags.li("Press Run analysis."),
                ),
                ui.p(ui.tags.small(
                    "Model selection uses a held-out A-learning loss and never a responder "
                    "label, so the workflow is valid on real trial data."
                )),
            )
        r = result()
        cards = [ui.value_box("Rows uploaded", f"{len(df):,}"),
                 ui.value_box("Columns", f"{df.shape[1]:,}")]
        if r is not None:
            i = r.info
            cards = [
                ui.value_box("Analysed", f"{i['n_rows']:,}",
                             ui.tags.small(f"{i['n_dropped']} dropped for missingness")),
                ui.value_box("Design columns", f"{i['n_features']:,}"),
                ui.value_box("Split (fit/val/test)",
                             f"{i['n_fit']}/{i['n_val']}/{i['n_test']}"),
                ui.value_box("Treated", f"{i['treated_frac']*100:.0f}%",
                             ui.tags.small(
                                 f"events {i['event_rate']*100:.0f}%"
                                 if i.get("event_rate") is not None else "")),
            ]
        return ui.layout_columns(*cards, col_widths=[3, 3, 3, 3])

    @render.ui
    def messages_ui():
        r = result()
        if r is None or not r.messages:
            return None
        return ui.div(
            ui.h6("Notes from the run"),
            ui.tags.ul(*[ui.tags.li(ui.tags.small(m)) for m in r.messages]),
            class_="alert alert-warning py-2",
        )

    @render.data_frame
    def preview_tbl():
        df = raw()
        if df is None:
            return pd.DataFrame()
        return render.DataGrid(df.head(50).round(4), height="320px")

    # ---------------------------------------------------------- Tuning panel #
    @render.ui
    def best_card():
        try:
            r = R()
        except Exception:
            return ui.help_text("Run an analysis to see tuning results.")
        b = r.best
        i = r.info
        extra = []
        if "auc_test" in i:
            extra.append(ui.value_box("AUC vs known label (test)", f"{i['auc_test']:.3f}"))
        close = near_ties_note(r)
        return ui.div(
            ui.layout_columns(
                ui.value_box("Selected K", str(b["K"])),
                ui.value_box("Selected learning rate", f"{b['lr']:g}"),
                ui.value_box("Hidden layers", arch_label(b["hidden"])),
                ui.value_box("Best mean validation loss", f"{b['val_loss']:.4f}"),
                ui.value_box("Ensembled fits", str(b["n_ensemble"])),
                *extra,
                col_widths=[3] * (5 + len(extra)),
            ),
            ui.p(ui.tags.small(
                f"Searched {i['n_configs_fitted']} configurations "
                f"({i['search']}) x {i['restarts']} restart(s); "
                f"epochs = {b['epochs']}, min_temp = {b['min_temp']:g}, "
                f"dropout = {b['dropout']:g}, L2 = {b['l2']:g}. "
                f"Configurations are ranked by MEAN validation loss across their "
                f"restarts, not by the single best fit.{close}"
            )),
            ui.p(ui.tags.small(
                f"Winning fit selected: {b['features']}. Concrete-layer sharpness "
                f"mean_max = {b['mean_max']:.3f}; a value well below 1 is normal and not "
                f"a failure -- the ranking of the selection logits settles long before "
                f"the distribution becomes one-hot."
            )),
        )

    @render.plot
    def tuning_plot():
        return tuning_figure(R())

    @render.data_frame
    def configs_tbl():
        return render.DataGrid(configs_table(R()), height="260px")

    @render.data_frame
    def runs_tbl():
        return render.DataGrid(runs_table(R()), height="300px")

    # -------------------------------------------------------- Features panel #
    @render.ui
    def feature_note():
        try:
            r = R()
        except Exception:
            return ui.help_text("Run an analysis to see the feature ranking.")
        return ui.div(
            ui.h5("Probability of selecting each candidate biomarker"),
            ui.p(ui.tags.small(
                "Selection probability is the concrete layer's own softmax mass: for each "
                "fit, the largest probability any selection slot puts on that feature, "
                "averaged over the ensembled fits. Selection frequency is the fraction of "
                "fits in which the feature was actually chosen (stability selection). "
                "Read them together: a feature with high probability AND frequency is a "
                "robust effect modifier. These are predictive (treatment-modifying) "
                "features, not prognostic ones -- the A-learning loss removes the "
                "prognostic signal before selection."
            )),
        )

    @render.plot
    def feature_plot():
        r = R()
        d = r.features.head(18).iloc[::-1]
        fig, ax = _fig(7.2, max(3.4, 0.34 * len(d)))
        ypos = np.arange(len(d))
        ax.barh(ypos, d.sel_prob_top, color=ACCENT, height=0.62,
                label="selection probability (ensemble)")
        ax.plot(d.sel_freq_top, ypos, "o", color="#dc7633", ms=7,
                label="selection frequency")
        ax.set_yticks(ypos)
        ax.set_yticklabels(d.feature)
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("probability / frequency")
        ax.grid(axis="x", color="#e2e8f0", lw=0.8)
        ax.grid(axis="y", visible=False)
        ax.legend(frameon=False, fontsize=9, loc="lower right")
        fig.tight_layout()
        return fig

    @render.data_frame
    def features_tbl():
        r = R()
        d = r.features.copy()
        d.columns = ["feature", "sel. probability (ensemble)", "sel. frequency (ensemble)",
                     "sel. probability (all fits)", "sel. frequency (all fits)"]
        return render.DataGrid(d.round(4), height="420px")

    # ------------------------------------------------------- Subgroups panel #
    @render.ui
    def subgroup_controls():
        try:
            r = R()
        except Exception:
            return ui.help_text("Run an analysis to see predicted subgroups.")
        s = r.subjects.benefit_score.to_numpy()
        lo, hi = float(np.quantile(s, 0.02)), float(np.quantile(s, 0.98))
        rng = max(abs(lo), abs(hi), 1e-6)
        return ui.div(
            ui.input_slider("cut", "Benefit-score threshold defining the subgroup",
                            round(-rng, 3), round(rng, 3), 0.0,
                            step=round(2 * rng / 100, 4)),
            ui.help_text(
                "Subjects with benefit score above the threshold are the predicted "
                "benefit subgroup. Zero is the natural cut: it is the point where the "
                "estimated treatment effect changes sign. Move it to trade subgroup size "
                "against effect size."
            ),
        )

    @reactive.calc
    def sub_summary():
        r = R()
        cut = float(iget("cut", 0.0) or 0.0)
        s = r.subjects
        oc = r.info["outcome"]
        y = (s["time"] if oc == "tte" else s["outcome"]).to_numpy(dtype=float)
        ev = s["event"].to_numpy(dtype=float) if oc == "tte" else None
        a = s["treatment"].to_numpy(dtype=float)
        pi = s["propensity"].to_numpy(dtype=float)
        bs = s["benefit_score"].to_numpy(dtype=float)

        label = "log hazard ratio" if oc == "tte" else (
            "risk difference" if oc == "binary" else "mean difference")
        out = []
        for name, mask in (("Predicted benefit", bs > cut),
                           ("Predicted no benefit", bs <= cut),
                           ("All subjects", np.ones(len(bs), dtype=bool))):
            est, lo, hi, nn, nev = subgroup_effect(oc, y, ev, a, pi, mask,
                                                   n_boot=r.info.get("n_boot", 100),
                                                   seed=r.info.get("seed", 0))
            row = {"subgroup": name, "n": nn,
                   "% of total": round(100 * nn / len(bs), 1),
                   f"treatment effect ({label})": est,
                   "95% CI low": lo, "95% CI high": hi}
            if oc == "tte":
                row["events"] = int(nev) if np.isfinite(nev) else None
                row["hazard ratio"] = float(np.exp(est)) if np.isfinite(est) else None
            out.append(row)
        return pd.DataFrame(out), label, cut

    @render.ui
    def subgroup_cards():
        try:
            tbl, label, cut = sub_summary()
            r = R()
        except Exception:
            return None
        ben = tbl.iloc[0]
        non = tbl.iloc[1]
        col = f"treatment effect ({label})"
        cards = [
            ui.value_box("Predicted benefit subgroup",
                         f"{int(ben['n'])} ({ben['% of total']:.0f}%)"),
            ui.value_box(f"Effect there ({label})",
                         "n/a" if pd.isna(ben[col]) else f"{ben[col]:+.3f}"),
            ui.value_box(f"Effect elsewhere ({label})",
                         "n/a" if pd.isna(non[col]) else f"{non[col]:+.3f}"),
        ]
        g = r.info.get("test_gain", np.nan)
        if np.isfinite(g):
            cards.append(ui.value_box("Separation on held-out test", f"{g:+.3f}"))
        return ui.div(
            ui.layout_columns(*cards, col_widths=[3] * len(cards)),
            ui.p(ui.tags.small(
                "The two effects above are estimated inside groups the model itself "
                "defined, so they are optimistically biased on the data used to fit. The "
                "held-out separation is the honest number; use the test split for any "
                "claim you intend to report."
                + ("  For a hazard ratio below 1, treatment reduces the hazard."
                   if r.info["outcome"] == "tte" else "")
            )),
        )

    @render.data_frame
    def subgroup_tbl():
        tbl, _, _ = sub_summary()
        return render.DataGrid(tbl.round(4), height="200px")

    @render.plot
    def score_plot():
        r = R()
        _, label, cut = sub_summary()
        bs = r.subjects.benefit_score.to_numpy()
        fig, ax = _fig()
        bins = np.linspace(bs.min(), bs.max(), 44)
        ax.hist(bs[bs <= cut], bins=bins, color=GREY, label="predicted no benefit")
        ax.hist(bs[bs > cut], bins=bins, color=ACCENT, label="predicted benefit")
        ax.axvline(cut, color="#111827", lw=1.6, ls="--")
        ax.set_xlabel(f"benefit score   (oriented so higher = more benefit; {label} scale)")
        ax.set_ylabel("subjects")
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        return fig

    # -------------------------------------------------------------- downloads #
    @render.download(filename="deeprab_subject_predictions.csv")
    def dl_subjects():
        yield R().subjects.to_csv(index=False)

    @render.download(filename="deeprab_feature_probabilities.csv")
    def dl_features():
        yield R().features.to_csv(index=False)

    @render.download(filename="deeprab_tuning.csv")
    def dl_configs():
        yield R().configs.to_csv(index=False)

    @render.download(filename="deeprab_run_summary.txt")
    def dl_report():
        yield report_text(R())

    # ------------------------------------------------------------------ help #
    @render.ui
    def help_ui():
        return ui.div(
            ui.h5("What the model does"),
            ui.p(
                "It looks for a small set of baseline variables that change the TREATMENT "
                "EFFECT, not variables that predict the outcome. Those are different "
                "questions: a strong prognostic marker can be useless for deciding who to "
                "treat. The model removes the prognostic part of the signal first (a "
                "cross-fitted baseline model, entered as an offset), then a concrete "
                "autoencoder selects K features and a small network maps them to a "
                "contrast f(x) - the estimated treatment effect for a subject with "
                "covariates x."
            ),
            ui.h5("How a model gets chosen"),
            ui.p(
                "Every candidate is scored by its A-learning loss on a validation split "
                "that played no part in fitting it. That loss needs no responder label, "
                "which is what makes the procedure usable on a real trial. Minimising it "
                "is equivalent to minimising squared error against the true contrast "
                "function, so it is not a heuristic stand-in."
            ),
            ui.p(
                "Configurations are compared on the MEAN validation loss over their "
                "restarts, and the winner is the best configuration - not the single "
                "luckiest fit. Restarts of one configuration differ only by random "
                "initialisation and Gumbel noise, so averaging them first is what makes "
                "the comparison about the hyperparameters instead of about the seed."
            ),
            ui.h5("Grid versus random search"),
            ui.p(
                "Fast, Balanced and Thorough vary two things: K and the learning rate. "
                "Thorough is 5 K x 2 learning rates x 3 restarts = 30 fits, so only 10 "
                "distinct configurations; the network itself is fixed at two hidden "
                "layers of 32 and 16 nodes. Wide and Exhaustive additionally search the "
                "architecture (depth and width together), the epoch budget and min_temp, "
                "drawing configurations at random rather than enumerating them - above "
                "two or three dimensions a grid coarse enough to afford covers the space "
                "worse than the same number of random draws. Dropout and L2 are available "
                "under Advanced but pinned off by default."
            ),
            ui.p(ui.HTML(
                "<b>A wider search is not automatically a better answer.</b> Taking the "
                "minimum over 100 candidates on one validation split is a more optimistic "
                "statistic than the minimum over 10, and the winner's validation loss is "
                "biased low by an amount that grows with the number of candidates. "
                "Ranking on the restart mean and ensembling the winner's fits both damp "
                "this, but they do not remove it. Whenever the top few configurations sit "
                "within one restart-to-restart sd of each other, the honest reading is "
                "that the search could not tell them apart - look at whether the feature "
                "ranking is stable across them rather than at which one came first."
            )),
            ui.h5("Reading the numbers"),
            ui.tags.ul(
                ui.tags.li(ui.HTML(
                    "<b>Selection probability</b> and <b>frequency</b> answer different "
                    "questions - how much softmax mass a feature attracts, versus how "
                    "often it is actually picked. Trust features that score high on both.")),
                ui.tags.li(ui.HTML(
                    "<b>Effects inside predicted subgroups are optimistic</b> on the data "
                    "that defined them. The held-out test split is the honest number.")),
                ui.tags.li(ui.HTML(
                    "<b>K slightly above the number of real modifiers is safer</b> than K "
                    "exactly right: spare slots act as insurance, and the extra features "
                    "show up with low selection frequency.")),
                ui.tags.li(ui.HTML(
                    "<b>mean_max well below 1 is normal.</b> The ordering of the selection "
                    "logits stabilises long before the softmax becomes one-hot.")),
                ui.tags.li(ui.HTML(
                    "<b>Time-to-event sign convention:</b> the raw contrast is a log hazard "
                    "ratio, so negative means benefit. The app already flips it, so a "
                    "higher benefit score always means more benefit.")),
            ),
            ui.h5("Limits worth stating"),
            ui.tags.ul(
                ui.tags.li("A single train/test split, so the reported numbers carry "
                           "split-to-split variability. Re-run with different seeds."),
                ui.tags.li("Findings are exploratory and hypothesis-generating; they are "
                           "not a pre-specified subgroup analysis."),
                ui.tags.li("With 'Estimate from covariates' the propensity model must be "
                           "adequate. For a randomised trial prefer the known ratio."),
                ui.tags.li("A categorical variable with C levels becomes C-1 columns, each "
                           "ranked separately, meaning 'this level vs the reference'."),
            ),
            class_="mt-2",
        )


app = App(app_ui, server)
