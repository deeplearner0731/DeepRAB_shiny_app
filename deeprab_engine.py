"""
Analysis engine behind the DeepRAB Shiny app.

Deliberately free of any Shiny import so it can be unit-tested, scripted, or
called from a notebook.  `run_analysis(df, spec, progress)` does everything and
returns an `AnalysisResult`.

Design notes for the app setting (differs from demo_code_Simple_tunning_v2.py)
-----------------------------------------------------------------------------
* ONE fit/validation/test split, not 10 resamples.  Ten outer splits are the
  right way to estimate the variance of a *method*; a clinician analysing one
  real trial wants one answer, plus honest uncertainty on that answer.
* tryout_limit = 1.  The selector's retry-with-2x-epochs loop triples runtime
  chasing mean_max >= 0.998.  Empirically the *ranking* of the concrete logits
  settles long before the softmax sharpens (the best configs on our benchmark
  sat at mean_max ~ 0.5 and still selected the right features in 100% of runs),
  so the retry buys almost nothing and costs 3x.  mean_max is reported so you
  can see where it landed.
* The ensemble averages the RAW contrast f, not its ranks.  f is on an
  interpretable scale in every endpoint (mean difference / log OR / log HR) and
  all runs estimate the same tau(x), so the average is itself a valid estimate
  and -- unlike a rank average -- it preserves the f = 0 threshold that defines
  the subgroup.
* Nuisances are cross-fitted within the training portion, and a separate
  full-training-portion fit supplies the offset for held-out rows.  No test row
  ever contributes to its own nuisance estimate.

Hyperparameter search
---------------------
Two modes, both driven by the same `build_configs` / job-list code path:

* "grid"   -- full factorial over K x lr with the architecture, epochs, dropout,
              l2 and min_temp pinned.  This is the original two-dimensional
              behaviour, kept so the fast / balanced / thorough presets still
              reproduce previously reported numbers exactly.
* "random" -- seeded draws from SEARCH_SPACE, which also varies the decoder
              architecture (depth and width together), the epoch budget (as a
              multiplier on the endpoint's base, never an absolute count) and
              min_temp.  Above two or three dimensions a full factorial coarse
              enough to afford covers the space worse than the same number of
              random draws, so the wide presets sample rather than enumerate.

Configs are ranked by MEAN validation loss across their restarts, and the
winner is the rank-1 config -- not the single best individual fit.  The minimum
over many single fits on one validation split is a badly biased statistic, and
the bias grows with the number of candidates, so a wider search needs the
sturdier selection rule to be an improvement rather than a regression.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- endpoints #
ENDPOINTS = {
    "continuous": "Continuous",
    "binary": "Binary (0/1)",
    "tte": "Time-to-event (Cox)",
}

# Architectures offered to the search.  Depth and width are ONE dimension, not
# two: a 3-layer net needs 3 widths, so "number of layers" and "nodes per layer"
# cannot be varied independently without generating nonsense like (16, 128).
ARCHITECTURES = ((16,), (32,), (32, 16), (64, 32), (64, 32, 16), (128, 64, 32))


def arch_label(hidden) -> str:
    """(64, 32, 16) -> '64-32-16'.  Groupable, CSV-safe, readable in a table."""
    return "-".join(str(int(h)) for h in hidden)


def arch_from_label(label: str) -> tuple:
    return tuple(int(p) for p in str(label).split("-") if p != "")


# The random-search space.  Every entry is the *support* of one dimension; a
# single-valued tuple pins that dimension and contributes nothing to the search,
# which is how dropout and l2 stay switched off by default.
#
# epochs_mult is a MULTIPLIER, never an absolute epoch count: Cox trains full
# batch (1 epoch == 1 gradient step) and needs ~1200 where continuous needs
# ~400, so an absolute value that suits one endpoint starves the other.
SEARCH_SPACE = dict(
    K=(1, 2, 3, 4, 5, 6, 8, 10),
    lr=(5e-4, 1e-3, 2e-3, 5e-3, 1e-2),
    hidden=ARCHITECTURES,
    epochs_mult=(0.5, 0.75, 1.0, 1.5, 2.0),
    min_temp=(0.01, 0.02, 0.05, 0.1),   # the selector needs <= 0.1 to sharpen
    dropout=(0.0,),
    l2=(0.0,),
)

# Speed presets.
#   search="grid"   -> full factorial over K x lr, everything else pinned.
#                      n_fits = len(K) * len(lr) * restarts
#   search="random" -> n_configs draws from the space below, x restarts.
#
# fast / balanced / thorough are unchanged from the two-dimensional original so
# that a previously reported run still reproduces exactly; the wider searches
# are new presets rather than a redefinition of the old ones.
PRESETS = {
    "fast": dict(search="grid", K=(2, 3, 4), lr=(1e-3, 5e-3), restarts=1,
                 epochs_std=250, epochs_tte=600, label="Fast"),
    "balanced": dict(search="grid", K=(2, 3, 4, 6), lr=(1e-3, 5e-3), restarts=2,
                     epochs_std=300, epochs_tte=900, label="Balanced"),
    "thorough": dict(search="grid", K=(2, 3, 4, 6, 8), lr=(1e-3, 5e-3), restarts=3,
                     epochs_std=400, epochs_tte=1200, label="Thorough"),
    "wide": dict(search="random", n_configs=50, restarts=2,
                 epochs_std=400, epochs_tte=1200,
                 label="Wide (random, ~100 fits)"),
    "exhaustive": dict(search="random", n_configs=100, restarts=2,
                       epochs_std=400, epochs_tte=1200,
                       label="Exhaustive (random, ~200 fits)"),
}


def _space_from(spec, p: dict) -> dict:
    """
    The search space actually used: SEARCH_SPACE defaults, overridden by any
    per-dimension tuple the caller set on the spec, and -- in grid mode -- with
    K and lr taken from the preset so the old two-dimensional behaviour is
    reproduced by the same code path.
    """
    sp = dict(SEARCH_SPACE)
    for dim, val in (("K", spec.K_values), ("lr", spec.lr_values),
                     ("hidden", spec.hidden_values),
                     ("epochs_mult", spec.epochs_mult_values),
                     ("min_temp", spec.min_temp_values),
                     ("dropout", spec.dropout_values),
                     ("l2", spec.l2_values)):
        if val:
            sp[dim] = tuple(val)
    if p["search"] == "grid":
        # Grid mode varies K and lr only; everything else collapses to the
        # single legacy default so len(grid) stays len(K) * len(lr).
        sp["K"] = tuple(spec.K_values or p["K"])
        sp["lr"] = tuple(spec.lr_values or p["lr"])
        sp["hidden"] = (tuple(spec.hidden),)
        sp["epochs_mult"] = (1.0,)
        sp["min_temp"] = (float(spec.min_temp),)
        sp["dropout"] = (float(spec.dropout),)
        sp["l2"] = (float(spec.l2),)
    return sp


def _mk_config(K, lr, hidden, epochs, min_temp, dropout, l2) -> dict:
    return dict(K=int(K), lr=float(lr), hidden=tuple(int(h) for h in hidden),
                epochs=int(epochs), min_temp=float(min_temp),
                dropout=float(dropout), l2=float(l2))


def _config_key(c: dict) -> tuple:
    return (c["K"], c["lr"], c["hidden"], c["epochs"], c["min_temp"],
            c["dropout"], c["l2"])


def build_configs(space: dict, search: str, n_configs: int, seed: int,
                  n_features: int, base_epochs: int) -> tuple:
    """
    The list of hyperparameter configurations to fit, plus any notes.

    K is clamped to `n_features` BEFORE deduplication.  Without that, sampling
    K = 10 against a 5-column design matrix yields several configs that the fit
    silently collapses to K = 5 -- paying full price for duplicate work and
    inflating the apparent size of the search.

    Random search rather than a full factorial: with five live dimensions the
    product runs to thousands, and above two or three dimensions random draws
    cover the space better per fit than a grid coarse enough to afford.
    Seeded from `seed`, so a run is reproducible.
    """
    notes = []
    Ks = tuple(sorted({min(int(k), int(n_features)) for k in space["K"]}))
    dropped = sorted({int(k) for k in space["K"] if int(k) > n_features})
    if dropped:
        notes.append(f"K value(s) {', '.join(map(str, dropped))} exceed the "
                     f"{n_features} design columns and were clamped to {n_features}.")

    def _epochs(mult):
        return max(1, int(round(float(mult) * base_epochs)))

    if search == "grid":
        cfgs = [_mk_config(k, lr, h, _epochs(m), mt, dr, l2)
                for k in Ks
                for lr in space["lr"]
                for h in space["hidden"]
                for m in space["epochs_mult"]
                for mt in space["min_temp"]
                for dr in space["dropout"]
                for l2 in space["l2"]]
        seen, out = set(), []
        for c in cfgs:
            if _config_key(c) not in seen:
                seen.add(_config_key(c))
                out.append(c)
        return out, notes

    rng = np.random.RandomState(int(seed) % (2**31 - 1))
    n_target = max(1, int(n_configs))
    total = (len(Ks) * len(space["lr"]) * len(space["hidden"])
             * len(space["epochs_mult"]) * len(space["min_temp"])
             * len(space["dropout"]) * len(space["l2"]))
    seen, out = set(), []
    # Rejection sampling with a bounded attempt count: the space is far larger
    # than n_configs in normal use, but a user who pins most dimensions can make
    # it smaller than the budget, and that must terminate rather than spin.
    for _ in range(200 * n_target):
        if len(out) >= min(n_target, total):
            break
        c = _mk_config(
            K=Ks[rng.randint(len(Ks))],
            lr=space["lr"][rng.randint(len(space["lr"]))],
            hidden=space["hidden"][rng.randint(len(space["hidden"]))],
            epochs=_epochs(space["epochs_mult"][rng.randint(len(space["epochs_mult"]))]),
            min_temp=space["min_temp"][rng.randint(len(space["min_temp"]))],
            dropout=space["dropout"][rng.randint(len(space["dropout"]))],
            l2=space["l2"][rng.randint(len(space["l2"]))],
        )
        k = _config_key(c)
        if k not in seen:
            seen.add(k)
            out.append(c)
    if len(out) < n_target:
        notes.append(f"Requested {n_target} configurations but the search space "
                     f"only holds {len(out)} distinct ones; fitting {len(out)}.")
    return out, notes


@dataclass
class AnalysisSpec:
    """Everything the user chooses in the sidebar."""
    outcome: str = "continuous"          # continuous | binary | tte
    y_col: Optional[str] = None          # continuous / binary outcome
    time_col: Optional[str] = None       # tte
    event_col: Optional[str] = None      # tte
    trt_col: str = ""
    x_cols: tuple = ()
    truth_col: Optional[str] = None      # optional known responder label

    pi_mode: str = "estimate"            # "known" | "estimate"
    pi_known: float = 0.5

    preset: str = "fast"
    search: Optional[str] = None         # "grid" | "random"; None = preset's
    n_configs: Optional[int] = None      # random search budget (distinct configs)
    K_values: Optional[tuple] = None     # override the preset / search space
    lr_values: Optional[tuple] = None
    hidden_values: Optional[tuple] = None        # tuple of tuples
    epochs_mult_values: Optional[tuple] = None
    min_temp_values: Optional[tuple] = None
    dropout_values: Optional[tuple] = None
    l2_values: Optional[tuple] = None
    restarts: Optional[int] = None
    epochs: Optional[int] = None         # base epochs; 0/None = preset's
    hidden: tuple = (32, 16)             # grid mode: the pinned architecture
    l2: float = 0.0                      # grid mode: the pinned penalty
    dropout: float = 0.0                 # grid mode: the pinned dropout rate
    min_temp: float = 0.05               # grid mode: the pinned final temperature
    top_m: int = 3

    val_frac: float = 0.25
    test_frac: float = 0.20
    seed: int = 1
    n_boot: int = 100                    # bootstrap reps for subgroup effects

    def resolved(self) -> dict:
        p = PRESETS[self.preset]
        search = self.search or p["search"]
        base = self.epochs or (p["epochs_tte"] if self.outcome == "tte"
                               else p["epochs_std"])
        space = _space_from(self, dict(p, search=search))
        return dict(
            search=search,
            space=space,
            base_epochs=int(base),
            # 50 is the fallback for a spec that asks for random search under a
            # grid preset, which has no n_configs of its own.
            n_configs=int(self.n_configs or p.get("n_configs", 0) or
                          (50 if search == "random" else 0)),
            restarts=int(self.restarts or p["restarts"]),
            # kept for callers that only want the two legacy dimensions
            K=tuple(space["K"]),
            lr=tuple(space["lr"]),
        )

    def n_fits(self) -> int:
        """
        Planned fits.  In random mode this is an UPPER BOUND: K is clamped to the
        number of design columns and configs are deduped, both of which need the
        data and so can only lower the count at run time.
        """
        r = self.resolved()
        if r["search"] == "random":
            return max(1, r["n_configs"]) * r["restarts"]
        sp = r["space"]
        n_cfg = (len(sp["K"]) * len(sp["lr"]) * len(sp["hidden"])
                 * len(sp["epochs_mult"]) * len(sp["min_temp"])
                 * len(sp["dropout"]) * len(sp["l2"]))
        return n_cfg * r["restarts"]


@dataclass
class AnalysisResult:
    runs: pd.DataFrame               # one row per (config, restart)
    configs: pd.DataFrame            # val-loss aggregated per config
    features: pd.DataFrame           # selection probability per feature
    subjects: pd.DataFrame           # per-subject contrast + subgroup
    best: dict                       # winning hyperparameters
    info: dict                       # counts, split sizes, diagnostics
    messages: list = field(default_factory=list)


# =================================================================== helpers #
def _set_seed(seed: int) -> None:
    import tensorflow as tf
    seed = int(seed) % (2**31 - 1)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_design_matrix(df: pd.DataFrame, x_cols) -> tuple:
    """
    Numeric columns pass through; low-cardinality non-numeric columns are
    one-hot encoded (first level dropped).  Returns (X_df, notes).

    Real clinical exports are rarely all-numeric, so this is not optional.  A
    categorical with C levels contributes C-1 columns, each ranked separately in
    the feature table -- read those as "this level vs the reference", not as one
    variable.
    """
    parts, notes = [], []
    for c in x_cols:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            parts.append(s.astype(float).rename(c))
            continue
        nun = s.nunique(dropna=True)
        if nun < 2:
            notes.append(f"dropped '{c}': constant")
            continue
        if nun > 12:
            notes.append(f"dropped '{c}': non-numeric with {nun} levels (too many to encode)")
            continue
        dums = pd.get_dummies(s.astype("category"), prefix=c, drop_first=True)
        parts.append(dums.astype(float))
        notes.append(f"one-hot encoded '{c}' into {dums.shape[1]} column(s)")
    if not parts:
        raise ValueError("no usable covariates after encoding")
    X = pd.concat(parts, axis=1)
    const = [c for c in X.columns if X[c].nunique() < 2]
    if const:
        X = X.drop(columns=const)
        notes.append(f"dropped constant column(s): {', '.join(const)}")
    if X.shape[1] == 0:
        raise ValueError("all covariates were constant")
    return X, notes


def _make_decoder(hidden, l2=0.0, dropout=0.0):
    """
    The (B, K) -> (B, 1) head.  `dropout` is applied after every ReLU, not after
    the linear output.  Inference is unaffected: predict_score() calls the model
    with training=False, so Dropout is a no-op there.
    """
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Dropout, ReLU

    reg = tf.keras.regularizers.l2(l2) if l2 else None
    rate = float(dropout or 0.0)

    def decoder(x):
        for h in hidden:
            x = Dense(h, kernel_regularizer=reg)(x)
            x = ReLU()(x)
            if rate > 0.0:
                x = Dropout(rate)(x)
        return Dense(1, kernel_regularizer=reg)(x)

    return decoder


def _nuisance_pi(X_tr, a_tr, X_all, spec):
    """Propensity for training rows (cross-fitted) and for all rows."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    if spec.pi_mode == "known":
        p = float(spec.pi_known)
        return np.full(len(a_tr), p), np.full(len(X_all), p)

    pi_tr = np.zeros(len(a_tr))
    n_min = int(np.bincount(a_tr.astype(int), minlength=2).min())
    if n_min < 2:
        raise ValueError("one treatment arm has fewer than 2 subjects")
    skf = StratifiedKFold(n_splits=max(2, min(5, n_min)),
                          shuffle=True, random_state=spec.seed)
    for tr, te in skf.split(X_tr, a_tr):
        m = LogisticRegression(max_iter=2000).fit(X_tr[tr], a_tr[tr])
        pi_tr[te] = m.predict_proba(X_tr[te])[:, 1]
    full = LogisticRegression(max_iter=2000).fit(X_tr, a_tr)
    pi_all = full.predict_proba(X_all)[:, 1]
    return np.clip(pi_tr, 0.05, 0.95), np.clip(pi_all, 0.05, 0.95)


def _nuisance_offset(outcome, X_tr, y_tr, e_tr, X_all, seed):
    """
    Cross-fitted baseline eta for training rows + a full-train model applied to
    every row, so held-out rows get an offset that never saw their own outcome.
    """
    from sklearn.linear_model import LogisticRegressionCV, RidgeCV
    from sklearn.model_selection import KFold, StratifiedKFold

    from Deep_learning_subgroup_v2 import cox_fit_ridge

    n = len(y_tr)
    off_tr = np.zeros(n)

    if outcome == "continuous":
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        for tr, te in kf.split(X_tr):
            m = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(X_tr[tr], y_tr[tr])
            off_tr[te] = m.predict(X_tr[te])
        full = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(X_tr, y_tr)
        return off_tr, full.predict(X_all)

    if outcome == "binary":
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr, te in skf.split(X_tr, y_tr):
            m = LogisticRegressionCV(Cs=10, cv=3, max_iter=5000,
                                     scoring="neg_log_loss").fit(X_tr[tr], y_tr[tr])
            p = np.clip(m.predict_proba(X_tr[te])[:, 1], 1e-4, 1 - 1e-4)
            off_tr[te] = np.log(p / (1 - p))
        full = LogisticRegressionCV(Cs=10, cv=3, max_iter=5000,
                                    scoring="neg_log_loss").fit(X_tr, y_tr)
        p_all = np.clip(full.predict_proba(X_all)[:, 1], 1e-4, 1 - 1e-4)
        return off_tr, np.log(p_all / (1 - p_all))

    # tte
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X_tr, e_tr):
        beta = cox_fit_ridge(X_tr[tr], y_tr[tr], e_tr[tr], alpha=1e-2)
        off_tr[te] = X_tr[te] @ beta
    beta_full = cox_fit_ridge(X_tr, y_tr, e_tr, alpha=1e-2)
    return off_tr, X_all @ beta_full


# =========================================================== subgroup effects #
def subgroup_effect(outcome, y, event, trt, pi, mask, n_boot=100, seed=0):
    """
    Treatment effect inside `mask`, with a bootstrap CI.

    continuous / binary : IPW mean difference (risk difference if binary)
    tte                 : treatment log hazard ratio from a 1-covariate Cox fit
                          (exp() it for the HR)
    Returns (estimate, lo, hi, n, n_events_or_nan).
    """
    from Deep_learning_subgroup_v2 import cox_fit_ridge, ipw_effect

    idx = np.flatnonzero(mask)
    n = len(idx)
    nev = float(np.sum(event[idx])) if event is not None else np.nan
    if n < 10 or len(np.unique(trt[idx])) < 2:
        return np.nan, np.nan, np.nan, n, nev

    def est(ix):
        if outcome == "tte":
            if np.sum(event[ix]) < 3 or len(np.unique(trt[ix])) < 2:
                return np.nan
            return float(cox_fit_ridge(trt[ix], y[ix], event[ix], alpha=1e-6)[0])
        return ipw_effect(y[ix], trt[ix], pi[ix], None)

    point = est(idx)
    rng = np.random.RandomState(seed)
    boots = []
    for _ in range(int(n_boot)):
        b = rng.choice(idx, size=n, replace=True)
        v = est(b)
        if np.isfinite(v):
            boots.append(v)
    if len(boots) < 20:
        return point, np.nan, np.nan, n, nev
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi), n, nev


# ================================================================== analysis #
def run_analysis(df: pd.DataFrame, spec: AnalysisSpec,
                 progress: Optional[Callable[[float, str], None]] = None
                 ) -> AnalysisResult:
    import tensorflow as tf

    from Deep_learning_subgroup_v2 import (
        ConcreteAutoencoderFeatureSelector,
        benefit_score,
        check_y01,
        cox_subgroup_gain,
        held_out_loss,
        outcome_to_loss_name,
        subgroup_gain,
        to_a01,
    )

    def tick(frac, msg):
        if progress:
            progress(frac, msg)

    msgs = []
    oc = spec.outcome
    res = spec.resolved()
    tick(0.02, "Validating columns")

    # ---------------------------------------------------- column validation #
    need = [spec.trt_col] + list(spec.x_cols)
    need += [spec.time_col, spec.event_col] if oc == "tte" else [spec.y_col]
    if spec.truth_col:
        need.append(spec.truth_col)
    need = [c for c in need if c]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"columns not found in the data: {', '.join(missing)}")

    n0 = len(df)
    data = df[need].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if len(data) < n0:
        msgs.append(f"Dropped {n0 - len(data)} of {n0} rows with missing or "
                    f"infinite values in the selected columns.")
    if len(data) < 60:
        raise ValueError(f"only {len(data)} complete rows; need at least 60")

    X_df, enc_notes = build_design_matrix(data, spec.x_cols)
    msgs.extend(enc_notes)
    feat_names = list(X_df.columns)
    X_raw = X_df.to_numpy(dtype=float)

    a_all = to_a01(data[spec.trt_col].to_numpy())
    if len(np.unique(a_all)) != 2:
        raise ValueError(f"'{spec.trt_col}' must have exactly 2 levels")

    if oc == "tte":
        y_all = data[spec.time_col].to_numpy(dtype=float)
        e_all = check_y01(data[spec.event_col].to_numpy(), "event")
        if np.any(y_all <= 0):
            raise ValueError("follow-up times must be strictly positive")
        if e_all.sum() < 20:
            raise ValueError(f"only {int(e_all.sum())} events; need at least 20")
    else:
        y_all = data[spec.y_col].to_numpy(dtype=float)
        e_all = None
        if oc == "binary":
            y_all = check_y01(y_all, spec.y_col)

    truth = (data[spec.truth_col].to_numpy() if spec.truth_col else None)

    # ------------------------------------------------------------- splitting #
    n = len(data)
    rng = np.random.RandomState(spec.seed)
    perm = rng.permutation(n)
    n_test = int(round(spec.test_frac * n))
    n_val = int(round(spec.val_frac * (n - n_test)))
    test_idx = perm[:n_test]
    val_idx = perm[n_test:n_test + n_val]
    fit_idx = perm[n_test + n_val:]
    train_idx = np.concatenate([fit_idx, val_idx])
    if len(fit_idx) < 40:
        raise ValueError("fitting split too small; lower the validation/test fractions")

    # standardize on the training portion only
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_raw[train_idx])
    X = scaler.transform(X_raw)

    tick(0.08, "Fitting nuisance models (propensity and baseline)")
    pi_tr_only, pi_all = _nuisance_pi(X[train_idx], a_all[train_idx], X, spec)
    pi_all[train_idx] = pi_tr_only
    off_tr_only, off_all = _nuisance_offset(
        oc, X[train_idx], y_all[train_idx],
        None if e_all is None else e_all[train_idx], X, spec.seed)
    off_all[train_idx] = off_tr_only

    loss_name = outcome_to_loss_name(oc)
    is_cox = loss_name == "A_cox"

    def sl(ix):
        return (X[ix], y_all[ix], a_all[ix], pi_all[ix], off_all[ix],
                None if e_all is None else e_all[ix])

    Xf, yf, af, pf, of, ef = sl(fit_idx)
    Xv, yv, av, pv, ov, ev = sl(val_idx)

    # --------------------------------------------------------------- search #
    cfg_list, cfg_notes = build_configs(
        res["space"], res["search"], res["n_configs"], spec.seed,
        n_features=X.shape[1], base_epochs=res["base_epochs"])
    msgs.extend(cfg_notes)

    # One flat job list, so progress can be weighted by cost.  epochs is now a
    # searched dimension, and a 2.0x config costs 4x a 0.5x one -- advancing the
    # bar by fit count would make it crawl through the long configs and jump
    # through the short ones.
    jobs = [(ci, cfg, r) for ci, cfg in enumerate(cfg_list)
            for r in range(res["restarts"])]
    total = len(jobs)
    work_total = float(sum(cfg["epochs"] for _, cfg, _ in jobs)) or 1.0
    rows, f_all_runs, prob_runs, idx_runs = [], [], [], []
    work_done = 0.0

    for done, (ci, hp, r) in enumerate(jobs, start=1):
        tick(0.10 + 0.80 * work_done / work_total,
             f"Fitting {done}/{total}  (K={hp['K']}, lr={hp['lr']:g}, "
             f"hidden={arch_label(hp['hidden'])}, restart {r + 1})")
        _set_seed(7919 * spec.seed + 131 * ci + r)
        try:
            sel = ConcreteAutoencoderFeatureSelector(
                K=min(hp["K"], X.shape[1]),
                output_function=_make_decoder(hp["hidden"], hp["l2"], hp["dropout"]),
                num_epochs=hp["epochs"],
                batch_size=None if is_cox else 128,
                learning_rate=hp["lr"],
                start_temp=10.0,
                min_temp=hp["min_temp"],
                tryout_limit=1,          # see module docstring
                loss_name=loss_name,
                per_sample_noise=True,
                shuffle=True,
                verbose=0,
            ).fit(Xf, yf, af, pf, offset=of, event=ef)

            f_val = sel.predict_score(Xv)
            vloss = held_out_loss(oc, yv, av, pv, f_val, ov, ev)
            rows.append(dict(
                config=ci, restart=r, K=hp["K"], lr=hp["lr"],
                hidden=arch_label(hp["hidden"]), epochs=hp["epochs"],
                min_temp=hp["min_temp"], dropout=hp["dropout"], l2=hp["l2"],
                val_loss=float(vloss), mean_max=float(sel.mean_max),
                n_selected=int(len(np.unique(sel.get_indices()))),
                features=", ".join(feat_names[i]
                                   for i in sorted(set(sel.get_indices()))),
            ))
            f_all_runs.append(sel.predict_score(X))
            prob_runs.append(sel.feature_scores("max"))
            idx_runs.append(np.asarray(sel.get_indices(), dtype=int))
        except Exception as exc:                      # keep the app alive
            msgs.append(f"Fit {done} (K={hp['K']}, lr={hp['lr']:g}, "
                        f"hidden={arch_label(hp['hidden'])}) failed: {exc}")
        finally:
            work_done += hp["epochs"]
            tf.keras.backend.clear_session()

    if not rows:
        raise RuntimeError("every fit failed; see the messages above")

    runs = pd.DataFrame(rows)
    ok = np.isfinite(runs.val_loss.to_numpy())
    if not ok.any():
        raise RuntimeError("validation loss was not finite for any fit")
    runs = runs[ok].reset_index(drop=True)
    f_all_runs = [f for f, k in zip(f_all_runs, ok) if k]
    prob_runs = [p for p, k in zip(prob_runs, ok) if k]
    idx_runs = [i for i, k in zip(idx_runs, ok) if k]

    tick(0.92, "Aggregating")

    # -------------------------------------------------- config-level summary #
    CFG_KEYS = ["config", "K", "lr", "hidden", "epochs", "min_temp",
                "dropout", "l2"]
    configs = (runs.groupby(CFG_KEYS)
               .agg(val_loss_mean=("val_loss", "mean"),
                    val_loss_min=("val_loss", "min"),
                    val_loss_sd=("val_loss", "std"),
                    mean_max=("mean_max", "mean"),
                    n_selected=("n_selected", "mean"),
                    n_runs=("val_loss", "size"))
               .reset_index()
               .sort_values("val_loss_mean")
               .reset_index(drop=True))
    configs["rank"] = np.arange(1, len(configs) + 1)

    # Select by CONFIG MEAN, not by best individual run.  Averaging the restarts
    # before comparing configs is the entire point of running restarts, and with
    # a wide search it matters: the minimum over ~100 single fits on one
    # validation split is a noisier statistic than the minimum over ~50 means,
    # and chasing it is how a wider search makes the answer worse instead of
    # better.  The ensemble is then filled by walking configs in rank order --
    # all restarts of rank 1 first, spilling into rank 2, 3, ... only if top_m
    # asks for more runs than the winning config has.
    want = max(1, min(int(spec.top_m), len(runs)))
    top = []
    for cid in configs.config.to_numpy():
        cand = runs.index[runs.config == cid].to_numpy()
        cand = cand[np.argsort(runs.val_loss.to_numpy()[cand])]
        for i in cand:
            if len(top) >= want:
                break
            top.append(int(i))
        if len(top) >= want:
            break

    win = configs.iloc[0]
    win_runs = runs[runs.config == win.config].sort_values("val_loss")
    best_row = win_runs.iloc[0]
    best = dict(K=int(win.K), lr=float(win.lr), hidden=arch_from_label(win.hidden),
                epochs=int(win.epochs), min_temp=float(win.min_temp),
                dropout=float(win.dropout), l2=float(win.l2),
                restart=int(best_row.restart),
                val_loss=float(win.val_loss_mean),
                val_loss_best_run=float(best_row.val_loss),
                val_loss_sd=(float(win.val_loss_sd)
                             if np.isfinite(win.val_loss_sd) else np.nan),
                mean_max=float(win.mean_max), features=best_row.features,
                n_runs=int(win.n_runs), n_ensemble=len(top),
                n_configs_ensembled=int(runs.iloc[top].config.nunique()))

    # ------------------------------------------------- feature probabilities #
    d = len(feat_names)

    def _freq(idx_list):
        fr = np.zeros(d)
        for ix in idx_list:
            for j in np.unique(ix):
                fr[int(j)] += 1
        return fr / max(1, len(idx_list))

    # float64 throughout: these arrays arrive from TF as float32, and a float32
    # rounded to 4 dp still serialises as 0.49070000648498535 in the data grid.
    features = pd.DataFrame({
        "feature": feat_names,
        "sel_prob_top": np.mean([prob_runs[i] for i in top], axis=0).astype(np.float64),
        "sel_freq_top": _freq([idx_runs[i] for i in top]),
        "sel_prob_all": np.mean(prob_runs, axis=0).astype(np.float64),
        "sel_freq_all": _freq(idx_runs),
    }).sort_values("sel_prob_top", ascending=False).reset_index(drop=True)
    for c in features.columns[1:]:
        features[c] = features[c].astype(np.float64).round(6)

    # --------------------------------------------------- ensemble + subgroup #
    f_ens = np.mean([f_all_runs[i] for i in top], axis=0).astype(np.float64)
    bscore = benefit_score(f_ens, oc).astype(np.float64)

    split_lab = np.empty(n, dtype=object)
    split_lab[fit_idx] = "fit"
    split_lab[val_idx] = "validation"
    split_lab[test_idx] = "test"

    subjects = pd.DataFrame({
        "row": np.arange(n),
        "split": split_lab,
        "treatment": a_all,
        "contrast_f": f_ens,
        "benefit_score": bscore,
        "predicted_subgroup": np.where(bscore > 0, "benefit", "no benefit"),
        "propensity": pi_all,
        "baseline_offset": off_all,
    })
    if oc == "tte":
        subjects["time"] = y_all
        subjects["event"] = e_all
    else:
        subjects["outcome"] = y_all
    if truth is not None:
        subjects["known_label"] = truth

    gain_fn = (lambda ix: cox_subgroup_gain(y_all[ix], e_all[ix], a_all[ix], f_ens[ix])
               if oc == "tte" else
               subgroup_gain(y_all[ix], a_all[ix], pi_all[ix], f_ens[ix]))

    info = dict(
        n_rows=n, n_dropped=n0 - len(data), n_features=d,
        n_fit=len(fit_idx), n_val=len(val_idx), n_test=len(test_idx),
        n_fits_done=len(runs), n_fits_planned=total,
        outcome=oc, loss_name=loss_name,
        seed=int(spec.seed), n_boot=int(spec.n_boot),
        preset=spec.preset, search=res["search"],
        epochs_base=int(res["base_epochs"]),
        n_configs_planned=len(cfg_list), n_configs_fitted=len(configs),
        n_configs_requested=(int(res["n_configs"]) if res["search"] == "random"
                             else len(cfg_list)),
        restarts=int(res["restarts"]),
        # the searched dimensions, for the tuning plot and the report
        searched=[c for c in ("K", "lr", "hidden", "epochs", "min_temp",
                              "dropout", "l2")
                  if runs[c].nunique() > 1],
        event_rate=(float(e_all.mean()) if e_all is not None else None),
        treated_frac=float(a_all.mean()),
        feature_names=feat_names,
        val_gain=float(gain_fn(val_idx)) if len(val_idx) > 30 else np.nan,
        test_gain=float(gain_fn(test_idx)) if len(test_idx) > 30 else np.nan,
    )

    if truth is not None:
        from sklearn.metrics import roc_auc_score
        try:
            t01 = check_y01(truth, spec.truth_col)
            info["auc_all"] = float(roc_auc_score(t01, bscore))
            if len(test_idx) > 30 and len(np.unique(t01[test_idx])) == 2:
                info["auc_test"] = float(roc_auc_score(t01[test_idx], bscore[test_idx]))
        except Exception as exc:
            msgs.append(f"Could not compute AUC against '{spec.truth_col}': {exc}")

    tick(1.0, "Done")
    return AnalysisResult(runs=runs, configs=configs, features=features,
                          subjects=subjects, best=best, info=info, messages=msgs)
