"""
Worker: runs the full DeepRAB hyperparameter grid x restarts for ONE outer
train/test split of synthetic_clinical_data.csv, and dumps every run's
diagnostics to disk.

Nothing here uses the true subgroup label `sigpos` for fitting or for model
selection.  `sigpos` is only read to compute reported AUCs at the end.

Usage:  python run_synthetic_worker.py --rep 0
"""

import argparse
import os
import random
import warnings

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, ReLU

from Deep_learning_subgroup_v2 import (
    ConcreteAutoencoderFeatureSelector,
    r_loss_np,
    subgroup_gain,
    to_a01,
)

warnings.filterwarnings("ignore")

# ============================================================ configuration #
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "synthetic_clinical_data.csv")
OUTDIR = os.path.join(HERE, "results_synthetic")

FEATURES = ["age", "sex", "bmi"] + [f"X{i}" for i in range(1, 11)]
Y_COL, TRT_COL, TRUTH_COL = "y", "treatment", "sigpos"

N_TRAIN = 800
INNER_VAL_FRAC = 0.25       # of the training set -> 600 fit / 200 inner-val
N_RESTARTS = 4              # random initializations per config

PI_MODE = "known"           # "known"    : pi = mean(A) on the training split
                            #              (valid: this trial is randomized)
                            # "crossfit" : 5-fold logistic P(A=1|X)
PI_CLIP = 0.05
RESIDUALIZE = True          # R-learner:  y_res = y - m_hat(X), m_hat cross-fitted

START_TEMP = 10.0
MIN_TEMP = 0.05
EPOCHS = 300
BATCH = 128
TRYOUT_LIMIT = 2
PER_SAMPLE_NOISE = True

# ------------------------------------------------------------------ HP grid #
# Factorial over the three knobs that matter, so each is attributable:
#   K    : number of concrete selection slots (true answer is 2 modifiers)
#   lr   : Adam step size on standardized inputs
#   arch : decoder width + L2 (kept as a 2-level bundle)
K_GRID = [2, 3, 4, 6, 8]
LR_GRID = [1e-3, 5e-3]
ARCH_GRID = [((32, 16), 0.0), ((64, 32), 1e-4)]

HP_GRID = [
    {"K": k, "lr": lr, "hidden": h, "l2": l2, "epochs": EPOCHS,
     "batch": BATCH, "min_temp": MIN_TEMP}
    for k in K_GRID
    for lr in LR_GRID
    for (h, l2) in ARCH_GRID
]


# =================================================================== helpers #
def set_seed(seed):
    seed = int(seed) % (2**31 - 1)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def make_decoder(hidden, l2=0.0):
    reg = tf.keras.regularizers.l2(l2) if l2 else None

    def decoder(x):
        for h in hidden:
            x = Dense(h, kernel_regularizer=reg)(x)
            x = ReLU()(x)
        return Dense(1, kernel_regularizer=reg)(x)

    return decoder


def crossfit_propensity(X, a01, seed=0, n_splits=5):
    pi = np.zeros(len(a01))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, a01):
        m = LogisticRegression(max_iter=2000).fit(X[tr], a01[tr])
        pi[te] = m.predict_proba(X[te])[:, 1]
    return np.clip(pi, PI_CLIP, 1.0 - PI_CLIP)


def crossfit_baseline(X, y, seed=0, n_splits=5):
    m_hat = np.zeros(len(y))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        m = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(X[tr], y[tr])
        m_hat[te] = m.predict(X[te])
    return m_hat


# ==================================================================== driver #
def run_rep(rep):
    os.makedirs(OUTDIR, exist_ok=True)

    data = pd.read_csv(DATA)
    data = data.loc[:, ~data.columns.str.startswith("Unnamed")]
    n_total = len(data)

    # ---- outer split ----------------------------------------------------- #
    rng = np.random.RandomState(1000 + rep)
    train_idx = rng.choice(n_total, size=N_TRAIN, replace=False)
    test_idx = np.setdiff1d(np.arange(n_total), train_idx)

    X_all_raw = data[FEATURES].to_numpy(dtype=float)
    y_all = data[Y_COL].to_numpy(dtype=float)
    a_all = to_a01(data[TRT_COL].to_numpy())
    g_all = data[TRUTH_COL].to_numpy().astype(int)      # oracle: reporting only

    X_tr_raw, X_te_raw = X_all_raw[train_idx], X_all_raw[test_idx]
    y_tr = y_all[train_idx]
    a_tr = a_all[train_idx]

    # ---- standardize on the training split only --------------------------- #
    scaler = StandardScaler().fit(X_tr_raw)
    X_tr = scaler.transform(X_tr_raw)
    X_te = scaler.transform(X_te_raw)

    # ---- nuisance models -------------------------------------------------- #
    if PI_MODE == "known":
        pi_tr = np.full(len(y_tr), float(a_tr.mean()))
    else:
        pi_tr = crossfit_propensity(X_tr, a_tr, seed=rep)

    if RESIDUALIZE:
        y_res = y_tr - crossfit_baseline(X_tr, y_tr, seed=rep)
    else:
        y_res = y_tr - y_tr.mean()

    # ---- inner split used for label-free model selection ------------------ #
    n_val = int(round(INNER_VAL_FRAC * len(y_res)))
    perm = rng.permutation(len(y_res))
    val_sel, fit_sel = perm[:n_val], perm[n_val:]

    Xf, yf, af, pf = X_tr[fit_sel], y_res[fit_sel], a_tr[fit_sel], pi_tr[fit_sel]
    Xv, yv, av, pv = X_tr[val_sel], y_res[val_sel], a_tr[val_sel], pi_tr[val_sel]

    g_tr, g_te = g_all[train_idx], g_all[test_idx]

    # ---- grid x restarts -------------------------------------------------- #
    rows, f_train, f_test, f_val_store = [], [], [], []

    for cfg_i, hp in enumerate(HP_GRID):
        for restart in range(N_RESTARTS):
            set_seed(7919 * rep + 131 * cfg_i + restart)

            sel = ConcreteAutoencoderFeatureSelector(
                K=hp["K"],
                output_function=make_decoder(hp["hidden"], hp["l2"]),
                num_epochs=hp["epochs"],
                batch_size=hp["batch"],
                learning_rate=hp["lr"],
                start_temp=START_TEMP,
                min_temp=hp["min_temp"],
                tryout_limit=TRYOUT_LIMIT,
                loss_name="A",
                per_sample_noise=PER_SAMPLE_NOISE,
                shuffle=True,
                verbose=0,
            ).fit(Xf, yf, af, pf)
            # NOTE: validation_data is deliberately NOT passed to .fit().
            # Keras runs a full validation pass every epoch, which costs ~5x the
            # whole fit, and `history` is never used.  The selection criterion is
            # computed below from predict_score(Xv), which is what actually
            # matters and is identical either way.

            f_v = sel.predict_score(Xv)
            f_t = sel.predict_score(X_tr)
            f_e = sel.predict_score(X_te)
            prob = sel.feature_scores("max")
            idx = np.asarray(sel.get_indices(), dtype=int)

            row = {
                "rep": rep,
                "cfg": cfg_i,
                "restart": restart,
                "K": hp["K"],
                "lr": hp["lr"],
                "hidden": "x".join(str(h) for h in hp["hidden"]),
                "l2": hp["l2"],
                # ---- label-free selection criteria (computed on inner val) --
                "val_rloss": r_loss_np(yv, av, pv, f_v),
                "val_gain": subgroup_gain(yv, av, pv, f_v),
                # ---- diagnostics -------------------------------------------
                "mean_max": float(sel.mean_max),
                "epochs_used": int(len(sel.history.history["loss"])),
                "n_distinct": int(len(np.unique(idx))),
                "indices": ";".join(map(str, sorted(idx))),
                # ---- oracle, REPORTING ONLY --------------------------------
                "train_auc": roc_auc_score(g_tr, f_t),
                "test_auc": roc_auc_score(g_te, f_e),
            }
            for j, name in enumerate(FEATURES):
                row[f"prob_{name}"] = float(prob[j])
            rows.append(row)

            f_train.append(f_t)
            f_test.append(f_e)
            f_val_store.append(f_v)

            tf.keras.backend.clear_session()

        print(f"[rep {rep}] cfg {cfg_i + 1}/{len(HP_GRID)} done "
              f"(K={hp['K']}, lr={hp['lr']}, {row['hidden']})", flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, f"runs_rep{rep}.csv"), index=False)
    np.savez_compressed(
        os.path.join(OUTDIR, f"scores_rep{rep}.npz"),
        f_train=np.stack(f_train),
        f_test=np.stack(f_test),
        f_val=np.stack(f_val_store),
        train_idx=train_idx,
        test_idx=test_idx,
        val_sel=val_sel,
        g_train=g_tr,
        g_test=g_te,
        y_val=yv, a_val=av, pi_val=pv,
    )
    print(f"[rep {rep}] COMPLETE ({len(rows)} runs)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", type=int, required=True)
    run_rep(ap.parse_args().rep)
