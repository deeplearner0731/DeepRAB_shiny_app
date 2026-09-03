"""
Aggregates the per-rep grid results written by run_synthetic_worker.py.

Reports
-------
1. Hyperparameter tuning
   - validation R-loss marginals per knob (K, lr, architecture)
   - the config selected in each outer split, and the modal / overall best config
2. AUC vs the true subgroup label `sigpos`, on TRAIN and TEST, for
   - the single selected run
   - the top-M rank-averaged ensemble
   - every config (so you can see how much the choice of K matters)
3. Feature selection
   - prob_score  : mean over runs of max_k P[k, j]
   - select_freq : fraction of runs in which j is an argmax index
   - count_top2  : legacy duplicate-count ranking, selected run only
   reported both pooled over all runs and restricted to the selected config.
"""

import os

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "results_synthetic")
FEATURES = ["age", "sex", "bmi"] + [f"X{i}" for i in range(1, 11)]
D = len(FEATURES)
TOP_M = 5
SELECT_BY = "val_rloss"      # label-free.  lower is better
TRUE_MODIFIERS = {"X1", "X2"}


def load():
    reps = sorted(
        int(f.split("rep")[1].split(".")[0])
        for f in os.listdir(OUTDIR)
        if f.startswith("runs_rep") and f.endswith(".csv")
    )
    runs = pd.concat(
        [pd.read_csv(os.path.join(OUTDIR, f"runs_rep{r}.csv")) for r in reps],
        ignore_index=True,
    )
    scores = {r: np.load(os.path.join(OUTDIR, f"scores_rep{r}.npz")) for r in reps}
    return reps, runs, scores


def auc(labels, s):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(labels, s)


def rank_pct(v):
    order = np.argsort(np.argsort(v))
    return (order + 1) / len(v)


def main():
    reps, runs, scores = load()
    runs["arch"] = runs["hidden"] + " / l2=" + runs["l2"].astype(str)
    n_cfg = runs["cfg"].nunique()
    print(f"loaded {len(runs)} runs = {len(reps)} splits x {n_cfg} configs "
          f"x {runs.restart.nunique()} restarts\n")

    # =============================================== 1. tuning: knob marginals #
    print("=" * 100)
    print("1a. VALIDATION R-LOSS BY HYPERPARAMETER  (label-free criterion; lower = better)")
    print("=" * 100)
    for knob in ["K", "lr", "arch"]:
        t = (runs.groupby(knob)
                 .agg(val_rloss_mean=("val_rloss", "mean"),
                      val_rloss_sd=("val_rloss", "std"),
                      val_gain_mean=("val_gain", "mean"),
                      mean_max=("mean_max", "mean"),
                      test_auc_mean=("test_auc", "mean"),
                      test_auc_sd=("test_auc", "std"),
                      n=("val_rloss", "size"))
                 .round(4))
        print(f"\n-- marginal over {knob} --")
        print(t.to_string())

    print("\n" + "=" * 100)
    print("1b. FULL CONFIG TABLE, ranked by mean validation R-loss")
    print("=" * 100)
    cfg_tab = (runs.groupby(["cfg", "K", "lr", "arch"])
                   .agg(val_rloss=("val_rloss", "mean"),
                        val_gain=("val_gain", "mean"),
                        mean_max=("mean_max", "mean"),
                        n_distinct=("n_distinct", "mean"),
                        train_auc=("train_auc", "mean"),
                        test_auc=("test_auc", "mean"),
                        test_auc_sd=("test_auc", "std"))
                   .reset_index()
                   .sort_values("val_rloss")
                   .round(4))
    print(cfg_tab.to_string(index=False))

    # ============================ 1c. per-split selection (properly nested) ==== #
    print("\n" + "=" * 100)
    print("1c. CONFIG SELECTED IN EACH OUTER SPLIT (by inner-validation R-loss)")
    print("=" * 100)
    sel_rows, per_rep = [], []
    for r in reps:
        sub = runs[runs.rep == r].sort_values(SELECT_BY).reset_index(drop=True)
        best = sub.iloc[0]
        sel_rows.append(best)

        z = scores[r]
        gtr, gte = z["g_train"], z["g_test"]
        # positional index of each run within this rep's stacked score arrays
        pos = (runs[runs.rep == r].reset_index(drop=True)
               .assign(p=lambda t: np.arange(len(t)))
               .set_index(["cfg", "restart"])["p"])
        best_p = pos.loc[(best.cfg, best.restart)]

        top = sub.iloc[:TOP_M]
        top_p = [pos.loc[(row.cfg, row.restart)] for row in top.itertuples()]

        ens_tr = np.mean([rank_pct(z["f_train"][p]) for p in top_p], axis=0)
        ens_te = np.mean([rank_pct(z["f_test"][p]) for p in top_p], axis=0)

        per_rep.append({
            "rep": r,
            "best_cfg": int(best.cfg), "K": int(best.K), "lr": best.lr,
            "arch": best.arch, "restart": int(best.restart),
            "val_rloss": round(best.val_rloss, 4),
            "mean_max": round(best.mean_max, 4),
            "sel_features": ",".join(
                FEATURES[i] for i in sorted(set(int(v) for v in str(best.indices).split(";")))),
            "train_auc_single": round(auc(gtr, z["f_train"][best_p]), 4),
            "test_auc_single": round(auc(gte, z["f_test"][best_p]), 4),
            "train_auc_ens": round(auc(gtr, ens_tr), 4),
            "test_auc_ens": round(auc(gte, ens_te), 4),
            "test_auc_sd_over_runs": round(
                float(np.std([auc(gte, z["f_test"][p]) for p in range(len(pos))])), 4),
        })
    per_rep = pd.DataFrame(per_rep)
    print(per_rep.to_string(index=False))

    print("\n-- how often each config was selected --")
    freq = (per_rep.groupby(["best_cfg", "K", "lr", "arch"]).size()
            .rename("times_selected").reset_index()
            .sort_values("times_selected", ascending=False))
    print(freq.to_string(index=False))

    # ================================================================= 2. AUC #
    print("\n" + "=" * 100)
    print("2. AUC vs TRUE SUBGROUP LABEL sigpos  (mean +- sd over the 10 outer splits)")
    print("=" * 100)
    for col, lab in [("train_auc_single", "TRAIN  single selected run"),
                     ("test_auc_single", "TEST   single selected run"),
                     ("train_auc_ens", f"TRAIN  top-{TOP_M} ensemble"),
                     ("test_auc_ens", f"TEST   top-{TOP_M} ensemble")]:
        v = per_rep[col]
        print(f"  {lab:32s} {v.mean():.4f}  +-{v.std():.4f}   "
              f"[min {v.min():.4f}, max {v.max():.4f}]")
    print(f"\n  across-run test-AUC sd within a split (all {n_cfg * runs.restart.nunique()} "
          f"runs): {per_rep.test_auc_sd_over_runs.mean():.4f}")
    print("  -> this is the seed/config sensitivity the ensemble is averaging away")

    # sensitivity: what if you had selected by the oracle instead?
    print("\n-- sensitivity of the selection criterion (TEST AUC of the run it picks) --")
    for crit, asc, lab in [("val_rloss", True, "val R-loss   (label-free, used)"),
                           ("val_gain", False, "val IPW gain (label-free)"),
                           ("train_auc", False, "train AUC    (ORACLE, cheating)")]:
        picked = [runs[runs.rep == r].sort_values(crit, ascending=asc).iloc[0].test_auc
                  for r in reps]
        print(f"  {lab:34s} {np.mean(picked):.4f}  +-{np.std(picked, ddof=1):.4f}")

    # ==================================================== 3. feature selection #
    print("\n" + "=" * 100)
    print("3. FEATURE SELECTION")
    print("=" * 100)

    def ranking(df, label):
        prob = df[[f"prob_{f}" for f in FEATURES]].to_numpy().mean(axis=0)
        freq_ = np.zeros(D)
        for s in df["indices"]:
            for j in set(int(v) for v in str(s).split(";")):
                freq_[j] += 1
        freq_ /= len(df)
        out = pd.DataFrame({"feature": FEATURES, "prob_score": prob,
                            "select_freq": freq_})
        out["true_modifier"] = ["YES" if f in TRUE_MODIFIERS else "" for f in FEATURES]
        return out.sort_values("prob_score", ascending=False).reset_index(drop=True), label

    # (a) pooled over every run in the grid
    tab_all, _ = ranking(runs, "all")
    # (b) restricted to the selected config family (the tuned model)
    best_cfg_mode = int(per_rep.best_cfg.mode().iloc[0])
    tab_best, _ = ranking(runs[runs.cfg == best_cfg_mode], "best cfg")
    # (c) the 10 selected runs only
    sel_df = pd.DataFrame(sel_rows)
    tab_sel, _ = ranking(sel_df, "selected runs")

    merged = (tab_all.rename(columns={"prob_score": "prob_ALL", "select_freq": "freq_ALL"})
              .merge(tab_best.rename(columns={"prob_score": "prob_BESTCFG",
                                              "select_freq": "freq_BESTCFG"})
                     [["feature", "prob_BESTCFG", "freq_BESTCFG"]], on="feature")
              .merge(tab_sel.rename(columns={"prob_score": "prob_SEL",
                                             "select_freq": "freq_SEL"})
                     [["feature", "prob_SEL", "freq_SEL"]], on="feature"))

    # legacy count_top2, on the selected run of each split
    count_top2 = np.zeros(D)
    for s in sel_df["indices"]:
        idx = np.array([int(v) for v in str(s).split(";")], dtype=int)
        c = np.bincount(idx, minlength=D)
        for j in np.argsort(-c)[:2]:
            count_top2[j] += 1.0
    count_top2 /= len(sel_df)
    merged = merged.merge(
        pd.DataFrame({"feature": FEATURES, "legacy_count_top2": count_top2}), on="feature")

    merged = merged.sort_values("prob_SEL", ascending=False).reset_index(drop=True)
    print(f"\nALL = all {len(runs)} runs | BESTCFG = the {len(runs[runs.cfg == best_cfg_mode])} "
          f"runs of the modal selected config (cfg {best_cfg_mode}) | SEL = the "
          f"{len(sel_df)} selected runs")
    print(merged.round(4).to_string(index=False))

    print("\n-- selected feature SET of each split's chosen model --")
    print(per_rep[["rep", "K", "sel_features"]].to_string(index=False))

    print("\n-- exact-recovery rate of {X1, X2} --")
    for name, df in [("all runs", runs),
                     (f"modal best cfg ({best_cfg_mode})", runs[runs.cfg == best_cfg_mode]),
                     ("selected runs", sel_df)]:
        sets = [set(FEATURES[int(v)] for v in str(s).split(";")) for s in df["indices"]]
        exact = np.mean([s == TRUE_MODIFIERS for s in sets])
        contain = np.mean([TRUE_MODIFIERS <= s for s in sets])
        print(f"  {name:28s} exact = {exact:.3f}   contains both = {contain:.3f}   (n={len(sets)})")

    print("\n-- recovery by K (does K matter for feature recovery?) --")
    rec = []
    for k, sub in runs.groupby("K"):
        sets = [set(FEATURES[int(v)] for v in str(s).split(";")) for s in sub["indices"]]
        rec.append({"K": k, "n": len(sets),
                    "exact_X1X2": np.mean([s == TRUE_MODIFIERS for s in sets]),
                    "contains_both": np.mean([TRUE_MODIFIERS <= s for s in sets]),
                    "mean_distinct": sub.n_distinct.mean(),
                    "mean_test_auc": sub.test_auc.mean()})
    print(pd.DataFrame(rec).round(3).to_string(index=False))

    per_rep.to_csv(os.path.join(OUTDIR, "summary_per_split.csv"), index=False)
    cfg_tab.to_csv(os.path.join(OUTDIR, "summary_configs.csv"), index=False)
    merged.to_csv(os.path.join(OUTDIR, "summary_features.csv"), index=False)
    print(f"\nwrote summary_per_split.csv / summary_configs.csv / summary_features.csv to {OUTDIR}")


if __name__ == "__main__":
    main()
