"""Colt Wood - Decision Tree, CS 3368, UNSW-NB15 (v2).

Run from any location:  python path/to/train_decision_tree.py

Task: predict HTTP, DNS, SMTP or FTP for *known benign* network flows.

What this script does, per scenario (S1 / S2 / S3, see traffic_common.py):
  1. Loads the official UNSW-NB15 train and test CSVs (kept separate).
  2. Chooses the base Decision Tree's settings by 5-fold cross-validation on
     the TRAINING file only (the test file is never used for any choice).
  3. Fits baselines (majority, stump, depth-3 tree, logistic regression),
     the base tree, and the two improvement steps on the full training file.
  4. Scores everything once on the official test file, with bootstrap 95% CIs.
  5. Runs extra checks: swapped train/test direction, learning curve,
     permutation importance.
  6. Writes tables, figures, and the saved base-tree models to results/.
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.tree import DecisionTreeClassifier

from make_figures import (plot_confusions, plot_importance, plot_ladder_zoom,
                          plot_learning_curve, plot_model_comparison)
from traffic_common import (RANDOM_STATE, RESULTS_DIR, SCENARIOS, TEST_CSV,
                            TRAIN_CSV, fast_scores, load_normal_services,
                            make_pipeline, scenario_xy)

N_BOOTSTRAP = 1000
TREE_GRID = [
    {"max_depth": d, "min_samples_leaf": leaf, "class_weight": cw}
    for d in (None, 4, 6, 8, 12)
    for leaf in (1, 5, 10, 20)
    for cw in (None, "balanced")
]


def tree(params: dict) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(random_state=RANDOM_STATE, **params)


def select_base_params(X, y, scenario_key: str):
    """5-fold CV on the training file. Ties within 0.0005 macro-F1 go to the simpler tree."""
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for params in TREE_GRID:
        res = cross_validate(make_pipeline(tree(params)), X, y, cv=cv,
                             scoring=("accuracy", "f1_macro"), n_jobs=-1)
        rows.append({"scenario": scenario_key, **params,
                     "cv_accuracy": res["test_accuracy"].mean(),
                     "cv_macro_f1": res["test_f1_macro"].mean(),
                     "cv_macro_f1_std": res["test_f1_macro"].std()})
    table = pd.DataFrame(rows)
    best = table["cv_macro_f1"].max()
    eligible = table[table["cv_macro_f1"] >= best - 0.0005].copy()
    eligible["depth_rank"] = eligible["max_depth"].fillna(10_000)
    eligible["balanced"] = eligible["class_weight"].notna()
    chosen = eligible.sort_values(["depth_rank", "balanced", "min_samples_leaf"],
                                  ascending=[True, True, False]).iloc[0]
    params = {"max_depth": None if pd.isna(chosen["max_depth"]) else int(chosen["max_depth"]),
              "min_samples_leaf": int(chosen["min_samples_leaf"]),
              "class_weight": None if pd.isna(chosen["class_weight"]) else chosen["class_weight"]}
    table["selected"] = [all((a == b) or (pd.isna(a) and b is None)
                             for a, b in [(r["max_depth"], params["max_depth"]),
                                          (r["min_samples_leaf"], params["min_samples_leaf"]),
                                          (r["class_weight"], params["class_weight"])])
                         for _, r in table.iterrows()]
    return params, table


def model_zoo(params: dict):
    """(name, role, pipeline). 'mine' rows are the base tree and its two improvements."""
    return [
        ("Majority class", "baseline", make_pipeline(DummyClassifier(strategy="most_frequent"))),
        ("Depth-1 stump", "baseline", make_pipeline(tree({"max_depth": 1}))),
        ("Depth-3 tree", "baseline", make_pipeline(tree({"max_depth": 3}))),
        ("Logistic regression", "baseline",
         make_pipeline(LogisticRegression(max_iter=3000), scale=True)),
        ("Decision tree (base)", "mine", make_pipeline(tree(params))),
        ("Tree + flow features (+1)", "mine", make_pipeline(tree(params), add_features=True)),
        ("Random forest + flow features (+2)", "mine",
         make_pipeline(RandomForestClassifier(300, random_state=RANDOM_STATE, n_jobs=-1),
                       add_features=True)),
    ]


def bootstrap_ci(y_idx, pred_idx, n_classes, boot_rows):
    accs, f1s = [], []
    for rows in boot_rows:
        a, f = fast_scores(y_idx[rows], pred_idx[rows], n_classes)
        accs.append(a)
        f1s.append(f)
    return (np.percentile(accs, 2.5), np.percentile(accs, 97.5),
            np.percentile(f1s, 2.5), np.percentile(f1s, 97.5))


def learning_curve(params, X_tr, y_tr, X_te, y_te, classes, scenario_key):
    rows = []
    for frac in (0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0):
        seeds = [0] if frac == 1.0 else range(5)
        for seed in seeds:
            if frac == 1.0:
                X_s, y_s = X_tr, y_tr
            else:
                X_s, _, y_s, _ = train_test_split(X_tr, y_tr, train_size=frac,
                                                  stratify=y_tr, random_state=seed)
            model = make_pipeline(tree(params)).fit(X_s, y_s)
            pred = model.predict(X_te)
            _, f1 = fast_scores(pd.Categorical(y_te, categories=classes).codes,
                                pd.Categorical(pred, categories=classes).codes, len(classes))
            rows.append({"scenario": scenario_key, "train_fraction": frac,
                         "train_rows": len(X_s), "macro_f1": f1})
    return pd.DataFrame(rows)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "models").mkdir(exist_ok=True)
    train_df = load_normal_services(TRAIN_CSV)
    test_df = load_normal_services(TEST_CSV)
    print(f"Normal-traffic rows: train={len(train_df)}, test={len(test_df)}")
    print("Train classes:", train_df["target"].value_counts().to_dict())
    print("Test classes: ", test_df["target"].value_counts().to_dict())

    all_rows, cv_tables, reverse_rows, curves, ladder_cv_rows = [], [], [], [], []
    confusions, importances, per_class = {}, {}, []
    chosen_params, class_counts = {}, {}

    for key, scen in SCENARIOS.items():
        print(f"\n=== {scen.title} ===")
        X_tr, y_tr = scenario_xy(train_df, scen)
        X_te, y_te = scenario_xy(test_df, scen)
        assert list(X_tr.columns) == list(X_te.columns)
        classes = list(scen.classes)
        class_counts[key] = {"train_rows": len(y_tr), "test_rows": len(y_te),
                             "n_features": X_tr.shape[1],
                             "train_classes": y_tr.value_counts().to_dict(),
                             "test_classes": y_te.value_counts().to_dict()}
        print(f"features={X_tr.shape[1]}  train={len(y_tr)}  test={len(y_te)}")

        params, cv_table = select_base_params(X_tr, y_tr, key)
        chosen_params[key] = params
        cv_tables.append(cv_table)
        print("Base tree chosen by 5-fold CV on training data:", params)

        y_idx = pd.Categorical(y_te, categories=classes).codes
        rng = np.random.default_rng(RANDOM_STATE)
        boot_rows = [rng.integers(0, len(y_idx), len(y_idx)) for _ in range(N_BOOTSTRAP)]

        for name, role, pipe in model_zoo(params):
            start = time.perf_counter()
            pipe.fit(X_tr, y_tr)
            fit_s = time.perf_counter() - start
            pred = pipe.predict(X_te)
            pred_idx = pd.Categorical(pred, categories=classes).codes
            acc, f1 = fast_scores(y_idx, pred_idx, len(classes))
            a_lo, a_hi, f_lo, f_hi = bootstrap_ci(y_idx, pred_idx, len(classes), boot_rows)
            all_rows.append({"scenario": key, "model": name, "role": role,
                             "accuracy": acc, "acc_ci_low": a_lo, "acc_ci_high": a_hi,
                             "macro_f1": f1, "f1_ci_low": f_lo, "f1_ci_high": f_hi,
                             "train_seconds": round(fit_s, 3)})
            print(f"  {name:36s} acc={acc:.4f} [{a_lo:.4f},{a_hi:.4f}]  macro-F1={f1:.4f}")

            if name == "Decision tree (base)":
                base_pipe, base_pred = pipe, pred
                cm = confusion_matrix(y_te, pred, labels=classes)
                confusions[key] = (cm, classes)
                pd.DataFrame(cm, index=classes, columns=classes).to_csv(
                    RESULTS_DIR / f"confusion_matrix_{key}.csv")
                rep = pd.DataFrame(classification_report(
                    y_te, pred, labels=classes, zero_division=0, output_dict=True)).T
                rep.insert(0, "scenario", key)
                per_class.append(rep.reset_index().rename(columns={"index": "class"}))
                (RESULTS_DIR / f"classification_report_{key}.txt").write_text(
                    classification_report(y_te, pred, labels=classes, zero_division=0))
                pd.DataFrame({"actual": y_te.values, "predicted": pred}).to_csv(
                    RESULTS_DIR / f"test_predictions_{key}.csv", index=False)
                joblib.dump({"pipeline": pipe, "scenario": key,
                             "feature_columns": list(X_tr.columns), "classes": classes},
                            RESULTS_DIR / "models" / f"decision_tree_{key}.joblib")

        # Check: swap the direction (train on the test file, evaluate on the train file).
        rev = make_pipeline(tree(params)).fit(X_te, y_te)
        r_pred = rev.predict(X_tr)
        r_acc, r_f1 = fast_scores(pd.Categorical(y_tr, categories=classes).codes,
                                  pd.Categorical(r_pred, categories=classes).codes, len(classes))
        reverse_rows.append({"scenario": key, "direction": "train on test CSV -> test on train CSV",
                             "accuracy": r_acc, "macro_f1": r_f1})
        print(f"  swapped-direction check: acc={r_acc:.4f} macro-F1={r_f1:.4f}")

        # Extra analysis uses a validation split cut from the TRAINING file, so the
        # official test file is only used for the final scores above.
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=RANDOM_STATE)
        curves.append(learning_curve(params, X_fit, y_fit, X_val, y_val, classes, key))

        # Cross-validation on the training file for the base tree and its two
        # improvements (no test data involved).
        cv5 = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
        for name, role, pipe in model_zoo(params):
            if role != "mine":
                continue
            res = cross_validate(pipe, X_tr, y_tr, cv=cv5, scoring=("accuracy", "f1_macro"))
            ladder_cv_rows.append({"scenario": key, "model": name,
                                   "cv_accuracy": res["test_accuracy"].mean(),
                                   "cv_macro_f1": res["test_f1_macro"].mean(),
                                   "cv_macro_f1_std": res["test_f1_macro"].std()})
            print(f"  5-fold CV (train only) {name:36s} macro-F1={ladder_cv_rows[-1]['cv_macro_f1']:.4f}")

        if key in ("S2", "S3"):
            imp_pipe = make_pipeline(tree(params)).fit(X_fit, y_fit)
            imp = permutation_importance(imp_pipe, X_val, y_val, scoring="f1_macro",
                                         n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
            importances[key] = pd.DataFrame({
                "feature": X_val.columns, "importance": imp.importances_mean,
                "std": imp.importances_std}).sort_values("importance", ascending=False)

    results = pd.DataFrame(all_rows)
    results.to_csv(RESULTS_DIR / "all_results.csv", index=False)
    pd.concat(cv_tables).to_csv(RESULTS_DIR / "cv_tree_selection.csv", index=False)
    pd.DataFrame(reverse_rows).to_csv(RESULTS_DIR / "swapped_split_check.csv", index=False)
    pd.DataFrame(ladder_cv_rows).to_csv(RESULTS_DIR / "ladder_cv_train_only.csv", index=False)
    curve_df = pd.concat(curves)
    curve_df.to_csv(RESULTS_DIR / "learning_curve.csv", index=False)
    pd.concat(per_class).to_csv(RESULTS_DIR / "per_class_base_tree.csv", index=False)
    for key, df in importances.items():
        df.to_csv(RESULTS_DIR / f"permutation_importance_{key}.csv", index=False)

    # Compact table for slides.
    slide = results.assign(
        accuracy_95ci=lambda d: d.apply(
            lambda r: f"{r.accuracy:.2%} ({r.acc_ci_low:.2%}-{r.acc_ci_high:.2%})", axis=1),
        macro_f1_value=lambda d: d.macro_f1.round(3))[
        ["scenario", "model", "role", "accuracy_95ci", "macro_f1_value", "train_seconds"]]
    slide.to_csv(RESULTS_DIR / "results_table_for_slides.csv", index=False)

    plot_model_comparison(results, RESULTS_DIR / "fig_model_comparison.png")
    plot_ladder_zoom(results, RESULTS_DIR / "fig_improvement_ladder_zoom.png")
    plot_confusions(confusions, RESULTS_DIR / "fig_confusion_matrices.png")
    plot_learning_curve(curve_df, RESULTS_DIR / "fig_learning_curve.png")
    plot_importance(importances, RESULTS_DIR / "fig_permutation_importance.png")

    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "problem": "Classify known-benign flows into HTTP / DNS / SMTP / FTP service categories",
            "dataset": "UNSW-NB15 official training and testing CSVs (normal flows only)",
            "scenarios": {k: {"title": SCENARIOS[k].title, **class_counts[k],
                              "base_tree_params": chosen_params[k]} for k in SCENARIOS},
            "selection": "Base tree chosen by 5-fold CV macro-F1 on the training CSV only; "
                         "all other models use fixed settings and are not tuned on the test CSV.",
            "results": results.round(5).to_dict(orient="records"),
            "swapped_split_check": reverse_rows,
            "ladder_cv_train_only": ladder_cv_rows,
            "caution": "Published intrusion-detection scores (e.g. More et al., 2024) "
                       "solve a different task and are not directly comparable.",
        }, f, indent=2, default=str)

    print("\nSaved tables, figures, and models to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
