from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
    "axes.spines.top": False, "axes.spines.right": False,
})
GRAY, BLUE, ORANGE = "#8a8f98", "#1f5fa8", "#d9730d"
SCENARIO_TITLES = {
    "S1": "S1: full 4-class",
    "S2": "S2: TCP-only 3-class",
    "S3": "S3: behaviour-only 3-class",
}


def plot_model_comparison(results, path):
    """Dot plot of macro-F1 (with 95% bootstrap CI) for every model in every scenario."""
    order = list(dict.fromkeys(results["model"]))
    keys = list(SCENARIO_TITLES)
    fig, axes = plt.subplots(1, len(keys), figsize=(17, 6.2), sharey=True)
    for ax, key in zip(axes, keys):
        sub = results[results["scenario"] == key].set_index("model").loc[order]
        y = np.arange(len(order))[::-1]
        for yi, (name, row) in zip(y, sub.iterrows()):
            colour = GRAY if row["role"] == "baseline" else (
                BLUE if name == "Decision tree (base)" else ORANGE)
            ax.errorbar(row["macro_f1"], yi,
                        xerr=[[row["macro_f1"] - row["f1_ci_low"]],
                              [row["f1_ci_high"] - row["macro_f1"]]],
                        fmt="o", color=colour, ms=9, capsize=4, lw=2)
            ax.text(min(row["macro_f1"] + 0.025, 0.985), yi + 0.28,
                    f"{row['macro_f1']:.3f}", fontsize=12, color=colour, ha="left")
        ax.set_yticks(y)
        ax.set_yticklabels(order)
        ax.set_xlim(0, 1.08)
        ax.set_xlabel("Macro-F1 on test CSV (95% CI)")
        ax.set_title(SCENARIO_TITLES[key], pad=10)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Decision tree vs. baselines and improvements (gray = baseline, blue = base tree, "
                 "orange = my improvements)", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusions(confusions, path):
    keys = [k for k in ("S1", "S3") if k in confusions]
    fig, axes = plt.subplots(1, len(keys), figsize=(7.2 * len(keys), 6.4))
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, keys):
        cm, classes = confusions[key]
        share = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted service")
        ax.set_ylabel("True service")
        ax.set_title(f"{SCENARIO_TITLES[key]}\nbase tree, test CSV (counts)")
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=15,
                        color="white" if share[i, j] > 0.55 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_learning_curve(curve_df, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, colour in zip(SCENARIO_TITLES, (GRAY, BLUE, ORANGE)):
        sub = curve_df[curve_df["scenario"] == key]
        g = sub.groupby("train_rows")["macro_f1"].agg(["mean", "std"]).reset_index()
        ax.errorbar(g["train_rows"], g["mean"], yerr=g["std"].fillna(0), marker="o",
                    lw=2, capsize=3, color=colour, label=SCENARIO_TITLES[key])
    ax.set_xscale("log")
    ax.set_xlabel("Training rows used (log scale)")
    ax.set_ylabel("Macro-F1 on validation split (from training CSV)")
    ax.set_title("Base tree: how much training data is needed?")
    ax.set_ylim(0.5, 1.01)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_importance(importances, path):
    keys = [k for k in ("S2", "S3") if k in importances]
    fig, axes = plt.subplots(1, len(keys), figsize=(8.5 * len(keys), 6.2))
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, keys):
        top = importances[key].head(8).iloc[::-1]
        ax.barh(top["feature"], top["importance"], xerr=top["std"], color=BLUE, capsize=3)
        ax.set_xlabel("Drop in validation macro-F1 when feature is shuffled")
        ax.set_ylabel("Input feature")
        ax.set_title(f"{SCENARIO_TITLES[key]}\npermutation importance, base tree")
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_ladder_zoom(results, path):
    """Base tree -> +1 flow features -> +2 random forest, zoomed per scenario."""
    names = ["Decision tree (base)", "Tree + flow features (+1)",
             "Random forest + flow features (+2)"]
    labels = ["Base\ntree", "+1 flow\nfeatures", "+2 random\nforest"]
    colours = ["#0072B2", "#D55E00", "#009E73"]
    keys = list(SCENARIO_TITLES)
    subtitles = {"S1": "(HTTP/DNS/SMTP/FTP)", "S2": "(DNS giveaway removed)",
                 "S3": "(no volume / seq. numbers)"}
    fig, axes = plt.subplots(1, len(keys), figsize=(17, 5.8))
    for ax, key in zip(axes, keys):
        sub = results[results["scenario"] == key].set_index("model").loc[names]
        for x, (name, row), colour in zip(range(3), sub.iterrows(), colours):
            ax.errorbar(x, row["macro_f1"],
                        yerr=[[row["macro_f1"] - row["f1_ci_low"]],
                              [row["f1_ci_high"] - row["macro_f1"]]],
                        fmt="o", color=colour, ms=10, capsize=5, lw=2)
            ax.text(x + 0.12, row["macro_f1"], f"{row['macro_f1']:.3f}", va="center",
                    fontsize=14)
        lo = (sub["f1_ci_low"].min())
        hi = (sub["f1_ci_high"].max())
        pad = (hi - lo) * 0.25
        ax.set_ylim(lo - pad, min(hi + pad, 1.0005))
        ax.set_xlim(-0.5, 2.7)
        ax.set_xticks(range(3))
        ax.set_xticklabels(labels)
        ax.set_ylabel("Macro-F1 on official test CSV")
        ax.set_title(f"{SCENARIO_TITLES[key]}\n{subtitles[key]}", pad=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Improvement ladder, zoomed (error bars: 95% bootstrap CI; "
                 "y-axis range differs per panel)", fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
