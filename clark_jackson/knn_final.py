import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

y_train = train["service"]
x_train = train.drop(columns=["service"])
y_test = test["service"]
x_test = test.drop(columns=["service"])

x_train = pd.get_dummies(x_train, columns=["proto", "state"])
x_test = pd.get_dummies(x_test, columns=["proto", "state"])

x_test = x_test.reindex(columns=x_train.columns, fill_value=0)

num_cols = list(x_train.select_dtypes("number").columns)

x_train_log = x_train.copy()
x_test_log = x_test.copy()
for col in num_cols:
    x_train_log[col] = np.log1p(x_train_log[col])
    x_test_log[col] = np.log1p(x_test_log[col])


def scale(x_tr, x_te):
    scaler = StandardScaler()
    x_tr = x_tr.copy()
    x_te = x_te.copy()
    x_tr[num_cols] = scaler.fit_transform(x_tr[num_cols])
    x_te[num_cols] = scaler.transform(x_te[num_cols])
    return x_tr, x_te


(
    x_train_s,
    x_test_s,
) = scale(x_train, x_test)
x_train_log_s, x_test_log_s = scale(x_train_log, x_test_log)

tcp_train = y_train != "dns"
tcp_test = y_test != "dns"
y_train_tcp = y_train[tcp_train]
y_test_tcp = y_test[tcp_test]

proto_cols = []
for col in x_train.columns:
    if col.startswith("proto_"):
        proto_cols.append(col)

x_train_tcp_s, x_test_tcp_s = scale(
    x_train[tcp_train].drop(columns=proto_cols),
    x_test[tcp_test].drop(columns=proto_cols),
)
x_train_log_tcp_s, x_test_log_tcp_s = scale(
    x_train_log[tcp_train].drop(columns=proto_cols),
    x_test_log[tcp_test].drop(columns=proto_cols),
)

print("x_train:", x_train.shape)
print("x_test:", x_test.shape)
print("x_train_tcp:", x_train_tcp_s.shape)
print("x_test_tcp:", x_test_tcp_s.shape)

search = GridSearchCV(
    KNeighborsClassifier(),
    {
        "n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    },
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=3368),
    scoring="f1_macro",
    n_jobs=-1,
)

search.fit(x_train_log_s, y_train)
best = search.best_params_
print("Best cv macro F1:", round(search.best_score_, 4))
print("Best Settings:", best)

results = []
confusions = {}


def run_model(model_name, dataset, model, x_tr, y_tr, x_te, y_te):
    model.fit(x_tr, y_tr)
    start = time.perf_counter()
    pred = model.predict(x_te)
    predict_time = time.perf_counter() - start

    cm = confusion_matrix(y_te, pred, labels=["http", "ftp"])
    http_ftp_errors = cm[0, 1] + cm[1, 0]

    print(model_name, "|", dataset)
    print("Accuracy:", round(accuracy_score(y_te, pred), 4))
    print("Macro F1:", round(f1_score(y_te, pred, average="macro"), 4))

    row = {}
    row["model"] = model_name
    row["dataset"] = dataset
    row["accuracy"] = round(accuracy_score(y_te, pred), 4)
    row["macro_f1"] = round(f1_score(y_te, pred, average="macro"), 4)
    row["http_ftp_errors"] = http_ftp_errors
    row["total_errors"] = (pred != y_te).sum()
    row["predict_seconds"] = round(predict_time, 4)
    results.append(row)
    confusions[(model_name, dataset)] = pred


model_c_settings = dict(
    n_neighbors=best["n_neighbors"], weights=best["weights"], metric=best["metric"]
)

run_model(
    "A: interim",
    "all classes",
    KNeighborsClassifier(n_neighbors=3),
    x_train_s,
    y_train,
    x_test_s,
    y_test,
)
run_model(
    "B: log",
    "all classes",
    KNeighborsClassifier(n_neighbors=3),
    x_train_log_s,
    y_train,
    x_test_log_s,
    y_test,
)
run_model(
    "C: log + tuned",
    "all classes",
    KNeighborsClassifier(**model_c_settings),
    x_train_log_s,
    y_train,
    x_test_log_s,
    y_test,
)

run_model(
    "A: interim",
    "tcp only",
    KNeighborsClassifier(n_neighbors=3),
    x_train_tcp_s,
    y_train_tcp,
    x_test_tcp_s,
    y_test_tcp,
)
run_model(
    "B: log",
    "tcp only",
    KNeighborsClassifier(n_neighbors=3),
    x_train_log_tcp_s,
    y_train_tcp,
    x_test_log_tcp_s,
    y_test_tcp,
)
run_model(
    "C: log + tuned",
    "tcp only",
    KNeighborsClassifier(**model_c_settings),
    x_train_log_tcp_s,
    y_train_tcp,
    x_test_log_tcp_s,
    y_test_tcp,
)

results_table = pd.DataFrame(results)
print(results_table)

results_folder = "clark_jackson/results"
os.makedirs(results_folder, exist_ok=True)
results_table.to_csv(results_folder + "/knn_final_results.csv", index=False)

model_names = ["A: interim", "B: log", "C: log + tuned"]
all_scores = []
tcp_scores = []
for name in model_names:
    for row in results:
        if row["model"] == name and row["dataset"] == "all classes":
            all_scores.append(row["macro_f1"])
        if row["model"] == name and row["dataset"] == "tcp only":
            tcp_scores.append(row["macro_f1"])
positions = np.arange(len(model_names))
width = 0.40

fig, ax = plt.subplots(figsize=(10, 6))
bars_all = ax.bar(
    positions - width / 2, all_scores, width, label="All 4 classes", color="#098eb7"
)
bars_tcp = ax.bar(
    positions + width / 2, tcp_scores, width, label="TCP only (no DNS)", color="#09b731"
)

for bar in list(bars_all) + list(bars_tcp):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.001,
        f"{bar.get_height():.3f}",
        ha="center",
        fontsize=14,
    )

ax.set_title("KNN models: macro F1 on the UNSW-NB15 test set", fontsize=18)
ax.set_xlabel("Model", fontsize=16)
ax.set_ylabel("Macro F1 (higher is better)", fontsize=16)
ax.set_xticks(positions)
ax.set_xticklabels(model_names, fontsize=14)

ax.set_ylim(0.95, 1.005)
ax.tick_params(axis="y", labelsize=14)
ax.legend(fontsize=14, loc="upper left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig(results_folder + "/knn_final_macro_f1.png", dpi=200)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, name in zip(axes, ["A: interim", "C: log + tuned"]):
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        confusions[(name, "all classes")],
        labels=["dns", "http", "smtp", "ftp"],
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(name)

fig.tight_layout()
fig.savefig(results_folder + "/knn_final_confusion.png", dpi=200)
print("Saved results to", results_folder)
