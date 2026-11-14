import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

RESULTS = Path("clark_jackson/results")
RESULTS.mkdir(exist_ok=True)

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")
X_train, y_train = train.drop(columns=["service"]), train["service"]
X_test, y_test = test.drop(columns=["service"]), test["service"]

TEXT_COLS = ["proto", "state"]
NUM_COLS = [c for c in X_train.columns if c not in TEXT_COLS]
CLASSES = ["dns", "http", "smtp", "ftp"]

SCALERS = {
    "min-max": MinMaxScaler(),
    "standard": StandardScaler(),
    "none": "passthrough",
}
K_VALUES = list(range(1, 26, 2))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=3368)


def make_model(scaler):
    prep = ColumnTransformer(
        [
            ("text", OneHotEncoder(handle_unknown="ignore"), TEXT_COLS),
            ("num", scaler, NUM_COLS),
        ]
    )
    return Pipeline([("prep", prep), ("knn", KNeighborsClassifier())])


rows = []
curves = {}
best_overall = None

for name, scaler in SCALERS.items():
    search = GridSearchCV(
        make_model(scaler),
        {"knn__n_neighbors": K_VALUES},
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    curves[name] = search.cv_results_["mean_test_score"]
    best_k = search.best_params_["knn__n_neighbors"]

    model = make_model(scaler).set_params(knn__n_neighbors=best_k)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start
    start = time.perf_counter()
    pred = model.predict(X_test)
    predict_time = time.perf_counter() - start

    row = {
        "scaler": name,
        "best_k": best_k,
        "cv_macro_f1": search.best_score_,
        "test_accuracy": accuracy_score(y_test, pred),
        "test_macro_f1": f1_score(y_test, pred, average="macro"),
        "train_s": train_time,
        "predict_s": predict_time,
    }
    rows.append(row)
    print(row)

    if best_overall is None or row["cv_macro_f1"] > best_overall[0]:
        best_overall = (row["cv_macro_f1"], name, best_k, pred)

results = pd.DataFrame(rows).round(3)
results.to_csv(RESULTS / "knn_results.csv", index=False)
print(results.to_string(index=False))

plt.figure(figsize=(8, 5))
for name, scores in curves.items():
    plt.plot(K_VALUES, scores, marker="o", label=name)
plt.xlabel("k (number of neighbors)")
plt.ylabel("Mean 5 fold CV macro F1")
plt.title("KNN macro F1 vs k by scaling method")
plt.legend(title="Scaling")
plt.grid(alpha=0.3)
plt.savefig(RESULTS / "knn_f1_vs_k.png", dpi=200, bbox_inches="tight")

_, name, k, pred = best_overall
ConfusionMatrixDisplay.from_predictions(y_test, pred, labels=CLASSES, cmap="Blues")
plt.title(f"KNN confusion matrix (test set, {name} scaling, k = {k})")
plt.savefig(RESULTS / "knn_confusion_matrix.png", dpi=200, bbox_inches="tight")
print("best: ", name, "k = ", k)
