import time

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# load
train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")
X_train, y_train = train.drop(columns=["service"]), train["service"]
X_test, y_test = test.drop(columns=["service"]), test["service"]

# prepare
TEXT_COLS = ["proto", "state"]
NUM_COLS = [c for c in X_train.columns if c not in TEXT_COLS]
prep = ColumnTransformer(
    [
        ("text", OneHotEncoder(handle_unknown="ignore"), TEXT_COLS),
        ("num", MinMaxScaler(), NUM_COLS),
    ]
)

# First version: min-max scaling, k = 5
# Final model: standard scaling, k = 3 is chosen in knn_tune.py and checked in knn_checks.py
# pipeline
model = Pipeline([("prep", prep), ("knn", KNeighborsClassifier(n_neighbors=5))])

# train, time, and score it
start = time.perf_counter()
model.fit(X_train, y_train)
train_time = time.perf_counter() - start

start = time.perf_counter()
pred = model.predict(X_test)
predict_time = time.perf_counter() - start

print(f"accuracy: {accuracy_score(y_test, pred):.3f}")
print(f"macro F1: {f1_score(y_test, pred, average='macro'):.3f}")
print(f"train: {train_time:.3f}s predict: {predict_time:.3f}s")
print(classification_report(y_test, pred))
