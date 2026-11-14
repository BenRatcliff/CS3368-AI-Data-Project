import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")
X_train, y_train = train.drop(columns=["service"]), train["service"]
X_test, y_test = test.drop(columns=["service"]), test["service"]

# prepare
TEXT_COLS = ["proto", "state"]
NUM_COLS = [c for c in X_train.columns if c not in TEXT_COLS]


def make_model():
    prep = ColumnTransformer(
        [
            ("text", OneHotEncoder(handle_unknown="ignore"), TEXT_COLS),
            ("num", StandardScaler(), NUM_COLS),
        ]
    )
    return Pipeline([("prep", prep), ("knn", KNeighborsClassifier(n_neighbors=3))])


model = make_model().fit(X_train, y_train)
real_f1 = f1_score(y_test, model.predict(X_test), average="macro")

y_shuffled = y_train.sample(frac=1, random_state=0).values
shuffled_model = make_model().fit(X_train, y_shuffled)
shuffled_f1 = f1_score(y_test, shuffled_model.predict(X_test), average="macro")

print(f"Check 1 real labels: {real_f1:.3f} shuffled labels: {shuffled_f1:.3f}")

overlap = X_test.merge(X_train.drop_duplicates(), how="inner")
print(f"Check 2 test rows also in train: {len(overlap)} of {len(X_test)}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=3368)
cv_scores = cross_val_score(make_model(), X_train, y_train, cv=cv, scoring="f1_macro")
print(
    f"Check 3 CV macro F1: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})  "
    f"test macro F1: {real_f1:.3f}"
)
