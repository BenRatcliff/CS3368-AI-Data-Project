import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

X_train, y_train = train.drop(columns=["service"]), train["service"]
X_test, y_test = test.drop(columns=["service"]), test["service"]

model = DummyClassifier(strategy="most_frequent")
model.fit(X_train, y_train)
pred = model.predict(X_test)

print(f"accuracy: {accuracy_score(y_test, pred):.3f}")
print(f"macro F1: {f1_score(y_test, pred, average='macro'):.3f}")
