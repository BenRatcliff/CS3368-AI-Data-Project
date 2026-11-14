import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# file reading
train = pd.read_csv("data/processed/train.csv")
test= pd.read_csv("data/processed/test.csv")

# split features x and y
y_train = train["service"]
x_train = train.drop(columns = ["service"])
y_test = test["service"]
x_test = test.drop(columns = ["service"])

# encode the text columns
x_train = pd.get_dummies(x_train, columns = ["proto", "state"])
x_test = pd.get_dummies(x_test, columns = ["proto", "state"])
# make test have the same number of columns in the same order
x_test = x_test.reindex(columns = x_train.columns, fill_value = 0)

# tco only no dns
tcp_train = y_train != "dns"
x_train_tcp = x_train[tcp_train]
y_train_tcp = y_train[tcp_train]

tcp_test = y_test != "dns"
x_test_tcp = x_test[tcp_test]
y_test_tcp = y_test[tcp_test]

# every row is tcp, so the proto columns get dropped
proto_cols = []
for col in x_train_tcp.columns:
    if col.startswith("proto_"):
        proto_cols.append(col)

x_train_tcp = x_train_tcp.drop(columns = proto_cols)
x_test_tcp = x_test_tcp.drop(columns = proto_cols)

print("x_train:", x_train.shape)
print("x_test:", x_test.shape)
print("x_train_tcp:", x_train_tcp.shape)
print("x_test_tcp:", x_test_tcp.shape)
print(y_train_tcp.value_counts())

# Model a: all 4 classes
model_a = GaussianNB()
model_a.fit(x_train, y_train)
pred_a = model_a.predict(x_test)

print("Model a: all 4 classes")
print("Accuracy:", round(accuracy_score(y_test, pred_a), 4))
print("Macro F1:", round(f1_score(y_test, pred_a, average = "macro"), 4))
print(confusion_matrix(y_test, pred_a, labels = ["dns", "http", "smtp", "ftp"]))

# Model a: tcp only
model_a_tcp = GaussianNB()
model_a_tcp.fit(x_train_tcp, y_train_tcp)
pred_a_tcp = model_a_tcp.predict(x_test_tcp)

print("Model a: tcp only")
print("Accuracy:", round(accuracy_score(y_test_tcp, pred_a_tcp), 4))
print("Macro F1:", round(f1_score(y_test_tcp, pred_a_tcp, average = "macro"), 4))
print(confusion_matrix(y_test_tcp, pred_a_tcp, labels = ["http", "smtp", "ftp"]))

print(model_a.classes_, model_a.class_prior_)
