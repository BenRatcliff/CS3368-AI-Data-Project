import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
from fcbf import discretize_table, fcbf

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

# find the numeric columns first
num_cols = list(x_train.select_dtypes("number").columns)

# test to make sure log(negative number) doesn't exist so nothing breaks
print("Numeric columns:", len(num_cols))
# sum is used here twice because the first sum returns a series and the second one returns a single number
print("Negative values found in train:", (x_train[num_cols] < 0).sum().sum())
print("Negative values found in test:", (x_test[num_cols] < 0).sum().sum())

# copy so that the raw tables stay untouched (model a still needs them)
x_train_log = x_train.copy()
x_test_log = x_test.copy()

for col in num_cols:
    x_train_log[col] = np.log1p(x_train_log[col])
    x_test_log[col] = np.log1p(x_test_log[col])

# for tcp only
x_train_log_tcp = x_train_log[tcp_train].drop(columns = proto_cols)
x_test_log_tcp = x_test_log[tcp_test].drop(columns = proto_cols)

# model b all 4 classes
model_b = GaussianNB()
model_b.fit(x_train_log, y_train)
pred_b = model_b.predict(x_test_log)

print("Model b: all 4 classes")
print("Accuracy:", round(accuracy_score(y_test, pred_b), 4))
print("Macro F1:", round(f1_score(y_test, pred_b, average = "macro"), 4))
print(confusion_matrix(y_test, pred_b, labels = ["dns", "http", "smtp", "ftp"]))

# model b tcp only
model_b_tcp = GaussianNB()
model_b_tcp.fit(x_train_log_tcp, y_train_tcp)
pred_b_tcp = model_b_tcp.predict(x_test_log_tcp)

print("Model b: tcp only")
print("Accuracy:", round(accuracy_score(y_test_tcp, pred_b_tcp), 4))
print("Macro F1:", round(f1_score(y_test_tcp, pred_b_tcp, average = "macro"), 4))
print(confusion_matrix(y_test_tcp, pred_b_tcp, labels = ["http", "smtp", "ftp"]))

# Model c: FCBF + log + GaussianNB, all 4 classes
# bin the training data and let FCBF pick features (train only, no test data involved)
x_train_disc = discretize_table(x_train, x_train)[0]
selected_c, scores_c = fcbf(x_train_disc, y_train)
print("FCBF selected (all classes):", selected_c)

# top 10 most relevant features and their scores
for col in sorted(scores_c, key = lambda col: scores_c[col], reverse = True)[:10]:
    print(" ", col, round(scores_c[col], 4))

# train on the log data, but only the columns FCBF kept
model_c = GaussianNB()
model_c.fit(x_train_log[selected_c], y_train)
pred_c = model_c.predict(x_test_log[selected_c])

print("Model c: all 4 classes")
print("Accuracy:", round(accuracy_score(y_test, pred_c), 4))
print("Macro F1:", round(f1_score(y_test, pred_c, average = "macro"), 4))
print(confusion_matrix(y_test, pred_c, labels = ["dns", "http", "smtp", "ftp"]))

# Model c: tcp only
# FCBF runs again here because different data can mean different features
x_train_tcp_disc = discretize_table(x_train_tcp, x_train_tcp)[0]
selected_c_tcp, scores_c_tcp = fcbf(x_train_tcp_disc, y_train_tcp)
print("FCBF selected (tcp only):", selected_c_tcp)

for col in sorted(scores_c_tcp, key = lambda col: scores_c_tcp[col], reverse = True)[:10]:
    print(" ", col, round(scores_c_tcp[col], 4))

model_c_tcp = GaussianNB()
model_c_tcp.fit(x_train_log_tcp[selected_c_tcp], y_train_tcp)
pred_c_tcp = model_c_tcp.predict(x_test_log_tcp[selected_c_tcp])

print("Model c: tcp only")
print("Accuracy:", round(accuracy_score(y_test_tcp, pred_c_tcp), 4))
print("Macro F1:", round(f1_score(y_test_tcp, pred_c_tcp, average = "macro"), 4))
print(confusion_matrix(y_test_tcp, pred_c_tcp, labels = ["http", "smtp", "ftp"]))
