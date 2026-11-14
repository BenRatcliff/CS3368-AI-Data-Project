import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
from fcbf import discretize_table, fcbf
import time
import os
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report

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
start = time.perf_counter()
model_a.fit(x_train, y_train)
fit_time_a = time.perf_counter() - start
pred_a = model_a.predict(x_test)

print("Model a: all 4 classes")
print("Accuracy:", round(accuracy_score(y_test, pred_a), 4))
print("Macro F1:", round(f1_score(y_test, pred_a, average = "macro"), 4))
print(confusion_matrix(y_test, pred_a, labels = ["dns", "http", "smtp", "ftp"]))

# Model a: tcp only
model_a_tcp = GaussianNB()
start = time.perf_counter()
model_a_tcp.fit(x_train_tcp, y_train_tcp)
fit_time_a_tcp = time.perf_counter() - start
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
start = time.perf_counter()
model_b.fit(x_train_log, y_train)
fit_time_b = time.perf_counter() - start
pred_b = model_b.predict(x_test_log)

print("Model b: all 4 classes")
print("Accuracy:", round(accuracy_score(y_test, pred_b), 4))
print("Macro F1:", round(f1_score(y_test, pred_b, average = "macro"), 4))
print(confusion_matrix(y_test, pred_b, labels = ["dns", "http", "smtp", "ftp"]))

# model b tcp only
model_b_tcp = GaussianNB()
start = time.perf_counter()
model_b_tcp.fit(x_train_log_tcp, y_train_tcp)
fit_time_b_tcp = time.perf_counter() - start
pred_b_tcp = model_b_tcp.predict(x_test_log_tcp)

print("Model b: tcp only")
print("Accuracy:", round(accuracy_score(y_test_tcp, pred_b_tcp), 4))
print("Macro F1:", round(f1_score(y_test_tcp, pred_b_tcp, average = "macro"), 4))
print(confusion_matrix(y_test_tcp, pred_b_tcp, labels = ["http", "smtp", "ftp"]))

# Model c: FCBF + log + GaussianNB, all 4 classes
# bin the training data and let FCBF pick features (train only, no test data involved)
x_train_disc = discretize_table(x_train, x_train)[0]
start = time.perf_counter()
selected_c, scores_c = fcbf(x_train_disc, y_train)
fcbf_time = time.perf_counter() - start
print("FCBF selected (all classes):", selected_c)

# top 10 most relevant features and their scores
for col in sorted(scores_c, key = lambda col: scores_c[col], reverse = True)[:10]:
    print(" ", col, round(scores_c[col], 4))

# train on the log data, but only the columns FCBF kept
model_c = GaussianNB()
start = time.perf_counter()
model_c.fit(x_train_log[selected_c], y_train)
fit_time_c = time.perf_counter() - start
pred_c = model_c.predict(x_test_log[selected_c])

print("Model c: all 4 classes")
print("Accuracy:", round(accuracy_score(y_test, pred_c), 4))
print("Macro F1:", round(f1_score(y_test, pred_c, average = "macro"), 4))
print(confusion_matrix(y_test, pred_c, labels = ["dns", "http", "smtp", "ftp"]))

# Model c: tcp only
# FCBF runs again here because different data can mean different features
x_train_tcp_disc = discretize_table(x_train_tcp, x_train_tcp)[0]
start = time.perf_counter()
selected_c_tcp, scores_c_tcp = fcbf(x_train_tcp_disc, y_train_tcp)
fcbf_time_tcp = time.perf_counter() - start
print("FCBF selected (tcp only):", selected_c_tcp)

for col in sorted(scores_c_tcp, key = lambda col: scores_c_tcp[col], reverse = True)[:10]:
    print(" ", col, round(scores_c_tcp[col], 4))

model_c_tcp = GaussianNB()
start = time.perf_counter()
model_c_tcp.fit(x_train_log_tcp[selected_c_tcp], y_train_tcp)
fit_time_c_tcp = time.perf_counter() - start
pred_c_tcp = model_c_tcp.predict(x_test_log_tcp[selected_c_tcp])

print("Model c: tcp only")
print("Accuracy:", round(accuracy_score(y_test_tcp, pred_c_tcp), 4))
print("Macro F1:", round(f1_score(y_test_tcp, pred_c_tcp, average = "macro"), 4))
print(confusion_matrix(y_test_tcp, pred_c_tcp, labels = ["http", "smtp", "ftp"]))

# majority class baseline
baseline = DummyClassifier(strategy = "most_frequent")
start = time.perf_counter()
baseline.fit(x_train, y_train)
fit_time_base = time.perf_counter() - start
pred_base = baseline.predict(x_test)

baseline_tcp = DummyClassifier(strategy = "most_frequent")
start = time.perf_counter()
baseline_tcp.fit(x_train_tcp, y_train_tcp)
fit_time_base_tcp = time.perf_counter() - start
pred_base_tcp = baseline_tcp.predict(x_test_tcp)

results = []

def add_result(model_name, dataset, y_true, y_pred, fit_seconds, n_features):
    # print the per class report
    print(model_name, "|", dataset)
    print(classification_report(y_true, y_pred, zero_division = 0))

    row = {}
    row["model"] = model_name
    row["dataset"] = dataset
    row["features"] = n_features
    row["accuracy"] = round(accuracy_score(y_true, y_pred), 4)
    row["macro_f1"] = round(f1_score(y_true, y_pred, average ="macro", zero_division = 0), 4)

    for service in ["dns", "http", "smtp", "ftp"]:
        if service in list(y_true):
            row["f1_" + service] = round(f1_score(y_true, y_pred, labels = [service], average = "macro", zero_division = 0), 4)
        else: 
            row["f1_" + service] = ""
    row["fit_seconds"] = round(fit_seconds, 4)
    results.append(row)

add_result("Majority baseline", "all classes", y_test, pred_base, fit_time_base, x_train.shape[1])
add_result("A: raw", "all classes", y_test, pred_a, fit_time_a, x_train.shape[1])
add_result("B: log", "all classes", y_test, pred_b, fit_time_b, x_train_log.shape[1])
add_result("C: FCBF + log", "all classes", y_test, pred_c, fcbf_time + fit_time_c, len(selected_c))

add_result("Majority baseline", "tcp only", y_test_tcp, pred_base_tcp, fit_time_base_tcp, x_train_tcp.shape[1])
add_result("A: raw", "tcp only", y_test_tcp, pred_a_tcp, fit_time_a_tcp, x_train_tcp.shape[1])
add_result("B: log", "tcp only", y_test_tcp, pred_b_tcp, fit_time_b_tcp, x_train_log_tcp.shape[1])
add_result("C: FCBF + log", "tcp only", y_test_tcp, pred_c_tcp, fcbf_time_tcp + fit_time_c_tcp, len(selected_c_tcp))

results_table = pd.DataFrame(results)
print(results_table)

# save everything to a folder
results_folder = "results/naive_bayes"
os.makedirs(results_folder, exist_ok = True)
results_table.to_csv(results_folder + "/results.csv", index = False)

model_names = ["Majority baseline", "A: raw", "B: log", "C: FCBF + log"]
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

fig, ax = plt.subplots(figsize = (10, 6))
bars_all = ax.bar(positions - width / 2, all_scores, width, label = "All 4 classes", color = "#098eb7")
bars_tcp = ax.bar(positions + width / 2, tcp_scores, width, label = "TCP only (no DNS)", color = "#09b731")

# write each score on the top of its bar
for bar in list(bars_all) + list(bars_tcp):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
        f"{bar.get_height():.2f}", ha = "center", fontsize = 14)

ax.set_title("Naive Bayes models: macro F1 on the UNSW-NB15 test set", fontsize = 18)
ax.set_xlabel("Model", fontsize = 16)
ax.set_ylabel("Macro F1 (higher is better)", fontsize = 16)
ax.set_xticks(positions)
ax.set_xticklabels(model_names, fontsize = 14)
ax.set_ylim(0, 1.1)
ax.tick_params(axis = "y", labelsize = 14)
ax.legend(fontsize = 14, loc = "upper left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis = "y", alpha = 0.3)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig(results_folder + "/macro_f1.png", dpi = 200)
print("Saved results.csv and macro_f1.png to", results_folder)
