# FCBF removes irrelevant and redundant features from the dataset, giving it really good synergy with Naive Bayes
# This is accomplished by scoring every feature against the class, drop features that scored below the threshold, sort highest to lowest, 
# take the top feature and use it to remove redundant features below it, and move to the next feature that's still in the list, repeat

import numpy as np
import pandas as pd

def entropy(x):
    # H(X) = -sum p * log2(p) for every value x can take
    counts = {}
    for value in x:
        if value in counts:
            counts[value] += 1
        else: 
            counts[value] = 1

    total = len(x)
    h = 0.0
    for value in counts:
        p = counts[value] / total
        h = h - p * np.log2(p)
    return h

def conditional_entropy(x, y):
    # H(X|Y) = sum over each value v of y: P(y = v) * H(x where y = v)
    x = list(x)
    y = list(y)
    total = len(y)

    y_values = []
    for v in y:
        if v not in y_values:
            y_values.append(v)

    h = 0.0
    for v in y_values:
        x_subset = []
        for i in range(total):
            if y[i] == v:
                x_subset.append(x[i])
        weight = len(x_subset) / total
        h = h + weight * entropy(x_subset)
    return h

def symmetric_uncertainty(x, y):
    # SU = 2 * (H(X) - H(X|Y)) / (H(X) + H(Y)) between 0 and 1, higher = better
    h_x = entropy(x)
    h_y = entropy(y)
    if h_x + h_y == 0:
        return 0.0
    info_gain = h_x - conditional_entropy(x, y)

    return 2 * info_gain / (h_x + h_y)

def discrete(train_col, test_col, n_bins = 10):
    # T/F columns become 0 and 1
    train_col = train_col.astype(float)
    test_col = test_col.astype(float)

    if train_col.nunique() <= n_bins:
        return train_col, test_col

    edges = pd.qcut(train_col, n_bins, retbins = True, duplicates = "drop")[1]

    edges[0] = -np.inf
    edges[-1] = np.inf

    train_binned = pd.cut(train_col, bins = edges, labels = False)
    test_binned = pd.cut(test_col, bins = edges, labels = False)

    return train_binned, test_binned

def discretize_table(x_train, x_test, n_bins = 10):
    # bin every column of both tables using the train edges for both
    x_train_disc = pd.DataFrame(index = x_train.index)
    x_test_disc = pd.DataFrame(index = x_test.index)
    for col in x_train.columns:
        train_binned, test_binned = discrete(x_train[col], x_test[col], n_bins)
        x_train_disc[col] = train_binned
        x_test_disc[col] = test_binned

    return x_train_disc, x_test_disc

def fcbf(x_disc, y, delta = 0.0):
    # score every feature against the class
    scores = {}
    for col in x_disc.columns:
        scores[col] = symmetric_uncertainty(x_disc[col], y)

    # drop the irrelevant features
    candidates = []
    for col in scores:
        if scores[col] > delta:
            candidates.append(col)

    # sort best to worst
    candidates = sorted(candidates, key = lambda col: scores[col], reverse = True)

    # each surviving feature removes the feature below it that it makes redundant
    i = 0
    while i < len(candidates):
        f_p = candidates[i]
        j = i + 1
        while j < len(candidates):
            f_q = candidates[j]
            if symmetric_uncertainty(x_disc[f_p], x_disc[f_q]) >= scores[f_q]:
                candidates.pop(j)
            else:
                j += 1
        i += 1

    return candidates, scores

#-------Test-------#

if __name__ == "__main__":
    # math testing with known answers
    import random as rd
    print("Math testing:")
    print(entropy([0, 0, 1, 1])) # 1.0
    print(entropy([0, 0, 0, 0])) # 0.0
    print(entropy(["a", "b", "c", "d"])) # 2.0
    print(conditional_entropy([0, 0, 1, 1], ["a", "a", "b", "b"])) # 0.0
    print(conditional_entropy([0, 1, 0, 1], ["a", "a", "b", "b"])) # 1.0
    x = [0, 0, 1, 1, 2, 2]
    print(symmetric_uncertainty(x, x)) # 1.0
    rd.seed(0)
    a = [rd.randint(0, 3) for i in range(5000)]
    b = [rd.randint(0, 3) for i in range(5000)]
    print(round(symmetric_uncertainty(a, b), 4)) # 0.0003

    # test on the tcp only dataset 
    print("FCBF on the tcp only dataset:")
    train = pd.read_csv("data/processed/train.csv")
    train = train[train["service"] != "dns"]
    y = train["service"]
    x = train.drop(columns = ["service", "proto", "state"])
    x_disc = discretize_table(x, x)[0]
    selected, scores = fcbf(x_disc, y)
    print("stcpb score:", round(scores["stcpb"], 4))
    print("selected:", selected)
