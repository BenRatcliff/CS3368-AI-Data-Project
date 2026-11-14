import pandas as pd

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

# every row is tcp, so the proto columns get dropped like me as a baby
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
