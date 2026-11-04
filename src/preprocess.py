import pandas as pd

train = pd.read_csv("data/raw/UNSW_NB15_training-set.csv")
test = pd.read_csv("data/raw/UNSW_NB15_testing-set.csv")

print("raw train:", train.shape)
print("raw test:", test.shape)

CLASSES = ["dns", "http", "smtp", "ftp"]


def filter_rows(df):
    mask = (df["label"] == 0) & (df["service"].isin(CLASSES))
    return df[mask]


train = filter_rows(train)
test = filter_rows(test)

NOT_FEATURES = ["id", "label", "attack_cat"]
LEAKY = [
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "trans_depth",
    "response_body_len",
]

train = train.drop(columns=NOT_FEATURES + LEAKY)
test = test.drop(columns=NOT_FEATURES + LEAKY)

train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)

for name, df in [("train", train), ("test", test)]:
    print(f"{name}: {df.shape}")
    print(df["service"].value_counts(), "\n")
