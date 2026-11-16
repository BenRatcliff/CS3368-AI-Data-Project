from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
TRAIN_CSV = DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_CSV = DATA_DIR / "UNSW_NB15_testing-set.csv"
RANDOM_STATE = 42

SERVICE_TO_CLASS = {
    "http": "HTTP",
    "dns": "DNS",
    "smtp": "SMTP",
    "ftp": "FTP",  # FTP control flows only; ftp-data flows are left out on purpose.
}

# Identifiers, targets, attack labels, and columns derived from the service
# itself. Removing them stops the model from reading its answer directly.
EXCLUDE_COLUMNS = [
    "id", "service", "label", "attack_cat",
    "ct_srv_src", "ct_srv_dst",                              # same-service counters
    "ct_flw_http_mthd", "trans_depth", "response_body_len",  # HTTP specific
    "is_ftp_login", "ct_ftp_cmd",                            # FTP specific
]

# Columns that describe HOW MUCH data moved (volume, loss, rate). Dropped in
# the hardest scenario to see whether timing/handshake behaviour alone works.
VOLUME_COLUMNS = [
    "sbytes", "dbytes", "spkts", "dpkts", "sloss", "dloss",
    "smean", "dmean", "sload", "dload", "rate",
]
# TCP initial sequence numbers are arbitrary values, not traffic behaviour.
SEQUENCE_COLUMNS = ["stcpb", "dtcpb"]


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    classes: tuple
    tcp_only: bool
    drop: tuple


SCENARIOS = {
    "S1": Scenario("S1", "S1 Full 4-class (original task)",
                   ("HTTP", "DNS", "SMTP", "FTP"), False, ()),
    "S2": Scenario("S2", "S2 TCP-only 3-class (DNS/UDP giveaway removed)",
                   ("HTTP", "SMTP", "FTP"), True, ("proto",)),
    "S3": Scenario("S3", "S3 Behaviour-only 3-class (no volume or sequence numbers)",
                   ("HTTP", "SMTP", "FTP"), True,
                   ("proto", *VOLUME_COLUMNS, *SEQUENCE_COLUMNS)),
}


def load_normal_services(path: Path) -> pd.DataFrame:
    """Normal (label == 0) flows of HTTP, DNS, SMTP or FTP, plus a `target` column."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path.name}. Put it inside {DATA_DIR}")
    df = pd.read_csv(path)
    missing = {"service", "label", "proto"}.difference(df.columns)
    if missing:
        raise ValueError(f"Missing dataset columns: {sorted(missing)}")
    df = df.loc[(df["label"] == 0) & df["service"].isin(SERVICE_TO_CLASS)].copy()
    df["target"] = df["service"].map(SERVICE_TO_CLASS)
    return df


def scenario_xy(df: pd.DataFrame, scenario: Scenario) -> tuple[pd.DataFrame, pd.Series]:
    if scenario.tcp_only:
        df = df.loc[df["proto"] == "tcp"]
    y = df["target"]
    X = df.drop(columns=["target", *EXCLUDE_COLUMNS, *scenario.drop], errors="ignore")
    X = X.replace([np.inf, -np.inf], np.nan)
    return X, y


class FlowFeatureAdder(BaseEstimator, TransformerMixin):
    """Improvement #1: ratio/shape features describing the two directions of a flow.

    A feature is only added when the columns it needs are present, so the same
    step works in every scenario.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        eps = 1e-9

        def lg(col):
            return np.log1p(X[col].clip(lower=0))

        cols = set(X.columns)
        if {"sload", "dload"} <= cols:
            X["fx_load_ratio"] = lg("sload") - lg("dload")
        if {"sinpkt", "dinpkt"} <= cols:
            X["fx_inter_arrival_ratio"] = lg("sinpkt") - lg("dinpkt")
        if {"sjit", "djit"} <= cols:
            X["fx_jitter_ratio"] = lg("sjit") - lg("djit")
        if {"synack", "ackdat", "tcprtt"} <= cols:
            X["fx_handshake_share"] = (X["synack"] + X["ackdat"]) / (X["tcprtt"] + eps)
        if {"tcprtt", "dur"} <= cols:
            X["fx_rtt_over_duration"] = X["tcprtt"] / (X["dur"] + eps)
        if "dur" in cols:
            X["fx_log_duration"] = lg("dur")
        if {"sbytes", "dbytes"} <= cols:
            X["fx_byte_balance"] = lg("sbytes") - lg("dbytes")
        if {"sbytes", "spkts"} <= cols:
            X["fx_src_bytes_per_pkt"] = X["sbytes"] / (X["spkts"] + eps)
        if {"dbytes", "dpkts"} <= cols:
            X["fx_dst_bytes_per_pkt"] = X["dbytes"] / (X["dpkts"] + eps)
        return X.replace([np.inf, -np.inf], np.nan)


def make_pipeline(estimator, add_features: bool = False, scale: bool = False) -> Pipeline:
    """Preprocessing (fit on training data only) followed by `estimator`."""
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    prep = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps),
             make_column_selector(dtype_include="number")),
            ("categorical", Pipeline([
                ("fill", SimpleImputer(strategy="most_frequent")),
                ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), make_column_selector(dtype_exclude="number")),
        ],
        remainder="drop",
    )
    steps = []
    if add_features:
        steps.append(("flow_features", FlowFeatureAdder()))
    steps += [("prep", prep), ("model", estimator)]
    return Pipeline(steps)


def fast_scores(y_true_idx: np.ndarray, y_pred_idx: np.ndarray, n_classes: int):
    """Accuracy and macro-F1 from integer-coded labels (fast, for bootstrapping)."""
    cm = np.bincount(y_true_idx * n_classes + y_pred_idx,
                     minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(axis=0), out=np.zeros(n_classes), where=cm.sum(axis=0) > 0)
    recall = np.divide(tp, cm.sum(axis=1), out=np.zeros(n_classes), where=cm.sum(axis=1) > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(n_classes), where=denom > 0)
    return tp.sum() / cm.sum(), f1.mean()
