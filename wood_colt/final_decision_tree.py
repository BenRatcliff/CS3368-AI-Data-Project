"""Assignment 6: Colt Wood's updated Decision Tree (timing features + class balancing).

Run: python final_decision_tree.py
Uses the existing S3 definition (TCP-only HTTP/SMTP/FTP; excludes easy volume
and TCP sequence-number features). Selects the improved model using 5-fold
cross-validation on the training file. Evaluates on the official test file once.
Does not change or overwrite the Assignment 5 results.
"""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from traffic_common import (RANDOM_STATE, TRAIN_CSV, TEST_CSV, RESULTS_DIR,
                            SCENARIOS, load_normal_services, make_pipeline, scenario_xy)
from train_decision_tree import select_base_params


class TimingFeatureAdder(BaseEstimator, TransformerMixin):
    """Derived TCP flow-timing features; calculations use no labels/test stats."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for a, b, name in [
            ('sinpkt', 'dinpkt', 'interarrival'),
            ('sjit', 'djit', 'jitter'),
            ('synack', 'ackdat', 'handshake'),
        ]:
            if a in X and b in X:
                X[f'timing_{name}_log_difference'] = (np.log1p(X[a].clip(lower=0))
                                                    - np.log1p(X[b].clip(lower=0)))
                X[f'timing_{name}_log_sum'] = np.log1p((X[a] + X[b]).clip(lower=0))
        for a, b, name in [
            ('tcprtt', 'dur', 'rtt_duration'),
            ('synack', 'tcprtt', 'syn_rtt'),
            ('ackdat', 'tcprtt', 'ack_rtt'),
            ('sjit', 'sinpkt', 'source_jitter_gap'),
            ('djit', 'dinpkt', 'destination_jitter_gap'),
            ('swin', 'dwin', 'windows'),
            ('sttl', 'dttl', 'ttl'),
        ]:
            if a in X and b in X:
                X[f'timing_{name}_log_difference'] = (np.log1p(X[a].clip(lower=0))
                                                   - np.log1p(X[b].clip(lower=0)))
        return X.replace([np.inf, -np.inf], np.nan)


def improved_pipeline(params):
    pipeline = make_pipeline(DecisionTreeClassifier(random_state=RANDOM_STATE, **params))
    pipeline.steps.insert(0, ('timing_features', TimingFeatureAdder()))
    return pipeline


def main():
    output = RESULTS_DIR / 'final_model'
    output.mkdir(parents=True, exist_ok=True)
    train = load_normal_services(TRAIN_CSV)
    test = load_normal_services(TEST_CSV)
    x_train, y_train = scenario_xy(train, SCENARIOS['S3'])
    x_test, y_test = scenario_xy(test, SCENARIOS['S3'])
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

    # Original model as implemented for Assignment 5.
    base_params, _ = select_base_params(x_train, y_train, 'S3')
    candidates = [
        ('Original Decision Tree', make_pipeline(DecisionTreeClassifier(random_state=RANDOM_STATE, **base_params))),
        ('Timing features: unbalanced', improved_pipeline({'max_depth': None, 'min_samples_leaf': 1})),
        ('Timing features: balanced', improved_pipeline({'max_depth': None, 'min_samples_leaf': 1,
                                                         'class_weight': 'balanced'})),
        ('Timing features: balanced depth 12', improved_pipeline({'max_depth': 12, 'min_samples_leaf': 1,
                                                                  'class_weight': 'balanced'})),
    ]
    cv_rows = []
    for name, pipe in candidates:
        scores = cross_val_score(pipe, x_train, y_train, cv=cv, scoring='f1_macro', n_jobs=1)
        cv_rows.append({'model': name, 'cv_macro_f1': scores.mean(), 'cv_std': scores.std()})
    pd.DataFrame(cv_rows).to_csv(output / 'training_cv_comparison.csv', index=False)

    # Selection uses *training CV only*, not the official test file.
    # Baseline stays in comparison; choose the best of the timing candidates.
    winner_i = max(range(1, len(candidates)), key=lambda i: cv_rows[i]['cv_macro_f1'])
    winner_name = candidates[winner_i][0]
    report_rows = []
    class_labels = ['HTTP', 'SMTP', 'FTP']
    for name, pipe in [candidates[0], candidates[winner_i]]:
        pipe.fit(x_train, y_train)
        predictions = pipe.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        macro_f1 = f1_score(y_test, predictions, average='macro')
        report_rows.append({'model': name, 'accuracy': accuracy, 'macro_f1': macro_f1,
                            'errors': int((predictions != y_test).sum()),
                            'test_rows': len(y_test)})
        prefix = 'original' if name == 'Original Decision Tree' else 'final'
        cm = confusion_matrix(y_test, predictions, labels=class_labels)
        pd.DataFrame(cm, index=class_labels, columns=class_labels).to_csv(output / f'{prefix}_confusion.csv')
        (output / f'{prefix}_classification_report.txt').write_text(
            classification_report(y_test, predictions, labels=class_labels, zero_division=0))
        if prefix == 'final':
            joblib.dump(pipe, output / 'final_timing_tree.joblib')
    pd.DataFrame(report_rows).to_csv(output / 'final_test_comparison.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = ['Original tree', 'Improved tree']
    for ax, metric, title in zip(axes, ['accuracy', 'macro_f1'], ['Test accuracy', 'Test macro F1']):
        vals = [r[metric] for r in report_rows]
        ax.bar(labels, vals)
        ax.set_ylim(0.9 if metric == 'accuracy' else 0.85, 1.005)
        ax.set_ylabel(title)
        ax.set_title(title + ' (S3: HTTP / SMTP / FTP)')
        for i, v in enumerate(vals):
            ax.text(i, v + 0.002, f'{v:.3f}', ha='center', fontsize=10)
    fig.tight_layout()
    fig.savefig(output / 'final_tree_comparison.png', dpi=160)
    plt.close(fig)

    print(f'Train rows: {len(y_train)}, test rows: {len(y_test)}')
    print('Training-only CV:', pd.DataFrame(cv_rows).to_string(index=False))
    print(f'Chosen from training CV: {winner_name}')
    print('Official test:', pd.DataFrame(report_rows).to_string(index=False))
    print('Saved results to:', output)


if __name__ == '__main__':
    main()
