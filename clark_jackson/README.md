# K-Nearest Neighbors

Jackson Clark · CS 3368 Intro to AI · Fall 2026 · The Port Authority

Classifies normal UNSW-NB15 traffic as DNS, HTTP, SMTP, or FTP using K-Nearest Neighbors, and tests how much feature scaling matters.

## How to run

Run from the repo root:

```bash
pip install -r requirements.txt
python clark_jackson/knn.py          # first version
python clark_jackson/knn_tune.py     # tuning and scaling comparison
python clark_jackson/knn_checks.py   # leakage checks on the final model
```

## Files

- `knn.py`: first version (min-max scaling, k = 5)
- `knn_tune.py`: tunes k (1–25) with 5-fold cross-validation for min-max, standard, and no scaling; saves the table and charts
- `knn_checks.py`: leakage checks on the final model
- `results/knn_results.csv`: results table
- `results/knn_f1_vs_k.png`: macro F1 vs. k for each scaling method
- `results/knn_confusion_matrix.png`: confusion matrix for the final model

## Models

Each model one-hot encodes `proto` and `state`, scales the numeric columns, then runs KNN.

| Scaling              | Best k (5-fold CV) | Accuracy  | Macro F1  |
| -------------------- | ------------------ | --------- | --------- |
| Min-max              | 5                  | 0.993     | 0.988     |
| **Standard (final)** | **3**              | **0.990** | **0.983** |
| None                 | 1                  | 0.762     | 0.590     |

The final model is chosen by cross-validation score, not test score. Without scaling, large columns such as byte counts dominate the distance, and macro F1 drops to 0.59.

## Leakage checks (final model)

- Shuffled training labels: macro F1 drops to 0.22 (chance level)
- Test rows that also appear in train: 0 of 8,474
- 5-fold CV macro F1 0.987 vs. test macro F1 0.983

## Sources

- Scaling comparison for KNN: Serag et al. (2025)
- 5-fold CV and dropping leaky columns: Antari et al. (2025)
- Dataset: UNSW-NB15 (Moustafa & Slay, 2015)
- scikit-learn documentation: Nearest Neighbors
