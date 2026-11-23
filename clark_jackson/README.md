# K-Nearest Neighbors

Jackson Clark · CS 3368 Intro to AI · Fall 2026 · The Port Authority

Classifies normal UNSW-NB15 traffic as DNS, HTTP, SMTP, or FTP using K-Nearest Neighbors.

## How to run

Run from the repo root (the folder with `data/` in it):

```bash
pip install -r requirements.txt
python clark_jackson/knn_final.py
```

Interim scripts (still runnable):

```bash
python clark_jackson/knn.py
python clark_jackson/knn_tune.py
python clark_jackson/knn_checks.py
```

## Files

- `knn_final.py`: final model; trains models A, B, and C, prints the results, and saves the table and charts
- `knn.py`: first version (min-max scaling, k = 5)
- `knn_tune.py`: interim tuning of k (1–25) with 5-fold CV for min-max, standard, and no scaling
- `knn_checks.py`: leakage checks on the interim model
- `results/knn_final_results.csv`: final results table
- `results/knn_final_macro_f1.png`: macro F1 for each final model
- `results/knn_final_confusion.png`: confusion matrices, model A vs. model C
- `results/knn_results.csv`, `knn_f1_vs_k.png`, `knn_confusion_matrix.png`: interim results

## Final models

Each model one-hot encodes `proto` and `state` and scales the numeric columns with standard scaling, then runs KNN.

- **Model A:** interim model (standard scaling, k = 3)
- **Model B:** A + log transform (`log1p`) on the numeric columns before scaling
- **Model C:** B + k, vote weighting, and distance metric chosen by 5-fold CV on the training data only

Model C's CV pick: k = 7, distance-weighted votes, Manhattan distance.

Each model is tested on all four services and on TCP only (DNS removed).

| Model              | Test            | Accuracy  | Macro F1  | HTTP↔FTP errors | Total errors |
| ------------------ | --------------- | --------- | --------- | --------------- | ------------ |
| A: interim         | all classes     | 0.990     | 0.983     | 80              | 85           |
| B: log             | all classes     | 0.996     | 0.993     | 35              | 35           |
| **C: log + tuned** | **all classes** | **0.999** | **0.998** | **11**          | **11**       |
| A: interim         | tcp only        | 0.983     | 0.976     | 86              | 90           |
| B: log             | tcp only        | 0.993     | 0.989     | 36              | 39           |
| **C: log + tuned** | **tcp only**    | **0.999** | **0.998** | **7**           | **8**        |

At the interim, 80 of the 85 errors were HTTP and FTP mixed up. Byte counts are very skewed, so a few huge flows dominated the distance. The log transform (taken from Ben's Naive Bayes model) cut those errors from 80 to 35. Distance-weighted votes and Manhattan distance cut them to 11.

## Interim models

| Scaling                | Best k (5-fold CV) | Accuracy  | Macro F1  |
| ---------------------- | ------------------ | --------- | --------- |
| Min-max                | 5                  | 0.993     | 0.988     |
| **Standard (interim)** | **3**              | **0.990** | **0.983** |
| None                   | 1                  | 0.762     | 0.590     |

Without scaling, large columns such as byte counts dominate the distance, and macro F1 drops to 0.59.

Leakage checks (interim model):

- Shuffled training labels: macro F1 drops to 0.22 (chance level)
- Test rows that also appear in train: 0 of 8,474
- 5-fold CV macro F1 0.987 vs. test macro F1 0.983

## Sources

- Scaling comparison for KNN: Serag et al. (2025)
- 5-fold CV and dropping leaky columns: Antari et al. (2025)
- Dataset: UNSW-NB15 (Moustafa & Slay, 2015)
- scikit-learn documentation: Nearest Neighbors
