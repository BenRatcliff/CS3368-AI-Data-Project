# Classifying HTTP, DNS, SMTP, and FTP Traffic: Decision Tree vs. KNN vs. Naïve Bayes

CS 3368 Intro to AI · Fall 2026 · Team The Port Authority

## Overview

We classify normal network flows from the UNSW-NB15 dataset into four service types (DNS, HTTP, SMTP, and FTP) and compare three machine-learning models that each come from a different paradigm.

| Model               | Paradigm       | Owner         | Folder                                      |
| ------------------- | -------------- | ------------- | ------------------------------------------- |
| Decision Tree       | Rule-based     | Colt Wood     | [`wood_colt/`](wood_colt/README.md)         |
| K-Nearest Neighbors | Instance-based | Jackson Clark | [`clark_jackson/`](clark_jackson/README.md) |
| Naïve Bayes         | Probabilistic  | Ben Ratcliff  | [`ratcliff_ben/`](ratcliff_ben/README.md)   |

Each folder has its own README with how to run that model and its full results.

## Setup

```bash
pip install -r requirements.txt
```

## Data

[UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) (Moustafa & Slay, 2015). The original CSVs are in `data/raw/`. The shared, cleaned files every model uses are in `data/processed/`.

Preprocessing (`shared/preprocess.py`):

- Keeps normal traffic (`label == 0`) for dns, http, smtp, and ftp (`ftp-data` is excluded).
- Drops `id`, `label`, and `attack_cat`.
- Drops `is_ftp_login`, `ct_ftp_cmd`, `ct_flw_http_mthd`, `trans_depth`, and `response_body_len`. Each is only nonzero for one protocol, so it gives away the answer.
- The target is `service`.

|       | dns   | http  | smtp  | ftp   | total  |
| ----- | ----- | ----- | ----- | ----- | ------ |
| train | 7,493 | 5,348 | 1,579 | 1,218 | 15,638 |
| test  | 3,068 | 4,013 | 635   | 758   | 8,474  |

## Shared code

Run from the repo root:

```bash
python shared/preprocess.py   # rebuild data/processed from data/raw (output is already committed)
python shared/baseline.py     # majority-class baseline
```

## Results (test set, 8,474 flows)

| Model                            | Accuracy | Macro F1 |
| -------------------------------- | -------- | -------- |
| Majority baseline (always `dns`) | 0.362    | 0.133    |
| Decision Tree                    | 0.999    | 0.998    |
| KNN (standard scaling, k = 3)    | 0.990    | 0.983    |
| Naïve Bayes (FCBF + log)         | 0.981    | 0.965    |
