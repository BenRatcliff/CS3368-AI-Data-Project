# Classifying HTTP, DNS, SMTP, and FTP Traffic

CS 3368 Fall 2026 · Team The Port Authority

- Colt Wood: Decision Tree
- Ben Ratcliff: Naïve Bayes
- Jackson Clark: KNN

## Setup

```bash
pip install -r requirements.txt
python shared/preprocess.py
python shared/baseline.py
```

Run the commands from the repo root.

## Data

[UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) (Moustafa & Slay, 2015). The original files are in `data/raw/`. The cleaned files are in `data/processed/`.

Preprocessing:

- Keeps normal traffic (`label == 0`) for dns, http, smtp, and ftp.
- Drops `id`, `label`, and `attack_cat`.
- Drops `is_ftp_login`, `ct_ftp_cmd`, `ct_flw_http_mthd`, `trans_depth`, and `response_body_len`. Each is only nonzero for one protocol, so it gives away the answer.
- The target is `service`.

|       | dns   | http  | smtp  | ftp   | total  |
| ----- | ----- | ----- | ----- | ----- | ------ |
| train | 7,493 | 5,348 | 1,579 | 1,218 | 15,638 |
| test  | 3,068 | 4,013 | 635   | 758   | 8,474  |

## Baseline

The baseline always predicts `dns`: accuracy 0.362, macro F1 0.133.
