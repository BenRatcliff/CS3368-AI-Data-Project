# Naive Bayes + FCBF

Benjamin Ratcliff · CS 3368 Intro to AI · Fall 2026 · The Port Authority

Classifies normal UNSW-NB15 traffic as DNS, HTTP, SMTP, or FTP using Gaussian Naive Bayes.

## How to run

Run from the repo root (the folder with `data/` in it):

```bash
pip install -r requirements.txt
python ratcliff_ben/run_nb.py
```

To run only the FCBF tests:

```bash
python ratcliff_ben/fcbf.py
```

## Files

- `run_nb.py`: trains every model, prints the results, and saves the table and chart
- `fcbf.py`: FCBF feature selection, with built-in tests

## Models

- **Model A:** Naive Bayes on all features
- **Model B:** A + log transform
- **Model C:** B + FCBF

Each model is tested on all four services and on TCP only (DNS removed).

Results are saved to `results/naive_bayes/`.

## Sources

- FCBF algorithm: Yu & Liu (2003)
- FCBF with Naive Bayes for traffic classification: Moore & Zuev (2005)
- Methods for handling skewed data: Radečić (2020) (Not Scopus indexed)
- RSM and machine learning for predicting pavement stress under fixed traffic loads: Alnadish et al. (2025)
- Dataset: UNSW-NB15 (Moustafa & Slay, 2015)
