# CS 3368 - Colt Wood: Decision Tree Network Traffic Classifier

## Project Overview

I built a Decision Tree to classify normal network traffic as HTTP, DNS, SMTP, or FTP using the UNSW-NB15 dataset. This project does not detect attacks or monitor live traffic.

We use **FTP control traffic (`ftp`) only**. Records labeled `ftp-data` are excluded from training and testing. Jackson is working on KNN, and Ben is working on Naive Bayes. The other models in my script are comparisons or additional experiments.

## How to Run

1. Extract the ZIP and open the `wood_colt_software_v4` folder in VS Code.
2. Make sure Python 3 is installed.
3. Open a terminal in that folder and run:

   `python -m pip install -r requirements.txt`

   `python train_decision_tree.py`

On Windows you may need to use `py` instead of `python`. The program uses relative paths, so no paths need to be changed. It can take a few minutes to finish all tests. Results and saved models will be written to `results/`.

## What the Code Does

- Loads the UNSW-NB15 CSVs and keeps normal HTTP, DNS, SMTP, and FTP flows: **15,638 training** and **8,474 testing** records.
- Removes features that directly give away the traffic type.
- Uses cross-validation on the training data to choose Decision Tree settings.
- Tests the Decision Tree against simple baselines and Logistic Regression.
- Tries extra flow features (+1) and Random Forest with those features (+2).
- Runs additional tests and saves result tables, charts, and trained models.

## Main Results (Four Traffic Types)

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Majority-class baseline | 36.2% | 0.133 |
| Depth-1 Decision Tree | 83.6% | 0.463 |
| Depth-3 Decision Tree | 97.9% | 0.967 |
| Logistic Regression | 99.7% | 0.995 |
| **My Decision Tree** | **99.93%** | **0.998** |
| Tree + flow features | 99.92% | 0.998 |
| Random Forest + flow features | 99.95% | 0.999 |

The main Decision Tree made **6 mistakes out of 8,474 test records**. Accuracy is the percent classified correctly. Macro F1 gives each traffic type equal weight.

## Extra Tests

The main task was surprisingly easy because all DNS flows in our selected records used UDP, while the other three services used TCP. I also tested:

- **S2 (TCP only):** HTTP, SMTP, and FTP without DNS. Decision Tree: **99.91%** accuracy.
- **S3 (reduced features):** The same three classes, but with many of the easier data-volume and sequence-number features removed. Decision Tree: **96.6%**, Tree + flow features: **98.6%**, Random Forest + flow features: **99.5%** accuracy.

S3 made the differences between the models easier to see. The script also runs training-data cross-validation and a swapped train/test experiment.

## Limitations

- The dataset comes from a controlled lab environment, so these scores may not carry over to other networks.
- I designed some additional tests after reviewing early test results. Their scores may be optimistic.
- In the swapped-direction check, the original test file is used for training; this is a separate experiment, not the main evaluation.
- The classifier always chooses a known category, even for an unknown service such as `ftp-data`.
- More et al. (2024) studied attack detection, not normal-service classification, so the scores are not directly comparable.

## Included Files

- `train_decision_tree.py` - trains and compares the models.
- `traffic_common.py` - loads and prepares the datasets.
- `make_figures.py` - generates graphs.
- `requirements.txt` - Python packages.
- `data/` - the original UNSW-NB15 training and testing CSVs.
- `results/all_results.csv` - saved numerical results.
- `results/fig_model_comparison.png`, `fig_confusion_matrices.png`, and `fig_improvement_ladder_zoom.png` - selected graphs.

More result files and `.joblib` models are **generated when you run the main script**. They are intentionally not included in this smaller submission. No separate prediction tool is included.

## Sources

- UNSW-NB15 dataset: https://research.unsw.edu.au/projects/unsw-nb15-dataset
- Moustafa, N., & Slay, J. (2015). UNSW-NB15: A comprehensive data set for network intrusion detection systems. *MilCIS*, IEEE.
- More, S., Idrissi, M., Mahmoud, H., & Asyhari, A. T. (2024). Enhanced intrusion detection systems performance with UNSW-NB15 data analysis. *Algorithms*, 17(2), Article 64. https://doi.org/10.3390/a17020064
- Python library documentation: scikit-learn, pandas, NumPy, matplotlib, joblib.
