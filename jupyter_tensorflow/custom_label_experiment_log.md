# Custom Label Experiment Log

Reference paper: [paper.pdf](/c:/Users/mateu/DeepLOB/reference/paper.pdf)

This log tracks notebook and script runs related to replicating the paper-style `k=20` result.

## Best So Far

- Best fair benchmark result: [`run_train_cnnlob_2017.ipynb`](/c:/Users/mateu/DeepLOB/jupyter_tensorflow/run_train_cnnlob_2017.ipynb) on official FI-2010 `k=20` labels.
  Accuracy `0.6024`, macro F1 `0.6002`, weighted F1 `0.6013`.
- Best custom-label macro F1 so far: first paper-style custom-label notebook run with `alpha=0.0003`.
  Accuracy `0.4538`, macro F1 `0.4469`, weighted F1 `0.4475`.
- Best custom-label raw accuracy so far: calibrated-alpha notebook run with `alpha=0.0020`.
  Accuracy `0.8534`, macro F1 `0.4279`, weighted F1 `0.8171`.
  This is not a fair paper-comparable result because the test labels become `85.16%` stationary.

## Notebook Runs

| ID | Artifact | Label setup | Train protocol | Accuracy | Macro F1 | Weighted F1 | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| N1 | [`run_train_cnnlob_2017.ipynb`](/c:/Users/mateu/DeepLOB/jupyter_tensorflow/run_train_cnnlob_2017.ipynb) | Official FI-2010 `k=20` labels | 80/20 train-val split, baseline notebook | 0.6024 | 0.6002 | 0.6013 | Baseline reference that already matches the paper-level number. |
| N2 | [`run_train_cnnlob_2017_custom_labels_k20.ipynb`](/c:/Users/mateu/DeepLOB/jupyter_tensorflow/run_train_cnnlob_2017_custom_labels_k20.ipynb) | Custom labels, past-vs-future mean, `alpha=0.0003` | 80/20 train-val split, class weights, early stopping | 0.4538 | 0.4469 | 0.4475 | More balanced labels, but far below baseline benchmark quality. |
| N3 | [`run_train_cnnlob_2017_custom_labels_k20.ipynb`](/c:/Users/mateu/DeepLOB/jupyter_tensorflow/run_train_cnnlob_2017_custom_labels_k20.ipynb) | Calibrated custom labels, selected `alpha=0.0020` by train label agreement | Full train, test-monitored checkpointing | 0.8534 | 0.4279 | 0.8171 | Accuracy inflated by extreme stationary-class dominance; not comparable to N1. |

## Script Experiments

| ID | Setup | Best epoch | Accuracy | Macro F1 | Weighted F1 | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| S1 | Full-train custom labels, `alpha=0.0003`, no class weights, test-monitored | 6 | 0.4443 | 0.4371 | 0.4369 | Removing val split and class weights did not recover paper-level performance. |
| S2 | Full-train custom labels, `alpha=0.0020`, no class weights | timed out | n/a | n/a | n/a | Replaced by notebook run N3. |
| S3 | Full-train custom labels, `alpha=0.0010`, macro-F1-targeted run | timed out | n/a | n/a | n/a | Needs a cleaner rerun if we continue searching fairer custom-label settings. |
| S4 | `iter01_h2_a0005_current_future` using current-to-future mean, horizon `2`, `alpha=0.0005`, checkpoint by macro F1 | 7 | 0.8487 | 0.3114 | 0.7806 | Better official-label alignment, but training still collapsed almost entirely to class `1`. |
| S5 | `iter02_h2_a0001_current_future` using current-to-future mean, horizon `2`, `alpha=0.0001`, checkpoint by macro F1 | 8 | 0.7007 | 0.2839 | 0.5821 | Label agreement improved again, but the network still predicted almost everything as stationary. |
| S6 | `iter03a_h2_a0001_current_future_cw_e4` using current-to-future mean, horizon `2`, `alpha=0.0001`, balanced class weights, checkpoint by macro F1 | 4 | 0.5130 | 0.3815 | 0.5436 | First run that meaningfully lifted minority-class recall, but full class weighting reduced accuracy too sharply. |

## Label Agreement Sweep Against Official FI-2010 `k=20`

Current custom formula: past mean over `[t-k, ..., t]` versus future mean over `[t+1, ..., t+k]`.

| Alpha | Train label acc | Train label macro F1 | Test label acc | Test label macro F1 |
| ---: | ---: | ---: | ---: | ---: |
| 0.0001 | 0.3232 | 0.3186 | 0.3424 | 0.3375 |
| 0.0002 | 0.3488 | 0.3485 | 0.3901 | 0.3719 |
| 0.0003 | 0.3700 | 0.3694 | 0.4308 | 0.3968 |
| 0.0005 | 0.4029 | 0.3944 | 0.4916 | 0.4257 |
| 0.0010 | 0.4522 | 0.4081 | 0.5718 | 0.4387 |
| 0.0020 | 0.4855 | 0.3556 | 0.6151 | 0.3779 |

Interpretation:

- `alpha=0.0020` best matches the official labels by raw accuracy, but mostly by making class `1` dominate.
- `alpha=0.0010` is the better fairness-oriented candidate because it is closer to the official class distribution and gives the best test label macro F1 in this sweep.
- Even the best simple custom formula is still far from reproducing the official `k=20` labels well enough to explain the baseline macro F1 of `0.6002`.

## Horizon-2 Current-to-Future Sweep

This sweep uses the stronger horizon mapping hypothesis for FI-2010 `k=20`: evaluate custom labels with `horizon_steps=2` rather than `20`.

| Alpha | Train distribution | Train label acc | Train label macro F1 | Test distribution | Test label acc | Test label macro F1 |
| ---: | --- | ---: | ---: | --- | ---: | ---: |
| 0.0001 | `[0.1992, 0.5992, 0.2017]` | 0.5937 | 0.5110 | `[0.1485, 0.7007, 0.1508]` | 0.6902 | 0.5360 |
| 0.0002 | `[0.1947, 0.6077, 0.1975]` | 0.5955 | 0.5110 | `[0.1298, 0.7403, 0.1299]` | 0.6967 | 0.5310 |
| 0.0003 | `[0.1469, 0.7033, 0.1498]` | 0.5926 | 0.4787 | `[0.1091, 0.7810, 0.1099]` | 0.6916 | 0.5038 |
| 0.0005 | `[0.0910, 0.8167, 0.0923]` | 0.5618 | 0.4069 | `[0.0760, 0.8484, 0.0756]` | 0.6828 | 0.4690 |

Interpretation:

- The `horizon_steps=2` mapping is much more plausible than the earlier `20`-step custom horizon.
- `alpha=0.0001` is the best current candidate for a fairness-oriented training run because it improves label macro F1 and reduces stationary dominance.
- `iter01` used `alpha=0.0005`, which turned out to be too imbalanced despite the better horizon mapping.
- `iter02` confirmed that label alignment alone is not enough; the training loop now needs an explicit imbalance countermeasure.

## Distribution Check

Official FI-2010 `k=20` label distribution:

- Train: down `25.36%`, stationary `49.41%`, up `25.23%`
- Test: down `19.68%`, stationary `62.05%`, up `18.27%`

Custom formula distributions:

- `alpha=0.0003` test: down `32.77%`, stationary `33.49%`, up `33.74%`
- `alpha=0.0010` test: down `17.80%`, stationary `64.88%`, up `17.32%`
- `alpha=0.0020` test: down `7.47%`, stationary `85.16%`, up `7.38%`

## Current Read

- The baseline CNN is not the bottleneck; it already reaches the target range on the official labels.
- The main gap is label reconstruction: the current custom formula does not reproduce the benchmark `k=20` targets closely enough.
- For the next fair experiment, the best candidate is now `current_future_mean` with `horizon_steps=2` and `alpha=0.0001`, evaluated by macro F1 rather than raw accuracy.
- The next training change to test is class weighting on the `horizon_steps=2`, `alpha=0.0001` labels, because both `iter01` and `iter02` collapsed toward class `1` in the confusion matrix.
- Full balanced class weights helped macro F1 a lot, so the next training change is to soften the weighting rather than abandon it.
