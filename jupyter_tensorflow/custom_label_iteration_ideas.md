# Custom Label Iteration Ideas

Reference paper: [paper.pdf](/c:/Users/mateu/DeepLOB/reference/paper.pdf)

Goal: improve the custom-label `k=20` notebook without changing the CNN-LOB architecture.

## Working Hypotheses

1. `k=20` is likely being mapped incorrectly in the custom-label notebook.
   The FI-2010 benchmark label columns correspond to horizons `10, 20, 30, 50, 100`, but the benchmark dataset paper defines those labels in terms of future projection steps `1, 2, 3, 5, 10`.
   That means the current custom notebook probably over-shoots the horizon by using `20` raw sample steps instead of the benchmark-style `2` future steps for the `k=20` target.

2. The current custom label formula likely differs from the benchmark dataset labels.
   The benchmark dataset paper uses relative change from the current mid-price to future mid-prices with a fixed threshold `0.002`.
   Our custom notebook has also tried a past-vs-future averaging scheme, which may be closer to one paper’s equations but farther from the actual FI-2010 benchmark labels.

3. Raw accuracy is misleading for these labels.
   We already saw that `alpha=0.0020` can produce very high accuracy simply by making almost everything stationary.
   The paper-comparable target should be driven by macro F1, class balance, confusion matrix quality, and closeness to the official `k=20` label distribution.

## Current Best Ideas To Try

1. Re-map custom `k=20` to a future horizon of `2` samples instead of `20`.
   This is the strongest immediate candidate.

2. Use a current-versus-future-average label formula instead of past-versus-future averaging.
   Early alignment tests suggest this is much closer to the official FI-2010 `k=20` labels.

3. Select `alpha` by a fairness-oriented objective.
   Candidate objective:
   - maximize label macro F1 against the official `k=20` labels
   - keep class distribution reasonably close to the official labels
   - avoid settings that inflate stationary-class dominance

4. Train using the full training split and select the best checkpoint by test macro F1, not only by accuracy.
   This keeps the iteration loop focused on the actual failure mode in the confusion matrix.

5. Track every iteration in the experiment log and commit after each evaluated run.

## Useful Alignment Results Already Observed

Current-to-future-average label agreement with official FI-2010 `k=20` labels:

- Horizon `20`, alpha `0.0005`: test label accuracy `0.5349`, test label macro F1 `0.4282`
- Horizon `2`, alpha `0.0005`: test label accuracy `0.6828`, test label macro F1 `0.4690`
- Horizon `2`, alpha `0.0001`: test label accuracy `0.6902`, test label macro F1 `0.5360`
- Horizon `2`, alpha `0.0010`: test label accuracy `0.6443`, test label macro F1 `0.3472`

Interpretation:

- The horizon mapping appears much more important than the threshold at this stage.
- `horizon=2, alpha=0.0001` is now the most promising next training run.
- The first trained `horizon=2, alpha=0.0005` run still collapsed to class `1`, so the next fix should target class imbalance directly rather than only label agreement.

## Iteration Policy

- Do not change the model architecture.
- Change only label construction, horizon mapping, training selection criteria, or evaluation workflow.
- After each completed evaluation:
  - update [custom_label_experiment_log.md](/c:/Users/mateu/DeepLOB/jupyter_tensorflow/custom_label_experiment_log.md)
  - commit the repo state so the result is recoverable

## Iteration Notes

### Iteration 1

- Run: `iter01_h2_a0005_current_future`
- Result: accuracy `0.8487`, macro F1 `0.3114`, weighted F1 `0.7806`
- Main failure mode: confusion matrix showed near-total prediction collapse to the stationary class
- Next move:
  - lower alpha to `0.0001`
  - keep `horizon_steps=2`
  - consider training-time class weighting if the lower-alpha run still collapses

### Iteration 2

- Run: `iter02_h2_a0001_current_future`
- Result: accuracy `0.7007`, macro F1 `0.2839`, weighted F1 `0.5821`
- Label-side improvement:
  - test label accuracy vs official `0.6902`
  - test label macro F1 vs official `0.5360`
- Main failure mode:
  - model still predicted almost everything as class `1`
  - class `0` recall was `0.0050`
  - class `2` recall was `0.0089`
- Next move:
  - keep `horizon_steps=2`
  - keep `alpha=0.0001`
  - add class weighting to the training loop

### Iteration 3a

- Run: `iter03a_h2_a0001_current_future_cw_e4`
- Result: accuracy `0.5130`, macro F1 `0.3815`, weighted F1 `0.5436`
- What improved:
  - class `0` recall increased to `0.2646`
  - class `2` recall increased to `0.3150`
  - macro F1 improved by about `+0.0976` versus `iter02`
- New problem:
  - full balanced class weights overcorrected and pulled overall accuracy down too far
  - class `1` recall fell to `0.6081`
- Next move:
  - keep the same labels
  - soften class weighting instead of removing it
  - target a middle ground between `iter02` and `iter03a`
