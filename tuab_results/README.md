# TUAB abnormal-EEG detection

**Eval AUROC 0.8824 ± 0.0028, balanced accuracy 79.90 ± 0.45** (mean ± population std over seeds 0-4), 20,250
parameters, selected by validation AUROC from an 18-cell grid.

| seed | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| eval AUROC | 0.8786 | 0.8851 | 0.8805 | 0.8860 | 0.8820 |
| best step | 5,000 | 5,000 | 4,000 | 5,000 | 4,000 |

## Data

The TUH EEG Abnormal Corpus v3.0.1, which is distributed under its own data-use agreement and is not included here. It
is preprocessed with BIOT's released code: 200 Hz, 16 bipolar channels, non-overlapping
10 s windows, each window and channel scaled by its 95th percentile of |x|. The train and eval splits are the corpus's
own; validation is 15% of the training patients (whole patients, seed 0), giving 316,854 / 55,656 / 36,945 windows.
Metrics are per window: AUROC, and balanced accuracy at the threshold that makes the number of predicted positives on
validation equal to the number of validation positives.

## Model and training (`config.json`)

Eight EWS layers side by side on the same input, each with its own lift from the 16 channels to width 8, a depth-2
signature with A made of two complex Jordan chains of size 4, and a learnable blended clock. Each head's features are
averaged over time and scaled per coordinate; the eight heads give 584 features, followed by dropout 0.1 and an MLP
584 → 32 → 2. AdamW at a constant learning rate 4e-3, weight decay 1e-3, gradient clipping 1.0, batch 64, 5,000 steps,
validation every 1,000 steps; the reported checkpoint is the best one on validation.

The same model at depth 1 (72 features, 3,354 parameters) reaches an eval AUROC of 0.6665 ± 0.0469 and a balanced
accuracy of 62.47 ± 3.48.

## Selection (`grid.json`)

dropout ∈ {0.1, 0.2, 0.3} × lr ∈ {2e-3, 3e-3, 4e-3} × head width ∈ {32, 64}, every cell on seeds 0, 1 and 2; the cell
with the highest mean validation AUROC is selected and then run on seeds 3 and 4.
