# Boundary localisation on EEG Eye State

Question: can a class-balanced LightGBM model with multi-scale causal context localise state-change boundaries more accurately than a standard MLP baseline on the OpenML EEG Eye State recording?

Setup: chronological 70/30 split (first 70% train, last 30% test), 3 seeds (42, 43, 44), boundary F1 with a ±50-sample tolerance.

Run: `pip install -r requirements.txt && python analysis.py`. Writes `results.json` and `figures/boundary_f1.png`.

Fix note: the label column is now located by name (`eyeDetection`, `Class`, `label`, `target`, case-insensitive), falling back to the last ARFF attribute, because the attribute name in the downloaded ARFF did not match `eyeDetection` exactly. The method, split, baseline and metrics are unchanged.

## Results (latest run)

On the public OpenML EEG Eye State recording (9 test boundaries), a class-balanced LightGBM reached boundary F1 0.365 versus 0.238 for a standard MLP classifier on windowed features, but its frame-level F1 fell from 0.448 to 0.201.

| Metric | Value |
|---|---|
| baseline_score | 0.2378 |
| improved_score | 0.3654 |
| improvement | 0.1276 |
| baseline_score_std | 0.04666 |
| improved_score_std | 0.009147 |
| selected_smoothing_window | 31 |
| test_true_boundaries | 9 |
| baseline_boundary_precision | 0.1379 |
| improved_boundary_precision | 0.2389 |
| baseline_boundary_recall | 0.8889 |
| improved_boundary_recall | 0.7778 |
| baseline_mean_offset | 21.81 |
| improved_mean_offset | 45.59 |
| baseline_frame_accuracy | 0.4538 |
| improved_frame_accuracy | 0.4068 |
| baseline_frame_f1 | 0.4479 |
| improved_frame_f1 | 0.2013 |
| baseline_f1 | 0.4479 |
| improved_f1 | 0.2013 |

Run on 2026-10-20. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Single-recording EEG eye-state labels are used as a stand-in for segment boundaries; they are not action boundaries. Rolling features are trailing (causal); test rows may use training rows as past context, as in streaming use. Median smoothing is centred and therefore uses predictions from neighbouring future frames within the test recording, an offline-segmentation assumption. Only the smoothing window is tuned (3-fold TimeSeriesSplit on training data); the baseline is untuned. Three seeds only; dataset is 14,980 rows so no subsampling was needed. The paper's result is not stated, so no comparison to it is made.

## Reference

This project is motivated by the following paper:

> H Wang, M Mirmehdi, D Damen, T Perrett (2023). *Centre stage: Centricity-based audio-visual temporal action detection*. arXiv preprint arXiv. https://arxiv.org/abs/2311.16446

Paper author: Prof. Dima Damen, University of Bristol (dima.damen@bristol.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
