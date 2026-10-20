# Multi-scale LightGBM vs MLP for state-change boundary localisation on EEG Eye State

Motivation: "Centre stage: Centricity-based audio-visual temporal action detection" (2023) proposes a one-stage anchor-free detector that fuses audio and visual encoders with multi-scale cross-attention and a centricity head to localise where actions start and end in untrimmed sequences. Its network is too heavy to re-implement here, so this experiment asks a simpler question about the boundary-localisation part of the problem.

Question: Can a class-balanced LightGBM model with multi-scale temporal context localise state-change boundaries more accurately than a standard MLP stand-in for the paper's network on a public time-ordered sensor recording?

Dataset: OpenML EEG Eye State (data_id 1471), 14,980 time-ordered EEG samples with 14 channels and a binary eye-state label. Boundaries are label changes. The data is chronologically split (first 70% train, last 30% test). This is not the paper's own dataset.

Baseline: scikit-learn MLPClassifier on 14 channels plus a short sliding window, with per-frame predictions. It is a stand-in for the paper's neural encoder, not a re-implementation.

Improved: class-balanced LightGBM on causal multi-scale rolling mean, std and deviation features, with median smoothing of predicted labels. The smoothing window is chosen by time-ordered cross-validation on the training set.

## How to run

```
pip install -r requirements.txt
python analysis.py
```

Results: written to `results.json` (charts saved to `figures/`).

## Reference

This project is motivated by the following paper:

> H Wang, M Mirmehdi, D Damen, T Perrett (2023). *Centre stage: Centricity-based audio-visual temporal action detection*. arXiv preprint arXiv. https://arxiv.org/abs/2311.16446

Paper author: Prof. Dima Damen, University of Bristol (dima.damen@bristol.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
