# Boundary localisation on EEG Eye State

Question: can a class-balanced LightGBM model with multi-scale causal context localise state-change boundaries more accurately than a standard MLP baseline on the OpenML EEG Eye State recording?

Setup: chronological 70/30 split (first 70% train, last 30% test), 3 seeds (42, 43, 44), boundary F1 with a ±50-sample tolerance.

Run: `pip install -r requirements.txt && python analysis.py`. Writes `results.json` and `figures/boundary_f1.png`.

Fix note: the label column is now located by name (`eyeDetection`, `Class`, `label`, `target`, case-insensitive), falling back to the last ARFF attribute, because the attribute name in the downloaded ARFF did not match `eyeDetection` exactly. The method, split, baseline and metrics are unchanged.

## Reference

This project is motivated by the following paper:

> H Wang, M Mirmehdi, D Damen, T Perrett (2023). *Centre stage: Centricity-based audio-visual temporal action detection*. arXiv preprint arXiv. https://arxiv.org/abs/2311.16446

Paper author: Prof. Dima Damen, University of Bristol (dima.damen@bristol.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
