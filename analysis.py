"""Boundary localisation on EEG Eye State: LightGBM with multi-scale causal context vs an MLP baseline."""
import io, json, os, sys, time
import numpy as np
import pandas as pd
import requests
import lightgbm as lgb
from scipy.io import arff
from scipy.ndimage import median_filter
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
SEEDS = [42, 43, 44]          # both models are randomised, so both run with 3 seeds
TOL = 50                      # boundary tolerance in samples
SMOOTH_CANDS = [1, 15, 31, 61]  # median-filter windows, selected by CV on training data only
META_URL = "https://www.openml.org/api/v1/json/data/1471"
t0 = time.time()


def get(url, retries=5):
    """GET with timeout=60 and exponential backoff on 429/5xx or network errors."""
    for i in range(retries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if i == retries - 1:
                raise
            print(f"  download failed ({e}); retrying in {2 ** i}s")
            time.sleep(2 ** i)


def load_data():
    """Download EEG Eye State (ARFF) from OpenML; rows are already in time order."""
    meta = get(META_URL).json()["data_set_description"]
    url = meta.get("url") or f"https://www.openml.org/data/v1/download/{meta['file_id']}"
    print("Downloading", url)
    data, _ = arff.loadarff(io.StringIO(get(url).text))
    names = list(data.dtype.names)
    # The label attribute name can differ between ARFF exports (e.g. eyeDetection / Class),
    # so locate it by name, falling back to the last attribute (the target in OpenML's ARFF).
    label = next((c for c in names if c.lower() in ("eyedetection", "class", "label", "target")), names[-1])
    print(f"Label column: {label!r}; feature columns: {len(names) - 1}")
    feats = [c for c in names if c != label]
    X = np.column_stack([data[c].astype(float) for c in feats])
    y = np.array([int(float(v.decode() if isinstance(v, bytes) else v)) for v in data[label]])
    return X, y


def window_feats(X, w=5):
    """Baseline features: current frame plus w-1 previous frames (causal; start padded by back-fill)."""
    df = pd.DataFrame(X)
    return np.hstack([df.shift(k).bfill().values for k in range(w)])


def causal_feats(X, windows=(5, 25, 125)):
    """Improved features: raw channels plus trailing rolling mean/std/deviation at several scales."""
    df = pd.DataFrame(X)
    feats = [X]
    for w in windows:
        r = df.rolling(w, min_periods=1)
        m = r.mean().values
        feats += [m, r.std().fillna(0).values, X - m]
    return np.hstack(feats)


def boundaries(lab):
    """Indices where the label changes (first sample of each new segment)."""
    return np.flatnonzero(np.diff(lab) != 0) + 1


def boundary_scores(y_true, y_pred, tol=TOL):
    """Boundary F1 with +-tol tolerance, precision, recall and mean abs offset (true -> nearest predicted)."""
    gt, pr = boundaries(y_true), boundaries(y_pred)
    if len(gt) == 0 or len(pr) == 0:
        return 0.0, 0.0, 0.0, float(len(y_true))
    d = np.abs(gt[:, None] - pr[None, :])
    prec = float(np.mean(d.min(0) <= tol))
    rec = float(np.mean(d.min(1) <= tol))
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return f1, prec, rec, float(d.min(1).mean())


def evaluate(y_true, y_pred):
    """Shared evaluation for both models: boundary metrics plus frame accuracy and macro F1."""
    f1, p, r, off = boundary_scores(y_true, y_pred)
    return dict(boundary_f1=f1, boundary_precision=p, boundary_recall=r, mean_offset=off,
                frame_accuracy=accuracy_score(y_true, y_pred),
                frame_f1=f1_score(y_true, y_pred, average="macro"))


def smooth(p, k):
    """Threshold probabilities at 0.5 and apply a median filter of width k (k=1: no smoothing)."""
    lab = (p >= 0.5).astype(int)
    return median_filter(lab, size=k) if k > 1 else lab


def fit_mlp(Xtr, ytr, Xte, seed):
    """Baseline: standard scikit-learn MLP on standardised windowed features."""
    sc = StandardScaler().fit(Xtr)
    m = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, early_stopping=True, random_state=seed)
    m.fit(sc.transform(Xtr), ytr)
    return m.predict(sc.transform(Xte))


def fit_lgb(Xtr, ytr, Xte, seed):
    """Improved: class-balanced LightGBM; returns positive-class probabilities."""
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, class_weight="balanced",
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           random_state=seed, verbose=-1)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def main():
    X, y = load_data()
    n = len(y)
    n_tr = int(0.7 * n)  # chronological split: first 70% train, last 30% test
    print(f"Loaded {n} rows, {X.shape[1]} channels; train={n_tr}, test={n - n_tr}")
    Xb, Xi = window_feats(X), causal_feats(X)

    # Select the smoothing window by time-ordered CV on the training set only
    print("Selecting smoothing window by CV on training data...")
    cv = {k: [] for k in SMOOTH_CANDS}
    for tr_i, va_i in TimeSeriesSplit(n_splits=3).split(np.arange(n_tr)):
        p = fit_lgb(Xi[tr_i], y[tr_i], Xi[va_i], SEED)
        for k in SMOOTH_CANDS:
            cv[k].append(boundary_scores(y[va_i], smooth(p, k))[0])
    best_k = max(SMOOTH_CANDS, key=lambda k: np.mean(cv[k]))
    print(f"  CV boundary F1 per window: { {k: round(float(np.mean(v)), 4) for k, v in cv.items()} }")
    print(f"  selected smoothing window = {best_k}")

    # Final test evaluation: both models, 3 seeds, same split and metric
    base, imp = [], []
    for s in SEEDS:
        print(f"Seed {s}: baseline MLP...")
        base.append(evaluate(y[n_tr:], fit_mlp(Xb[:n_tr], y[:n_tr], Xb[n_tr:], s)))
        print(f"Seed {s}: improved LightGBM...")
        imp.append(evaluate(y[n_tr:], smooth(fit_lgb(Xi[:n_tr], y[:n_tr], Xi[n_tr:], s), best_k)))

    def agg(runs, key):
        v = np.array([r[key] for r in runs], dtype=float)
        return float(v.mean()), float(v.std())

    bm, bs = agg(base, "boundary_f1")
    im, is_ = agg(imp, "boundary_f1")
    n_te_bnd = len(boundaries(y[n_tr:]))
    metrics = {
        "baseline_score": bm, "improved_score": im, "improvement": im - bm,
        "baseline_score_std": bs, "improved_score_std": is_,
        "selected_smoothing_window": best_k, "test_true_boundaries": n_te_bnd,
    }
    for key in ["boundary_precision", "boundary_recall", "mean_offset", "frame_accuracy", "frame_f1"]:
        metrics[f"baseline_{key}"] = agg(base, key)[0]
        metrics[f"improved_{key}"] = agg(imp, key)[0]
    metrics["baseline_f1"] = metrics["baseline_frame_f1"]
    metrics["improved_f1"] = metrics["improved_frame_f1"]

    # Sanity gate: refuse to report if metrics are not meaningful
    vals = [v for v in metrics.values()]
    if n_te_bnd == 0 or not np.all(np.isfinite(vals)):
        print("ERROR: no meaningful metrics produced")
        sys.exit(1)

    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(["baseline MLP", "improved LGBM"], [bm, im], yerr=[bs, is_], capsize=6, color=["grey", "tab:blue"])
    ax.set_ylabel(f"boundary F1 (±{TOL} samples)")
    ax.set_title("Test boundary F1 (mean ± std over 3 seeds)")
    fig.tight_layout()
    fig.savefig("figures/boundary_f1.png", dpi=120)

    results = {
        "question": "Can a class-balanced LightGBM model with multi-scale temporal context localise state-change boundaries more accurately than a standard MLP stand-in for the paper's network on a public time-ordered sensor recording?",
        "method": "Chronological 70/30 split of OpenML EEG Eye State; baseline MLPClassifier on 14 channels + 5-sample window; improved class-balanced LightGBM on causal multi-scale rolling features with CV-selected median smoothing; boundary F1 with +-50-sample tolerance on label changes.",
        "dataset": {
            "name": "OpenML EEG Eye State (data_id 1471)",
            "source_url": "https://www.openml.org/d/1471",
            "n_rows": n, "n_train": n_tr, "n_test": n - n_tr,
            "same_as_paper": False, "baseline_is_papers_method": False,
        },
        "metric_name": f"boundary_f1_tol{TOL}",
        "higher_is_better": True,
        "baseline": "scikit-learn MLPClassifier (hidden 64-32, early stopping) on standardised 14 channels plus a 5-sample sliding window; per-frame predictions with no smoothing. Stand-in for the paper's neural encoder, not a re-implementation.",
        "improved": "LightGBM with balanced class weights on causal multi-scale rolling mean/std/deviation features (windows 5, 25, 125 samples); probabilities median-smoothed with a window chosen by time-ordered CV on the training set.",
        "metrics": metrics,
        "notes": ("Single-recording EEG eye-state labels are used as a stand-in for segment boundaries; they are not action boundaries. "
                  "Rolling features are trailing (causal); test rows may use training rows as past context, as in streaming use. "
                  "Median smoothing is centred and therefore uses predictions from neighbouring future frames within the test recording, an offline-segmentation assumption. "
                  "Only the smoothing window is tuned (3-fold TimeSeriesSplit on training data); the baseline is untuned. "
                  "Three seeds only; dataset is 14,980 rows so no subsampling was needed. The paper's result is not stated, so no comparison to it is made."),
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Done in {time.time() - t0:.0f}s. Baseline {bm:.4f}±{bs:.4f}, improved {im:.4f}±{is_:.4f}. Wrote results.json")


if __name__ == "__main__":
    main()
