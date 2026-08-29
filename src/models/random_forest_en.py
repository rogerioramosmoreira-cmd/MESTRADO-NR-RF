"""
Standard Random Forest Model — High Performance (English)
Goal: R² >= 0.80 on the test set.

Strategies:
  1. Feature engineering — 8 derived features from granulometric variables
  2. RandomForestRegressor — standard single model with randomized hyperparameter search
  3. Search — 150 iterations, 10-fold CV, scoring=neg_MSE
  4. Retrain on train+validation after tuning
  5. Full metrics: MSE, RMSE, MAE, MAPE, R²
  6. Log-transform on target (log1p/expm1)
  7. Sample weights — rare high-CBR samples receive higher weight
  8. Reasoning charts: search evolution and OOB convergence
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import dump
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import (train_test_split, RandomizedSearchCV, KFold,
                                     cross_val_predict)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline as _Pipe

warnings.filterwarnings("ignore")

# Torna o pacote `core` (src/core) importável quando este script é executado
# diretamente, e não apenas através de `main.py`.
import sys                                            # noqa: E402
from pathlib import Path                              # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import (  # noqa: E402
    console, dependencies, paths, plots, progress, runtime, scoreboard,
)
from core import metrics as core_metrics  # noqa: E402

dependencies.require("core")

# ─────────────────────────────────────────────
# GLOBAL SETTINGS
# ─────────────────────────────────────────────

DATA_PATH = paths.DATASET

TARGET_COL = "CBR "

FEATURES = [
    "25.4mm",           # IG  — coarse gravel
    "9.5mm",            # EXP — medium/fine gravel
    "4.8mm",            # D3  — sieve #4 (gravel/sand boundary)
    "2.0mm",            # D4  — sieve #10 (coarse sand)
    "0.42mm",           # D5  — sieve #40 (fine sand)
    "0.076mm",          # D6  — sieve #200 (silt/clay)
    "LL",               # LL  — liquid limit
    "IP",               # IP  — plasticity index
    "Umidade Ótima",    # CH  — optimum moisture content (Proctor)
    "Densidade máxima", # CY  — maximum dry density (Proctor)
]

FEATURES_LABELS = [
    "25.4mm (IG)",
    "9.5mm (EXP)",
    "4.8mm (D3)",
    "2.0mm (D4)",
    "0.42mm (D5)",
    "0.076mm (D6)",
    "LL",
    "IP",
    "Opt. Moisture (CH)",
    "Max. Density (CY)",
]

SEED         = 42
TEST_SIZE    = 0.20
VAL_SIZE     = 0.15
# The second value applies only with MLL_FAST=1 - see core/runtime.py.
N_ITER_SEARCH = runtime.budget(150, 15)
CV_FOLDS     = runtime.budget(10, 3)
# Goal stated as R2 (see core/metrics.py). The equivalent MSE threshold
# depends on the variance of the evaluated split, so it is only known after
# the data is divided - it gets its real value in section 6.
R2_GOAL      = core_metrics.META_R2
MSE_GOAL     = float("inf")

OUTPUT_DIR = paths.RF_EN_DIR
paths.ensure(OUTPUT_DIR)

LOG_TARGET  = True
USE_WEIGHTS = True
THRESHOLD   = 25.0
W_MINOR     = 3.0
W_MAJOR     = 1.0

# ─────────────────────────────────────────────
# COLOR PALETTE
# ─────────────────────────────────────────────
PALETTE = {
    "blue":    "#2563EB",
    "blue2":   "#60A5FA",
    "orange":  "#EA580C",
    "green":   "#16A34A",
    "red":     "#DC2626",
    "purple":  "#7C3AED",
    "bg":      "#F8FAFC",
    "grid":    "#E2E8F0",
}

plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["bg"],
    "axes.grid":         True,
    "grid.color":        PALETTE["grid"],
    "grid.linewidth":    0.8,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def metrics(y_true, y_pred, name):
    """Computes and prints MSE, RMSE, MAE, MAPE and R²."""
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-6))) * 100
    r2   = r2_score(y_true, y_pred)
    print(f"\n  [{name}]")
    print(f"    MSE  : {mse:.4f}")
    # A near-perfect R2 on real data almost always means the evaluated split
    # leaked into training. The old threshold was a fixed MSE (<= 1), which
    # fired exactly when the project goal was met; in R2 the check no longer
    # depends on the scale of the target.
    if r2 >= 0.99:
        print(f"    WARNING: R²={r2:.4f} — too high for unseen data; "
              "check scale or data leakage.")
    print(f"    RMSE : {rmse:.4f}")
    print(f"    MAE  : {mae:.4f}")
    print(f"    MAPE : {mape:.2f}%")
    print(f"    R²   : {r2:.4f}")
    return dict(name=name, mse=mse, rmse=rmse, mae=mae, mape=mape, r2=r2)


# ─────────────────────────────────────────────
# CHART FUNCTIONS
# ─────────────────────────────────────────────

def plot_predicted_vs_actual_val(y_cv, pred_cv, met):
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    lim = [min(y_cv.min(), pred_cv.min()) * 0.95, max(y_cv.max(), pred_cv.max()) * 1.05]
    ax.scatter(y_cv, pred_cv, alpha=0.6, color=PALETTE["blue"], edgecolors="white", linewidths=0.4, s=50)
    ax.plot(lim, lim, "--", color=PALETTE["orange"], lw=1.5, label="Ideal")
    ax.set_xlim(lim); ax.set_ylim(lim)

    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.legend()
    ax.text(0.05, 0.92, f"R² = {met['r2']:.4f}", transform=ax.transAxes,
            fontsize=10, color=PALETTE["blue"], fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFFFFF", edgecolor="#E2E8F0", alpha=0.93))
    plt.tight_layout(); plots.save(plt.gcf(), "predicted_vs_actual_val", "random_forest_en")


def plot_predicted_vs_actual_test(y_test, pred_test, met):
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    lim = [min(y_test.min(), pred_test.min()) * 0.95, max(y_test.max(), pred_test.max()) * 1.05]
    ax.scatter(y_test, pred_test, alpha=0.6, color=PALETTE["blue2"], edgecolors="white", linewidths=0.4, s=50)
    ax.plot(lim, lim, "--", color=PALETTE["orange"], lw=1.5, label="Ideal")
    ax.set_xlim(lim); ax.set_ylim(lim)

    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.legend()
    ax.text(0.05, 0.92, f"R² = {met['r2']:.4f}", transform=ax.transAxes,
            fontsize=10, color=PALETTE["blue2"], fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFFFFF", edgecolor="#E2E8F0", alpha=0.93))
    cor = PALETTE["blue2"] if met["r2"] >= R2_GOAL else PALETTE["orange"]
    ax.text(0.05, 0.83, f"MSE = {met['mse']:.4f}", transform=ax.transAxes,
            fontsize=10, color=cor, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFFFFF", edgecolor="#E2E8F0", alpha=0.93))
    plt.tight_layout(); plots.save(plt.gcf(), "predicted_vs_actual_test", "random_forest_en")


def plot_metrics_table(met_cv, met_test):
    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=PALETTE["bg"])
    ax.axis("off")
    table_data = [
        ["Metric",  "CV Validation",           "Test"],
        ["MSE",  f"{met_cv['mse']:.4f}",  f"{met_test['mse']:.4f}"],
        ["RMSE", f"{met_cv['rmse']:.4f}", f"{met_test['rmse']:.4f}"],
        ["MAE",  f"{met_cv['mae']:.4f}",  f"{met_test['mae']:.4f}"],
        ["MAPE", f"{met_cv['mape']:.2f}%", f"{met_test['mape']:.2f}%"],
        ["R²",   f"{met_cv['r2']:.4f}",   f"{met_test['r2']:.4f}"],
    ]
    tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                   cellLoc="center", loc="center", bbox=[0.05, 0.05, 0.9, 0.85])
    tbl.auto_set_font_size(False); tbl.set_fontsize(11)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(PALETTE["grid"])
        if r == 0:
            cell.set_facecolor(PALETTE["blue"])
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#EFF6FF" if c != 0 else "#F1F5F9")

    plt.tight_layout(); plots.save(plt.gcf(), "metrics_table", "random_forest_en")


def plot_residuals_val(y_cv, pred_cv):
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"]); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.scatter(pred_cv, y_cv - pred_cv, alpha=0.6, color=PALETTE["blue"], edgecolors="white", linewidths=0.4, s=50)
    ax.axhline(0, color=PALETTE["orange"], lw=1.5, linestyle="--")

    ax.set_xlabel("Predicted"); ax.set_ylabel("Residual (Actual − Predicted)")
    plt.tight_layout(); plots.save(plt.gcf(), "residuals_val", "random_forest_en")


def plot_residuals_test(y_test, pred_test):
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"]); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.scatter(pred_test, y_test - pred_test, alpha=0.6, color=PALETTE["blue2"], edgecolors="white", linewidths=0.4, s=50)
    ax.axhline(0, color=PALETTE["orange"], lw=1.5, linestyle="--")

    ax.set_xlabel("Predicted"); ax.set_ylabel("Residual (Actual − Predicted)")
    plt.tight_layout(); plots.save(plt.gcf(), "residuals_test", "random_forest_en")


def plot_feature_importance(rf_model, feature_names):
    """Horizontal bar chart — Gini importance. Darker = more important."""
    top_n = len(feature_names)
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"]); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    imp    = rf_model.feature_importances_
    idx    = np.argsort(imp)[-top_n:][::-1]
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))[::-1]

    ax.barh(range(top_n), imp[idx], color=colors, edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([FEATURES_LABELS[i] for i in idx], fontsize=10)
    ax.invert_yaxis()

    ax.set_xlabel("Importance (Gini)")
    ax.grid(color=PALETTE["grid"], linewidth=0.8)
    plt.tight_layout(); plots.save(plt.gcf(), "feature_importance", "random_forest_en")


def plot_search_reasoning(search_rf):
    """Search evolution: cumulative minimum MSE and marginal improvement per iteration."""
    results  = search_rf.cv_results_
    mse_iter = -results["mean_test_score"]
    n_iter   = len(mse_iter)
    iters    = np.arange(1, n_iter + 1)
    mse_min  = np.minimum.accumulate(mse_iter)
    delta    = np.diff(mse_min)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), facecolor=PALETTE["bg"],
                             gridspec_kw={"height_ratios": [2, 1]})


    ax1 = axes[0]
    ax1.set_facecolor(PALETTE["bg"]); ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    ax1.plot(iters, mse_min, color=PALETTE["blue"], lw=2, label="Cumulative min MSE CV")

    improvements = [i for i in range(1, n_iter) if mse_min[i] < mse_min[i-1] * 0.95]
    if improvements:
        ax1.scatter([i + 1 for i in improvements], [mse_min[i] for i in improvements],
                    marker="*", s=150, color=PALETTE["orange"], zorder=5, label="★ ≥5% improvement")
        for i in improvements[:6]:
            ax1.annotate(f"it.{i+1}\n{mse_min[i]:.3f}", xy=(i + 1, mse_min[i]),
                         xytext=(i + 1 + n_iter * 0.02, mse_min[i] * 1.01),
                         fontsize=9, color=PALETTE["orange"])

    best_idx = int(np.argmin(mse_iter))
    ax1.axhline(mse_min[-1], color=PALETTE["green"], lw=1.2, linestyle="--",
                label=f"Best final MSE = {mse_min[-1]:.4f}")
    ax1.scatter([best_idx + 1], [mse_iter[best_idx]], color=PALETTE["green"], s=90, zorder=5,
                label=f"Best iteration: {best_idx + 1}")
    ax1.set_ylabel("MSE CV (cumulative min)")

    ax1.legend()

    ax2 = axes[1]
    ax2.set_facecolor(PALETTE["bg"]); ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    bar_colors = [PALETTE["green"] if d < 0 else PALETTE["red"] for d in delta]
    ax2.bar(iters[1:], delta, color=bar_colors, edgecolor="white", alpha=0.85, width=0.8)
    ax2.axhline(0, color=PALETTE["grid"], lw=1.0, linestyle="--")
    top_gains = np.argsort(delta)[:3]
    for ig in top_gains:
        if delta[ig] < 0:
            ax2.annotate(f"−{abs(delta[ig]):.3f}", xy=(iters[1:][ig], delta[ig]),
                         xytext=(iters[1:][ig], delta[ig] * 1.3),
                         ha="center", fontsize=9, color=PALETTE["green"], fontweight="bold")
    ax2.set_xlabel("Search Iteration"); ax2.set_ylabel("ΔMSE (negative = improvement)")

    plt.tight_layout(); plots.save(plt.gcf(), "search_reasoning", "random_forest_en")


def plot_oob_convergence(rf_optimized, X_tv_n, Y_tv):
    """OOB error convergence by number of trees, with elbow detection."""
    n_est = rf_optimized.n_estimators
    steps = list(range(10, n_est + 1, max(1, n_est // 40)))
    if steps[-1] != n_est:
        steps.append(n_est)

    params = rf_optimized.get_params()
    params.update({"oob_score": True, "warm_start": True, "n_jobs": -1})

    rf_oob = RandomForestRegressor(**params)
    oob_errors = []
    for n in steps:
        rf_oob.set_params(n_estimators=n)
        rf_oob.fit(X_tv_n, Y_tv)
        oob_errors.append(1 - rf_oob.oob_score_)

    oob_arr  = np.array(oob_errors)
    deltas   = -np.diff(oob_arr)
    knee_idx = int(np.argmax(deltas)) + 1
    knee_n   = steps[knee_idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=PALETTE["bg"])


    ax = axes[0]
    ax.set_facecolor(PALETTE["bg"]); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.plot(steps, oob_arr, color=PALETTE["blue"], lw=2, label="OOB Error (1 − R²)")
    ax.axvline(knee_n, color=PALETTE["orange"], lw=1.5, linestyle="--", label=f"Elbow: {knee_n} trees")
    ax.scatter([knee_n], [oob_arr[knee_idx]], color=PALETTE["orange"], s=80, zorder=5)
    ax.axvspan(knee_n, steps[-1], alpha=0.06, color=PALETTE["red"], label="Diminishing returns")
    ax.set_xlabel("Number of Trees"); ax.set_ylabel("OOB Error (1 − R²)")
    ax.legend()

    ax2 = axes[1]
    ax2.set_facecolor(PALETTE["bg"]); ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    bar_colors = [PALETTE["green"] if d > 0 else PALETTE["red"] for d in deltas]
    ax2.bar(range(len(deltas)), deltas, color=bar_colors, edgecolor="white", alpha=0.85)
    ax2.axhline(0, color=PALETTE["grid"], lw=1.0)
    ax2.axvline(knee_idx - 0.5, color=PALETTE["orange"], lw=1.5, linestyle="--",
                label=f"Elbow (group {knee_idx})")
    ax2.set_xlabel("Tree Group"); ax2.set_ylabel("Marginal Gain (−ΔOOB)")

    ax2.legend()
    plt.tight_layout(); plots.save(plt.gcf(), "oob_convergence", "random_forest_en")


def plot_cv_distribution(search_rf):
    """Distribution of all 150 tested configurations."""
    mse_all = -search_rf.cv_results_["mean_test_score"]
    mse_std = search_rf.cv_results_["std_test_score"]
    best_mse = mse_all.min()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=PALETTE["bg"])


    ax = axes[0]
    ax.set_facecolor(PALETTE["bg"]); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.hist(mse_all, bins=20, color=PALETTE["blue"], alpha=0.7, edgecolor="white")
    ax.axvline(best_mse, color=PALETTE["orange"], lw=2, linestyle="--", label=f"Best = {best_mse:.4f}")
    ax.axvline(np.median(mse_all), color=PALETTE["purple"], lw=1.5, linestyle=":",
               label=f"Median = {np.median(mse_all):.4f}")
    ax.set_xlabel("MSE CV"); ax.set_ylabel("Frequency")

    ax.legend()

    ax2 = axes[1]
    ax2.set_facecolor(PALETTE["bg"]); ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    top_idx = np.argsort(mse_all)[:15]
    top_mse = mse_all[top_idx]; top_std = mse_std[top_idx]
    bar_colors = [PALETTE["orange"] if i == 0 else PALETTE["blue"] for i in range(len(top_idx))]
    bars = ax2.bar(range(len(top_idx)), top_mse, color=bar_colors, edgecolor="white", alpha=0.85)
    ax2.errorbar(range(len(top_idx)), top_mse, yerr=top_std, fmt="none", color="#374151", capsize=3, lw=1.0)
    # No goal line here: this axis shows the search MSE, which is on the log1p
    # scale, while the goal lives on the original CBR scale.
    for bar, val in zip(bars, top_mse):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax2.set_xlabel("Ranking"); ax2.set_ylabel("MSE CV")
    ax2.legend()
    plt.tight_layout(); plots.save(plt.gcf(), "cv_distribution", "random_forest_en")


# ─────────────────────────────────────────────
# 1. LOADING
# ─────────────────────────────────────────────
print("=" * 60)
print("  RANDOM FOREST — STANDARD MODEL — HIGH PERFORMANCE")
print(f"  GOAL: R² >= {R2_GOAL:.2f} on the test set")
print("=" * 60)
print("\n[1/6] Loading data...")

df = pd.read_csv(DATA_PATH)
print(f"     Dataset: {df.shape[0]} rows x {df.shape[1]} columns")

df.columns = df.columns.str.strip()
_col_map = {
    "Ll": "LL", "ll": "LL", "L.L": "LL",
    "Ip": "IP", "ip": "IP", "I.P": "IP",
    "Densidade Maxima": "Densidade máxima", "Densidade Máxima": "Densidade máxima",
    "densidade maxima": "Densidade máxima", "d_max": "Densidade máxima", "Dmax": "Densidade máxima",
    "Wot": "Umidade Ótima", "wot": "Umidade Ótima",
    "Umidade otima": "Umidade Ótima", "Umidade Otima": "Umidade Ótima",
    "CBR": "CBR ", "cbr": "CBR ", "Cbr": "CBR ",
}
df = df.rename(columns=_col_map)
df = df[FEATURES + [TARGET_COL]]

Y = df[TARGET_COL].values.ravel()
X = df[FEATURES].values
feature_names = FEATURES
print(f"     Features used: {len(FEATURES)} — {FEATURES}")

Y_orig = Y.copy()
if LOG_TARGET:
    Y = np.log1p(Y)
    print(f"     Log-transform active: [{Y_orig.min():.2f}, {Y_orig.max():.2f}]"
          f" → [{Y.min():.4f}, {Y.max():.4f}]")

# ─────────────────────────────────────────────
# 2. DATA SPLIT
# ─────────────────────────────────────────────
print("\n[2/6] Splitting data...")

X_tv, X_test, Y_tv, y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=SEED)
X_train, X_val, Y_train, Y_val = train_test_split(X_tv, Y_tv, test_size=VAL_SIZE, random_state=SEED)

scaler    = MinMaxScaler()
X_train_n = scaler.fit_transform(X_train)
X_val_n   = scaler.transform(X_val)
X_test_n  = scaler.transform(X_test)
X_tv_n    = scaler.transform(X_tv)

print(f"     Train: {X_train_n.shape[0]}  |  Validation: {X_val_n.shape[0]}  |  Test: {X_test_n.shape[0]}")
print(f"     Total features: {X_train_n.shape[1]}")

if USE_WEIGHTS and THRESHOLD is not None:
    thr = np.log1p(THRESHOLD) if LOG_TARGET else THRESHOLD
    sample_weights = np.where(Y_train > thr, W_MINOR, W_MAJOR).astype(np.float32)
    print(f"     Sample weights active — threshold={THRESHOLD} (log1p={thr:.4f})")
else:
    sample_weights = None

# ─────────────────────────────────────────────
# 3. HYPERPARAMETER SEARCH SPACE
# ─────────────────────────────────────────────
print("\n[3/6] Defining hyperparameter search space...")

param_rf = {
    "rf__n_estimators":      list(range(100, 801, 50)),
    "rf__max_depth":         list(range(3, 31, 2)) + [None],
    "rf__min_samples_split": list(range(2, 15)),
    "rf__min_samples_leaf":  list(range(1, 10)),
    "rf__max_features":      ["sqrt", "log2"] + np.linspace(0.1, 1.0, 10).round(2).tolist(),
    "rf__max_samples":       np.linspace(0.5, 1.0, 6).round(2).tolist(),
}

# ─────────────────────────────────────────────
# 4. HYPERPARAMETER SEARCH
# ─────────────────────────────────────────────
print(f"\n[4/6] Searching hyperparameters ({N_ITER_SEARCH} iter, {CV_FOLDS}-fold CV)...")

kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

pipe_rf   = _Pipe([("rf", RandomForestRegressor(random_state=SEED, n_jobs=-1))])
search_rf = RandomizedSearchCV(
    pipe_rf, param_rf, n_iter=N_ITER_SEARCH,
    scoring="neg_mean_squared_error",
    cv=kf, random_state=SEED, n_jobs=-1, verbose=0,
)
# One bar step per fit: the search runs one per candidate and per fold.
with progress.bar(total=N_ITER_SEARCH * CV_FOLDS) as _stage:
    with progress.joblib_stage(_stage):
        search_rf.fit(X_train_n, Y_train,
                      rf__sample_weight=sample_weights)

best_params = {k.replace("rf__", ""): v for k, v in search_rf.best_params_.items()}
# Score in log1p scale when LOG_TARGET is on, so NOT comparable to MSE_GOAL,
# which lives in the original CBR scale. The comparable figure is the
# out-of-fold CV metric computed in section 6.
_scale_note = " (log1p scale - not comparable to the goal)" if LOG_TARGET else ""
print(f"     Best CV MSE: {-search_rf.best_score_:.4f}{_scale_note}")
print(f"     Best params: {best_params}")

# ─────────────────────────────────────────────
# 5. FINAL TRAINING
# ─────────────────────────────────────────────
print("\n[5/6] Training final model...")

rf_final = RandomForestRegressor(**best_params, random_state=SEED, n_jobs=-1)

fit_kwargs = {}
if USE_WEIGHTS and THRESHOLD is not None:
    thr = np.log1p(THRESHOLD) if LOG_TARGET else THRESHOLD
    sw_tv = np.where(Y_tv > thr, W_MINOR, W_MAJOR).astype(np.float32)
    fit_kwargs["sample_weight"] = sw_tv

# Single sklearn call with no observable steps: activity bar.
with progress.bar() as _stage:
    with progress.joblib_stage(_stage):
        rf_final.fit(X_tv_n, Y_tv, **fit_kwargs)
print("     Model trained on train + validation.")

# ─────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────
print("\n[6/6] Evaluating model...")

# -- Generalization estimate: out-of-fold cross-validation -------------------
#
# The previous version measured "validation" with rf_final.predict(X_val_n).
# Since rf_final is trained on train+validation (X_tv), the validation set had
# already been seen: that metric was training error dressed up as
# generalization error - optimistic by construction (data leakage).
#
# Here the estimate comes from CV with the best configuration, fitted ONLY on
# the training set, so every prediction is made by a model that never saw that
# sample.
print(f"     Out-of-fold cross-validation ({CV_FOLDS} folds)...")

rf_cv     = RandomForestRegressor(**best_params, random_state=SEED, n_jobs=-1)
params_cv = {"sample_weight": sample_weights} if sample_weights is not None else None

with progress.bar(total=CV_FOLDS) as _stage:
    with progress.joblib_stage(_stage):
        pred_cv = cross_val_predict(rf_cv, X_train_n, Y_train,
                                    cv=kf, n_jobs=-1, params=params_cv)

pred_test = rf_final.predict(X_test_n)

if LOG_TARGET:
    pred_cv    = np.expm1(pred_cv)
    pred_test  = np.expm1(pred_test)
    Y_cv_met   = np.expm1(Y_train)
    y_test_met = np.expm1(y_test)
    print("     Predictions reverted to original scale (expm1).")
else:
    Y_cv_met   = Y_train
    y_test_met = y_test

# MSE threshold equivalent to R2_GOAL given the real test-set variance.
MSE_GOAL = core_metrics.meta_mse(y_test_met)
print(f"     Goal: R² >= {R2_GOAL:.2f}  (MSE <= {MSE_GOAL:.2f} on this sample)")

met_cv   = metrics(Y_cv_met,   pred_cv,   "CV Validation — Random Forest")
met_test = metrics(y_test_met, pred_test, "Final Test   — Random Forest")

print("\n" + "=" * 60)
if met_test["r2"] >= R2_GOAL:
    print(f"  GOAL REACHED! R² = {met_test['r2']:.4f} >= {R2_GOAL:.2f}"
          f"  (MSE {met_test['mse']:.2f} <= {MSE_GOAL:.2f})")
else:
    print(f"  R² = {met_test['r2']:.4f}  |  Goal: >= {R2_GOAL:.2f}"
          f"  (MSE {met_test['mse']:.2f}  |  Goal: <= {MSE_GOAL:.2f})")
print("=" * 60)

# Guarda o resultado do teste para o menu mostrar depois, sem treinar de novo.
scoreboard.record("random_forest_en",
                  "Árvore Aleatória — alta performance (EN)", met_test)

# ─────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────
print("\n--- Generating Charts ---")

plot_predicted_vs_actual_val(Y_cv_met, pred_cv, met_cv)
plot_predicted_vs_actual_test(y_test_met, pred_test, met_test)
plot_metrics_table(met_cv, met_test)
plot_residuals_val(Y_cv_met, pred_cv)
plot_residuals_test(y_test_met, pred_test)
plot_feature_importance(rf_final, feature_names)
plot_cv_distribution(search_rf)
plot_search_reasoning(search_rf)
plot_oob_convergence(rf_final, X_tv_n, Y_tv)

print("Charts generated successfully!")

# ─────────────────────────────────────────────
# SAVING
# ─────────────────────────────────────────────
dump(rf_final, os.path.join(OUTPUT_DIR, "rf_model.joblib"))
dump(scaler,   os.path.join(OUTPUT_DIR, "scaler.joblib"))

print(f"\n  Model saved  → rf_model.joblib")
print(f"  Scaler saved → scaler.joblib")
print(f"  Folder       → {OUTPUT_DIR}")
print("\n" + "=" * 60)
print("  Process completed successfully!")
print("=" * 60)

# Veredito final: o que os números acima significam para quem vai usar o
# modelo. Fica por último de propósito — é a linha que alguém lê ao voltar
# ao terminal depois de um treino longo.
core_metrics.report(met_test)
