"""
Standard Random Forest Model — High Performance (English)
Goal: MSE < 0.780 on the test set.

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
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline as _Pipe

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# GLOBAL SETTINGS
# ─────────────────────────────────────────────

DATA_PATH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed", "dados_processados_1.csv"
))

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
N_ITER_SEARCH = 150
CV_FOLDS     = 10
MSE_GOAL     = 0.780

OUTPUT_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "models", "rf_en"
))
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    if mse <= 1:
        print(f"    WARNING: MSE={mse:.4f} <= 1 — check scale or data leakage.")
    print(f"    RMSE : {rmse:.4f}")
    print(f"    MAE  : {mae:.4f}")
    print(f"    MAPE : {mape:.2f}%")
    print(f"    R²   : {r2:.4f}")
    return dict(name=name, mse=mse, rmse=rmse, mae=mae, mape=mape, r2=r2)


# ─────────────────────────────────────────────
# CHART FUNCTIONS
# ─────────────────────────────────────────────

def plot_predicted_vs_actual_val(y_val, pred_val, met):
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    lim = [min(y_val.min(), pred_val.min()) * 0.95, max(y_val.max(), pred_val.max()) * 1.05]
    ax.scatter(y_val, pred_val, alpha=0.6, color=PALETTE["blue"], edgecolors="white", linewidths=0.4, s=50)
    ax.plot(lim, lim, "--", color=PALETTE["orange"], lw=1.5, label="Ideal")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title("Predicted vs Actual — Validation", fontweight="bold")
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.legend(framealpha=0)
    ax.text(0.05, 0.92, f"R² = {met['r2']:.4f}", transform=ax.transAxes,
            fontsize=10, color=PALETTE["blue"], fontweight="bold")
    plt.tight_layout(); plt.show()


def plot_predicted_vs_actual_test(y_test, pred_test, met):
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    lim = [min(y_test.min(), pred_test.min()) * 0.95, max(y_test.max(), pred_test.max()) * 1.05]
    ax.scatter(y_test, pred_test, alpha=0.6, color=PALETTE["blue2"], edgecolors="white", linewidths=0.4, s=50)
    ax.plot(lim, lim, "--", color=PALETTE["orange"], lw=1.5, label="Ideal")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title("Predicted vs Actual — Test", fontweight="bold")
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.legend(framealpha=0)
    ax.text(0.05, 0.92, f"R² = {met['r2']:.4f}", transform=ax.transAxes,
            fontsize=10, color=PALETTE["blue2"], fontweight="bold")
    cor = PALETTE["blue2"] if met["mse"] < MSE_GOAL else PALETTE["orange"]
    ax.text(0.05, 0.83, f"MSE = {met['mse']:.4f}", transform=ax.transAxes,
            fontsize=9, color=cor, fontweight="bold")
    plt.tight_layout(); plt.show()


def plot_metrics_table(met_val, met_test):
    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=PALETTE["bg"])
    ax.axis("off")
    table_data = [
        ["Metric",  "Validation",              "Test"],
        ["MSE",  f"{met_val['mse']:.4f}",  f"{met_test['mse']:.4f}"],
        ["RMSE", f"{met_val['rmse']:.4f}", f"{met_test['rmse']:.4f}"],
        ["MAE",  f"{met_val['mae']:.4f}",  f"{met_test['mae']:.4f}"],
        ["MAPE", f"{met_val['mape']:.2f}%", f"{met_test['mape']:.2f}%"],
        ["R²",   f"{met_val['r2']:.4f}",   f"{met_test['r2']:.4f}"],
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
    ax.set_title("Metrics Comparison — Random Forest", fontweight="bold", pad=14)
    plt.tight_layout(); plt.show()


def plot_residuals_val(y_val, pred_val):
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"]); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.scatter(pred_val, y_val - pred_val, alpha=0.6, color=PALETTE["blue"], edgecolors="white", linewidths=0.4, s=50)
    ax.axhline(0, color=PALETTE["orange"], lw=1.5, linestyle="--")
    ax.set_title("Residuals — Validation", fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Residual (Actual − Predicted)")
    plt.tight_layout(); plt.show()


def plot_residuals_test(y_test, pred_test):
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"]); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.scatter(pred_test, y_test - pred_test, alpha=0.6, color=PALETTE["blue2"], edgecolors="white", linewidths=0.4, s=50)
    ax.axhline(0, color=PALETTE["orange"], lw=1.5, linestyle="--")
    ax.set_title("Residuals — Test", fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Residual (Actual − Predicted)")
    plt.tight_layout(); plt.show()


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
    ax.set_yticklabels([FEATURES_LABELS[i] for i in idx], fontsize=9)
    ax.invert_yaxis()
    ax.set_title(f"Feature Importance — {top_n} features (Random Forest Gini)", fontweight="bold")
    ax.set_xlabel("Importance (Gini)")
    ax.grid(color=PALETTE["grid"], linewidth=0.8)
    plt.tight_layout(); plt.show()


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
    fig.suptitle("Random Forest Search — Hyperparameter Tuning Evolution",
                 fontsize=14, fontweight="bold")

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
                         fontsize=7, color=PALETTE["orange"])

    best_idx = int(np.argmin(mse_iter))
    ax1.axhline(mse_min[-1], color=PALETTE["green"], lw=1.2, linestyle="--",
                label=f"Best final MSE = {mse_min[-1]:.4f}")
    ax1.scatter([best_idx + 1], [mse_iter[best_idx]], color=PALETTE["green"], s=90, zorder=5,
                label=f"Best iteration: {best_idx + 1}")
    ax1.set_ylabel("MSE CV (cumulative min)")
    ax1.set_title("Search Learning Curve — Where the Reasoning Changed", fontweight="bold")
    ax1.legend(framealpha=0, fontsize=9)

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
                         ha="center", fontsize=7, color=PALETTE["green"], fontweight="bold")
    ax2.set_xlabel("Search Iteration"); ax2.set_ylabel("ΔMSE (negative = improvement)")
    ax2.set_title("Marginal Change per Iteration (green = improvement, red = stagnation)", fontweight="bold")
    plt.tight_layout(); plt.show()


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
    fig.suptitle("Random Forest Convergence — OOB Error by Number of Trees",
                 fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.set_facecolor(PALETTE["bg"]); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.plot(steps, oob_arr, color=PALETTE["blue"], lw=2, label="OOB Error (1 − R²)")
    ax.axvline(knee_n, color=PALETTE["orange"], lw=1.5, linestyle="--", label=f"Elbow: {knee_n} trees")
    ax.scatter([knee_n], [oob_arr[knee_idx]], color=PALETTE["orange"], s=80, zorder=5)
    ax.axvspan(knee_n, steps[-1], alpha=0.06, color=PALETTE["red"], label="Diminishing returns")
    ax.set_xlabel("Number of Trees"); ax.set_ylabel("OOB Error (1 − R²)")
    ax.set_title("OOB Convergence", fontweight="bold"); ax.legend(framealpha=0)

    ax2 = axes[1]
    ax2.set_facecolor(PALETTE["bg"]); ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    bar_colors = [PALETTE["green"] if d > 0 else PALETTE["red"] for d in deltas]
    ax2.bar(range(len(deltas)), deltas, color=bar_colors, edgecolor="white", alpha=0.85)
    ax2.axhline(0, color=PALETTE["grid"], lw=1.0)
    ax2.axvline(knee_idx - 0.5, color=PALETTE["orange"], lw=1.5, linestyle="--",
                label=f"Elbow (group {knee_idx})")
    ax2.set_xlabel("Tree Group"); ax2.set_ylabel("Marginal Gain (−ΔOOB)")
    ax2.set_title("Marginal Return — Where to Stop Adding Trees", fontweight="bold")
    ax2.legend(framealpha=0)
    plt.tight_layout(); plt.show()


def plot_cv_distribution(search_rf):
    """Distribution of all 150 tested configurations."""
    mse_all = -search_rf.cv_results_["mean_test_score"]
    mse_std = search_rf.cv_results_["std_test_score"]
    best_mse = mse_all.min()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=PALETTE["bg"])
    fig.suptitle("Performance Distribution — All Tested Configurations",
                 fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.set_facecolor(PALETTE["bg"]); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.hist(mse_all, bins=20, color=PALETTE["blue"], alpha=0.7, edgecolor="white")
    ax.axvline(best_mse, color=PALETTE["orange"], lw=2, linestyle="--", label=f"Best = {best_mse:.4f}")
    ax.axvline(np.median(mse_all), color=PALETTE["purple"], lw=1.5, linestyle=":",
               label=f"Median = {np.median(mse_all):.4f}")
    ax.set_xlabel("MSE CV"); ax.set_ylabel("Frequency")
    ax.set_title("MSE CV Distribution across Configurations", fontweight="bold")
    ax.legend(framealpha=0)

    ax2 = axes[1]
    ax2.set_facecolor(PALETTE["bg"]); ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    top_idx = np.argsort(mse_all)[:15]
    top_mse = mse_all[top_idx]; top_std = mse_std[top_idx]
    bar_colors = [PALETTE["orange"] if i == 0 else PALETTE["blue"] for i in range(len(top_idx))]
    bars = ax2.bar(range(len(top_idx)), top_mse, color=bar_colors, edgecolor="white", alpha=0.85)
    ax2.errorbar(range(len(top_idx)), top_mse, yerr=top_std, fmt="none", color="#374151", capsize=3, lw=1.0)
    ax2.axhline(MSE_GOAL, color="red", lw=1.2, linestyle="--", label=f"MSE Goal = {MSE_GOAL}")
    for bar, val in zip(bars, top_mse):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax2.set_title("Top 15 Configurations (orange = best)\nError bars = CV std", fontweight="bold")
    ax2.set_xlabel("Ranking"); ax2.set_ylabel("MSE CV")
    ax2.legend(framealpha=0)
    plt.tight_layout(); plt.show()


# ─────────────────────────────────────────────
# 1. LOADING
# ─────────────────────────────────────────────
print("=" * 60)
print("  RANDOM FOREST — STANDARD MODEL — HIGH PERFORMANCE")
print(f"  GOAL: MSE < {MSE_GOAL}")
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
search_rf.fit(X_train_n, Y_train, rf__sample_weight=sample_weights)

best_params = {k.replace("rf__", ""): v for k, v in search_rf.best_params_.items()}
print(f"     Best CV MSE: {-search_rf.best_score_:.4f}")
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

rf_final.fit(X_tv_n, Y_tv, **fit_kwargs)
print("     Model trained on train + validation.")

# ─────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────
print("\n[6/6] Evaluating model...")

pred_val  = rf_final.predict(X_val_n)
pred_test = rf_final.predict(X_test_n)

if LOG_TARGET:
    pred_val   = np.expm1(pred_val)
    pred_test  = np.expm1(pred_test)
    Y_val_met  = np.expm1(Y_val)
    y_test_met = np.expm1(y_test)
    print("     Predictions reverted to original scale (expm1).")
else:
    Y_val_met  = Y_val
    y_test_met = y_test

met_val  = metrics(Y_val_met,  pred_val,  "Validation  — Random Forest")
met_test = metrics(y_test_met, pred_test, "Final Test   — Random Forest")

print("\n" + "=" * 60)
if met_test["mse"] < MSE_GOAL:
    print(f"  GOAL REACHED! MSE = {met_test['mse']:.4f} < {MSE_GOAL}")
else:
    print(f"  MSE = {met_test['mse']:.4f}  |  Goal: < {MSE_GOAL}")
print("=" * 60)

# ─────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────
print("\n--- Generating Charts ---")

plot_predicted_vs_actual_val(Y_val_met, pred_val, met_val)
plot_predicted_vs_actual_test(y_test_met, pred_test, met_test)
plot_metrics_table(met_val, met_test)
plot_residuals_val(Y_val_met, pred_val)
plot_residuals_test(y_test_met, pred_test)
plot_feature_importance(rf_final, feature_names)
plot_cv_distribution(search_rf)
plot_search_reasoning(search_rf)
plot_oob_convergence(rf_final, X_tv_n, Y_tv)

print("Charts generated successfully!")

# ─────────────────────────────────────────────
# SAVING
# ─────────────────────────────────────────────
dump(rf_final, os.path.join(OUTPUT_DIR, "rf_model_final.joblib"))
dump(scaler,   os.path.join(OUTPUT_DIR, "scaler.joblib"))

print(f"\n  Model saved  → rf_model_final.joblib")
print(f"  Scaler saved → scaler.joblib")
print(f"  Folder       → {OUTPUT_DIR}")
print("\n" + "=" * 60)
print("  Process completed successfully!")
print("=" * 60)
