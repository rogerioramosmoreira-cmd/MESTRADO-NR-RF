"""
Random Forest Model — High Performance
Goal: MSE < 0.780 on the test set.

Strategies applied:
  1. Feature Engineering — ratios and interactions between granulometric variables
  2. Ensemble — VotingRegressor (RF + GradientBoosting + ExtraTrees)
  3. Extensive search — 150 iterations, 10-fold CV, scoring=neg_MSE
  4. Retraining on train+validation set after tuning
  5. Full metrics: MSE, RMSE, MAE, R²
  6. Alert if MSE <= 1
  7. Individual and standardized charts
  8. Log-transform on target (log1p/expm1) — compresses right skew of CBR distribution
  9. Sample weights — rare high-CBR samples receive higher weight during training
"""

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
import os                                        # Path manipulation and folder creation
import warnings                                  # Suppression of non-critical Python warnings
import numpy as np                               # Numerical operations and multidimensional arrays
import pandas as pd                              # CSV reading and tabular manipulation
import matplotlib.pyplot as plt                  # Chart generation

from joblib import dump                          # Serialization of model and scaler to .joblib files
from sklearn.preprocessing import MinMaxScaler  # Normalizes features to the [0, 1] interval
from sklearn.model_selection import (
    train_test_split,                            # Stratified data split into sets
    RandomizedSearchCV,                          # Random hyperparameter search with internal CV
    KFold,                                       # K-fold cross-validation strategy
)
from sklearn.ensemble import (
    RandomForestRegressor,                       # Random forest: average of N decision trees
    GradientBoostingRegressor,                   # Boosting: each tree corrects errors of the previous
    ExtraTreesRegressor,                         # RF variant with fully random splits
    VotingRegressor,                             # Combines predictions of multiple models by average
)
from sklearn.metrics import (
    mean_squared_error,                          # MSE: penalizes large errors quadratically
    mean_absolute_error,                         # MAE: mean absolute errors, robust to outliers
    r2_score,                                    # R²: proportion of variance explained by the model
)

warnings.filterwarnings("ignore")               # Remove unnecessary warnings from terminal

# ─────────────────────────────────────────────
# GLOBAL SETTINGS
# ─────────────────────────────────────────────

# Path relative to script: goes up one level (code/ -> ML/) and enters data/
CAMINHO_DADOS = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "dados_processados_1.csv"
))

COLUNA_ALVO   = "CBR "          # Exact name of the target column in CSV (note trailing space)
SEED          = 42               # Global seed: ensures reproducible splits, searches and models
TEST_SIZE     = 0.20             # 20% of dataset reserved for final test (never seen in training)
VAL_SIZE      = 0.15             # 15% of the remainder reserved for validation (separate from test)
N_ITER_BUSCA  = 150              # Combinations tested per model in search
CV_FOLDS      = 10               # 10-fold CV: more stable estimate than 5-fold on smaller datasets
META_MSE      = 0.780            # Performance goal: MSE below this value on the final test

# Folder where ensemble and scaler will be saved
OUTPUT_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Modelo_salvo_RF"
))
os.makedirs(OUTPUT_DIR, exist_ok=True)          # exist_ok=True avoids error if folder already exists

# ── Log-transform on target ──────────────────────────────────────────────────
# PROBLEM: CBR distribution is right-skewed — most samples cluster at low/medium
# CBR and few cases have CBR > 50%. MSE penalizes outlier errors quadratically,
# pulling the metric up even when the model is otherwise accurate.
#
# SOLUTION: train on log1p(CBR), predict on expm1(pred).
# - log1p compresses the right tail, making the distribution more symmetric
# - The model treats relative errors equally across the full CBR range
# - Metrics are always computed on the original scale (expm1 applied before evaluation)
LOG_ALVO = True                  # True = train on log1p(CBR), evaluate on expm1(pred)

# ── Sample weights ────────────────────────────────────────────────────────────
# PROBLEM: high-CBR samples are under-represented — the model systematically
# underestimates them because it sees too few examples in that region.
#
# SOLUTION: samples with CBR > THRESHOLD receive weight W_MINOR during training.
# THRESHOLD = 25.0: approx. upper quartile of typical Brazilian lateritic soils.
# Adjust based on the actual distribution of your dataset (check a histogram).
USE_WEIGHTS = True               # True = high-CBR samples receive extra attention
THRESHOLD   = 25.0               # Samples with CBR > 25% are considered rare/important
W_MINOR     = 3.0                # Weight of samples above threshold (rare → more important)
W_MAJOR     = 1.0                # Weight of samples below threshold (common → normal influence)

# ─────────────────────────────────────────────
# GLOBAL COLOR PALETTE
# ─────────────────────────────────────────────
# roxo (purple) → laranja (orange)
# verde (green) → azul (blue) — uses a second blue shade to distinguish from primary

PALETTE = {
    "azul":    "#2563EB",   # Validation set / primary blue
    "azul2":   "#60A5FA",   # Test set (was verde/green) → light blue
    "laranja": "#EA580C",   # Reference line and highlights (also replaces roxo/purple)
    "fundo":   "#F8FAFC",   # Chart background color
    "grade":   "#E2E8F0",   # Grid lines
}

# Applies global style to all charts created from this point
plt.rcParams.update({
    "figure.facecolor":  PALETTE["fundo"],   # Full figure background
    "axes.facecolor":    PALETTE["fundo"],   # Each individual axis background
    "axes.grid":         True,               # Enable grid on all axes
    "grid.color":        PALETTE["grade"],   # Grid line color
    "grid.linewidth":    0.8,                # Grid line thickness
    "font.family":       "DejaVu Sans",      # Default font
    "axes.spines.top":   False,              # Remove top border (cleaner look)
    "axes.spines.right": False,              # Remove right border
})

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def engenharia_features(df: pd.DataFrame, coluna_alvo: str) -> pd.DataFrame:
    """
    Creates new derived features from granulometric variables and physical indices.

    Logic of each feature:
      - ratio_X_Y: ratio between adjacent sieves — captures the shape of the
        gradation curve relatively, independent of absolute values
      - atividade: LL - IP — soils with high difference are more plastic and less resistant
      - compacidade: Density / Moisture — proxy of soil compaction energy
      - finos_sq: 0.076mm² — square of fine fraction, amplifies differences in clayey soils

    Args:
        df:           DataFrame with original columns.
        coluna_alvo:  Name of the target column (preserved without modification).

    Returns:
        DataFrame with original columns + 8 new derived features.
    """
    df  = df.copy()                                     # Avoid modifying the original DataFrame
    eps = 1e-6                                          # Epsilon: avoids division by zero in ratios

    # Ratios between consecutive sieves — describe gradient of gradation curve
    df["ratio_9_25"]    = df["9.5mm"]   / (df["25.4mm"]  + eps)
    df["ratio_4_9"]     = df["4.8mm"]   / (df["9.5mm"]   + eps)
    df["ratio_2_4"]     = df["2.0mm"]   / (df["4.8mm"]   + eps)
    df["ratio_042_2"]   = df["0.42mm"]  / (df["2.0mm"]   + eps)
    df["ratio_076_042"] = df["0.076mm"] / (df["0.42mm"]  + eps)

    # Activity index: higher = more plastic and less resistant soil
    df["atividade"]     = df["LL "] - df["IP "]

    # Compactness: denser soils with lower moisture tend to have higher CBR
    df["compacidade"]   = df["Densidade máxima "] / (df["Umidade Ótima"] + eps)

    # Fines squared: amplifies difference between soils with high/low fine material
    df["finos_sq"]      = df["0.076mm"] ** 2

    return df


def metricas(y_true: np.ndarray, y_pred: np.ndarray, nome: str) -> dict:
    """
    Calculates and prints MSE, RMSE, MAE and R² for a set of predictions.
    Warns if MSE <= 1, indicating possible scale issue or data leakage.

    Args:
        y_true: Real target values.
        y_pred: Values predicted by the model.
        nome:   Name of the evaluated set (e.g., "Validation", "Final Test").

    Returns:
        Dictionary with all calculated metrics.
    """
    mse  = mean_squared_error(y_true, y_pred)   # Penalizes large errors more than small ones
    rmse = np.sqrt(mse)                          # Root of MSE: same unit as target (%)
    mae  = mean_absolute_error(y_true, y_pred)  # Mean absolute errors — intuitive to interpret
    r2   = r2_score(y_true, y_pred)             # R²: 1.0 = perfect, 0.0 = constant model

    print(f"\n  [{nome}]")
    print(f"    MSE  : {mse:.4f}")

    if mse <= 1:
        print(f"    WARNING: MSE={mse:.4f} <= 1 — suspicious value. "
              "Check target scale or possible data leakage.")

    print(f"    RMSE : {rmse:.4f}")
    print(f"    MAE  : {mae:.4f}")
    print(f"    R²   : {r2:.4f}")
    return dict(nome=nome, mse=mse, rmse=rmse, mae=mae, r2=r2)


# ─────────────────────────────────────────────
# CHART FUNCTIONS
# ─────────────────────────────────────────────

def grafico_previsto_vs_real_validacao(Y_val: np.ndarray, pred_val_ens: np.ndarray,
                                       met_val: dict) -> None:
    """
    Scatter chart: ensemble predicted values vs. real values
    on the VALIDATION set.

    Points near the orange diagonal indicate precise predictions.
    R² is annotated in the upper left corner.
    """
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=PALETTE["fundo"])
    ax.set_facecolor(PALETTE["fundo"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    lim = [min(Y_val.min(), pred_val_ens.min()) * 0.95,
           max(Y_val.max(), pred_val_ens.max()) * 1.05]

    ax.scatter(Y_val, pred_val_ens, alpha=0.6, color=PALETTE["azul"],
               edgecolors="white", linewidths=0.4, s=50)

    ax.plot(lim, lim, "--", color=PALETTE["laranja"], lw=1.5, label="Ideal")  # y=x line

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_title("Predicted vs Actual — Validation", fontweight="bold")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.legend(framealpha=0)

    ax.text(0.05, 0.92, f"R² = {met_val['r2']:.4f}", transform=ax.transAxes,
            fontsize=10, color=PALETTE["azul"], fontweight="bold")

    plt.tight_layout()
    plt.show()


def grafico_previsto_vs_real_teste(y_teste: np.ndarray, pred_teste_ens: np.ndarray,
                                   met_teste: dict) -> None:
    """
    Scatter chart: ensemble predicted values vs. real values
    on the TEST set (final generalization evaluation).

    MSE is displayed in blue if goal was reached, orange otherwise.
    """
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=PALETTE["fundo"])
    ax.set_facecolor(PALETTE["fundo"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    lim = [min(y_teste.min(), pred_teste_ens.min()) * 0.95,
           max(y_teste.max(), pred_teste_ens.max()) * 1.05]

    ax.scatter(y_teste, pred_teste_ens, alpha=0.6, color=PALETTE["azul2"],  # was verde
               edgecolors="white", linewidths=0.4, s=50)
    ax.plot(lim, lim, "--", color=PALETTE["laranja"], lw=1.5, label="Ideal")

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_title("Predicted vs Actual — Test", fontweight="bold")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.legend(framealpha=0)

    ax.text(0.05, 0.92, f"R² = {met_teste['r2']:.4f}", transform=ax.transAxes,
            fontsize=10, color=PALETTE["azul2"], fontweight="bold")

    # MSE goal indicator: blue = reached, orange = not reached
    cor_meta = PALETTE["azul2"] if met_teste["mse"] < META_MSE else PALETTE["laranja"]
    ax.text(0.05, 0.83, f"MSE = {met_teste['mse']:.4f}", transform=ax.transAxes,
            fontsize=9, color=cor_meta, fontweight="bold")

    plt.tight_layout()
    plt.show()


def grafico_tabela_metricas(met_val: dict, met_teste: dict) -> None:
    """
    Comparative metrics table (MSE, RMSE, MAE, R²) side by side
    for validation and test sets.

    Displayed as a matplotlib figure (no axes), with blue header
    and alternating cells for easy reading.
    """
    fig, ax = plt.subplots(figsize=(6, 3), facecolor=PALETTE["fundo"])
    ax.axis("off")

    table_data = [
        ["Metric",  "Validation",               "Test"],
        ["MSE",  f"{met_val['mse']:.4f}",  f"{met_teste['mse']:.4f}"],
        ["RMSE", f"{met_val['rmse']:.4f}", f"{met_teste['rmse']:.4f}"],
        ["MAE",  f"{met_val['mae']:.4f}",  f"{met_teste['mae']:.4f}"],
        ["R²",   f"{met_val['r2']:.4f}",   f"{met_teste['r2']:.4f}"],
    ]

    tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                   cellLoc="center", loc="center", bbox=[0.05, 0.1, 0.9, 0.8])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(PALETTE["grade"])
        if r == 0:                               # Header row: blue background, white bold text
            cell.set_facecolor(PALETTE["azul"])
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#EFF6FF" if c != 0 else "#F1F5F9")

    ax.set_title("Metrics Comparison — Ensemble", fontweight="bold", pad=14)

    plt.tight_layout()
    plt.show()


def grafico_residuos_validacao(Y_val: np.ndarray, pred_val_ens: np.ndarray) -> None:
    """
    Residuals chart for the VALIDATION set.

    Residual = Actual - Predicted. Points randomly distributed around
    the zero line indicate a model without systematic bias.
    """
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=PALETTE["fundo"])
    ax.set_facecolor(PALETTE["fundo"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    residuos = Y_val - pred_val_ens              # Difference between actual and predicted

    ax.scatter(pred_val_ens, residuos, alpha=0.6, color=PALETTE["azul"],
               edgecolors="white", linewidths=0.4, s=50)

    ax.axhline(0, color=PALETTE["laranja"], lw=1.5, linestyle="--")  # Zero residual reference

    ax.set_title("Residuals — Validation", fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (Actual - Predicted)")

    plt.tight_layout()
    plt.show()


def grafico_residuos_teste(y_teste: np.ndarray, pred_teste_ens: np.ndarray) -> None:
    """
    Residuals chart for the TEST set.

    Systematic patterns (funnel, curve) indicate bias or lack of fit
    in specific CBR ranges.
    """
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=PALETTE["fundo"])
    ax.set_facecolor(PALETTE["fundo"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    residuos = y_teste - pred_teste_ens

    ax.scatter(pred_teste_ens, residuos, alpha=0.6, color=PALETTE["azul2"],  # was verde
               edgecolors="white", linewidths=0.4, s=50)
    ax.axhline(0, color=PALETTE["laranja"], lw=1.5, linestyle="--")

    ax.set_title("Residuals — Test", fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (Actual - Predicted)")

    plt.tight_layout()
    plt.show()


def grafico_comparativo_mse(met_rf: dict, met_gb: dict, met_et: dict,
                             met_teste: dict) -> None:
    """
    Bar chart comparing MSE of each individual model
    (RF, GB, ET) against the ensemble on the test set.

    roxo (purple) replaced by laranja (orange) for GradientBoosting bar.
    verde (green) replaced by azul2 (light blue) for ExtraTrees bar.
    """
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=PALETTE["fundo"])
    ax.set_facecolor(PALETTE["fundo"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    nomes_mod   = ["Random\nForest", "Gradient\nBoosting", "Extra\nTrees", "Ensemble"]
    mse_valores = [met_rf["mse"], met_gb["mse"], met_et["mse"], met_teste["mse"]]

    # Ensemble: dark blue (#1E40AF) if goal reached, orange if not
    cor_ensemble = PALETTE["laranja"] if met_teste["mse"] >= META_MSE else "#1E40AF"
    cores_bar = [
        PALETTE["azul"],      # Random Forest → primary blue
        PALETTE["laranja"],   # GradientBoosting → orange (was purple/roxo)
        PALETTE["azul2"],     # ExtraTrees → light blue (was green/verde)
        cor_ensemble,         # Ensemble
    ]

    bars = ax.bar(nomes_mod, mse_valores, color=cores_bar, edgecolor="white", width=0.5)

    # Dashed red line indicating the MSE goal
    ax.axhline(META_MSE, color="red", lw=1.5, linestyle="--",
               label=f"MSE Goal = {META_MSE}")

    # Annotate MSE value numerically above each bar
    for bar, val in zip(bars, mse_valores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_title("MSE by Model — Final Test", fontweight="bold")
    ax.set_ylabel("MSE")
    ax.legend(framealpha=0)

    plt.tight_layout()
    plt.show()


def grafico_importancia_features(rf_otimizado: RandomForestRegressor,
                                  et_otimizado: ExtraTreesRegressor,
                                  feature_names: list) -> None:
    """
    Horizontal bar chart with the top 15 most important features,
    calculated as the mean Gini importance between RF and ExtraTrees.
    """
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=PALETTE["fundo"])
    ax.set_facecolor(PALETTE["fundo"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    importancias_media = (
        rf_otimizado.feature_importances_ + et_otimizado.feature_importances_
    ) / 2

    top_n = min(15, len(feature_names))
    idx = np.argsort(importancias_media)[-top_n:][::-1]

    # Blue gradient: darker = more important
    cores = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))[::-1]

    ax.barh(range(top_n), importancias_media[idx], color=cores, edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in idx], fontsize=9)
    ax.invert_yaxis()                            # Most important feature at top

    ax.set_title(f"Top {top_n} Most Important Features (avg RF + ET)", fontweight="bold")
    ax.set_xlabel("Importance (Gini)")
    ax.grid(color=PALETTE["grade"], linewidth=0.8)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# 1. LOADING AND FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("=" * 60)
print("  RANDOM FOREST — HIGH PERFORMANCE")
print(f"  GOAL: MSE < {META_MSE}")
print("=" * 60)
print("\n[1/6] Loading data and applying feature engineering...")

df = pd.read_csv(CAMINHO_DADOS)
print(f"     Original dataset: {df.shape[0]} rows x {df.shape[1]} columns")

df_eng = engenharia_features(df, COLUNA_ALVO)
print(f"     Expanded dataset: {df_eng.shape[0]} rows x {df_eng.shape[1]} columns")
print(f"     New features: {df_eng.shape[1] - df.shape[1]} added")

Y = df_eng[COLUNA_ALVO].values.ravel()
X = df_eng.drop(columns=[COLUNA_ALVO])
feature_names = X.columns.tolist()
X = X.values

# Store original Y to compute metrics on the real CBR scale after prediction
Y_orig = Y.copy()
if LOG_ALVO:
    Y = np.log1p(Y)                              # log1p(x) = log(1+x): stable near zero
    print(f"     Log-transform active: target transformed to log1p(CBR)")
    print(f"     Original range: [{Y_orig.min():.2f}, {Y_orig.max():.2f}]"
          f"  →  log1p: [{Y.min():.4f}, {Y.max():.4f}]")

# ─────────────────────────────────────────────
# 2. DATA SPLIT
# ─────────────────────────────────────────────
print("\n[2/6] Splitting data...")

X_tv, X_teste, Y_tv, y_teste = train_test_split(
    X, Y, test_size=TEST_SIZE, random_state=SEED
)

X_treino, X_val, Y_treino, Y_val = train_test_split(
    X_tv, Y_tv, test_size=VAL_SIZE, random_state=SEED
)

# MinMaxScaler fitted ONLY on training set — avoids data leakage
scaler     = MinMaxScaler()
X_treino_n = scaler.fit_transform(X_treino)
X_val_n    = scaler.transform(X_val)
X_teste_n  = scaler.transform(X_teste)
X_tv_n     = scaler.transform(X_tv)

print(f"     Train: {X_treino_n.shape[0]}  |  Validation: {X_val_n.shape[0]}  |  Test: {X_teste_n.shape[0]}")
print(f"     Total features: {X_treino_n.shape[1]}")

if USE_WEIGHTS and THRESHOLD is not None:
    # When LOG_ALVO is active, Y_treino is in log scale — convert threshold accordingly
    thr_efetivo = np.log1p(THRESHOLD) if LOG_ALVO else THRESHOLD
    sample_weights = np.where(Y_treino > thr_efetivo, W_MINOR, W_MAJOR).astype(np.float32)
    print(f"     Sample weights active — threshold={THRESHOLD}"
          f"{' (log1p=' + f'{thr_efetivo:.4f})' if LOG_ALVO else ''}")
else:
    sample_weights = None

# ─────────────────────────────────────────────
# 3. HYPERPARAMETER SEARCH SPACES
# ─────────────────────────────────────────────
print("\n[3/6] Defining hyperparameter search spaces...")

param_rf = {
    "rf__n_estimators":      list(range(100, 801, 50)),
    "rf__max_depth":         list(range(3, 31, 2)) + [None],
    "rf__min_samples_split": list(range(2, 15)),
    "rf__min_samples_leaf":  list(range(1, 10)),
    "rf__max_features":      ["sqrt", "log2"] + np.linspace(0.1, 1.0, 10).round(2).tolist(),
    "rf__max_samples":       np.linspace(0.5, 1.0, 6).round(2).tolist(),
}

param_gb = {
    "gb__n_estimators":      list(range(100, 601, 50)),
    "gb__max_depth":         list(range(2, 9)),
    "gb__learning_rate":     [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2],
    "gb__min_samples_split": list(range(2, 10)),
    "gb__min_samples_leaf":  list(range(1, 8)),
    "gb__subsample":         np.linspace(0.6, 1.0, 5).round(2).tolist(),
    "gb__max_features":      ["sqrt", "log2", None],
}

param_et = {
    "et__n_estimators":      list(range(100, 601, 50)),
    "et__max_depth":         list(range(3, 31, 2)) + [None],
    "et__min_samples_split": list(range(2, 15)),
    "et__min_samples_leaf":  list(range(1, 10)),
    "et__max_features":      ["sqrt", "log2"] + np.linspace(0.1, 1.0, 10).round(2).tolist(),
}

# ─────────────────────────────────────────────
# 4. HYPERPARAMETER SEARCH PER MODEL
# ─────────────────────────────────────────────
print(f"\n[4/6] Searching best hyperparameters ({N_ITER_BUSCA} iter, {CV_FOLDS}-fold CV)...")

kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

from sklearn.pipeline import Pipeline as _Pipe

print("  Searching RF...")
pipe_rf   = _Pipe([("rf", RandomForestRegressor(random_state=SEED, n_jobs=-1))])
search_rf = RandomizedSearchCV(
    pipe_rf, param_rf, n_iter=N_ITER_BUSCA,
    scoring="neg_mean_squared_error",
    cv=kf, random_state=SEED, n_jobs=-1, verbose=0,
)
search_rf.fit(X_treino_n, Y_treino)
best_rf_params = {k.replace("rf__", ""): v for k, v in search_rf.best_params_.items()}
print(f"     Best CV MSE (RF): {-search_rf.best_score_:.4f}")

print("  Searching GradientBoosting...")
pipe_gb   = _Pipe([("gb", GradientBoostingRegressor(random_state=SEED))])
search_gb = RandomizedSearchCV(
    pipe_gb, param_gb, n_iter=N_ITER_BUSCA,
    scoring="neg_mean_squared_error",
    cv=kf, random_state=SEED, n_jobs=-1, verbose=0,
)
search_gb.fit(X_treino_n, Y_treino)
best_gb_params = {k.replace("gb__", ""): v for k, v in search_gb.best_params_.items()}
print(f"     Best CV MSE (GB): {-search_gb.best_score_:.4f}")

print("  Searching ExtraTrees...")
pipe_et   = _Pipe([("et", ExtraTreesRegressor(random_state=SEED, n_jobs=-1))])
search_et = RandomizedSearchCV(
    pipe_et, param_et, n_iter=N_ITER_BUSCA,
    scoring="neg_mean_squared_error",
    cv=kf, random_state=SEED, n_jobs=-1, verbose=0,
)
search_et.fit(X_treino_n, Y_treino)
best_et_params = {k.replace("et__", ""): v for k, v in search_et.best_params_.items()}
print(f"     Best CV MSE (ET): {-search_et.best_score_:.4f}")

# ─────────────────────────────────────────────
# 5. FINAL TRAINING — ENSEMBLE
# ─────────────────────────────────────────────
print("\n[5/6] Training final ensemble...")

rf_otimizado = RandomForestRegressor(**best_rf_params, random_state=SEED, n_jobs=-1)
gb_otimizado = GradientBoostingRegressor(**best_gb_params, random_state=SEED)
et_otimizado = ExtraTreesRegressor(**best_et_params,  random_state=SEED, n_jobs=-1)

ensemble = VotingRegressor(estimators=[
    ("rf", rf_otimizado),
    ("gb", gb_otimizado),
    ("et", et_otimizado),
])

fit_params_final = {}
if USE_WEIGHTS and THRESHOLD is not None:
    thr_efetivo = np.log1p(THRESHOLD) if LOG_ALVO else THRESHOLD
    sw_tv = np.where(Y_tv > thr_efetivo, W_MINOR, W_MAJOR).astype(np.float32)
    fit_params_final["sample_weight"] = sw_tv

ensemble.fit(X_tv_n, Y_tv, **fit_params_final)
print("     Ensemble trained on train + validation.")

# ─────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────
print("\n[6/6] Evaluating models...")

pred_val_ens   = ensemble.predict(X_val_n)
pred_teste_ens = ensemble.predict(X_teste_n)

# Revert log-transform on predictions and labels before computing metrics
# expm1(x) = exp(x)-1: exact inverse of log1p — restores original CBR scale
if LOG_ALVO:
    pred_val_ens   = np.expm1(pred_val_ens)
    pred_teste_ens = np.expm1(pred_teste_ens)
    Y_val_met   = np.expm1(Y_val)              # Validation labels on original scale
    y_teste_met = np.expm1(y_teste)            # Test labels on original scale
    print("     Predictions reverted to original scale (expm1)")
else:
    Y_val_met   = Y_val
    y_teste_met = y_teste

met_val   = metricas(Y_val_met,   pred_val_ens,   "Validation  — Ensemble")
met_teste = metricas(y_teste_met, pred_teste_ens, "Final Test   — Ensemble")

print("\n  --- Individual Models (Test) ---")
rf_otimizado.fit(X_tv_n, Y_tv)
gb_otimizado.fit(X_tv_n, Y_tv)
et_otimizado.fit(X_tv_n, Y_tv)

# Revert log-transform on individual model predictions as well
def _pred(model, X):
    p = model.predict(X)
    return np.expm1(p) if LOG_ALVO else p

met_rf = metricas(y_teste_met, _pred(rf_otimizado, X_teste_n), "Random Forest")
met_gb = metricas(y_teste_met, _pred(gb_otimizado, X_teste_n), "GradientBoosting")
met_et = metricas(y_teste_met, _pred(et_otimizado, X_teste_n), "ExtraTrees")

print("\n" + "=" * 60)
if met_teste["mse"] < META_MSE:
    print(f"  GOAL REACHED! MSE = {met_teste['mse']:.4f} < {META_MSE}")
else:
    print(f"  MSE = {met_teste['mse']:.4f}  |  Goal: < {META_MSE}")
print("=" * 60)

# ─────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────
print("\n--- Generating Charts ---")

grafico_previsto_vs_real_validacao(Y_val_met, pred_val_ens, met_val)
grafico_previsto_vs_real_teste(y_teste_met, pred_teste_ens, met_teste)
grafico_tabela_metricas(met_val, met_teste)
grafico_residuos_validacao(Y_val_met, pred_val_ens)
grafico_residuos_teste(y_teste_met, pred_teste_ens)
grafico_comparativo_mse(met_rf, met_gb, met_et, met_teste)
grafico_importancia_features(rf_otimizado, et_otimizado, feature_names)

print("Charts generated successfully!")

# ─────────────────────────────────────────────
# SAVING
# ─────────────────────────────────────────────

dump(ensemble, os.path.join(OUTPUT_DIR, "rf_modelo_final.joblib"))
dump(scaler,   os.path.join(OUTPUT_DIR, "scaler.joblib"))

print(f"\n  Ensemble saved -> rf_modelo_final.joblib")
print(f"  Scaler  saved  -> scaler.joblib")
print(f"  Folder         -> {OUTPUT_DIR}")

print("\n" + "=" * 60)
print("  Process completed successfully!")
print("=" * 60)