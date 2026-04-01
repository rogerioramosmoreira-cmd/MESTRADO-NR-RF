"""
Modelo Random Forest — Engenharia de Features (Tabela 2) + Cenários D1-D4
Baseado na configuração experimental da Seção 2.1 da dissertação.

Mudanças em relação à versão anterior:
  - Feature engineering restaurada com fórmulas exatas da Tabela 2
  - Correção: atividade = LL - LP  (LP = LL - IP, conforme Tabela 2)
  - Renomeado: compacidade → compactacao (nomenclatura da Tabela 2)
  - Adicionado: seleção de cenário D1/D2/D3/D4 conforme Seção 2.1
  - Pasta de saída por cenário: Modelo_salvo_RF_D1, _D2, _D3, _D4
  - Metadados do cenário salvos em metadados.json (lidos pelo PREVISAO.py)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import dump
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    KFold,
)
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    VotingRegressor,
)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.pipeline import Pipeline as _Pipe

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURAÇÕES GLOBAIS
# ─────────────────────────────────────────────

CAMINHO_DADOS = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "dados_processados_1.csv"
))

COLUNA_ALVO  = "CBR "
SEED         = 42
TEST_SIZE    = 0.20
VAL_SIZE     = 0.15
N_ITER_BUSCA = 150
CV_FOLDS     = 10
META_MSE     = 0.780

# ─────────────────────────────────────────────────────────────────────────────
# CENÁRIOS DE FEATURES — Seção 2.1 da dissertação
# ─────────────────────────────────────────────────────────────────────────────
#
# Mapeamento de abreviações da dissertação → nomes de coluna do CSV:
#   IG  = 25.4mm        (pedregulho grosso)
#   EXP = 9.5mm         (pedregulho fino / expansão granulométrica)
#   D3  = 4.8mm         (peneira #4)
#   D4  = 2.0mm         (peneira #10)
#   D5  = 0.42mm        (peneira #40)
#   D6  = 0.076mm       (peneira #200 — finos)
#   CH  = Umidade Ótima
#   CY  = Densidade máxima
#   IP  = IP
#   LL  = LL
#
# Subconjuntos (Seção 2.1):
#   D1: [D6, CH, CY]            → 3 features originais
#   D2: [D6, CH, CY, IP]        → 4 features originais
#   D3: [D6, CH, CY, IP, LL]    → 5 features originais
#   D4: [IG, EXP, D3, D4, D5, D6, CH, CY, IP, LL] → 10 features originais
#
# A feature engineering (Tabela 2) é aplicada com base nas features do cenário:
# ratios só são geradas se AMBAS as peneiras envolvidas estiverem no cenário.
# ─────────────────────────────────────────────────────────────────────────────

CENARIO = "D4"   # ← altere para "D1", "D2", "D3" ou "D4"

_ABREV = {
    "IG":  "25.4mm",
    "EXP": "9.5mm",
    "D3":  "4.8mm",
    "D4":  "2.0mm",
    "D5":  "0.42mm",
    "D6":  "0.076mm",
    "CH":  "Umidade Ótima",
    "CY":  "Densidade máxima",
    "IP":  "IP",
    "LL":  "LL",
}

CENARIOS = {
    "D1": [_ABREV[k] for k in ["D6", "CH", "CY"]],
    "D2": [_ABREV[k] for k in ["D6", "CH", "CY", "IP"]],
    "D3": [_ABREV[k] for k in ["D6", "CH", "CY", "IP", "LL"]],
    "D4": [_ABREV[k] for k in ["IG", "EXP", "D3", "D4", "D5", "D6", "CH", "CY", "IP", "LL"]],
}

# 10 features originais do CSV (necessárias para normalizar colunas)
FEATURES_ORIGINAIS = [
    "25.4mm", "9.5mm", "4.8mm", "2.0mm", "0.42mm", "0.076mm",
    "LL", "IP", "Umidade Ótima", "Densidade máxima",
]

OUTPUT_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), f"Modelo_salvo_RF_{CENARIO}"
))
os.makedirs(OUTPUT_DIR, exist_ok=True)

USE_WEIGHTS = False
THRESHOLD   = None
W_MINOR     = 3.0
W_MAJOR     = 1.0

PALETTE = {
    "azul":    "#2563EB",
    "verde":   "#16A34A",
    "laranja": "#EA580C",
    "roxo":    "#7C3AED",
    "fundo":   "#F8FAFC",
    "grade":   "#E2E8F0",
}

plt.rcParams.update({
    "figure.facecolor":  PALETTE["fundo"],
    "axes.facecolor":    PALETTE["fundo"],
    "axes.grid":         True,
    "grid.color":        PALETTE["grade"],
    "grid.linewidth":    0.8,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ─────────────────────────────────────────────
# FUNÇÕES AUXILIARES
# ─────────────────────────────────────────────

def normalizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove espaços extras nos nomes de colunas e aplica mapeamento de variações.
    Verifica se as 10 features originais + CBR estão presentes no CSV.
    """
    df.columns = df.columns.str.strip()

    mapa = {
        "25,4mm":"25.4mm","9,5mm":"9.5mm","4,8mm":"4.8mm",
        "2,0mm":"2.0mm","0,42mm":"0.42mm","0,076mm":"0.076mm",
        "P25.4":"25.4mm","P9.5":"9.5mm","P4.8":"4.8mm",
        "P2.0":"2.0mm","P0.42":"0.42mm","P0.076":"0.076mm",
        "Ll":"LL","ll":"LL","L.L":"LL","L.L.":"LL",
        "Limite de Liquidez":"LL",
        "Ip":"IP","ip":"IP","I.P":"IP","I.P.":"IP",
        "Índice de Plasticidade":"IP","Indice de Plasticidade":"IP",
        "Wot":"Umidade Ótima","wot":"Umidade Ótima","W_ot":"Umidade Ótima",
        "Umidade otima":"Umidade Ótima","Umidade Otima":"Umidade Ótima",
        "Umidade ótima":"Umidade Ótima","w_ot":"Umidade Ótima",
        "Densidade Maxima":"Densidade máxima","Densidade Máxima":"Densidade máxima",
        "densidade máxima":"Densidade máxima","densidade maxima":"Densidade máxima",
        "d_max":"Densidade máxima","Dmax":"Densidade máxima",
        "γdmax":"Densidade máxima","ydmax":"Densidade máxima",
        "CBR":"CBR ","cbr":"CBR ","Cbr":"CBR ",
    }
    df = df.rename(columns=mapa)

    colunas_esperadas = FEATURES_ORIGINAIS + [COLUNA_ALVO]
    faltando = [c for c in colunas_esperadas if c not in df.columns]
    if faltando:
        print("\n  AVISO: Colunas não encontradas no CSV:")
        for c in faltando:
            print(f"    - '{c}'")
        print("\n  Colunas reais (após strip):")
        for c in df.columns:
            print(f"    - '{c}'")
        raise SystemExit(1)
    return df


def engenharia_features(df: pd.DataFrame, features_cenario: list) -> pd.DataFrame:
    """
    Aplica a engenharia de features da Tabela 2 da dissertação.

    Fórmulas exatas conforme Tabela 2:
      ratio_9_25    = P9.5mm   / P25.4mm   — abertura da curva grossa
      ratio_4_9     = P4.8mm   / P9.5mm    — transição pedregulho/areia grossa
      ratio_2_4     = P2.0mm   / P4.8mm    — escalonamento interno médio
      ratio_042_2   = P0.42mm  / P2.0mm    — proporção areia fina
      ratio_076_042 = P0.076mm / P0.42mm   — limite silte/areia fina
      atividade     = LL - LP              — (LP = LL - IP; Tabela 2)
      compactacao   = Densidade máx / Wot — eficiência de compactação
      finos_sq      = (P0.076mm)²         — efeito não-linear dos finos

    Regra de geração condicional:
      Cada feature derivada só é criada se todas as features-base
      necessárias para seu cálculo estiverem no cenário selecionado.

    Args:
        df:               DataFrame com as 10 features originais.
        features_cenario: Lista de colunas originais do cenário ativo.

    Returns:
        DataFrame com colunas do cenário + derivadas aplicáveis.
    """
    df  = df.copy()
    eps = 1e-6

    # ── Ratios granulométricas (Tabela 2) ─────────────────────────────────────
    if {"9.5mm",  "25.4mm"}.issubset(features_cenario):
        df["ratio_9_25"]    = df["9.5mm"]   / (df["25.4mm"]  + eps)

    if {"4.8mm",  "9.5mm"}.issubset(features_cenario):
        df["ratio_4_9"]     = df["4.8mm"]   / (df["9.5mm"]   + eps)

    if {"2.0mm",  "4.8mm"}.issubset(features_cenario):
        df["ratio_2_4"]     = df["2.0mm"]   / (df["4.8mm"]   + eps)

    if {"0.42mm", "2.0mm"}.issubset(features_cenario):
        df["ratio_042_2"]   = df["0.42mm"]  / (df["2.0mm"]   + eps)

    if {"0.076mm","0.42mm"}.issubset(features_cenario):
        df["ratio_076_042"] = df["0.076mm"] / (df["0.42mm"]  + eps)

    # ── Atividade — LL - LP  (Tabela 2) ──────────────────────────────────────
    # LP = LL - IP  →  atividade = LL - (LL - IP) = IP
    # Calculado via LP para fidelidade à expressão da Tabela 2.
    if {"LL", "IP"}.issubset(features_cenario):
        LP = df["LL"] - df["IP"]             # Limite de Plasticidade
        df["atividade"] = df["LL"] - LP      # = IP conforme Tabela 2: LL - LP

    # ── Compactacao — ρdmax / Wot  (Tabela 2) ────────────────────────────────
    if {"Densidade máxima", "Umidade Ótima"}.issubset(features_cenario):
        df["compactacao"] = df["Densidade máxima"] / (df["Umidade Ótima"] + eps)

    # ── Finos ao quadrado — (P0.076mm)²  (Tabela 2) ──────────────────────────
    if "0.076mm" in features_cenario:
        df["finos_sq"] = df["0.076mm"] ** 2

    # Seleciona colunas do cenário + derivadas geradas (exclui features fora do cenário)
    colunas_finais = features_cenario + [
        c for c in df.columns
        if c not in FEATURES_ORIGINAIS and c != COLUNA_ALVO
    ]
    return df[colunas_finais]


def metricas(y_true: np.ndarray, y_pred: np.ndarray, nome: str) -> dict:
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"\n  [{nome}]")
    print(f"    MSE  : {mse:.4f}")
    if mse <= 1:
        print(f"    ATENCAO: MSE={mse:.4f} <= 1 — verifique data leakage.")
    print(f"    RMSE : {rmse:.4f}")
    print(f"    MAE  : {mae:.4f}")
    print(f"    R²   : {r2:.4f}")
    return dict(nome=nome, mse=mse, rmse=rmse, mae=mae, r2=r2)


# ─────────────────────────────────────────────
# FUNÇÕES DE GRÁFICO
# ─────────────────────────────────────────────

def grafico_previsto_vs_real_validacao(Y_val, pred, met):
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=PALETTE["fundo"])
    ax.set_facecolor(PALETTE["fundo"])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    lim = [min(Y_val.min(), pred.min()) * 0.95, max(Y_val.max(), pred.max()) * 1.05]
    ax.scatter(Y_val, pred, alpha=0.6, color=PALETTE["azul"],
               edgecolors="white", linewidths=0.4, s=50)
    ax.plot(lim, lim, "--", color=PALETTE["laranja"], lw=1.5, label="Ideal")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title(f"Previsto vs Real — Validação [{CENARIO}]", fontweight="bold")
    ax.set_xlabel("Real"); ax.set_ylabel("Previsto"); ax.legend(framealpha=0)
    ax.text(0.05, 0.92, f"R² = {met['r2']:.4f}", transform=ax.transAxes,
            fontsize=10, color=PALETTE["azul"], fontweight="bold")
    plt.tight_layout(); plt.show()


def grafico_previsto_vs_real_teste(y_teste, pred, met):
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=PALETTE["fundo"])
    ax.set_facecolor(PALETTE["fundo"])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    lim = [min(y_teste.min(), pred.min()) * 0.95, max(y_teste.max(), pred.max()) * 1.05]
    ax.scatter(y_teste, pred, alpha=0.6, color=PALETTE["verde"],
               edgecolors="white", linewidths=0.4, s=50)
    ax.plot(lim, lim, "--", color=PALETTE["laranja"], lw=1.5, label="Ideal")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title(f"Previsto vs Real — Teste [{CENARIO}]", fontweight="bold")
    ax.set_xlabel("Real"); ax.set_ylabel("Previsto"); ax.legend(framealpha=0)
    ax.text(0.05, 0.92, f"R² = {met['r2']:.4f}", transform=ax.transAxes,
            fontsize=10, color=PALETTE["verde"], fontweight="bold")
    cor = PALETTE["verde"] if met["mse"] < META_MSE else PALETTE["laranja"]
    ax.text(0.05, 0.83, f"MSE = {met['mse']:.4f}", transform=ax.transAxes,
            fontsize=9, color=cor, fontweight="bold")
    plt.tight_layout(); plt.show()


def grafico_tabela_metricas(met_val, met_teste, n_features):
    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=PALETTE["fundo"])
    ax.axis("off")
    td = [
        ["Métrica", "Validação",              "Teste"],
        ["MSE",  f"{met_val['mse']:.4f}",  f"{met_teste['mse']:.4f}"],
        ["RMSE", f"{met_val['rmse']:.4f}", f"{met_teste['rmse']:.4f}"],
        ["MAE",  f"{met_val['mae']:.4f}",  f"{met_teste['mae']:.4f}"],
        ["R²",   f"{met_val['r2']:.4f}",   f"{met_teste['r2']:.4f}"],
    ]
    tbl = ax.table(cellText=td[1:], colLabels=td[0],
                   cellLoc="center", loc="center", bbox=[0.05, 0.1, 0.9, 0.8])
    tbl.auto_set_font_size(False); tbl.set_fontsize(11)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(PALETTE["grade"])
        if r == 0:
            cell.set_facecolor(PALETTE["azul"])
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#EFF6FF" if c != 0 else "#F1F5F9")
    ax.set_title(f"Métricas — Cenário {CENARIO} | {n_features} features",
                 fontweight="bold", pad=14)
    plt.tight_layout(); plt.show()


def grafico_residuos(y_true, pred, titulo, cor):
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=PALETTE["fundo"])
    ax.set_facecolor(PALETTE["fundo"])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.scatter(pred, y_true - pred, alpha=0.6, color=cor,
               edgecolors="white", linewidths=0.4, s=50)
    ax.axhline(0, color=PALETTE["laranja"], lw=1.5, linestyle="--")
    ax.set_title(titulo, fontweight="bold")
    ax.set_xlabel("Previsto"); ax.set_ylabel("Resíduo (Real - Previsto)")
    plt.tight_layout(); plt.show()


def grafico_comparativo_mse(met_rf, met_gb, met_et, met_teste):
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=PALETTE["fundo"])
    ax.set_facecolor(PALETTE["fundo"])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    nomes = ["Random\nForest", "Gradient\nBoosting", "Extra\nTrees", "Ensemble"]
    vals  = [met_rf["mse"], met_gb["mse"], met_et["mse"], met_teste["mse"]]
    cor_e = PALETTE["laranja"] if met_teste["mse"] >= META_MSE else "#15803D"
    cores = [PALETTE["azul"], PALETTE["roxo"], PALETTE["verde"], cor_e]
    bars  = ax.bar(nomes, vals, color=cores, edgecolor="white", width=0.5)
    ax.axhline(META_MSE, color="red", lw=1.5, linestyle="--",
               label=f"Meta MSE = {META_MSE}")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title(f"MSE por Modelo — Teste [{CENARIO}]", fontweight="bold")
    ax.set_ylabel("MSE"); ax.legend(framealpha=0)
    plt.tight_layout(); plt.show()


def grafico_importancia_features(rf_ot, et_ot, feature_names):
    fig, ax = plt.subplots(figsize=(10, max(5, len(feature_names) * 0.45 + 1)),
                           facecolor=PALETTE["fundo"])
    ax.set_facecolor(PALETTE["fundo"])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    imp = (rf_ot.feature_importances_ + et_ot.feature_importances_) / 2
    n   = len(feature_names)
    idx = np.argsort(imp)[-n:][::-1]
    cores = plt.cm.Blues(np.linspace(0.4, 0.9, n))[::-1]
    ax.barh(range(n), imp[idx], color=cores, edgecolor="white")
    ax.set_yticks(range(n))
    ax.set_yticklabels([feature_names[i] for i in idx], fontsize=9)
    ax.invert_yaxis()
    ax.set_title(f"Importância das Features — Cenário {CENARIO} "
                 f"({n} features, média RF + ET)", fontweight="bold")
    ax.set_xlabel("Importância (Gini)")
    ax.grid(color=PALETTE["grade"], linewidth=0.8)
    plt.tight_layout(); plt.show()


# ─────────────────────────────────────────────
# 1. CARREGAMENTO E FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("=" * 60)
print(f"  RANDOM FOREST — CENÁRIO {CENARIO}")
print(f"  META: MSE < {META_MSE}")
print("=" * 60)

if CENARIO not in CENARIOS:
    print(f"ERRO: Cenário '{CENARIO}' inválido. Use D1, D2, D3 ou D4.")
    raise SystemExit(1)

features_cenario = CENARIOS[CENARIO]
print(f"\nCenário {CENARIO}: {len(features_cenario)} features originais")
print(f"  {features_cenario}")

print("\n[1/6] Carregando dados e aplicando engenharia de features (Tabela 2)...")

df = pd.read_csv(CAMINHO_DADOS)
print(f"     Dataset: {df.shape[0]} linhas x {df.shape[1]} colunas")

df = normalizar_colunas(df)

Y    = df[COLUNA_ALVO].values.ravel()           # Alvo preservado intacto
X_df = engenharia_features(df, features_cenario) # Aplica Tabela 2 ao cenário

feature_names = X_df.columns.tolist()
X = X_df.values

n_orig = len(features_cenario)
n_der  = len(feature_names) - n_orig
print(f"     Features originais no cenário : {n_orig}")
print(f"     Features derivadas (Tabela 2) : {n_der}")
print(f"     Total de features para treino : {len(feature_names)}")

# ─────────────────────────────────────────────
# 2. DIVISÃO E NORMALIZAÇÃO
# ─────────────────────────────────────────────
print("\n[2/6] Dividindo e normalizando dados...")

X_tv, X_teste, Y_tv, y_teste = train_test_split(
    X, Y, test_size=TEST_SIZE, random_state=SEED
)
X_treino, X_val, Y_treino, Y_val = train_test_split(
    X_tv, Y_tv, test_size=VAL_SIZE, random_state=SEED
)

scaler     = MinMaxScaler()
X_treino_n = scaler.fit_transform(X_treino)     # fit apenas no treino
X_val_n    = scaler.transform(X_val)
X_teste_n  = scaler.transform(X_teste)
X_tv_n     = scaler.transform(X_tv)

print(f"     Treino: {X_treino_n.shape[0]}  |  Val: {X_val_n.shape[0]}  |  Teste: {X_teste_n.shape[0]}")
print(f"     Features: {X_treino_n.shape[1]}")

sample_weights = None
if USE_WEIGHTS and THRESHOLD is not None:
    sample_weights = np.where(Y_treino > THRESHOLD, W_MINOR, W_MAJOR).astype(np.float32)
    print(f"     Pesos amostrais ativos — threshold={THRESHOLD}")

# ─────────────────────────────────────────────
# 3. ESPAÇOS DE HIPERPARÂMETROS
# ─────────────────────────────────────────────
print("\n[3/6] Definindo espaços de hiperparâmetros...")

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
# 4. BUSCA DE HIPERPARÂMETROS
# ─────────────────────────────────────────────
print(f"\n[4/6] Buscando hiperparâmetros ({N_ITER_BUSCA} iter, {CV_FOLDS}-fold CV)...")

kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

print("  Buscando RF...")
pipe_rf   = _Pipe([("rf", RandomForestRegressor(random_state=SEED, n_jobs=-1))])
search_rf = RandomizedSearchCV(pipe_rf, param_rf, n_iter=N_ITER_BUSCA,
    scoring="neg_mean_squared_error", cv=kf, random_state=SEED, n_jobs=-1, verbose=0)
search_rf.fit(X_treino_n, Y_treino)
best_rf = {k.replace("rf__", ""): v for k, v in search_rf.best_params_.items()}
print(f"     Melhor MSE CV (RF): {-search_rf.best_score_:.4f}")

print("  Buscando GradientBoosting...")
pipe_gb   = _Pipe([("gb", GradientBoostingRegressor(random_state=SEED))])
search_gb = RandomizedSearchCV(pipe_gb, param_gb, n_iter=N_ITER_BUSCA,
    scoring="neg_mean_squared_error", cv=kf, random_state=SEED, n_jobs=-1, verbose=0)
search_gb.fit(X_treino_n, Y_treino)
best_gb = {k.replace("gb__", ""): v for k, v in search_gb.best_params_.items()}
print(f"     Melhor MSE CV (GB): {-search_gb.best_score_:.4f}")

print("  Buscando ExtraTrees...")
pipe_et   = _Pipe([("et", ExtraTreesRegressor(random_state=SEED, n_jobs=-1))])
search_et = RandomizedSearchCV(pipe_et, param_et, n_iter=N_ITER_BUSCA,
    scoring="neg_mean_squared_error", cv=kf, random_state=SEED, n_jobs=-1, verbose=0)
search_et.fit(X_treino_n, Y_treino)
best_et = {k.replace("et__", ""): v for k, v in search_et.best_params_.items()}
print(f"     Melhor MSE CV (ET): {-search_et.best_score_:.4f}")

# ─────────────────────────────────────────────
# 5. TREINAMENTO FINAL — ENSEMBLE
# ─────────────────────────────────────────────
print("\n[5/6] Treinando ensemble final (treino + validação)...")

rf_ot = RandomForestRegressor(**best_rf, random_state=SEED, n_jobs=-1)
gb_ot = GradientBoostingRegressor(**best_gb, random_state=SEED)
et_ot = ExtraTreesRegressor(**best_et,  random_state=SEED, n_jobs=-1)

ensemble = VotingRegressor(estimators=[("rf", rf_ot), ("gb", gb_ot), ("et", et_ot)])

fit_params = {}
if USE_WEIGHTS and THRESHOLD is not None:
    fit_params["sample_weight"] = np.where(Y_tv > THRESHOLD, W_MINOR, W_MAJOR).astype(np.float32)

ensemble.fit(X_tv_n, Y_tv, **fit_params)
print("     Ensemble treinado com sucesso.")

# ─────────────────────────────────────────────
# 6. AVALIAÇÃO
# ─────────────────────────────────────────────
print("\n[6/6] Avaliando modelos...")

pred_val   = ensemble.predict(X_val_n)
pred_teste = ensemble.predict(X_teste_n)

met_val   = metricas(Y_val,   pred_val,   f"Validação  — Ensemble [{CENARIO}]")
met_teste = metricas(y_teste, pred_teste, f"Teste Final — Ensemble [{CENARIO}]")

print("\n  --- Modelos Individuais (Teste) ---")
rf_ot.fit(X_tv_n, Y_tv); gb_ot.fit(X_tv_n, Y_tv); et_ot.fit(X_tv_n, Y_tv)
met_rf = metricas(y_teste, rf_ot.predict(X_teste_n), "Random Forest")
met_gb = metricas(y_teste, gb_ot.predict(X_teste_n), "GradientBoosting")
met_et = metricas(y_teste, et_ot.predict(X_teste_n), "ExtraTrees")

print("\n" + "=" * 60)
if met_teste["mse"] < META_MSE:
    print(f"  META ATINGIDA! MSE = {met_teste['mse']:.4f} < {META_MSE}")
else:
    print(f"  MSE = {met_teste['mse']:.4f}  |  Meta: < {META_MSE}")
print("=" * 60)

# ─────────────────────────────────────────────
# VISUALIZAÇÕES
# ─────────────────────────────────────────────
print("\n--- Gerando Gráficos ---")
grafico_previsto_vs_real_validacao(Y_val, pred_val, met_val)
grafico_previsto_vs_real_teste(y_teste, pred_teste, met_teste)
grafico_tabela_metricas(met_val, met_teste, len(feature_names))
grafico_residuos(Y_val,   pred_val,   f"Resíduos — Validação [{CENARIO}]", PALETTE["azul"])
grafico_residuos(y_teste, pred_teste, f"Resíduos — Teste [{CENARIO}]",     PALETTE["verde"])
grafico_comparativo_mse(met_rf, met_gb, met_et, met_teste)
grafico_importancia_features(rf_ot, et_ot, feature_names)

# ─────────────────────────────────────────────
# SALVAMENTO
# ─────────────────────────────────────────────
dump(ensemble, os.path.join(OUTPUT_DIR, "rf_modelo_final.joblib"))
dump(scaler,   os.path.join(OUTPUT_DIR, "scaler.joblib"))

# metadados.json — lido pelo PREVISAO.py para replicar a engenharia corretamente
metadados = {
    "cenario":          CENARIO,
    "features_cenario": features_cenario,   # features originais do cenário
    "feature_names":    feature_names,      # originais + derivadas (ordem do scaler)
}
with open(os.path.join(OUTPUT_DIR, "metadados.json"), "w", encoding="utf-8") as f:
    json.dump(metadados, f, ensure_ascii=False, indent=2)

print(f"\n  Ensemble salvo  -> rf_modelo_final.joblib")
print(f"  Scaler  salvo   -> scaler.joblib")
print(f"  Metadados       -> metadados.json")
print(f"  Pasta           -> {OUTPUT_DIR}")
print("\n" + "=" * 60)
print(f"  Concluído — Cenário {CENARIO}")
print("=" * 60)