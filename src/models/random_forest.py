"""
Modelo Random Forest — Alta Performance (Modelo Único)
Meta: R² >= 0.80 no conjunto de teste.

Estratégias aplicadas:
  1. RandomForestRegressor único com busca extensa de hiperparâmetros
  2. Busca — 150 iterações, 10-fold CV, scoring=neg_MSE
  3. Retreino no conjunto treino+validação após o ajuste
  3b. Generalização medida por CV out-of-fold no treino — nunca prevendo
      dados que o modelo final já viu (evita data leakage otimista)
  4. Métricas completas: MSE, RMSE, MAE, R²
  5. Alerta se R² >= 0.99 (sinal de vazamento)
  6. Transformação logarítmica no alvo (log1p/expm1)
  7. Pesos amostrais — amostras raras com CBR alto recebem maior peso
  8. Gráficos de raciocínio do modelo durante a busca e convergência OOB
"""

# ─────────────────────────────────────────────
# IMPORTAÇÕES
# ─────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from joblib import dump
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    KFold,
    cross_val_predict,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.pipeline import Pipeline as _Pipe

warnings.filterwarnings("ignore")

# Torna o pacote `core` (src/core) importável quando este script é executado
# diretamente, e não apenas através de `main.py`.
import sys                                            # noqa: E402
from pathlib import Path                              # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import (  # noqa: E402
    console, dependencies, metrics, paths, plots, progress, runtime,
    scoreboard,
)

dependencies.require("core")

# ─────────────────────────────────────────────
# CONFIGURAÇÕES GLOBAIS
# ─────────────────────────────────────────────

CAMINHO_DADOS = paths.DATASET

COLUNA_ALVO = "CBR "

FEATURES = [
    "25.4mm",           # IG  — pedregulho grosso
    "9.5mm",            # EXP — pedregulho médio/fino
    "4.8mm",            # D3  — peneira nº4 (pedregulho/areia)
    "2.0mm",            # D4  — peneira nº10 (areia grossa)
    "0.42mm",           # D5  — peneira nº40 (areia fina)
    "0.076mm",          # D6  — peneira nº200 (silte/argila)
    "LL",               # LL  — limite de liquidez
    "IP",               # IP  — índice de plasticidade
    "Umidade Ótima",    # CH  — umidade ótima de compactação (Proctor)
    "Densidade máxima", # CY  — densidade seca máxima (Proctor)
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
    "Umidade Ótima (CH)",
    "Densidade máxima (CY)",
]

SEED         = 42
TEST_SIZE    = 0.20
VAL_SIZE     = 0.15
# O segundo valor vale apenas com MLL_FAST=1 - ver core/runtime.py.
N_ITER_BUSCA = runtime.budget(150, 15)
CV_FOLDS     = runtime.budget(10, 3)
# Meta declarada em R² (ver core/metrics.py). O limiar equivalente de MSE
# depende da variância do conjunto avaliado, então só pode ser calculado
# depois da divisão dos dados — recebe o valor real na seção 6.
META_R2      = metrics.META_R2
META_MSE     = float("inf")

OUTPUT_DIR = paths.RF_DIR
paths.ensure(OUTPUT_DIR)

# ── Log-transform no alvo ─────────────────────────────────────────────────────
LOG_ALVO  = True                 # True = treina em log1p(CBR), avalia em expm1(pred)

# ── Pesos amostrais ───────────────────────────────────────────────────────────
USE_WEIGHTS = True
THRESHOLD   = 25.0
W_MINOR     = 3.0
W_MAJOR     = 1.0

# ─────────────────────────────────────────────
# PALETA DE CORES
# ─────────────────────────────────────────────
PALETTE = {
    "azul":     "#2563EB",   # Validação / azul primário
    "azul2":    "#60A5FA",   # Teste / azul claro
    "laranja":  "#EA580C",   # Referência, destaques
    "verde":    "#16A34A",   # Melhora, cotovelo, ganho positivo
    "vermelho": "#DC2626",   # Alertas, perda
    "roxo":     "#7C3AED",   # Raciocínio, derivada
    "fundo":    "#F8FAFC",
    "grade":    "#E2E8F0",
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

def metricas(y_true, y_pred, nome):
    """Calcula e imprime MSE, RMSE, MAE, MAPE e R²."""
    mse   = mean_squared_error(y_true, y_pred)
    rmse  = np.sqrt(mse)
    mae   = mean_absolute_error(y_true, y_pred)
    r2    = r2_score(y_true, y_pred)
    mape  = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-6))) * 100
    print(f"\n  [{nome}]")
    print(f"    MSE  : {mse:.4f}")
    # Um R² quase perfeito em dados de verdade quase sempre significa que o
    # conjunto avaliado vazou para o treino. O limiar antigo era um MSE fixo
    # (<= 1), que disparava exatamente quando a meta do projeto fosse
    # atingida; em R² o alerta independe da escala do alvo.
    if r2 >= 0.99:
        print(f"    ATENÇÃO: R²={r2:.4f} — alto demais para dados novos; "
              "verifique escala ou data leakage.")
    print(f"    RMSE : {rmse:.4f}")
    print(f"    MAE  : {mae:.4f}")
    print(f"    MAPE : {mape:.2f}%")
    print(f"    R²   : {r2:.4f}")
    return dict(nome=nome, mse=mse, rmse=rmse, mae=mae, mape=mape, r2=r2)


# ─────────────────────────────────────────────
# FUNÇÕES DE GRÁFICO
# ─────────────────────────────────────────────

def grafico_previsto_vs_real_validacao(Y_cv, pred_cv, met_cv):
    """Dispersão previsto × real — Validação cruzada (out-of-fold)."""
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=PALETTE["fundo"])
    ax.set_facecolor(PALETTE["fundo"])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    lim = [min(Y_cv.min(), pred_cv.min()) * 0.95,
           max(Y_cv.max(), pred_cv.max()) * 1.05]
    ax.scatter(Y_cv, pred_cv, alpha=0.6, color=PALETTE["azul"],
               edgecolors="white", linewidths=0.4, s=50)
    ax.plot(lim, lim, "--", color=PALETTE["laranja"], lw=1.5, label="Ideal")
    ax.set_xlim(lim); ax.set_ylim(lim)

    ax.set_xlabel("Real"); ax.set_ylabel("Previsto")
    ax.legend()
    ax.text(0.05, 0.92, f"R² = {met_cv['r2']:.4f}", transform=ax.transAxes,
            fontsize=10, color=PALETTE["azul"], fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFFFFF", edgecolor="#E2E8F0", alpha=0.93))
    plt.tight_layout(); plots.save(plt.gcf(), "previsto_vs_real_validacao", "random_forest")


def grafico_previsto_vs_real_teste(y_teste, pred_teste, met_teste):
    """Dispersão previsto × real — Teste."""
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=PALETTE["fundo"])
    ax.set_facecolor(PALETTE["fundo"]); ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    lim = [min(y_teste.min(), pred_teste.min()) * 0.95,
           max(y_teste.max(), pred_teste.max()) * 1.05]
    ax.scatter(y_teste, pred_teste, alpha=0.6, color=PALETTE["azul2"],
               edgecolors="white", linewidths=0.4, s=50)
    ax.plot(lim, lim, "--", color=PALETTE["laranja"], lw=1.5, label="Ideal")
    ax.set_xlim(lim); ax.set_ylim(lim)

    ax.set_xlabel("Real"); ax.set_ylabel("Previsto")
    ax.legend()
    ax.text(0.05, 0.92, f"R² = {met_teste['r2']:.4f}", transform=ax.transAxes,
            fontsize=10, color=PALETTE["azul2"], fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFFFFF", edgecolor="#E2E8F0", alpha=0.93))
    cor_meta = PALETTE["azul2"] if met_teste["r2"] >= META_R2 else PALETTE["laranja"]
    ax.text(0.05, 0.83, f"MSE = {met_teste['mse']:.4f}", transform=ax.transAxes,
            fontsize=10, color=cor_meta, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFFFFF", edgecolor="#E2E8F0", alpha=0.93))
    plt.tight_layout(); plots.save(plt.gcf(), "previsto_vs_real_teste", "random_forest")


def grafico_tabela_metricas(met_cv, met_teste):
    """Tabela comparativa de métricas."""
    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=PALETTE["fundo"])
    ax.axis("off")
    table_data = [
        ["Métrica", "Validação CV", "Teste"],
        ["MSE",  f"{met_cv['mse']:.4f}",  f"{met_teste['mse']:.4f}"],
        ["RMSE", f"{met_cv['rmse']:.4f}", f"{met_teste['rmse']:.4f}"],
        ["MAE",  f"{met_cv['mae']:.4f}",  f"{met_teste['mae']:.4f}"],
        ["MAPE", f"{met_cv['mape']:.2f}%", f"{met_teste['mape']:.2f}%"],
        ["R²",   f"{met_cv['r2']:.4f}",   f"{met_teste['r2']:.4f}"],
    ]
    tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                   cellLoc="center", loc="center", bbox=[0.05, 0.1, 0.9, 0.8])
    tbl.auto_set_font_size(False); tbl.set_fontsize(11)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(PALETTE["grade"])
        if r == 0:
            cell.set_facecolor(PALETTE["azul"])
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#EFF6FF" if c != 0 else "#F1F5F9")

    plt.tight_layout(); plots.save(plt.gcf(), "tabela_metricas", "random_forest")


def grafico_residuos_validacao(Y_cv, pred_cv):
    """Resíduos — Validação cruzada (out-of-fold)."""
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=PALETTE["fundo"])
    ax.set_facecolor(PALETTE["fundo"]); ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.scatter(pred_cv, Y_cv - pred_cv, alpha=0.6, color=PALETTE["azul"],
               edgecolors="white", linewidths=0.4, s=50)
    ax.axhline(0, color=PALETTE["laranja"], lw=1.5, linestyle="--")

    ax.set_xlabel("Previsto"); ax.set_ylabel("Resíduo (Real − Previsto)")
    plt.tight_layout(); plots.save(plt.gcf(), "residuos_validacao", "random_forest")


def grafico_residuos_teste(y_teste, pred_teste):
    """Resíduos — Teste."""
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=PALETTE["fundo"])
    ax.set_facecolor(PALETTE["fundo"]); ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.scatter(pred_teste, y_teste - pred_teste, alpha=0.6, color=PALETTE["azul2"],
               edgecolors="white", linewidths=0.4, s=50)
    ax.axhline(0, color=PALETTE["laranja"], lw=1.5, linestyle="--")

    ax.set_xlabel("Previsto"); ax.set_ylabel("Resíduo (Real − Previsto)")
    plt.tight_layout(); plots.save(plt.gcf(), "residuos_teste", "random_forest")


def grafico_importancia_features(rf_modelo, feature_names):
    """
    Gráfico de barras horizontais — Importância das features (Gini do RF).
    Gradiente de azul: mais escuro = mais importante.
    """
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=PALETTE["fundo"])
    ax.set_facecolor(PALETTE["fundo"])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    importancias = rf_modelo.feature_importances_
    top_n = len(feature_names)
    idx   = np.argsort(importancias)[-top_n:][::-1]
    cores = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))[::-1]

    ax.barh(range(top_n), importancias[idx], color=cores, edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([FEATURES_LABELS[i] for i in idx], fontsize=10)
    ax.invert_yaxis()

    ax.set_xlabel("Importância (Gini)")
    ax.grid(color=PALETTE["grade"], linewidth=0.8)
    plt.tight_layout(); plots.save(plt.gcf(), "importancia_features", "random_forest")


def grafico_raciocinio_rf(search_rf):
    """
    Raciocínio do Random Forest durante a busca de hiperparâmetros.

    Painel superior: MSE CV mínimo acumulado por iteração.
      Cada ponto = melhor configuração encontrada até aquela iteração.
      Estrelas (★) marcam quedas ≥ 5% — momentos de descoberta real.

    Painel inferior: taxa de mudança (ΔMSE) entre iterações consecutivas.
      Barras verdes = melhora; vermelhas = piora ou estagnação.
      Permite ver em quais iterações o algoritmo mudou sua estratégia.
    """
    resultados   = search_rf.cv_results_
    mse_iter     = -resultados["mean_test_score"]   # negativo no sklearn → inverte
    n_iter       = len(mse_iter)
    iters        = np.arange(1, n_iter + 1)
    mse_min_acum = np.minimum.accumulate(mse_iter)  # mínimo acumulado
    delta        = np.diff(mse_min_acum)             # variação entre iterações

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), facecolor=PALETTE["fundo"],
                             gridspec_kw={"height_ratios": [2, 1]})


    # ── Painel superior: curva de melhora cumulativa ──────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor(PALETTE["fundo"])
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    ax1.plot(iters, mse_min_acum, color=PALETTE["azul"], lw=2,
             label="MSE CV mínimo acumulado")

    # Marca momentos de melhora significativa (queda > 5%)
    melhoras = [i for i in range(1, n_iter) if mse_min_acum[i] < mse_min_acum[i-1] * 0.95]
    if melhoras:
        ax1.scatter([i + 1 for i in melhoras],
                    [mse_min_acum[i] for i in melhoras],
                    marker="*", s=150, color=PALETTE["laranja"], zorder=5,
                    label="★ Melhora ≥ 5%")
        # Anota o número da iteração em cada estrela
        for i in melhoras[:6]:   # limita a 6 anotações para não poluir
            ax1.annotate(f"it.{i+1}\n{mse_min_acum[i]:.3f}",
                         xy=(i + 1, mse_min_acum[i]),
                         xytext=(i + 1 + n_iter * 0.02, mse_min_acum[i] * 1.01),
                         fontsize=9, color=PALETTE["laranja"])

    # Linha da melhor configuração final
    melhor_idx = int(np.argmin(mse_iter))
    ax1.axhline(mse_min_acum[-1], color=PALETTE["verde"], lw=1.2, linestyle="--",
                label=f"Melhor MSE final = {mse_min_acum[-1]:.4f}")
    ax1.scatter([melhor_idx + 1], [mse_iter[melhor_idx]],
                color=PALETTE["verde"], s=90, zorder=5,
                label=f"Melhor iteração: {melhor_idx + 1}")

    ax1.set_ylabel("MSE CV (mínimo acumulado)")

    ax1.legend()

    # ── Painel inferior: ΔMSE por iteração ───────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(PALETTE["fundo"])
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    cores_delta = [PALETTE["verde"] if d < 0 else PALETTE["vermelho"] for d in delta]
    ax2.bar(iters[1:], delta, color=cores_delta, edgecolor="white", alpha=0.85, width=0.8)
    ax2.axhline(0, color=PALETTE["grade"], lw=1.0, linestyle="--")

    # Destaca os maiores ganhos
    maiores_ganhos = np.argsort(delta)[:3]   # 3 maiores quedas (mais negativos)
    for idx_g in maiores_ganhos:
        if delta[idx_g] < 0:
            ax2.annotate(f"−{abs(delta[idx_g]):.3f}",
                         xy=(iters[1:][idx_g], delta[idx_g]),
                         xytext=(iters[1:][idx_g], delta[idx_g] * 1.3),
                         ha="center", fontsize=9, color=PALETTE["verde"],
                         fontweight="bold")

    ax2.set_xlabel("Iteração da Busca")
    ax2.set_ylabel("ΔMSE (negativo = melhora)")


    plt.tight_layout()
    plots.save(plt.gcf(), "raciocinio_rf", "random_forest")


def grafico_convergencia_oob(rf_otimizado, X_tv_n, Y_tv):
    """
    Convergência do Random Forest por Número de Árvores.

    Painel esquerdo: curva de erro OOB (Out-of-Bag) conforme árvores
      são adicionadas. Marca o ponto de cotovelo — onde adicionar mais
      árvores deixa de trazer ganho significativo.

    Painel direito: ganho marginal de cada grupo de árvores adicionadas.
      Verde = melhora; vermelho = sem ganho adicional.
    """
    n_est_final = rf_otimizado.n_estimators
    passos = list(range(10, n_est_final + 1, max(1, n_est_final // 40)))
    if passos[-1] != n_est_final:
        passos.append(n_est_final)

    params = rf_otimizado.get_params()
    params["oob_score"]  = True
    params["warm_start"] = True
    params["n_jobs"]     = -1

    rf_oob    = RandomForestRegressor(**params)
    oob_errors = []
    for n in passos:
        rf_oob.set_params(n_estimators=n)
        rf_oob.fit(X_tv_n, Y_tv)
        oob_errors.append(1 - rf_oob.oob_score_)   # 1 − R² ≈ erro relativo

    oob_arr  = np.array(oob_errors)
    deltas   = -np.diff(oob_arr)                    # ganho marginal (positivo = melhora)
    knee_idx = int(np.argmax(deltas)) + 1
    knee_n   = passos[knee_idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=PALETTE["fundo"])


    # Painel esquerdo: curva de convergência
    ax = axes[0]
    ax.set_facecolor(PALETTE["fundo"])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.plot(passos, oob_arr, color=PALETTE["azul"], lw=2, label="Erro OOB (1 − R²)")
    ax.axvline(knee_n, color=PALETTE["laranja"], lw=1.5, linestyle="--",
               label=f"Cotovelo: {knee_n} árvores")
    ax.scatter([knee_n], [oob_arr[knee_idx]], color=PALETTE["laranja"], s=80, zorder=5)
    # Destaca a zona após o cotovelo (retorno decrescente)
    ax.axvspan(knee_n, passos[-1], alpha=0.06, color=PALETTE["vermelho"],
               label="Retorno decrescente")
    ax.set_xlabel("Número de Árvores"); ax.set_ylabel("Erro OOB (1 − R²)")

    ax.legend()

    # Painel direito: ganho marginal
    ax2 = axes[1]
    ax2.set_facecolor(PALETTE["fundo"])
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    cores_barra = [PALETTE["verde"] if d > 0 else PALETTE["vermelho"] for d in deltas]
    ax2.bar(range(len(deltas)), deltas, color=cores_barra, edgecolor="white", alpha=0.85)
    ax2.axhline(0, color=PALETTE["grade"], lw=1.0)
    ax2.axvline(knee_idx - 0.5, color=PALETTE["laranja"], lw=1.5, linestyle="--",
                label=f"Cotovelo (grupo {knee_idx})")
    ax2.set_xlabel("Grupo de Árvores Adicionadas")
    ax2.set_ylabel("Ganho Marginal (−ΔOOB)")

    ax2.legend()

    plt.tight_layout()
    plots.save(plt.gcf(), "convergencia_oob", "random_forest")


def grafico_distribuicao_cv(search_rf):
    """
    Distribuição das performances de todas as 150 configurações testadas.

    Substitui o gráfico de comparativo de ensemble.
    Mostra a distribuição do MSE CV de todas as combinações testadas,
    destacando onde a melhor configuração se posiciona em relação ao
    espaço explorado.
    """
    mse_todas   = -search_rf.cv_results_["mean_test_score"]
    mse_std     = search_rf.cv_results_["std_test_score"]
    melhor_mse  = mse_todas.min()
    melhor_idx  = mse_todas.argmin()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=PALETTE["fundo"])


    # Painel esquerdo: histograma de MSE CV
    ax = axes[0]
    ax.set_facecolor(PALETTE["fundo"])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.hist(mse_todas, bins=20, color=PALETTE["azul"], alpha=0.7, edgecolor="white")
    ax.axvline(melhor_mse, color=PALETTE["laranja"], lw=2, linestyle="--",
               label=f"Melhor = {melhor_mse:.4f}")
    ax.axvline(np.median(mse_todas), color=PALETTE["roxo"], lw=1.5, linestyle=":",
               label=f"Mediana = {np.median(mse_todas):.4f}")
    ax.set_xlabel("MSE CV"); ax.set_ylabel("Frequência")

    ax.legend()

    # Painel direito: top 15 configurações (barras com barra de erro)
    ax2 = axes[1]
    ax2.set_facecolor(PALETTE["fundo"])
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    top_idx  = np.argsort(mse_todas)[:15]           # 15 melhores
    top_mse  = mse_todas[top_idx]
    top_std  = mse_std[top_idx]
    cores_top = [PALETTE["laranja"] if i == 0 else PALETTE["azul"]
                 for i in range(len(top_idx))]
    bars = ax2.bar(range(len(top_idx)), top_mse, color=cores_top,
                   edgecolor="white", alpha=0.85)
    ax2.errorbar(range(len(top_idx)), top_mse, yerr=top_std,
                 fmt="none", color="#374151", capsize=3, lw=1.0)
    # Sem linha de meta: este eixo mostra o MSE da busca, que está em escala
    # log1p, enquanto a meta vive na escala original do CBR. Traçar uma
    # contra a outra comparava grandezas diferentes.
    for bar, val in zip(bars, top_mse):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax2.set_xlabel("Ranking"); ax2.set_ylabel("MSE CV")
    ax2.legend()

    plt.tight_layout()
    plots.save(plt.gcf(), "distribuicao_cv", "random_forest")


# ─────────────────────────────────────────────
# 1. CARREGAMENTO
# ─────────────────────────────────────────────
print("=" * 60)
print("  RANDOM FOREST — MODELO ÚNICO — ALTA PERFORMANCE")
print(f"  META: R² >= {META_R2:.2f} no teste")
print("=" * 60)
print("\n[1/6] Carregando dados...")

df = pd.read_csv(CAMINHO_DADOS)
print(f"     Dataset: {df.shape[0]} linhas x {df.shape[1]} colunas brutas")

# Remove espaços nos nomes das colunas e renomeia variações conhecidas
df.columns = df.columns.str.strip()
_mapa = {
    "Ll":"LL","ll":"LL","L.L":"LL",
    "Ip":"IP","ip":"IP","I.P":"IP",
    "Densidade Maxima":"Densidade máxima","Densidade Máxima":"Densidade máxima",
    "densidade maxima":"Densidade máxima","d_max":"Densidade máxima","Dmax":"Densidade máxima",
    "Wot":"Umidade Ótima","wot":"Umidade Ótima",
    "Umidade otima":"Umidade Ótima","Umidade Otima":"Umidade Ótima",
    "CBR":"CBR ","cbr":"CBR ","Cbr":"CBR ",
}
df = df.rename(columns=_mapa)
df = df[FEATURES + [COLUNA_ALVO]]

Y             = df[COLUNA_ALVO].values.ravel()
X             = df[FEATURES].values
feature_names = FEATURES
print(f"     Features utilizadas: {len(FEATURES)} — {FEATURES}")

Y_orig = Y.copy()
if LOG_ALVO:
    Y = np.log1p(Y)
    print(f"     Log-transform ativo: [{Y_orig.min():.2f}, {Y_orig.max():.2f}]"
          f" → [{Y.min():.4f}, {Y.max():.4f}]")

# ─────────────────────────────────────────────
# 2. DIVISÃO DOS DADOS
# ─────────────────────────────────────────────
print("\n[2/6] Dividindo dados...")

X_tv, X_teste, Y_tv, y_teste = train_test_split(
    X, Y, test_size=TEST_SIZE, random_state=SEED
)
X_treino, X_val, Y_treino, Y_val = train_test_split(
    X_tv, Y_tv, test_size=VAL_SIZE, random_state=SEED
)

# MinMaxScaler ajustado APENAS no treino — evita data leakage
scaler     = MinMaxScaler()
X_treino_n = scaler.fit_transform(X_treino)
X_val_n    = scaler.transform(X_val)
X_teste_n  = scaler.transform(X_teste)
X_tv_n     = scaler.transform(X_tv)

print(f"     Treino: {X_treino_n.shape[0]}  |  Validação: {X_val_n.shape[0]}"
      f"  |  Teste: {X_teste_n.shape[0]}")

if USE_WEIGHTS and THRESHOLD is not None:
    thr_efetivo    = np.log1p(THRESHOLD) if LOG_ALVO else THRESHOLD
    sample_weights = np.where(Y_treino > thr_efetivo, W_MINOR, W_MAJOR).astype(np.float32)
    print(f"     Pesos amostrais: threshold={THRESHOLD} (log={thr_efetivo:.4f})")
else:
    sample_weights = None

# ─────────────────────────────────────────────
# 3. ESPAÇO DE HIPERPARÂMETROS
# ─────────────────────────────────────────────
print("\n[3/6] Definindo espaço de hiperparâmetros...")

param_rf = {
    "rf__n_estimators":      list(range(100, 801, 50)),
    "rf__max_depth":         list(range(3, 31, 2)) + [None],
    "rf__min_samples_split": list(range(2, 15)),
    "rf__min_samples_leaf":  list(range(1, 10)),
    "rf__max_features":      ["sqrt", "log2"] + np.linspace(0.1, 1.0, 10).round(2).tolist(),
    "rf__max_samples":       np.linspace(0.5, 1.0, 6).round(2).tolist(),
}

# ─────────────────────────────────────────────
# 4. BUSCA DE HIPERPARÂMETROS
# ─────────────────────────────────────────────
print(f"\n[4/6] Buscando hiperparâmetros ({N_ITER_BUSCA} iter, {CV_FOLDS}-fold CV)...")

kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

pipe_rf   = _Pipe([("rf", RandomForestRegressor(random_state=SEED, n_jobs=-1))])
search_rf = RandomizedSearchCV(
    pipe_rf, param_rf, n_iter=N_ITER_BUSCA,
    scoring="neg_mean_squared_error",
    cv=kf, random_state=SEED, n_jobs=-1, verbose=0,
)

# Uma etapa da barra por ajuste: a busca faz um por combinacao e por dobra.
with progress.bar(total=N_ITER_BUSCA * CV_FOLDS) as _stage:
    with progress.joblib_stage(_stage):
        search_rf.fit(X_treino_n, Y_treino,
                      rf__sample_weight=sample_weights)
best_rf_params = {k.replace("rf__", ""): v for k, v in search_rf.best_params_.items()}
# Este score sai em escala log1p (a busca treina sobre Y ja transformado),
# entao NAO e comparavel com META_MSE, que vive na escala original do CBR.
# A metrica comparavel e a de validacao cruzada calculada na secao 6.
_escala = " (escala log1p — nao comparavel com a meta)" if LOG_ALVO else ""
print(f"     Melhor MSE CV (RF): {-search_rf.best_score_:.4f}{_escala}")
print(f"     Melhores parâmetros: {best_rf_params}")

# ─────────────────────────────────────────────
# 5. TREINAMENTO FINAL
# ─────────────────────────────────────────────
print("\n[5/6] Treinando modelo final...")

rf_final = RandomForestRegressor(**best_rf_params, random_state=SEED, n_jobs=-1)

fit_kwargs = {}
if USE_WEIGHTS and THRESHOLD is not None:
    thr_efetivo = np.log1p(THRESHOLD) if LOG_ALVO else THRESHOLD
    sw_tv       = np.where(Y_tv > thr_efetivo, W_MINOR, W_MAJOR).astype(np.float32)
    fit_kwargs["sample_weight"] = sw_tv

# Retreina no conjunto treino+validação para maximizar os dados.
# Chamada unica do sklearn, sem etapas observaveis: barra de atividade.
with progress.bar() as _stage:
    with progress.joblib_stage(_stage):
        rf_final.fit(X_tv_n, Y_tv, **fit_kwargs)
print("     Modelo treinado em treino + validação.")

# ─────────────────────────────────────────────
# 6. AVALIAÇÃO
# ─────────────────────────────────────────────
print("\n[6/6] Avaliando modelo...")

# ── Estimativa de generalização: validação cruzada out-of-fold ──────────────
#
# A versão anterior media a "validação" com `rf_final.predict(X_val_n)`. Como
# `rf_final` é treinado em treino+validação (X_tv), o conjunto de validação já
# tinha sido visto: aquela métrica era erro de treino disfarçado de erro de
# generalização — otimista por construção (data leakage).
#
# Aqui a estimativa vem do CV com a melhor configuração, ajustada SOMENTE no
# conjunto de treino. Cada previsão é feita por um modelo que não viu aquela
# amostra, então é honesta. E, por usar todo o treino em vez de um holdout de
# ~12% das linhas, é também menos ruidosa que a validação antiga.
print(f"     Validação cruzada out-of-fold ({CV_FOLDS} dobras)...")

rf_cv     = RandomForestRegressor(**best_rf_params, random_state=SEED, n_jobs=-1)
params_cv = {"sample_weight": sample_weights} if sample_weights is not None else None

with progress.bar(total=CV_FOLDS) as _stage:
    with progress.joblib_stage(_stage):
        pred_cv = cross_val_predict(rf_cv, X_treino_n, Y_treino,
                                    cv=kf, n_jobs=-1, params=params_cv)

pred_teste = rf_final.predict(X_teste_n)

if LOG_ALVO:
    pred_cv     = np.expm1(pred_cv)
    pred_teste  = np.expm1(pred_teste)
    Y_cv_met    = np.expm1(Y_treino)
    y_teste_met = np.expm1(y_teste)
    print("     Previsões revertidas para escala original (expm1).")
else:
    Y_cv_met    = Y_treino
    y_teste_met = y_teste

# Limiar de MSE equivalente a META_R2 na variância real do conjunto de teste.
META_MSE = metrics.meta_mse(y_teste_met)
print(f"     Meta: R² >= {META_R2:.2f}  (MSE <= {META_MSE:.2f} nesta amostra)")

met_cv    = metricas(Y_cv_met,    pred_cv,    "Validação CV — Random Forest")
met_teste = metricas(y_teste_met, pred_teste, "Teste Final — Random Forest")

print("\n" + "=" * 60)
if met_teste["r2"] >= META_R2:
    print(f"  META ATINGIDA! R² = {met_teste['r2']:.4f} >= {META_R2:.2f}"
          f"  (MSE {met_teste['mse']:.2f} <= {META_MSE:.2f})")
else:
    print(f"  R² = {met_teste['r2']:.4f}  |  Meta: >= {META_R2:.2f}"
          f"  (MSE {met_teste['mse']:.2f}  |  Meta: <= {META_MSE:.2f})")
print("=" * 60)

# Guarda o resultado do teste para o menu mostrar depois, sem treinar de novo.
scoreboard.record("random_forest", "Árvore Aleatória — modelo único", met_teste)

# ─────────────────────────────────────────────
# GRÁFICOS
# ─────────────────────────────────────────────
print("\n--- Gerando Gráficos ---")

grafico_previsto_vs_real_validacao(Y_cv_met, pred_cv, met_cv)
grafico_previsto_vs_real_teste(y_teste_met, pred_teste, met_teste)
grafico_tabela_metricas(met_cv, met_teste)
grafico_residuos_validacao(Y_cv_met, pred_cv)
grafico_residuos_teste(y_teste_met, pred_teste)
grafico_importancia_features(rf_final, feature_names)

# ── Gráficos de raciocínio do modelo ─────────────────────────────────────────
grafico_distribuicao_cv(search_rf)             # Distribuição de todas as 150 configs
grafico_raciocinio_rf(search_rf)               # Evolução da busca + marcos de melhora
grafico_convergencia_oob(rf_final, X_tv_n, Y_tv)  # Convergência OOB por nº de árvores

print("\nGráficos gerados com sucesso!")

# ─────────────────────────────────────────────
# SALVAMENTO
# ─────────────────────────────────────────────
dump(rf_final, os.path.join(OUTPUT_DIR, "rf_model.joblib"))
dump(scaler,   os.path.join(OUTPUT_DIR, "scaler.joblib"))

print(f"\n  Modelo salvo → rf_model.joblib")
print(f"  Scaler salvo → scaler.joblib")
print(f"  Pasta        → {OUTPUT_DIR}")
print("\n" + "=" * 60)
print("  Processo concluído com sucesso!")
print("=" * 60)

# Veredito final: o que os números acima significam para quem vai usar o
# modelo. Fica por último de propósito — é a linha que alguém lê ao voltar
# ao terminal depois de um treino longo.
metrics.report(met_teste)