"""""
Random Forest — Divisão por Faixas de CBR (D1 a D5)
====================================================
Divide o dataset em 5 grupos (quintis) baseados no valor de CBR:
  D1 — CBR mais baixo  (0–20%)
  D2 — CBR baixo       (20–40%)
  D3 — CBR médio       (40–60%)
  D4 — CBR alto        (60–80%)
  D5 — CBR mais alto   (80–100%)

Para CADA grupo:
  - Split treino/validação/teste independente
  - Busca de hiperparâmetros própria (RandomizedSearchCV, 10-fold CV)
  - Modelo RF treinado e avaliado APENAS no seu grupo
  - Resultado isolado — sem misturar dados entre grupos

Objetivo: identificar em qual faixa de CBR o modelo tem mais
dificuldade e qual combinação de hiperparâmetros funciona melhor
para cada nível de resistência do solo.
"""

# ─────────────────────────────────────────────
# IMPORTAÇÕES
# ─────────────────────────────────────────────
import os
import json
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
    "4.8mm",            # D3  — peneira nº4
    "2.0mm",            # D4  — peneira nº10
    "0.42mm",           # D5  — peneira nº40
    "0.076mm",          # D6  — peneira nº200
    "LL",               # LL  — limite de liquidez
    "IP",               # IP  — índice de plasticidade
    "Umidade Ótima",    # CH  — umidade ótima (Proctor)
    "Densidade máxima", # CY  — densidade seca máxima (Proctor)
]

FEATURES_LABELS = [
    "25.4mm (IG)", "9.5mm (EXP)", "4.8mm (D3)", "2.0mm (D4)",
    "0.42mm (D5)", "0.076mm (D6)", "LL", "IP",
    "Umidade Ótima (CH)", "Densidade máxima (CY)",
]

SEED         = 42
TEST_SIZE    = 0.25    # 25% por grupo para teste (grupos menores exigem mais cuidado)
VAL_SIZE     = 0.20    # 20% do restante para validação
# O segundo valor vale apenas com MLL_FAST=1 - ver core/runtime.py.
N_ITER_BUSCA = runtime.budget(80, 10)   # Reduzido: grupos menores convergem mais rápido
CV_FOLDS     = runtime.budget(5, 3)     # 5-fold: mais robusto em grupos pequenos
# Meta por grupo declarada em R² (ver core/metrics.py). Um limiar de MSE fixo
# não serve aqui: a variância do CBR dentro de D1 é uma fração da de D5, então
# o mesmo MSE significa desempenho muito diferente em cada quintil. Cada
# grupo recebe seu próprio limiar, derivado da variância do seu teste.
META_R2      = metrics.META_R2

LOG_ALVO    = True
USE_WEIGHTS = True
W_MINOR     = 2.0      # Menor que o RF global — grupos menores precisam de regularização suave
W_MAJOR     = 1.0

OUTPUT_DIR = paths.RF_QUINTIS_DIR
paths.ensure(OUTPUT_DIR)

# ─────────────────────────────────────────────
# PALETA DE CORES
# ─────────────────────────────────────────────
PALETTE = {
    "azul":     "#2563EB",
    "azul2":    "#60A5FA",
    "laranja":  "#EA580C",
    "verde":    "#16A34A",
    "vermelho": "#DC2626",
    "roxo":     "#7C3AED",
    "fundo":    "#F8FAFC",
    "grade":    "#E2E8F0",
}

# Cor distinta para cada quintil — claramente diferenciadas
CORES_QUINTIL = {
    "D1": "#1D4ED8",   # azul escuro
    "D2": "#0891B2",   # ciano
    "D3": "#16A34A",   # verde
    "D4": "#D97706",   # âmbar
    "D5": "#DC2626",   # vermelho (D5 = CBR mais alto = mais difícil)
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
    """Calcula MSE, RMSE, MAE e R²."""
    if len(y_true) < 2:
        print(f"  [{nome}] — amostras insuficientes para avaliação.")
        return dict(nome=nome, mse=np.nan, rmse=np.nan, mae=np.nan, r2=np.nan)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"\n  [{nome}]")
    print(f"    MSE  : {mse:.4f}  {'✓ META' if r2 >= META_R2 else '✗'}")
    print(f"    RMSE : {rmse:.4f}")
    print(f"    MAE  : {mae:.4f}")
    print(f"    R²   : {r2:.4f}")
    return dict(nome=nome, mse=mse, rmse=rmse, mae=mae, r2=r2)


def treinar_quintil(rotulo, mask, X_full, Y_orig_full, Y_log_full):
    """
    Pipeline completo para um quintil de CBR.

    Recebe:
      - rotulo:      string "D1"..."D5"
      - mask:        máscara booleana das amostras do quintil
      - X_full:      matriz de features (escala original)
      - Y_orig_full: alvo em escala original (CBR %)
      - Y_log_full:  alvo em escala log (log1p(CBR))

    Retorna dicionário com métricas, previsões, modelos e histórico de busca.
    """
    X_g     = X_full[mask]
    Y_orig_g = Y_orig_full[mask]
    Y_log_g  = Y_log_full[mask]
    Y_g      = Y_log_g if LOG_ALVO else Y_orig_g

    n_amostras = len(X_g)
    print(f"\n{'═'*55}")
    print(f"  QUINTIL {rotulo}  |  {n_amostras} amostras")
    print(f"  CBR: [{Y_orig_g.min():.1f}, {Y_orig_g.max():.1f}]  "
          f"média={Y_orig_g.mean():.1f}  mediana={np.median(Y_orig_g):.1f}")
    print(f"{'═'*55}")

    # Verifica se há amostras suficientes para split
    if n_amostras < 20:
        print(f"  AVISO: grupo muito pequeno ({n_amostras} amostras) — pulando.")
        return None

    # ── Split ────────────────────────────────────────────────────────────────
    X_tv, X_teste, Y_tv, y_teste, Y_orig_tv, Y_orig_teste = train_test_split(
        X_g, Y_g, Y_orig_g, test_size=TEST_SIZE, random_state=SEED
    )
    X_treino, X_val, Y_treino, Y_val, Y_orig_treino, Y_orig_val = train_test_split(
        X_tv, Y_tv, Y_orig_tv, test_size=VAL_SIZE, random_state=SEED
    )

    # ── Normalização ─────────────────────────────────────────────────────────
    scaler     = MinMaxScaler()
    X_treino_n = scaler.fit_transform(X_treino)  # fit APENAS no treino
    X_val_n    = scaler.transform(X_val)
    X_teste_n  = scaler.transform(X_teste)
    X_tv_n     = scaler.transform(X_tv)

    print(f"  Treino: {len(X_treino_n)}  Val: {len(X_val_n)}  Teste: {len(X_teste_n)}")

    # ── Pesos amostrais ───────────────────────────────────────────────────────
    # Para D5 (CBR alto), todos os pesos são iguais — já é o grupo "raro"
    # Para D1-D4, peso extra nas amostras acima da mediana do grupo
    mediana_g  = np.median(Y_treino)
    sw_treino  = np.where(Y_treino > mediana_g, W_MINOR, W_MAJOR).astype(np.float32) \
                 if USE_WEIGHTS else None
    sw_tv      = np.where(Y_tv > mediana_g, W_MINOR, W_MAJOR).astype(np.float32) \
                 if USE_WEIGHTS else None

    # ── Espaço de hiperparâmetros ─────────────────────────────────────────────
    # Espaço mais compacto que o RF global — grupos menores convergem mais rápido
    param_rf = {
        "rf__n_estimators":      list(range(50, 501, 50)),
        "rf__max_depth":         list(range(2, 20, 2)) + [None],
        "rf__min_samples_split": list(range(2, 12)),
        "rf__min_samples_leaf":  list(range(1, 8)),
        "rf__max_features":      ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0],
        "rf__max_samples":       [0.6, 0.7, 0.8, 0.9, 1.0],
    }

    # ── Busca de hiperparâmetros ──────────────────────────────────────────────
    kf = KFold(n_splits=min(CV_FOLDS, len(X_treino_n) // 5),
               shuffle=True, random_state=SEED)

    pipe_rf   = _Pipe([("rf", RandomForestRegressor(random_state=SEED, n_jobs=-1))])
    search_rf = RandomizedSearchCV(
        pipe_rf, param_rf, n_iter=N_ITER_BUSCA,
        scoring="neg_mean_squared_error",
        cv=kf, random_state=SEED, n_jobs=-1, verbose=0,
    )

    fit_kwargs_busca = {"rf__sample_weight": sw_treino} if sw_treino is not None else {}

    # `kf` pode ter menos dobras que CV_FOLDS em grupos pequenos; o total da
    # barra usa a contagem real para nao ficar preso abaixo de 100%.
    print(f"  Busca de hiperparametros ({N_ITER_BUSCA} iter, "
          f"{kf.get_n_splits()}-fold)")
    with progress.bar(total=N_ITER_BUSCA * kf.get_n_splits()) as _stage:
        with progress.joblib_stage(_stage):
            search_rf.fit(X_treino_n, Y_treino, **fit_kwargs_busca)

    best_params = {k.replace("rf__", ""): v for k, v in search_rf.best_params_.items()}
    melhor_mse_cv = -search_rf.best_score_
    print(f"  Melhor MSE CV: {melhor_mse_cv:.4f}  |  Params: {best_params}")

    # ── Treinamento final ─────────────────────────────────────────────────────
    rf_final = RandomForestRegressor(**best_params, random_state=SEED, n_jobs=-1)
    fit_kwargs_final = {"sample_weight": sw_tv} if sw_tv is not None else {}
    print("  Treino final")
    with progress.bar() as _stage:
        with progress.joblib_stage(_stage):
            rf_final.fit(X_tv_n, Y_tv, **fit_kwargs_final)

    # ── Avaliação ────────────────────────────────────────────────────────
    #
    # `rf_final` foi treinado em treino+validação (X_tv), então prever X_val_n
    # devolveria erro de treino, não de generalização — a métrica sairia
    # otimista por construção (data leakage). A estimativa honesta vem do CV
    # out-of-fold com a melhor configuração, ajustada só no treino.
    rf_cv     = RandomForestRegressor(**best_params, random_state=SEED, n_jobs=-1)
    params_cv = {"sample_weight": sw_treino} if sw_treino is not None else None
    with progress.bar(total=kf.get_n_splits()) as _stage:
        with progress.joblib_stage(_stage):
            pred_cv = cross_val_predict(rf_cv, X_treino_n, Y_treino,
                                        cv=kf, n_jobs=-1, params=params_cv)

    pred_teste = rf_final.predict(X_teste_n)

    if LOG_ALVO:
        pred_cv     = np.expm1(pred_cv)
        pred_teste  = np.expm1(pred_teste)
        Y_cv_met    = Y_orig_treino
        y_teste_met = Y_orig_teste
    else:
        Y_cv_met    = Y_treino
        y_teste_met = y_teste

    meta_mse_g = metrics.meta_mse(y_teste_met)

    met_cv    = metricas(Y_cv_met,    pred_cv,    f"{rotulo} — Validação CV")
    met_teste = metricas(y_teste_met, pred_teste, f"{rotulo} — Teste")

    # Salva modelo e scaler do grupo
    pasta_g = os.path.join(OUTPUT_DIR, rotulo)
    os.makedirs(pasta_g, exist_ok=True)
    dump(rf_final, os.path.join(pasta_g, "rf_model.joblib"))
    dump(scaler,   os.path.join(pasta_g, "scaler.joblib"))

    # metadata.json descreve exatamente o que este modelo espera receber.
    # `predict_rf_quintis.py` se recusa a carregar o modelo sem ele, e sem a
    # flag `engenharia_features` o previsor não tem como saber que este
    # treinamento usa as 10 variáveis medidas, sem features derivadas.
    with open(os.path.join(pasta_g, "metadata.json"), "w", encoding="utf-8") as arquivo_meta:
        json.dump({
            "cenario":              rotulo,
            "features_cenario":     FEATURES,
            "feature_names":        FEATURES,
            "engenharia_features":  False,
            "log_alvo":             LOG_ALVO,
            "coluna_alvo":          COLUNA_ALVO,
            "n_amostras":           int(n_amostras),
            "cbr_min":              float(Y_orig_g.min()),
            "cbr_max":              float(Y_orig_g.max()),
        }, arquivo_meta, indent=2, ensure_ascii=False)


    return {
        "rotulo":        rotulo,
        "n_amostras":    n_amostras,
        "cbr_min":       Y_orig_g.min(),
        "cbr_max":       Y_orig_g.max(),
        "cbr_media":     Y_orig_g.mean(),
        "met_cv":        met_cv,
        "meta_mse":      meta_mse_g,
        "met_teste":     met_teste,
        "melhor_mse_cv": melhor_mse_cv,
        "best_params":   best_params,
        "rf":            rf_final,
        "search":        search_rf,
        "pred_cv":       pred_cv,
        "pred_teste":    pred_teste,
        "Y_cv_met":      Y_cv_met,
        "y_teste_met":   y_teste_met,
        "scaler":        scaler,
    }


# ─────────────────────────────────────────────
# FUNÇÕES DE GRÁFICO
# ─────────────────────────────────────────────

def grafico_distribuicao_quintis(Y_orig, limites, rotulos):
    """
    Histograma do CBR com as fronteiras dos quintis marcadas.
    Cada faixa colorida corresponde a um Dn.
    """
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=PALETTE["fundo"])
    ax.set_facecolor(PALETTE["fundo"])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax.hist(Y_orig, bins=30, color=PALETTE["azul"], alpha=0.5,
            edgecolor="white", label="Distribuição CBR")

    cores_list = list(CORES_QUINTIL.values())
    for i, (rot, cor) in enumerate(CORES_QUINTIL.items()):
        lo = limites[i]
        hi = limites[i + 1]
        ax.axvspan(lo, hi, alpha=0.12, color=cor, label=f"{rot} [{lo:.1f}–{hi:.1f}]")
        ax.axvline(lo, color=cor, lw=1.2, linestyle="--", alpha=0.7)

    ax.axvline(limites[-1], color=cores_list[-1], lw=1.2, linestyle="--", alpha=0.7)

    ax.set_xlabel("CBR (%)"); ax.set_ylabel("Frequência")
    ax.legend()
    plt.tight_layout(); plots.save(plt.gcf(), "distribuicao_quintis", "random_forest_quintis")


def grafico_previsto_vs_real_grupo(res):
    """
    Painel 1×2: Previsto vs Real para Validação e Teste de um grupo.
    """
    rot = res["rotulo"]
    cor = CORES_QUINTIL[rot]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=PALETTE["fundo"])


    dados = [
        (res["Y_cv_met"],    res["pred_cv"],    res["met_cv"],    "Validação CV"),
        (res["y_teste_met"], res["pred_teste"],  res["met_teste"], "Teste"),
    ]
    for ax, (y_true, y_pred, met, conjunto) in zip(axes, dados):
        ax.set_facecolor(PALETTE["fundo"])
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        lim = [min(y_true.min(), y_pred.min()) * 0.92,
               max(y_true.max(), y_pred.max()) * 1.08]
        ax.scatter(y_true, y_pred, alpha=0.65, color=cor,
                   edgecolors="white", linewidths=0.4, s=55)
        ax.plot(lim, lim, "--", color=PALETTE["laranja"], lw=1.5, label="Ideal")
        ax.set_xlim(lim); ax.set_ylim(lim)

        ax.set_xlabel("Real (CBR %)"); ax.set_ylabel("Previsto (CBR %)")
        ax.legend()
        # Indica meta
        status = "✓ META" if met["r2"] >= META_R2 else "✗ acima da meta"
        cor_meta = PALETTE["verde"] if met["r2"] >= META_R2 else PALETTE["vermelho"]
        ax.text(0.98, 0.05, status, transform=ax.transAxes, ha="right",
                fontsize=10, fontweight="bold", color=cor_meta,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFFFFF", edgecolor="#E2E8F0", alpha=0.93))

    plt.tight_layout(); plots.save(plt.gcf(), "previsto_vs_real_grupo", "random_forest_quintis")


def grafico_residuos_grupo(res):
    """
    Resíduos (Real − Previsto) para Validação e Teste de um grupo.
    """
    rot = res["rotulo"]
    cor = CORES_QUINTIL[rot]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=PALETTE["fundo"])


    dados = [
        (res["Y_cv_met"],    res["pred_cv"],    "Validação CV"),
        (res["y_teste_met"], res["pred_teste"], "Teste"),
    ]
    for ax, (y_true, y_pred, conjunto) in zip(axes, dados):
        ax.set_facecolor(PALETTE["fundo"])
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        res_arr = y_true - y_pred
        ax.scatter(y_pred, res_arr, alpha=0.65, color=cor,
                   edgecolors="white", linewidths=0.4, s=50)
        ax.axhline(0, color=PALETTE["laranja"], lw=1.5, linestyle="--")
        # Banda ±MAE
        mae = mean_absolute_error(y_true, y_pred)
        ax.axhspan(-mae, mae, alpha=0.08, color=cor, label=f"±MAE ({mae:.2f})")

        ax.set_xlabel("Previsto (CBR %)"); ax.set_ylabel("Resíduo (Real − Previsto)")
        ax.legend()

    plt.tight_layout(); plots.save(plt.gcf(), "residuos_grupo", "random_forest_quintis")


def grafico_busca_grupo(res):
    """
    Evolução da busca de hiperparâmetros para um grupo.
    Mostra MSE cumulativo mínimo + top 10 configurações.
    """
    rot        = res["rotulo"]
    cor        = CORES_QUINTIL[rot]
    search     = res["search"]
    mse_iter   = -search.cv_results_["mean_test_score"]
    mse_std    = search.cv_results_["std_test_score"]
    mse_acum   = np.minimum.accumulate(mse_iter)
    iters      = np.arange(1, len(mse_iter) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), facecolor=PALETTE["fundo"])


    # Painel esquerdo: curva de melhora cumulativa
    ax = axes[0]
    ax.set_facecolor(PALETTE["fundo"])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.plot(iters, mse_acum, color=cor, lw=2, label="MSE CV mínimo acumulado")

    melhoras = [i for i in range(1, len(mse_iter))
                if mse_acum[i] < mse_acum[i-1] * 0.95]
    if melhoras:
        ax.scatter([i + 1 for i in melhoras],
                   [mse_acum[i] for i in melhoras],
                   marker="*", s=130, color=PALETTE["laranja"], zorder=5,
                   label="★ Melhora ≥ 5%")

    ax.axhline(mse_acum[-1], color=PALETTE["verde"], lw=1.2, linestyle="--",
               label=f"Melhor MSE = {mse_acum[-1]:.4f}")
    # Sem linha de meta: este eixo está na escala log1p da busca, e a meta
    # vive na escala original do CBR.
    ax.set_xlabel("Iteração"); ax.set_ylabel("MSE CV (mínimo acumulado)")

    ax.legend()

    # Painel direito: top 10 configurações com barras de erro
    ax2 = axes[1]
    ax2.set_facecolor(PALETTE["fundo"])
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    top_idx = np.argsort(mse_iter)[:10]
    top_mse = mse_iter[top_idx]
    top_std = mse_std[top_idx]
    cores_top = [PALETTE["laranja"] if i == 0 else cor for i in range(len(top_idx))]
    bars = ax2.bar(range(len(top_idx)), top_mse, color=cores_top,
                   edgecolor="white", alpha=0.85)
    ax2.errorbar(range(len(top_idx)), top_mse, yerr=top_std,
                 fmt="none", color="#374151", capsize=3, lw=1.0)
    # Sem linha de meta: eixo na escala log1p da busca.
    for bar, val in zip(bars, top_mse):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax2.set_xlabel("Ranking"); ax2.set_ylabel("MSE CV")
    ax2.legend()

    plt.tight_layout(); plots.save(plt.gcf(), "busca_grupo", "random_forest_quintis")


def grafico_importancia_grupo(res):
    """
    Importância das features (Gini) para um grupo específico.
    Mostra se features diferentes são mais relevantes em cada faixa de CBR.
    """
    rot = res["rotulo"]
    cor = CORES_QUINTIL[rot]
    rf  = res["rf"]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=PALETTE["fundo"])
    ax.set_facecolor(PALETTE["fundo"])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    imp = rf.feature_importances_
    idx = np.argsort(imp)[::-1]

    # Gradiente da cor do quintil
    cores = plt.cm.Blues(np.linspace(0.35, 0.9, len(FEATURES)))[::-1]
    # Sobrepõe a cor do quintil
    cores_g = [cor if i == 0 else cores[i] for i in range(len(FEATURES))]

    ax.barh(range(len(FEATURES)), imp[idx], color=cores_g, edgecolor="white")
    ax.set_yticks(range(len(FEATURES)))
    ax.set_yticklabels([FEATURES_LABELS[i] for i in idx], fontsize=10)
    ax.invert_yaxis()

    ax.set_xlabel("Importância (Gini)")
    plt.tight_layout(); plots.save(plt.gcf(), "importancia_grupo", "random_forest_quintis")


def grafico_comparativo_final(resultados):
    """
    Painel comparativo com TODOS os grupos D1–D5.

    Linha 1: MSE e R² por grupo (barras)
    Linha 2: Scatter previsto vs real de TODOS os grupos sobrepostos
    Linha 3: MAE por grupo + tabela resumo com melhores hiperparâmetros
    """
    rotulos  = [r["rotulo"]             for r in resultados]
    mse_vals = [r["met_teste"]["mse"]   for r in resultados]
    r2_vals  = [r["met_teste"]["r2"]    for r in resultados]
    mae_vals = [r["met_teste"]["mae"]   for r in resultados]
    n_vals   = [r["n_amostras"]         for r in resultados]
    cores    = [CORES_QUINTIL[r]        for r in rotulos]

    fig = plt.figure(figsize=(20, 15), facecolor=PALETTE["fundo"])

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35)

    # ── A) MSE por grupo ─────────────────────────────────────────────────────
    ax_mse = fig.add_subplot(gs[0, 0])
    ax_mse.set_facecolor(PALETTE["fundo"])
    ax_mse.spines["top"].set_visible(False); ax_mse.spines["right"].set_visible(False)
    bars = ax_mse.bar(rotulos, mse_vals, color=cores, edgecolor="white", width=0.6)
    # Uma marca de meta por barra: o limiar equivalente a R² >= META_R2 muda
    # de grupo para grupo, junto com a variância do CBR dentro dele.
    metas_g = [r["meta_mse"] for r in resultados]
    for pos, meta_g in enumerate(metas_g):
        ax_mse.hlines(meta_g, pos - 0.3, pos + 0.3, color=PALETTE["vermelho"],
                      lw=1.5, linestyle="--",
                      label=f"Meta do grupo (R² {META_R2:.2f})" if pos == 0 else None)
    for bar, val in zip(bars, mse_vals):
        ax_mse.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax_mse.set_ylabel("MSE"); ax_mse.legend()

    # ── B) R² por grupo ──────────────────────────────────────────────────────
    ax_r2 = fig.add_subplot(gs[0, 1])
    ax_r2.set_facecolor(PALETTE["fundo"])
    ax_r2.spines["top"].set_visible(False); ax_r2.spines["right"].set_visible(False)
    bars2 = ax_r2.bar(rotulos, r2_vals, color=cores, edgecolor="white", width=0.6)
    ax_r2.axhline(META_R2, color=PALETTE["verde"], lw=1.2, linestyle="--",
                  alpha=0.7, label=f"Meta: R² = {META_R2:.2f}")
    ax_r2.axhline(0.0, color=PALETTE["vermelho"], lw=0.8, linestyle=":", alpha=0.5)
    for bar, val in zip(bars2, r2_vals):
        ax_r2.text(bar.get_x() + bar.get_width() / 2, max(val, 0) + 0.01,
                   f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax_r2.set_ylabel("R²"); ax_r2.legend()

    # ── C) MAE por grupo ─────────────────────────────────────────────────────
    ax_mae = fig.add_subplot(gs[0, 2])
    ax_mae.set_facecolor(PALETTE["fundo"])
    ax_mae.spines["top"].set_visible(False); ax_mae.spines["right"].set_visible(False)
    bars3 = ax_mae.bar(rotulos, mae_vals, color=cores, edgecolor="white", width=0.6)
    for bar, val in zip(bars3, mae_vals):
        ax_mae.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax_mae.set_ylabel("MAE (CBR %)")

    # ── D) Scatter sobrepostos — todos os grupos ──────────────────────────────
    ax_scatter = fig.add_subplot(gs[1, :2])
    ax_scatter.set_facecolor(PALETTE["fundo"])
    ax_scatter.spines["top"].set_visible(False); ax_scatter.spines["right"].set_visible(False)
    todos_real = np.concatenate([r["y_teste_met"] for r in resultados])
    todos_pred = np.concatenate([r["pred_teste"]  for r in resultados])
    lim_g = [todos_real.min() * 0.9, todos_real.max() * 1.1]
    for res_g in resultados:
        ax_scatter.scatter(res_g["y_teste_met"], res_g["pred_teste"],
                           alpha=0.65, color=CORES_QUINTIL[res_g["rotulo"]],
                           edgecolors="white", linewidths=0.3, s=40,
                           label=f"{res_g['rotulo']} (MAE={res_g['met_teste']['mae']:.2f})")
    ax_scatter.plot(lim_g, lim_g, "--", color=PALETTE["laranja"], lw=1.5, label="Ideal")
    ax_scatter.set_xlim(lim_g); ax_scatter.set_ylim(lim_g)

    ax_scatter.set_xlabel("Real (CBR %)"); ax_scatter.set_ylabel("Previsto (CBR %)")
    ax_scatter.legend()

    # ── E) N amostras por grupo ───────────────────────────────────────────────
    ax_n = fig.add_subplot(gs[1, 2])
    ax_n.set_facecolor(PALETTE["fundo"])
    ax_n.spines["top"].set_visible(False); ax_n.spines["right"].set_visible(False)
    bars4 = ax_n.bar(rotulos, n_vals, color=cores, edgecolor="white", width=0.6, alpha=0.85)
    for bar, val in zip(bars4, n_vals):
        ax_n.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                  str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax_n.set_ylabel("Amostras")

    # ── F) Tabela resumo ──────────────────────────────────────────────────────
    ax_tab = fig.add_subplot(gs[2, :])
    ax_tab.axis("off")
    cab = ["Grupo", "Faixa CBR", "N", "Melhor MSE CV",
           "MSE Teste", "RMSE", "MAE", "R²", "Meta", "n_estimators", "max_depth"]
    linhas = []
    for res_g in resultados:
        mt  = res_g["met_teste"]
        bp  = res_g["best_params"]
        ok  = "✓" if mt["r2"] >= META_R2 else "✗"
        linhas.append([
            res_g["rotulo"],
            f"{res_g['cbr_min']:.1f}–{res_g['cbr_max']:.1f}",
            str(res_g["n_amostras"]),
            f"{res_g['melhor_mse_cv']:.4f}",
            f"{mt['mse']:.4f}",
            f"{mt['rmse']:.4f}",
            f"{mt['mae']:.4f}",
            f"{mt['r2']:.4f}",
            ok,
            str(bp.get("n_estimators", "—")),
            str(bp.get("max_depth", "—")),
        ])
    tbl = ax_tab.table(cellText=linhas, colLabels=cab,
                       cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(PALETTE["grade"])
        if r == 0:
            cell.set_facecolor(PALETTE["azul"])
            cell.set_text_props(color="white", fontweight="bold")
        elif c == 8:   # coluna Meta
            val_cel = linhas[r-1][8]
            cell.set_facecolor("#DCFCE7" if val_cel == "✓" else "#FEE2E2")
        elif r < len(linhas) + 1:
            rot_cel = linhas[r-1][0]
            base = CORES_QUINTIL.get(rot_cel, "#FFFFFF")
            # Versão muito clara da cor do quintil
            cell.set_facecolor(base + "18")


    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plots.save(plt.gcf(), "comparativo_final", "random_forest_quintis")


def grafico_importancia_comparativa(resultados):
    """
    Mapa de calor comparando a importância de cada feature entre grupos.
    Revela quais variáveis são mais relevantes em diferentes faixas de CBR.
    """
    n_feat = len(FEATURES)
    matriz = np.zeros((5, n_feat))
    rotulos_g = []

    for i, res_g in enumerate(resultados):
        matriz[i] = res_g["rf"].feature_importances_
        rotulos_g.append(res_g["rotulo"])

    fig, ax = plt.subplots(figsize=(14, 5), facecolor=PALETTE["fundo"])
    ax.set_facecolor(PALETTE["fundo"])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    im = ax.imshow(matriz, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(n_feat))
    ax.set_xticklabels(FEATURES_LABELS, rotation=35, ha="right", fontsize=9.5)
    ax.set_yticks(range(len(rotulos_g)))
    ax.set_yticklabels([f"{r}\n{resultados[i]['cbr_min']:.0f}–{resultados[i]['cbr_max']:.0f}%"
                        for i, r in enumerate(rotulos_g)], fontsize=10)

    for i in range(len(rotulos_g)):
        for j in range(n_feat):
            ax.text(j, i, f"{matriz[i,j]:.3f}", ha="center", va="center",
                    fontsize=9, color="white" if matriz[i,j] > 0.15 else "black")

    plt.colorbar(im, ax=ax, shrink=0.8, label="Importância (Gini)")

    plt.tight_layout(); plots.save(plt.gcf(), "importancia_comparativa", "random_forest_quintis")


# ═════════════════════════════════════════════
# FLUXO PRINCIPAL
# ═════════════════════════════════════════════
print("=" * 60)
print("  RANDOM FOREST — DIVISÃO POR QUINTIS DE CBR (D1–D5)")
print(f"  META: R² >= {META_R2:.2f} por grupo")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. CARREGAMENTO
# ─────────────────────────────────────────────
print("\n[1/4] Carregando dados...")

df = pd.read_csv(CAMINHO_DADOS)
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

Y_orig = df[COLUNA_ALVO].values.ravel()   # CBR em escala original
X_full = df[FEATURES].values
Y_log  = np.log1p(Y_orig)                 # CBR em escala log

print(f"     Dataset: {len(Y_orig)} amostras")
print(f"     CBR: [{Y_orig.min():.1f}, {Y_orig.max():.1f}]  "
      f"média={Y_orig.mean():.1f}  mediana={np.median(Y_orig):.1f}")

# ─────────────────────────────────────────────
# 2. DEFINIÇÃO DOS QUINTIS
# ─────────────────────────────────────────────
print("\n[2/4] Calculando fronteiras dos quintis...")

percentis = [0, 20, 40, 60, 80, 100]
limites   = np.percentile(Y_orig, percentis)

# Ajuste: inclui o valor mínimo exato no D1 e o máximo exato no D5
limites[0]  = Y_orig.min() - 1e-6
limites[-1] = Y_orig.max() + 1e-6

rotulos_quintis = ["D1", "D2", "D3", "D4", "D5"]
mascaras        = []
for i in range(5):
    mask = (Y_orig > limites[i]) & (Y_orig <= limites[i+1])
    mascaras.append(mask)
    n = mask.sum()
    print(f"  {rotulos_quintis[i]}: {n:3d} amostras  "
          f"CBR [{Y_orig[mask].min():.1f}–{Y_orig[mask].max():.1f}]")

# Gráfico de distribuição antes do treino
grafico_distribuicao_quintis(Y_orig, limites, rotulos_quintis)

# ─────────────────────────────────────────────
# 3. TREINO POR GRUPO
# ─────────────────────────────────────────────
print("\n[3/4] Treinando um modelo por quintil...")

resultados = []
for rot, mask in zip(rotulos_quintis, mascaras):
    res = treinar_quintil(rot, mask, X_full, Y_orig, Y_log)
    if res is not None:
        resultados.append(res)
        # Gráficos individuais do grupo
        grafico_previsto_vs_real_grupo(res)
        grafico_residuos_grupo(res)
        grafico_busca_grupo(res)
        grafico_importancia_grupo(res)

# ─────────────────────────────────────────────
# 4. COMPARATIVO FINAL
# ─────────────────────────────────────────────
print("\n[4/4] Gerando gráficos comparativos...")

grafico_comparativo_final(resultados)
grafico_importancia_comparativa(resultados)

# Resumo no terminal
print("\n" + "=" * 60)
print("  RESUMO FINAL — TESTE")
print("=" * 60)
for res_g in resultados:
    mt  = res_g["met_teste"]
    ok  = "✓ META ATINGIDA" if mt["r2"] >= META_R2 else "✗ acima da meta"
    print(f"  {res_g['rotulo']}  CBR {res_g['cbr_min']:.1f}–{res_g['cbr_max']:.1f}%  "
          f"MSE={mt['mse']:.4f}  R²={mt['r2']:.4f}  [{ok}]")
print("=" * 60)
print(f"\n  Modelos salvos em: {OUTPUT_DIR}/")
print("  Processo concluído com sucesso!")

# Precisão do conjunto de modelos como um todo: as previsões de teste de
# todas as faixas juntas. A média dos R² de cada faixa não serviria — cada
# faixa mede uma amostra de tamanho diferente, e nenhuma delas sozinha cobre
# o intervalo de CBR que os modelos únicos enfrentam.
if resultados:
    todos_y = np.concatenate([r["y_teste_met"] for r in resultados])
    todos_pred = np.concatenate([r["pred_teste"] for r in resultados])
    met_todas = metricas(todos_y, todos_pred, "Teste — todas as faixas juntas")
    scoreboard.record("random_forest_quintis", "Árvore Aleatória — quintis D1–D5",
                      met_todas, faixas=len(resultados))

    # Veredito final: o que os números acima significam para quem vai usar o
    # modelo. Fica por último de propósito — é a linha que alguém lê ao voltar
    # ao terminal depois de um treino longo.
    metrics.report(met_todas,
                   notes=[f"Avaliação das {len(resultados)} faixas juntas; "
                          "cada faixa tem o próprio modelo."])
