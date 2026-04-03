"""
Modelo Random Forest — 10 Variáveis Originais (sem engenharia de features)
Meta: MSE < 0.780 no conjunto de teste.

Estrategias aplicadas:
  1. Apenas as 10 variáveis originais do banco de dados
  2. Ensemble — VotingRegressor (RF + GradientBoosting + ExtraTrees)
  3. Busca extensa — 150 iteracoes, 10-fold CV, scoring=neg_MSE
  4. Retreino no conjunto treino+validacao apos tuning
  5. Metricas completas: MSE, RMSE, MAE, R²
  6. Alerta se MSE <= 1
  7. Graficos individuais e padronizados
"""

# ─────────────────────────────────────────────
# IMPORTAÇÕES
# ─────────────────────────────────────────────
import os                                        # Manipulação de caminhos e criação de pastas
import warnings                                  # Supressão de avisos não críticos do Python
import numpy as np                               # Operações numéricas e arrays multidimensionais
import pandas as pd                              # Leitura do CSV e manipulação tabular
import matplotlib.pyplot as plt                  # Geração de todos os gráficos

from joblib import dump                          # Serialização do modelo e scaler em arquivos .joblib
from sklearn.preprocessing import MinMaxScaler  # Normaliza features para o intervalo [0, 1]
from sklearn.model_selection import (
    train_test_split,                            # Divisão estratificada dos dados em conjuntos
    RandomizedSearchCV,                          # Busca aleatória de hiperparâmetros com CV interno
    KFold,                                       # Estratégia de validação cruzada K-fold
)
from sklearn.ensemble import (
    RandomForestRegressor,                       # Floresta aleatória: média de N árvores de decisão
    GradientBoostingRegressor,                   # Boosting: cada árvore corrige os erros da anterior
    ExtraTreesRegressor,                         # Variante do RF com splits totalmente aleatórios
    VotingRegressor,                             # Combina previsões de múltiplos modelos pela média
)
from sklearn.metrics import (
    mean_squared_error,                          # MSE: penaliza erros grandes quadraticamente
    mean_absolute_error,                         # MAE: média dos erros absolutos, robusto a outliers
    r2_score,                                    # R²: proporção da variância explicada pelo modelo
)

warnings.filterwarnings("ignore")               # Remove avisos desnecessários do terminal

# ─────────────────────────────────────────────
# CONFIGURAÇÕES GLOBAIS
# ─────────────────────────────────────────────

CAMINHO_DADOS = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "dados_processados_1.csv"
))

COLUNA_ALVO  = "CBR "           # Nome exato da coluna alvo no CSV (atenção ao espaço)
SEED         = 42               # Semente global para reprodutibilidade
TEST_SIZE    = 0.20             # 20% do dataset reservado para teste final
VAL_SIZE     = 0.15             # 15% do restante para validação
N_ITER_BUSCA = 150              # Combinações testadas na busca de hiperparâmetros
CV_FOLDS     = 10               # 10-fold CV para estimativa estável
META_MSE     = 0.780            # Objetivo de MSE no teste final

# Pasta de saída do modelo e scaler
OUTPUT_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Modelo_salvo_RF"
))
os.makedirs(OUTPUT_DIR, exist_ok=True)          # Cria a pasta se não existir

# Pesos amostrais (opcional)
USE_WEIGHTS = False             # False = todas as amostras com peso igual
THRESHOLD   = None
W_MINOR     = 3.0
W_MAJOR     = 1.0

# ─────────────────────────────────────────────
# COLUNAS ESPERADAS NO CSV (10 features + alvo)
# ─────────────────────────────────────────────
# Lista das 10 variáveis de entrada originais do banco de dados.
# Nenhuma feature derivada é criada — o modelo trabalha diretamente
# com o que foi medido em laboratório.
COLUNAS_ENTRADA = [
    "25.4mm",          # Passante acumulado na peneira 25,4 mm (%)
    "9.5mm",           # Passante acumulado na peneira 9,5 mm (%)
    "4.8mm",           # Passante acumulado na peneira 4,8 mm (%)
    "2.0mm",           # Passante acumulado na peneira 2,0 mm (%)
    "0.42mm",          # Passante acumulado na peneira 0,42 mm (%)
    "0.076mm",         # Passante acumulado na peneira 0,076 mm (%)
    "LL",              # Limite de Liquidez (%)
    "IP",              # Índice de Plasticidade (%)
    "Umidade Ótima",   # Umidade ótima de compactação Proctor (%)
    "Densidade máxima",# Densidade seca máxima de compactação (kg/m³)
]

# ─────────────────────────────────────────────
# PALETA DE CORES GLOBAL
# ─────────────────────────────────────────────
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
    Remove espaços extras dos nomes de colunas e aplica mapeamento de
    variações conhecidas para os nomes padrão esperados pelo modelo.

    Resolve o problema de KeyError causado por nomes levemente diferentes
    no CSV (ex: "LL " com espaço vs "LL", "Wot" vs "Umidade Ótima").

    Args:
        df: DataFrame recém-carregado do CSV.

    Returns:
        DataFrame com colunas renomeadas para o padrão do modelo.
    """
    df.columns = df.columns.str.strip()         # Remove espaços no início e fim de cada nome

    mapa = {
        # Variações de granulometria com vírgula ou prefixo P
        "25,4mm": "25.4mm",  "9,5mm": "9.5mm",  "4,8mm": "4.8mm",
        "2,0mm":  "2.0mm",   "0,42mm": "0.42mm", "0,076mm": "0.076mm",
        "P25.4":  "25.4mm",  "P9.5":  "9.5mm",   "P4.8":   "4.8mm",
        "P2.0":   "2.0mm",   "P0.42": "0.42mm",  "P0.076": "0.076mm",

        # Variações de Limite de Liquidez
        "Ll": "LL", "ll": "LL", "L.L": "LL", "L.L.": "LL",
        "Limite de Liquidez": "LL",

        # Variações de Índice de Plasticidade
        "Ip": "IP", "ip": "IP", "I.P": "IP", "I.P.": "IP",
        "Índice de Plasticidade": "IP", "Indice de Plasticidade": "IP",

        # Variações de Umidade Ótima
        "Wot": "Umidade Ótima", "wot": "Umidade Ótima", "W_ot": "Umidade Ótima",
        "Umidade otima": "Umidade Ótima", "Umidade Otima": "Umidade Ótima",
        "Umidade ótima": "Umidade Ótima", "w_ot": "Umidade Ótima",

        # Variações de Densidade Seca Máxima
        "Densidade Maxima": "Densidade máxima", "Densidade Máxima": "Densidade máxima",
        "densidade máxima": "Densidade máxima", "densidade maxima": "Densidade máxima",
        "d_max": "Densidade máxima", "Dmax": "Densidade máxima",
        "γdmax": "Densidade máxima", "ydmax": "Densidade máxima",

        # Variações do alvo CBR
        "CBR": "CBR ", "cbr": "CBR ", "Cbr": "CBR ",
    }

    df = df.rename(columns=mapa)                # Aplica o mapeamento de variações

    # Verifica se todas as colunas esperadas (10 entradas + alvo) estão presentes
    colunas_esperadas = COLUNAS_ENTRADA + [COLUNA_ALVO]
    faltando = [c for c in colunas_esperadas if c not in df.columns]
    if faltando:
        print("\n  AVISO: Colunas não encontradas após mapeamento:")
        for c in faltando:
            print(f"    - '{c}'")
        print("\n  Colunas reais no CSV:")
        for c in df.columns:
            print(f"    - '{c}'")
        raise SystemExit(1)

    return df


def metricas(y_true: np.ndarray, y_pred: np.ndarray, nome: str) -> dict:
    """
    Calcula MSE, RMSE, MAE e R² e imprime no terminal.

    Args:
        y_true: Valores reais do alvo.
        y_pred: Valores previstos pelo modelo.
        nome:   Rótulo do conjunto avaliado.

    Returns:
        Dicionário com todas as métricas.
    """
    mse  = mean_squared_error(y_true, y_pred)   # Penaliza erros grandes quadraticamente
    rmse = np.sqrt(mse)                          # Mesma unidade do alvo (%)
    mae  = mean_absolute_error(y_true, y_pred)  # Média dos erros absolutos
    r2   = r2_score(y_true, y_pred)             # Proporção da variância explicada

    print(f"\n  [{nome}]")
    print(f"    MSE  : {mse:.4f}")
    if mse <= 1:
        print(f"    ATENCAO: MSE={mse:.4f} <= 1 — verifique escala ou data leakage.")
    print(f"    RMSE : {rmse:.4f}")
    print(f"    MAE  : {mae:.4f}")
    print(f"    R²   : {r2:.4f}")
    return dict(nome=nome, mse=mse, rmse=rmse, mae=mae, r2=r2)


# ─────────────────────────────────────────────
# FUNÇÕES DE GRÁFICO
# ─────────────────────────────────────────────

def grafico_previsto_vs_real_validacao(Y_val, pred_val_ens, met_val):
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=PALETTE["fundo"])
    ax.set_facecolor(PALETTE["fundo"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    lim = [min(Y_val.min(), pred_val_ens.min()) * 0.95,
           max(Y_val.max(), pred_val_ens.max()) * 1.05]
    ax.scatter(Y_val, pred_val_ens, alpha=0.6, color=PALETTE["azul"],
               edgecolors="white", linewidths=0.4, s=50)
    ax.plot(lim, lim, "--", color=PALETTE["laranja"], lw=1.5, label="Ideal")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title("Previsto vs Real — Validação", fontweight="bold")
    ax.set_xlabel("Real"); ax.set_ylabel("Previsto")
    ax.legend(framealpha=0)
    ax.text(0.05, 0.92, f"R² = {met_val['r2']:.4f}", transform=ax.transAxes,
            fontsize=10, color=PALETTE["azul"], fontweight="bold")
    plt.tight_layout(); plt.show()


def grafico_previsto_vs_real_teste(y_teste, pred_teste_ens, met_teste):
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=PALETTE["fundo"])
    ax.set_facecolor(PALETTE["fundo"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    lim = [min(y_teste.min(), pred_teste_ens.min()) * 0.95,
           max(y_teste.max(), pred_teste_ens.max()) * 1.05]
    ax.scatter(y_teste, pred_teste_ens, alpha=0.6, color=PALETTE["verde"],
               edgecolors="white", linewidths=0.4, s=50)
    ax.plot(lim, lim, "--", color=PALETTE["laranja"], lw=1.5, label="Ideal")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title("Previsto vs Real — Teste", fontweight="bold")
    ax.set_xlabel("Real"); ax.set_ylabel("Previsto")
    ax.legend(framealpha=0)
    ax.text(0.05, 0.92, f"R² = {met_teste['r2']:.4f}", transform=ax.transAxes,
            fontsize=10, color=PALETTE["verde"], fontweight="bold")
    cor_meta = PALETTE["verde"] if met_teste["mse"] < META_MSE else PALETTE["laranja"]
    ax.text(0.05, 0.83, f"MSE = {met_teste['mse']:.4f}", transform=ax.transAxes,
            fontsize=9, color=cor_meta, fontweight="bold")
    plt.tight_layout(); plt.show()


def grafico_tabela_metricas(met_val, met_teste):
    fig, ax = plt.subplots(figsize=(6, 3), facecolor=PALETTE["fundo"])
    ax.axis("off")
    table_data = [
        ["Métrica", "Validação", "Teste"],
        ["MSE",  f"{met_val['mse']:.4f}",  f"{met_teste['mse']:.4f}"],
        ["RMSE", f"{met_val['rmse']:.4f}", f"{met_teste['rmse']:.4f}"],
        ["MAE",  f"{met_val['mae']:.4f}",  f"{met_teste['mae']:.4f}"],
        ["R²",   f"{met_val['r2']:.4f}",   f"{met_teste['r2']:.4f}"],
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
    ax.set_title("Comparativo de Métricas — Ensemble", fontweight="bold", pad=14)
    plt.tight_layout(); plt.show()


def grafico_residuos_validacao(Y_val, pred_val_ens):
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=PALETTE["fundo"])
    ax.set_facecolor(PALETTE["fundo"])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    residuos = Y_val - pred_val_ens
    ax.scatter(pred_val_ens, residuos, alpha=0.6, color=PALETTE["azul"],
               edgecolors="white", linewidths=0.4, s=50)
    ax.axhline(0, color=PALETTE["laranja"], lw=1.5, linestyle="--")
    ax.set_title("Resíduos — Validação", fontweight="bold")
    ax.set_xlabel("Previsto"); ax.set_ylabel("Resíduo (Real - Previsto)")
    plt.tight_layout(); plt.show()


def grafico_residuos_teste(y_teste, pred_teste_ens):
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=PALETTE["fundo"])
    ax.set_facecolor(PALETTE["fundo"])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    residuos = y_teste - pred_teste_ens
    ax.scatter(pred_teste_ens, residuos, alpha=0.6, color=PALETTE["verde"],
               edgecolors="white", linewidths=0.4, s=50)
    ax.axhline(0, color=PALETTE["laranja"], lw=1.5, linestyle="--")
    ax.set_title("Resíduos — Teste", fontweight="bold")
    ax.set_xlabel("Previsto"); ax.set_ylabel("Resíduo (Real - Previsto)")
    plt.tight_layout(); plt.show()


def grafico_comparativo_mse(met_rf, met_gb, met_et, met_teste):
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=PALETTE["fundo"])
    ax.set_facecolor(PALETTE["fundo"])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    nomes_mod   = ["Random\nForest", "Gradient\nBoosting", "Extra\nTrees", "Ensemble"]
    mse_valores = [met_rf["mse"], met_gb["mse"], met_et["mse"], met_teste["mse"]]
    cor_ensemble = PALETTE["laranja"] if met_teste["mse"] >= META_MSE else "#15803D"
    cores_bar = [PALETTE["azul"], PALETTE["roxo"], PALETTE["verde"], cor_ensemble]
    bars = ax.bar(nomes_mod, mse_valores, color=cores_bar, edgecolor="white", width=0.5)
    ax.axhline(META_MSE, color="red", lw=1.5, linestyle="--",
               label=f"Meta MSE = {META_MSE}")
    for bar, val in zip(bars, mse_valores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title("MSE por Modelo — Teste Final", fontweight="bold")
    ax.set_ylabel("MSE"); ax.legend(framealpha=0)
    plt.tight_layout(); plt.show()


def grafico_importancia_features(rf_otimizado, et_otimizado, feature_names):
    """
    Importância das 10 features originais (média Gini entre RF e ExtraTrees).
    Permite identificar quais variáveis de laboratório mais influenciam o CBR.
    """
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=PALETTE["fundo"])
    ax.set_facecolor(PALETTE["fundo"])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    importancias_media = (
        rf_otimizado.feature_importances_ + et_otimizado.feature_importances_
    ) / 2                                        # Média suaviza vieses individuais de cada modelo

    top_n = min(10, len(feature_names))          # Máximo de 10 features (total disponível)
    idx   = np.argsort(importancias_media)[-top_n:][::-1]  # Ordena do mais para o menos importante
    cores = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))[::-1]  # Gradiente: mais escuro = mais importante

    ax.barh(range(top_n), importancias_media[idx], color=cores, edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in idx], fontsize=10)
    ax.invert_yaxis()                            # Feature mais importante no topo
    ax.set_title(f"Importância das {top_n} Features Originais (média RF + ET)",
                 fontweight="bold")
    ax.set_xlabel("Importância (Gini)")
    ax.grid(color=PALETTE["grade"], linewidth=0.8)
    plt.tight_layout(); plt.show()


# ═════════════════════════════════════════════
# FLUXO PRINCIPAL
# ═════════════════════════════════════════════

print("=" * 60)
print("  RANDOM FOREST — 10 VARIÁVEIS ORIGINAIS")
print(f"  META: MSE < {META_MSE}")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. CARREGAMENTO DOS DADOS
# ─────────────────────────────────────────────
# DIFERENÇA DO ORIGINAL: sem engenharia de features.
# O DataFrame carregado é usado diretamente — apenas normalização de nomes de colunas.
print("\n[1/6] Carregando dados...")

try:
    df = pd.read_csv(CAMINHO_DADOS)             # Lê o CSV bruto
    print(f"     Linhas: {df.shape[0]}  |  Colunas brutas: {df.shape[1]}")
except FileNotFoundError:
    print(f"ERRO: Arquivo nao encontrado em '{CAMINHO_DADOS}'.")
    raise SystemExit(1)

df = normalizar_colunas(df)                     # Padroniza nomes de colunas

# Seleciona apenas as 10 colunas de entrada + coluna alvo
# Descarta qualquer coluna extra que possa existir no CSV
colunas_usar = COLUNAS_ENTRADA + [COLUNA_ALVO]
df = df[colunas_usar]                           # Filtra para o subconjunto exato de colunas

Y            = df[COLUNA_ALVO].values.ravel()   # Alvo: array 1D de CBR (%)
X            = df[COLUNAS_ENTRADA]              # Features: DataFrame com as 10 colunas originais
feature_names = X.columns.tolist()             # Nomes para o gráfico de importância
X            = X.values                         # Converte para array NumPy (exigido pelo sklearn)

print(f"     Features de entrada : {len(feature_names)}")
print(f"     Feature names       : {feature_names}")
print(f"     Amostras            : {X.shape[0]}")

# ─────────────────────────────────────────────
# 2. DIVISÃO E NORMALIZAÇÃO
# ─────────────────────────────────────────────
print("\n[2/6] Dividindo e normalizando dados...")

# Split 1: separa 20% para teste final — esses dados nunca entram no tuning
X_tv, X_teste, Y_tv, y_teste = train_test_split(
    X, Y, test_size=TEST_SIZE, random_state=SEED
)

# Split 2: dos 80% restantes, 15% para validação — usado para comparar modelos
X_treino, X_val, Y_treino, Y_val = train_test_split(
    X_tv, Y_tv, test_size=VAL_SIZE, random_state=SEED
)

# MinMaxScaler ajustado APENAS no treino — evita data leakage
# (os limites do teste não podem influenciar a escala do treino)
scaler      = MinMaxScaler()
X_treino_n  = scaler.fit_transform(X_treino)   # Aprende min/max e normaliza o treino
X_val_n     = scaler.transform(X_val)          # Aplica a escala do treino na validação
X_teste_n   = scaler.transform(X_teste)        # Aplica a escala do treino no teste
X_tv_n      = scaler.transform(X_tv)           # Treino+val normalizado para o retreino final

print(f"     Treino: {X_treino_n.shape[0]}  |  Validação: {X_val_n.shape[0]}  |  Teste: {X_teste_n.shape[0]}")
print(f"     Features: {X_treino_n.shape[1]}")

# Pesos amostrais (opcional)
if USE_WEIGHTS and THRESHOLD is not None:
    sample_weights = np.where(Y_treino > THRESHOLD, W_MINOR, W_MAJOR).astype(np.float32)
    print(f"     Pesos amostrais ativos — threshold={THRESHOLD}")
else:
    sample_weights = None

# ─────────────────────────────────────────────
# 3. ESPAÇOS DE HIPERPARÂMETROS
# ─────────────────────────────────────────────
print("\n[3/6] Definindo espacos de hiperparametros...")

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
print(f"\n[4/6] Buscando melhores hiperparametros ({N_ITER_BUSCA} iter, {CV_FOLDS}-fold CV)...")

kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)  # 10-fold com embaralhamento

from sklearn.pipeline import Pipeline as _Pipe  # Pipeline local para evitar conflito de namespace

# ── Busca RF ─────────────────────────────────────────────────────────────────
print("  Buscando RF...")
pipe_rf   = _Pipe([("rf", RandomForestRegressor(random_state=SEED, n_jobs=-1))])
search_rf = RandomizedSearchCV(pipe_rf, param_rf, n_iter=N_ITER_BUSCA,
                               scoring="neg_mean_squared_error",
                               cv=kf, random_state=SEED, n_jobs=-1, verbose=0)
search_rf.fit(X_treino_n, Y_treino)
best_rf_params = {k.replace("rf__", ""): v for k, v in search_rf.best_params_.items()}
print(f"     Melhor MSE CV (RF): {-search_rf.best_score_:.4f}")

# ── Busca GradientBoosting ────────────────────────────────────────────────────
print("  Buscando GradientBoosting...")
pipe_gb   = _Pipe([("gb", GradientBoostingRegressor(random_state=SEED))])
search_gb = RandomizedSearchCV(pipe_gb, param_gb, n_iter=N_ITER_BUSCA,
                               scoring="neg_mean_squared_error",
                               cv=kf, random_state=SEED, n_jobs=-1, verbose=0)
search_gb.fit(X_treino_n, Y_treino)
best_gb_params = {k.replace("gb__", ""): v for k, v in search_gb.best_params_.items()}
print(f"     Melhor MSE CV (GB): {-search_gb.best_score_:.4f}")

# ── Busca ExtraTrees ──────────────────────────────────────────────────────────
print("  Buscando ExtraTrees...")
pipe_et   = _Pipe([("et", ExtraTreesRegressor(random_state=SEED, n_jobs=-1))])
search_et = RandomizedSearchCV(pipe_et, param_et, n_iter=N_ITER_BUSCA,
                               scoring="neg_mean_squared_error",
                               cv=kf, random_state=SEED, n_jobs=-1, verbose=0)
search_et.fit(X_treino_n, Y_treino)
best_et_params = {k.replace("et__", ""): v for k, v in search_et.best_params_.items()}
print(f"     Melhor MSE CV (ET): {-search_et.best_score_:.4f}")

# ─────────────────────────────────────────────
# 5. TREINAMENTO FINAL — ENSEMBLE
# ─────────────────────────────────────────────
print("\n[5/6] Treinando ensemble final...")

rf_otimizado = RandomForestRegressor(**best_rf_params, random_state=SEED, n_jobs=-1)
gb_otimizado = GradientBoostingRegressor(**best_gb_params, random_state=SEED)
et_otimizado = ExtraTreesRegressor(**best_et_params,  random_state=SEED, n_jobs=-1)

# VotingRegressor: média das previsões dos 3 modelos
# Erros descorrelacionados entre RF, GB e ET se cancelam → menor variância total
ensemble = VotingRegressor(estimators=[
    ("rf", rf_otimizado),
    ("gb", gb_otimizado),
    ("et", et_otimizado),
])

fit_params_final = {}
if sample_weights is not None:
    sw_tv = np.where(Y_tv > THRESHOLD, W_MINOR, W_MAJOR).astype(np.float32)
    fit_params_final["sample_weight"] = sw_tv

ensemble.fit(X_tv_n, Y_tv, **fit_params_final)  # Treina no conjunto treino+validação combinado
print("     Ensemble treinado em treino + validacao.")

# ─────────────────────────────────────────────
# 6. AVALIAÇÃO
# ─────────────────────────────────────────────
print("\n[6/6] Avaliando modelos...")

pred_val_ens   = ensemble.predict(X_val_n)      # Previsão na validação
pred_teste_ens = ensemble.predict(X_teste_n)    # Previsão no teste (avaliação final)

met_val   = metricas(Y_val,   pred_val_ens,   "Validacao  — Ensemble")
met_teste = metricas(y_teste, pred_teste_ens, "Teste Final — Ensemble")

# Retreina modelos individuais em treino+val para comparação justa
print("\n  --- Modelos Individuais (Teste) ---")
rf_otimizado.fit(X_tv_n, Y_tv)
gb_otimizado.fit(X_tv_n, Y_tv)
et_otimizado.fit(X_tv_n, Y_tv)

met_rf = metricas(y_teste, rf_otimizado.predict(X_teste_n), "Random Forest")
met_gb = metricas(y_teste, gb_otimizado.predict(X_teste_n), "GradientBoosting")
met_et = metricas(y_teste, et_otimizado.predict(X_teste_n), "ExtraTrees")

print("\n" + "=" * 60)
if met_teste["mse"] < META_MSE:
    print(f"  META ATINGIDA! MSE = {met_teste['mse']:.4f} < {META_MSE}")
else:
    print(f"  MSE = {met_teste['mse']:.4f}  |  Meta: < {META_MSE}")
print("=" * 60)

# ─────────────────────────────────────────────
# VISUALIZAÇÕES
# ─────────────────────────────────────────────
print("\n--- Gerando Graficos ---")

grafico_previsto_vs_real_validacao(Y_val, pred_val_ens, met_val)
grafico_previsto_vs_real_teste(y_teste, pred_teste_ens, met_teste)
grafico_tabela_metricas(met_val, met_teste)
grafico_residuos_validacao(Y_val, pred_val_ens)
grafico_residuos_teste(y_teste, pred_teste_ens)
grafico_comparativo_mse(met_rf, met_gb, met_et, met_teste)
grafico_importancia_features(rf_otimizado, et_otimizado, feature_names)

print("Graficos gerados com sucesso!")

# ─────────────────────────────────────────────
# SALVAMENTO
# ─────────────────────────────────────────────
dump(ensemble, os.path.join(OUTPUT_DIR, "rf_modelo_final.joblib"))  # Salva o ensemble completo
dump(scaler,   os.path.join(OUTPUT_DIR, "scaler.joblib"))           # Salva o scaler do treino

print(f"\n  Ensemble salvo -> rf_modelo_final.joblib")
print(f"  Scaler  salvo  -> scaler.joblib")
print(f"  Pasta          -> {OUTPUT_DIR}")

print("\n" + "=" * 60)
print("  Processo concluido com sucesso!")
print("=" * 60)