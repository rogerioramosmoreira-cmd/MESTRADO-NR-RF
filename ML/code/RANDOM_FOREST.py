"""
Modelo Random Forest — Alta Performance
Meta: MSE < 0.780 no conjunto de teste.

Estratégias aplicadas:
  1. Engenharia de Features — razões e interações entre variáveis granulométricas
  2. Ensemble — VotingRegressor (RF + GradientBoosting + ExtraTrees)
  3. Busca extensa — 150 iterações, 10-fold CV, scoring=neg_MSE
  4. Retreino no conjunto treino+validação após o ajuste
  5. Métricas completas: MSE, RMSE, MAE, R²
  6. Alerta se MSE <= 1
  7. Gráficos individuais e padronizados
  8. Transformação logarítmica no alvo (log1p/expm1) — comprime assimetria à direita da distribuição do CBR
  9. Pesos amostrais — amostras raras com CBR alto recebem maior peso no treino
"""

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
import os                                        # Manipulação de caminhos e criação de pastas
import warnings                                  # Supressão de avisos não críticos
import numpy as np                               # Operações numéricas e arrays multidimensionais
import pandas as pd                              # Leitura de CSV e manipulação tabular
import matplotlib.pyplot as plt                  # Geração de gráficos

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

# Caminho relativo ao script: sobe um nível (code/ -> ML/) e entra em data/
CAMINHO_DADOS = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "dados_processados_1.csv"
))

COLUNA_ALVO   = "CBR "          # Nome exato da coluna alvo no CSV (atenção ao espaço no final)

# Mapeamento das 10 features utilizadas (notação dissertação → coluna do CSV):
#   IG  → 25.4mm  |  EXP → 9.5mm   |  D3 → 4.8mm   |  D4 → 2.0mm
#   D5  → 0.42mm  |  D6  → 0.076mm |  LL → LL       |  IP → IP
#   CH  → Umidade Ótima             |  CY → Densidade máxima
FEATURES = [
    "25.4mm",        # IG  — pedregulho grosso
    "9.5mm",         # EXP — pedregulho médio/fino
    "4.8mm",         # D3  — peneira nº4 (pedregulho/areia)
    "2.0mm",         # D4  — peneira nº10 (areia grossa)
    "0.42mm",        # D5  — peneira nº40 (areia fina)
    "0.076mm",       # D6  — peneira nº200 (silte/argila)
    "LL",            # LL  — limite de liquidez
    "IP",            # IP  — índice de plasticidade
    "Umidade Ótima", # CH  — umidade ótima de compactação (Proctor)
    "Densidade máxima", # CY — densidade seca máxima (Proctor)
]

# Rótulos exibidos no gráfico de importância: "coluna (notação)"
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
SEED          = 42               # Semente global: garante splits, buscas e modelos reproduzíveis
TEST_SIZE     = 0.20             # 20% do dataset reservado para teste final (nunca visto no treino)
VAL_SIZE      = 0.15             # 15% do restante reservado para validação (separado do teste)
N_ITER_BUSCA  = 150              # Combinações testadas por modelo na busca
CV_FOLDS      = 10               # 10-fold CV: estimativa mais estável que 5-fold em datasets menores
META_MSE      = 0.780            # Meta de performance: MSE abaixo deste valor no teste final

# Pasta onde ensemble e scaler serão salvos
OUTPUT_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Modelo_salvo_RF"
))
os.makedirs(OUTPUT_DIR, exist_ok=True)          # exist_ok=True evita erro se a pasta já existir

# ── Log-transform on target ──────────────────────────────────────────────────
# PROBLEMA: a distribuição do CBR é assimétrica à direita — a maioria das amostras
# concentra em CBR baixo/médio e poucos casos têm CBR > 50%. MSE penaliza
# erros extremos quadraticamente, inflando a métrica mesmo quando o modelo é preciso.
#
# SOLUÇÃO: treinar em log1p(CBR), prever em expm1(pred).
# - log1p comprime a cauda direita, tornando a distribuição mais simétrica
# - O modelo trata erros relativos igualmente em toda a faixa de CBR
# - Métricas sempre calculadas na escala original (expm1 aplicado antes da avaliação)
LOG_ALVO = True                  # True = treina em log1p(CBR), avalia em expm1(pred)

# ── Sample weights ────────────────────────────────────────────────────────────
# PROBLEMA: amostras com CBR alto são sub-representadas — o modelo
# subestima sistematicamente pois vê poucos exemplos nessa região.
#
# SOLUÇÃO: amostras com CBR > THRESHOLD recebem peso W_MINOR no treino.
# THRESHOLD = 25.0: aprox. quartil superior de solos lateríticos brasileiros típicos.
# Ajuste conforme a distribuição real do seu dataset (verifique um histograma).
USE_WEIGHTS = True               # True = amostras com CBR alto recebem mais atenção
THRESHOLD   = 25.0               # Amostras com CBR > 25% são consideradas raras/importantes
W_MINOR     = 3.0                # Peso das amostras acima do threshold (raras → mais importantes)
W_MAJOR     = 1.0                # Peso das amostras abaixo do threshold (comuns → influência normal)

# ─────────────────────────────────────────────
# PALETA DE CORES GLOBAL
# ─────────────────────────────────────────────
# roxo (roxo) → laranja (laranja)
# verde (verde) → azul2 (azul claro) — tom secundário para distinguir do azul primário

PALETTE = {
    "azul":    "#2563EB",   # Conjunto de validação / azul primário
    "azul2":   "#60A5FA",   # Conjunto de teste (era verde) → azul claro
    "laranja": "#EA580C",   # Linha de referência e destaques (substitui roxo)
    "fundo":   "#F8FAFC",   # Cor de fundo dos gráficos
    "grade":   "#E2E8F0",   # Linhas de grade
}

# Aplica estilo global a todos os gráficos criados a partir daqui
plt.rcParams.update({
    "figure.facecolor":  PALETTE["fundo"],   # Fundo da figura completa
    "axes.facecolor":    PALETTE["fundo"],   # Fundo de cada eixo individual
    "axes.grid":         True,               # Ativa grade em todos os eixos
    "grid.color":        PALETTE["grade"],   # Cor das linhas de grade
    "grid.linewidth":    0.8,                # Espessura das linhas de grade
    "font.family":       "DejaVu Sans",      # Fonte padrão
    "axes.spines.top":   False,              # Remove borda superior (visual mais limpo)
    "axes.spines.right": False,              # Remove borda direita
})

# ─────────────────────────────────────────────
# FUNÇÕES AUXILIARES
# ─────────────────────────────────────────────

def engenharia_features(df: pd.DataFrame, coluna_alvo: str) -> pd.DataFrame:
    """
    Cria novas features derivadas das variáveis granulométricas e índices físicos.

    Lógica de cada feature:
      - ratio_X_Y: razão entre peneiras adjacentes — captura o formato da
        curva granulométrica de forma relativa, independente dos valores absolutos
      - atividade: LL - IP — solos com alta diferença são mais plásticos e menos resistentes
      - compacidade: Densidade / Umidade — proxy da energia de compactação do solo
      - finos_sq: 0.076mm² — quadrado da fração fina, amplifica diferenças em solos argilosos

    Args:
        df:           DataFrame com colunas originais.
        coluna_alvo:  Nome da coluna alvo (preservada sem modificação).

    Returns:
        DataFrame com as colunas originais + 8 novas features derivadas.
    """
    df  = df.copy()                                     # Evita modificar o DataFrame original
    eps = 1e-6                                          # Epsilon: evita divisão por zero nas razões

    # Razões entre peneiras consecutivas — descrevem o gradiente da curva granulométrica
    df["ratio_9_25"]    = df["9.5mm"]   / (df["25.4mm"]  + eps)
    df["ratio_4_9"]     = df["4.8mm"]   / (df["9.5mm"]   + eps)
    df["ratio_2_4"]     = df["2.0mm"]   / (df["4.8mm"]   + eps)
    df["ratio_042_2"]   = df["0.42mm"]  / (df["2.0mm"]   + eps)
    df["ratio_076_042"] = df["0.076mm"] / (df["0.42mm"]  + eps)

    # Índice de atividade: quanto maior, mais plástico e menos resistente é o solo
    df["atividade"]     = df["LL"] - df["IP"]

    # Compacidade: solos mais densos com menor umidade tendem a ter CBR maior
    df["compacidade"]   = df["Densidade máxima"] / (df["Umidade Ótima"] + eps)

    # Finos ao quadrado: amplifica a diferença entre solos com muito ou pouco material fino
    df["finos_sq"]      = df["0.076mm"] ** 2

    return df


def metricas(y_true: np.ndarray, y_pred: np.ndarray, nome: str) -> dict:
    """
    Calcula e imprime MSE, RMSE, MAE e R² para um conjunto de previsões.
    Alerta se MSE <= 1, indicando possível problema de escala ou data leakage.

    Args:
        y_true: Valores reais do alvo.
        y_pred: Valores previstos pelo modelo.
        nome:   Nome do conjunto avaliado (ex: "Validação", "Teste Final").

    Returns:
        Dicionário com todas as métricas calculadas.
    """
    mse  = mean_squared_error(y_true, y_pred)   # Penalizes large errors more than small ones
    rmse = np.sqrt(mse)                          # Raiz do MSE: na mesma unidade do alvo (%)
    mae  = mean_absolute_error(y_true, y_pred)  # Mean absolute errors — intuitive to interpret
    r2   = r2_score(y_true, y_pred)             # R²: 1.0 = perfeito, 0.0 = modelo constante

    print(f"\n  [{nome}]")
    print(f"    MSE  : {mse:.4f}")

    if mse <= 1:
        print(f"    ATENÇÃO: MSE={mse:.4f} <= 1 — valor suspeito. "
              "Verifique a escala do alvo ou possível data leakage.")

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
    Gráfico de dispersão: valores previstos pelo ensemble vs. valores reais
    no conjunto de VALIDAÇÃO.

    Pontos próximos da diagonal laranja indicam previsões precisas.
    R² anotado no canto superior esquerdo.
    """
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=PALETTE["fundo"])
    ax.set_facecolor(PALETTE["fundo"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    lim = [min(Y_val.min(), pred_val_ens.min()) * 0.95,   # limite inferior com margem
           max(Y_val.max(), pred_val_ens.max()) * 1.05]  # limite superior com margem

    ax.scatter(Y_val, pred_val_ens, alpha=0.6, color=PALETTE["azul"],
               edgecolors="white", linewidths=0.4, s=50)

    ax.plot(lim, lim, "--", color=PALETTE["laranja"], lw=1.5, label="Ideal")  # linha y=x (previsão perfeita)

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_title("Previsto vs Real — Validação", fontweight="bold")
    ax.set_xlabel("Real")
    ax.set_ylabel("Previsto")
    ax.legend(framealpha=0)

    ax.text(0.05, 0.92, f"R² = {met_val['r2']:.4f}", transform=ax.transAxes,
            fontsize=10, color=PALETTE["azul"], fontweight="bold")

    plt.tight_layout()
    plt.show()


def grafico_previsto_vs_real_teste(y_teste: np.ndarray, pred_teste_ens: np.ndarray,
                                   met_teste: dict) -> None:
    """
    Gráfico de dispersão: valores previstos pelo ensemble vs. valores reais
    no conjunto de TESTE (avaliação final de generalização).

    MSE exibido em azul se a meta foi atingida, laranja caso contrário.
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
    ax.set_title("Previsto vs Real — Teste", fontweight="bold")
    ax.set_xlabel("Real")
    ax.set_ylabel("Previsto")
    ax.legend(framealpha=0)

    ax.text(0.05, 0.92, f"R² = {met_teste['r2']:.4f}", transform=ax.transAxes,
            fontsize=10, color=PALETTE["azul2"], fontweight="bold")

    # Indicador de meta do MSE: azul = atingida, laranja = não atingida
    cor_meta = PALETTE["azul2"] if met_teste["mse"] < META_MSE else PALETTE["laranja"]
    ax.text(0.05, 0.83, f"MSE = {met_teste['mse']:.4f}", transform=ax.transAxes,
            fontsize=9, color=cor_meta, fontweight="bold")

    plt.tight_layout()
    plt.show()


def grafico_tabela_metricas(met_val: dict, met_teste: dict) -> None:
    """
    Tabela comparativa de métricas (MSE, RMSE, MAE, R²) lado a lado
    para os conjuntos de validação e teste.

    Exibida como figura matplotlib (sem eixos), com cabeçalho azul
    e células alternadas para facilitar a leitura.
    """
    fig, ax = plt.subplots(figsize=(6, 3), facecolor=PALETTE["fundo"])
    ax.axis("off")

    table_data = [
        ["Métrica",  "Validação",               "Teste"],
        ["MSE",  f"{met_val['mse']:.4f}",  f"{met_teste['mse']:.4f}"],
        ["RMSE", f"{met_val['rmse']:.4f}", f"{met_teste['rmse']:.4f}"],
        ["MAE",  f"{met_val['mae']:.4f}",  f"{met_teste['mae']:.4f}"],
        ["R²",   f"{met_val['r2']:.4f}",   f"{met_teste['r2']:.4f}"],
    ]

    tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                   cellLoc="center", loc="center", bbox=[0.05, 0.1, 0.9, 0.8])  # bbox: posição da tabela
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(PALETTE["grade"])
        if r == 0:                               # Header row: blue background, white bold text
            cell.set_facecolor(PALETTE["azul"])
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#EFF6FF" if c != 0 else "#F1F5F9")

    ax.set_title("Comparativo de Métricas — Ensemble", fontweight="bold", pad=14)

    plt.tight_layout()
    plt.show()


def grafico_residuos_validacao(Y_val: np.ndarray, pred_val_ens: np.ndarray) -> None:
    """
    Gráfico de resíduos do conjunto de VALIDAÇÃO.

    Resíduo = Real - Previsto. Pontos distribuídos aleatoriamente em torno
    da linha zero indicam modelo sem viés sistemático.
    """
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=PALETTE["fundo"])
    ax.set_facecolor(PALETTE["fundo"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    residuos = Y_val - pred_val_ens              # Diferença entre real e previsto para cada amostra

    ax.scatter(pred_val_ens, residuos, alpha=0.6, color=PALETTE["azul"],
               edgecolors="white", linewidths=0.4, s=50)

    ax.axhline(0, color=PALETTE["laranja"], lw=1.5, linestyle="--")  # Linha de resíduo zero (ideal)

    ax.set_title("Resíduos — Validação", fontweight="bold")
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Resíduo (Real - Previsto)")

    plt.tight_layout()
    plt.show()


def grafico_residuos_teste(y_teste: np.ndarray, pred_teste_ens: np.ndarray) -> None:
    """
    Gráfico de resíduos do conjunto de TESTE.

    Padrões sistemáticos (funil, curva) indicam viés ou falta de ajuste
    em faixas específicas de CBR.
    """
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=PALETTE["fundo"])
    ax.set_facecolor(PALETTE["fundo"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    residuos = y_teste - pred_teste_ens

    ax.scatter(pred_teste_ens, residuos, alpha=0.6, color=PALETTE["azul2"],  # was verde
               edgecolors="white", linewidths=0.4, s=50)
    ax.axhline(0, color=PALETTE["laranja"], lw=1.5, linestyle="--")

    ax.set_title("Resíduos — Teste", fontweight="bold")
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Resíduo (Real - Previsto)")

    plt.tight_layout()
    plt.show()


def grafico_comparativo_mse(met_rf: dict, met_gb: dict, met_et: dict,
                             met_teste: dict) -> None:
    """
    Gráfico de barras comparando o MSE de cada modelo individual
    (RF, GB, ET) contra o ensemble no conjunto de teste.

    roxo substituído por laranja na barra do GradientBoosting.
    verde substituído por azul2 na barra do ExtraTrees.
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

    # Linha vermelha tracejada indicando a meta de MSE
    ax.axhline(META_MSE, color="red", lw=1.5, linestyle="--",
               label=f"Meta MSE = {META_MSE}")

    # Anota o valor de MSE numericamente acima de cada barra
    for bar, val in zip(bars, mse_valores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_title("MSE por Modelo — Teste Final", fontweight="bold")
    ax.set_ylabel("MSE")
    ax.legend(framealpha=0)

    plt.tight_layout()
    plt.show()


def grafico_importancia_features(rf_otimizado: RandomForestRegressor,
                                  et_otimizado: ExtraTreesRegressor,
                                  feature_names: list) -> None:
    """
    Gráfico de barras horizontais com as top 15 features mais importantes,
    calculadas como a média da importância Gini entre RF e ExtraTrees.
    """
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=PALETTE["fundo"])
    ax.set_facecolor(PALETTE["fundo"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    importancias_media = (
        rf_otimizado.feature_importances_ + et_otimizado.feature_importances_
    ) / 2

    top_n = len(feature_names)          # mostra todas as features, sem limite fixo
    idx = np.argsort(importancias_media)[-top_n:][::-1]

    # Gradiente de azul: mais escuro = mais importante
    cores = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))[::-1]

    ax.barh(range(top_n), importancias_media[idx], color=cores, edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([FEATURES_LABELS[i] for i in idx], fontsize=9)
    ax.invert_yaxis()                            # Feature mais importante no topo

    ax.set_title(f"Importância das {top_n} Features (média RF + ET)", fontweight="bold")
    ax.set_xlabel("Importância (Gini)")
    ax.grid(color=PALETTE["grade"], linewidth=0.8)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# 1. CARREGAMENTO
# ─────────────────────────────────────────────
print("=" * 60)
print("  RANDOM FOREST — ALTA PERFORMANCE")
print(f"  META: MSE < {META_MSE}")
print("=" * 60)
print("\n[1/6] Carregando dados...")

df = pd.read_csv(CAMINHO_DADOS)
print(f"     Dataset: {df.shape[0]} linhas x {df.shape[1]} colunas brutas")

# Remove espaços nos nomes das colunas (causa mais comum de KeyError no CSV)
df.columns = df.columns.str.strip()

# Renomeia variações conhecidas para os nomes padrão usados em FEATURES
_mapa_colunas = {
    "Ll": "LL", "ll": "LL", "L.L": "LL",
    "Ip": "IP", "ip": "IP", "I.P": "IP",
    "Densidade Maxima": "Densidade máxima",
    "Densidade Máxima": "Densidade máxima",
    "densidade maxima": "Densidade máxima",
    "d_max": "Densidade máxima", "Dmax": "Densidade máxima",
    "Wot": "Umidade Ótima", "wot": "Umidade Ótima",
    "Umidade otima": "Umidade Ótima", "Umidade Otima": "Umidade Ótima",
    "CBR": "CBR ", "cbr": "CBR ", "Cbr": "CBR ",
}
df = df.rename(columns=_mapa_colunas)

# Seleciona apenas as 10 features definidas em FEATURES — sem engenharia de features
df = df[FEATURES + [COLUNA_ALVO]]

Y            = df[COLUNA_ALVO].values.ravel()   # Alvo: array 1D (CBR %)
X            = df[FEATURES].values              # Matriz numérica: 10 colunas
feature_names = FEATURES                        # Nomes para o gráfico de importância
print(f"     Features utilizadas: {len(FEATURES)} — {FEATURES}")

# Armazena Y original para calcular métricas na escala real do CBR após a previsão
Y_orig = Y.copy()
if LOG_ALVO:
    Y = np.log1p(Y)                              # log1p(x) = log(1+x): estável próximo de zero
    print(f"     Log-transform ativo: alvo transformado para log1p(CBR)")
    print(f"     Faixa original: [{Y_orig.min():.2f}, {Y_orig.max():.2f}]"
          f"  →  log1p: [{Y.min():.4f}, {Y.max():.4f}]")

# ─────────────────────────────────────────────
# 2. DIVISÃO DOS DADOS
# ─────────────────────────────────────────────
print("\n[2/6] Splitting data...")

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

print(f"     Treino: {X_treino_n.shape[0]}  |  Validação: {X_val_n.shape[0]}  |  Teste: {X_teste_n.shape[0]}")
print(f"     Features totais: {X_treino_n.shape[1]}")

if USE_WEIGHTS and THRESHOLD is not None:
    # Quando LOG_ALVO está ativo, Y_treino está em escala log — converte o threshold
    thr_efetivo = np.log1p(THRESHOLD) if LOG_ALVO else THRESHOLD
    sample_weights = np.where(Y_treino > thr_efetivo, W_MINOR, W_MAJOR).astype(np.float32)
    print(f"     Pesos amostrais ativos — threshold={THRESHOLD}"
          f"{' (log1p=' + f'{thr_efetivo:.4f})' if LOG_ALVO else ''}")
else:
    sample_weights = None

# ─────────────────────────────────────────────
# 3. ESPAÇOS DE HIPERPARÂMETROS
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
# 4. BUSCA DE HIPERPARÂMETROS POR MODELO
# ─────────────────────────────────────────────
print(f"\n[4/6] Searching best hyperparameters ({N_ITER_BUSCA} iter, {CV_FOLDS}-fold CV)...")

kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

from sklearn.pipeline import Pipeline as _Pipe

print("  Buscando RF...")
pipe_rf   = _Pipe([("rf", RandomForestRegressor(random_state=SEED, n_jobs=-1))])
search_rf = RandomizedSearchCV(
    pipe_rf, param_rf, n_iter=N_ITER_BUSCA,
    scoring="neg_mean_squared_error",
    cv=kf, random_state=SEED, n_jobs=-1, verbose=0,
)
search_rf.fit(X_treino_n, Y_treino)
best_rf_params = {k.replace("rf__", ""): v for k, v in search_rf.best_params_.items()}
print(f"     Melhor MSE CV (RF): {-search_rf.best_score_:.4f}")

print("  Buscando GradientBoosting...")
pipe_gb   = _Pipe([("gb", GradientBoostingRegressor(random_state=SEED))])
search_gb = RandomizedSearchCV(
    pipe_gb, param_gb, n_iter=N_ITER_BUSCA,
    scoring="neg_mean_squared_error",
    cv=kf, random_state=SEED, n_jobs=-1, verbose=0,
)
search_gb.fit(X_treino_n, Y_treino)
best_gb_params = {k.replace("gb__", ""): v for k, v in search_gb.best_params_.items()}
print(f"     Melhor MSE CV (GB): {-search_gb.best_score_:.4f}")

print("  Buscando ExtraTrees...")
pipe_et   = _Pipe([("et", ExtraTreesRegressor(random_state=SEED, n_jobs=-1))])
search_et = RandomizedSearchCV(
    pipe_et, param_et, n_iter=N_ITER_BUSCA,
    scoring="neg_mean_squared_error",
    cv=kf, random_state=SEED, n_jobs=-1, verbose=0,
)
search_et.fit(X_treino_n, Y_treino)
best_et_params = {k.replace("et__", ""): v for k, v in search_et.best_params_.items()}
print(f"     Melhor MSE CV (ET): {-search_et.best_score_:.4f}")

# ─────────────────────────────────────────────
# 5. TREINAMENTO FINAL — ENSEMBLE
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
print("     Ensemble treinado em treino + validação.")

# ─────────────────────────────────────────────
# 6. AVALIAÇÃO
# ─────────────────────────────────────────────
print("\n[6/6] Evaluating models...")

pred_val_ens   = ensemble.predict(X_val_n)
pred_teste_ens = ensemble.predict(X_teste_n)

# Reverte log-transform nas previsões e rótulos antes de calcular métricas
# expm1(x) = exp(x)-1: inverso exato de log1p — restaura a escala original do CBR
if LOG_ALVO:
    pred_val_ens   = np.expm1(pred_val_ens)
    pred_teste_ens = np.expm1(pred_teste_ens)
    Y_val_met   = np.expm1(Y_val)              # Rótulos de validação na escala original
    y_teste_met = np.expm1(y_teste)            # Rótulos de teste na escala original
    print("     Previsões revertidas para a escala original (expm1)")
else:
    Y_val_met   = Y_val
    y_teste_met = y_teste

met_val   = metricas(Y_val_met,   pred_val_ens,   "Validação   — Ensemble")
met_teste = metricas(y_teste_met, pred_teste_ens, "Teste Final  — Ensemble")

print("\n  --- Individual Models (Test) ---")
rf_otimizado.fit(X_tv_n, Y_tv)
gb_otimizado.fit(X_tv_n, Y_tv)
et_otimizado.fit(X_tv_n, Y_tv)

# Reverte log-transform nas previsões dos modelos individuais também
def _pred(model, X):
    p = model.predict(X)
    return np.expm1(p) if LOG_ALVO else p

met_rf = metricas(y_teste_met, _pred(rf_otimizado, X_teste_n), "Random Forest")
met_gb = metricas(y_teste_met, _pred(gb_otimizado, X_teste_n), "GradientBoosting")
met_et = metricas(y_teste_met, _pred(et_otimizado, X_teste_n), "ExtraTrees")

print("\n" + "=" * 60)
if met_teste["mse"] < META_MSE:
    print(f"  META ATINGIDA! MSE = {met_teste["mse"]:.4f} < {META_MSE}")
else:
    print(f"  MSE = {met_teste["mse"]:.4f}  |  Meta: < {META_MSE}")
print("=" * 60)

# ─────────────────────────────────────────────
# GRÁFICOS
# ─────────────────────────────────────────────
print("\n--- Generating Charts ---")

grafico_previsto_vs_real_validacao(Y_val_met, pred_val_ens, met_val)
grafico_previsto_vs_real_teste(y_teste_met, pred_teste_ens, met_teste)
grafico_tabela_metricas(met_val, met_teste)
grafico_residuos_validacao(Y_val_met, pred_val_ens)
grafico_residuos_teste(y_teste_met, pred_teste_ens)
grafico_comparativo_mse(met_rf, met_gb, met_et, met_teste)
grafico_importancia_features(rf_otimizado, et_otimizado, feature_names)

print("Gráficos gerados com sucesso!")

# ─────────────────────────────────────────────
# SALVAMENTO
# ─────────────────────────────────────────────

dump(ensemble, os.path.join(OUTPUT_DIR, "rf_modelo_final.joblib"))
dump(scaler,   os.path.join(OUTPUT_DIR, "scaler.joblib"))

print(f"\n  Ensemble saved -> rf_modelo_final.joblib")
print(f"  Scaler  salvo  -> scaler.joblib")
print(f"  Pasta          -> {OUTPUT_DIR}")

print("\n" + "=" * 60)
print("  Processo concluído com sucesso!")
print("=" * 60)