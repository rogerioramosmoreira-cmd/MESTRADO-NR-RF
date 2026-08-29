"""
Modelo MLP (Rede Neural) — Alta Precisão para Previsão de CBR.

Melhorias aplicadas:
  - Log-transform no alvo (log1p/expm1) — comprime assimetria da distribuição do CBR
  - Pesos amostrais — amostras com CBR alto recebem maior peso
  - Early stopping patience=25 — exploração mais longa antes de parar
  - N_ITER_BUSCA=30 — cobertura maior do espaço de hiperparâmetros
  - Sem data leakage: scaler fit apenas no treino; val separado do teste

Gráficos disponíveis:
  1. Painel principal (Previsto vs Real, Resíduos, Histórico MAE)
  2. Distribuição dos resíduos (histograma + KDE)
  3. Histórico de MSE
  4. Busca de hiperparâmetros
  5. Verificação de Data Leakage (gap treino/val por época)
  6. Outliers nos resíduos (boxplot + IQR)
  7. Detalhamento do Early Stopping (janela de patience)
  8. R² comparativo (treino, validação, teste)
  9. Raciocínio da Rede Neural (derivada da loss + marcos de aprendizado)
"""

# ─────────────────────────────────────────────
# IMPORTAÇÕES
# ─────────────────────────────────────────────
import os                                         # Manipulação de caminhos e criação de pastas
import warnings                                   # Supressão de avisos não críticos
import numpy as np                                # Operações numéricas e geração de arrays
import pandas as pd                               # Leitura do CSV e manipulação tabular
import matplotlib.pyplot as plt                   # Geração de todos os gráficos
import matplotlib.gridspec as gridspec            # Layout avançado para múltiplos subplots
import seaborn as sns                             # Histogramas com KDE (distribuição suavizada)

from joblib import dump                           # Serialização do scaler em arquivo .joblib
from sklearn.preprocessing import MinMaxScaler   # Normaliza features para o intervalo [0, 1]
from sklearn.model_selection import train_test_split  # Divisão dos dados
from sklearn.metrics import (
    r2_score,                                     # R²: proporção da variância explicada (0 a 1)
    mean_squared_error,                           # MSE: penaliza erros grandes quadraticamente
    mean_absolute_error,                          # MAE: média dos erros absolutos, robusto a outliers
)

import tensorflow as tf                           # Framework de deep learning
from tensorflow.keras.models import Sequential   # Modelo sequencial em camadas
from tensorflow.keras.layers import (
    Dense,                                        # Camada densa totalmente conectada
    Dropout,                                      # Regularização por desativação aleatória
    BatchNormalization,                           # Normalização por lote — estabiliza treino
    LeakyReLU,                                    # Ativação com gradiente negativo pequeno
)
from tensorflow.keras.callbacks import (
    EarlyStopping,                                # Para o treino quando val_loss estagna
    ReduceLROnPlateau,                            # Reduz LR quando o modelo estagna
)
from tensorflow.keras.regularizers import l2     # Penalização L2 nos pesos
from tensorflow.keras.optimizers import Adam     # Otimizador Adam adaptativo

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

dependencies.require("neural")
tf.get_logger().setLevel("ERROR")

# ─────────────────────────────────────────────
# CONFIGURAÇÕES GLOBAIS
# ─────────────────────────────────────────────

CAMINHO_DADOS = paths.DATASET

COLUNA_ALVO = "CBR "             # Nome exato da coluna alvo no CSV (atenção ao espaço)
SEED        = 42                  # Semente global para reprodutibilidade
TEST_SIZE   = 0.20                # 20% reservados para teste final
VAL_SIZE    = 0.15                # 15% do restante para validação

OUTPUT_DIR = paths.MLP_DIR
paths.ensure(OUTPUT_DIR)

# ── Log-transform no alvo ────────────────────────────────────────────────────
# A distribuição do CBR é assimétrica à direita. log1p comprime a cauda,
# equilibrando os erros em toda a faixa do alvo.
LOG_ALVO = True                   # True = treina em log1p(CBR), avalia em expm1(pred)

# ── Pesos amostrais ───────────────────────────────────────────────────────────
# Amostras com CBR > THRESHOLD recebem peso W_MINOR — forçam atenção nas regiões raras.
USE_WEIGHTS = True
THRESHOLD   = 20.0
W_MINOR     = 3.0
W_MAJOR     = 1.0

# ── Busca de hiperparâmetros ──────────────────────────────────────────────────
LEARNING_RATES = [1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4]
BATCH_SIZES    = [8, 16, 24, 32, 48, 64]
DROPOUTS       = [0.0, 0.05, 0.1, 0.15, 0.2]
# O segundo valor vale apenas com MLL_FAST=1 - ver core/runtime.py.
N_ITER_BUSCA   = runtime.budget(30, 4)   # Cobertura do espaco de hiperparametros

# ─────────────────────────────────────────────
# PALETA DE CORES
# ─────────────────────────────────────────────
PALETTE = {
    "azul":    "#2563EB",         # Treino / validação — azul primário
    "azul2":   "#60A5FA",         # Teste — azul claro
    "laranja": "#EA580C",         # Referência, destaques, Early Stopping
    "verde":   "#16A34A",         # R² positivo, melhora
    "vermelho":"#DC2626",         # Alertas, data leakage, outliers
    "roxo":    "#7C3AED",         # Derivada / raciocínio
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
    """Remove espaços e aplica mapeamento de variações de nomes de colunas."""
    df.columns = df.columns.str.strip()
    mapa = {
        "25,4mm":"25.4mm","9,5mm":"9.5mm","4,8mm":"4.8mm","2,0mm":"2.0mm",
        "0,42mm":"0.42mm","0,076mm":"0.076mm",
        "P25.4":"25.4mm","P9.5":"9.5mm","P4.8":"4.8mm",
        "P2.0":"2.0mm","P0.42":"0.42mm","P0.076":"0.076mm",
        "Ll":"LL","ll":"LL","L.L":"LL","L.L.":"LL",
        "Ip":"IP","ip":"IP","I.P":"IP","I.P.":"IP",
        "Wot":"Umidade Ótima","wot":"Umidade Ótima","W_ot":"Umidade Ótima",
        "Umidade otima":"Umidade Ótima","Umidade Otima":"Umidade Ótima",
        "Umidade ótima":"Umidade Ótima",
        "Densidade Maxima":"Densidade máxima","Densidade Máxima":"Densidade máxima",
        "d_max":"Densidade máxima","Dmax":"Densidade máxima",
        "CBR":"CBR ","cbr":"CBR ","Cbr":"CBR ",
    }
    df = df.rename(columns=mapa)
    esperadas = ["25.4mm","9.5mm","4.8mm","2.0mm","0.42mm","0.076mm",
                 "LL","IP","Umidade Ótima","Densidade máxima","CBR "]
    faltando = [c for c in esperadas if c not in df.columns]
    if faltando:
        print(f"\n  AVISO: colunas não encontradas: {faltando}")
        print(f"  Colunas reais: {list(df.columns)}")
        raise SystemExit(1)
    return df


def exibir_metricas(y_true: np.ndarray, y_pred: np.ndarray, nome: str) -> dict:
    """Calcula e imprime MSE, RMSE, MAE e R²."""
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"\n  [{nome}]")
    print(f"    MSE  : {mse:.4f}")
    # Um R² quase perfeito em dados de verdade quase sempre significa que o
    # conjunto avaliado vazou para o treino. O limiar antigo era um MSE fixo
    # (<= 1), que só fazia sentido na escala log1p; em R² o alerta independe
    # da escala do alvo.
    if r2 >= 0.99:
        print(f"    ATENÇÃO: R²={r2:.4f} — alto demais para dados novos; "
              "verifique escala ou data leakage.")
    print(f"    RMSE : {rmse:.4f}")
    print(f"    MAE  : {mae:.4f}")
    print(f"    R²   : {r2:.4f}")
    return dict(nome=nome, mse=mse, rmse=rmse, mae=mae, r2=r2)


def construir_modelo(n_features: int, dropout: float, lr: float) -> Sequential:
    """
    Constrói e compila o MLP.
    Arquitetura: 128 → 64 → 32 → 16 → 1
    Bloco padrão: Dense → BatchNorm → LeakyReLU → Dropout
    """
    modelo = Sequential([
        Dense(128, input_shape=(n_features,), kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(dropout),

        Dense(64, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(dropout * 0.7),

        Dense(32, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(dropout * 0.4),

        Dense(16),
        LeakyReLU(alpha=0.1),

        Dense(1),                                # Saída sem ativação = regressão livre
    ])
    modelo.compile(
        optimizer=Adam(learning_rate=lr),
        loss="mae",                              # MAE como função de perda principal
        metrics=["mse"],
    )
    return modelo


# ─────────────────────────────────────────────
# FUNÇÕES DE GRÁFICO
# ─────────────────────────────────────────────

def _ax_base(figsize=(10, 5)):
    """Cria figura e eixo com estilo padrão."""
    fig, ax = plt.subplots(figsize=figsize, facecolor=PALETTE["fundo"])
    ax.set_facecolor(PALETTE["fundo"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def grafico_data_leakage(hist):
    """
    Gráfico 5 — Verificação de Data Leakage.

    Plota o gap (val_loss - train_loss) por época.
    - Gap próximo de 0 = saudável.
    - Gap muito negativo (val << train) = possível leakage.
    - Gap crescente = overfitting.
    Destaca com faixa colorida a zona de risco.
    """
    train_loss = np.array(hist.history["loss"])
    val_loss   = np.array(hist.history["val_loss"])
    gap        = val_loss - train_loss             # positivo = overfitting; negativo = leakage
    epocas     = np.arange(1, len(gap) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=PALETTE["fundo"])


    # Painel esquerdo: curvas treino vs validação
    ax = axes[0]
    ax.set_facecolor(PALETTE["fundo"])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.plot(epocas, train_loss, color=PALETTE["azul"],    lw=1.5, label="MAE Treino")
    ax.plot(epocas, val_loss,   color=PALETTE["laranja"], lw=1.5, linestyle="--", label="MAE Validação")

    ax.set_xlabel("Época"); ax.set_ylabel("MAE")
    ax.legend()

    # Painel direito: gap com faixas de diagnóstico
    ax2 = axes[1]
    ax2.set_facecolor(PALETTE["fundo"])
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    ax2.plot(epocas, gap, color=PALETTE["roxo"], lw=2.0, label="Gap (val − treino)")
    ax2.axhline(0, color="#111827", lw=1.5, linestyle="--", label="Zero (saudável)")

    # Limiar baseado no desvio padrão do gap — mais robusto que percentual da loss
    limiar_leakage = -np.std(gap) * 1.0        # 1 desvio abaixo de zero
    limiar_overfit =  np.std(gap) * 1.0        # 1 desvio acima de zero

    # Zona de risco de DATA LEAKAGE (gap muito negativo) — vermelho forte
    ax2.axhspan(min(gap.min() * 1.2, limiar_leakage - 0.001),
                limiar_leakage, alpha=0.30,
                color=PALETTE["vermelho"], label="⚠ Risco Data Leakage")
    ax2.axhline(limiar_leakage, color=PALETTE["vermelho"], lw=1.0,
                linestyle=":", alpha=0.8)

    # Zona de risco de OVERFITTING (gap muito positivo) — laranja forte
    ax2.axhspan(limiar_overfit,
                max(gap.max() * 1.2, limiar_overfit + 0.001), alpha=0.30,
                color="#F59E0B", label="⚠ Risco Overfitting")   # amarelo-laranja distinto
    ax2.axhline(limiar_overfit, color="#F59E0B", lw=1.0,
                linestyle=":", alpha=0.8)

    # Zona saudável (entre os dois limiares) — verde claro
    ax2.axhspan(limiar_leakage, limiar_overfit, alpha=0.10,
                color=PALETTE["verde"], label="✓ Zona saudável")


    ax2.set_xlabel("Época"); ax2.set_ylabel("Diferença de MAE")
    ax2.legend()

    plt.tight_layout()
    plots.save(plt.gcf(), "data_leakage", "mlp")


def grafico_outliers(residuos_val, residuos_teste):
    """
    Gráfico 6 — Outliers nos Resíduos.

    Boxplot dos resíduos de validação e teste com pontos discrepantes
    (> 1.5 × IQR) marcados em vermelho e anotados com seu valor.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=PALETTE["fundo"])


    for ax, res, titulo, cor, cor_norm in zip(
        axes,
        [residuos_val, residuos_teste],
        ["Validação", "Teste"],
        [PALETTE["azul"], PALETTE["verde"]],        # azul vs verde — claramente distintos
        ["#BFDBFE", "#BBF7D0"],                     # fundo da caixa: azul claro vs verde claro
    ):
        ax.set_facecolor(PALETTE["fundo"])
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

        # Boxplot base — caixa com cor distinta por conjunto
        bp = ax.boxplot(res, vert=True, patch_artist=True,
                        boxprops=dict(facecolor=cor_norm, alpha=0.7,
                                      edgecolor=cor, linewidth=1.5),
                        medianprops=dict(color=cor, lw=2.5),          # mediana na cor do conjunto
                        whiskerprops=dict(color=cor, lw=1.2, linestyle="--"),
                        capprops=dict(color=cor, lw=1.5),
                        flierprops=dict(marker="o", markersize=0))

        # Cálculo IQR e identificação de outliers
        q1, q3 = np.percentile(res, [25, 75])
        iqr     = q3 - q1
        lim_inf = q1 - 1.5 * iqr
        lim_sup = q3 + 1.5 * iqr
        outliers = res[(res < lim_inf) | (res > lim_sup)]

        # Pontos normais — cor do conjunto, bem visíveis
        normais = res[(res >= lim_inf) & (res <= lim_sup)]
        ax.scatter(np.ones(len(normais)) + np.random.uniform(-0.08, 0.08, len(normais)),
                   normais, alpha=0.55, color=cor, s=25, zorder=3, edgecolors="white",
                   linewidths=0.3)

        # Outliers em vermelho com anotação
        for v in outliers:
            ax.scatter(1, v, color=PALETTE["vermelho"], s=60, zorder=5)
            ax.annotate(f"{v:.1f}", xy=(1, v), xytext=(1.15, v),
                        fontsize=9, color=PALETTE["vermelho"],
                        arrowprops=dict(arrowstyle="-", color=PALETTE["vermelho"], lw=0.8))

        n_out = len(outliers)

        ax.set_ylabel("Resíduo (Real − Previsto)")
        ax.set_xticks([])
        ax.text(0.98, 0.98, f"IQR = {iqr:.2f}\nLim inf = {lim_inf:.2f}\nLim sup = {lim_sup:.2f}",
                transform=ax.transAxes, va="top", ha="right", fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=PALETTE["fundo"],
                          edgecolor=PALETTE["grade"]))

    plt.tight_layout()
    plots.save(plt.gcf(), "outliers", "mlp")


def grafico_early_stopping(hist, epocas_reais, patience):
    """
    Gráfico 7 — Detalhamento do Early Stopping.

    Mostra a curva de val_loss com a janela de patience destacada,
    a melhor época (menor val_loss) e o ponto de parada.
    """
    val_loss   = np.array(hist.history["val_loss"])
    train_loss = np.array(hist.history["loss"])
    epocas     = np.arange(1, len(val_loss) + 1)

    melhor_epoca = int(np.argmin(val_loss)) + 1   # epoch 1-indexed
    melhor_val   = val_loss[melhor_epoca - 1]

    fig, ax = _ax_base(figsize=(13, 5))
    ax.plot(epocas, train_loss, color=PALETTE["azul"],    lw=1.5, label="MAE Treino",    alpha=0.8)
    ax.plot(epocas, val_loss,   color=PALETTE["laranja"], lw=1.5, linestyle="--", label="MAE Validação")

    # Marca a melhor época
    ax.scatter([melhor_epoca], [melhor_val], color=PALETTE["verde"],
               s=80, zorder=5, label=f"Melhor época: {melhor_epoca}")
    ax.annotate(f"  época {melhor_epoca}\n  MAE={melhor_val:.4f}",
                xy=(melhor_epoca, melhor_val),
                xytext=(melhor_epoca + len(epocas) * 0.05, melhor_val * 1.05),
                fontsize=9.5, color=PALETTE["verde"],
                arrowprops=dict(arrowstyle="->", color=PALETTE["verde"]))

    # Janela de patience após a melhor época
    inicio_patience = melhor_epoca
    fim_patience    = min(melhor_epoca + patience, epocas_reais)
    ax.axvspan(inicio_patience, fim_patience, alpha=0.08,
               color=PALETTE["vermelho"], label=f"Janela patience={patience}")
    ax.axvline(fim_patience, color=PALETTE["vermelho"], lw=1.2,
               linestyle=":", label=f"Parada (época {epocas_reais})")


    ax.set_xlabel("Época"); ax.set_ylabel("MAE")
    ax.legend()
    plt.tight_layout()
    plots.save(plt.gcf(), "early_stopping", "mlp")


def grafico_r2_comparativo(r2_treino, r2_val, r2_teste):
    """
    Gráfico 8 — R² Comparativo (Treino, Validação, Teste).

    Barras horizontais com linha de referência em 1.0 (perfeito).
    Cores indicam qualidade: verde >= 0.8, laranja >= 0.6, vermelho < 0.6.
    """
    fig, ax = _ax_base(figsize=(8, 4))

    conjuntos = ["Treino", "Validação", "Teste"]
    valores   = [r2_treino, r2_val, r2_teste]
    cores     = []
    for v in valores:
        if v >= 0.8:
            cores.append(PALETTE["verde"])
        elif v >= 0.6:
            cores.append(PALETTE["laranja"])
        else:
            cores.append(PALETTE["vermelho"])

    bars = ax.barh(conjuntos, valores, color=cores, edgecolor="white", height=0.5)
    ax.axvline(1.0, color=PALETTE["grade"], lw=1.2, linestyle="--", label="R²=1 (perfeito)")
    ax.axvline(0.0, color=PALETTE["vermelho"], lw=0.8, linestyle=":")

    for bar, val in zip(bars, valores):
        ax.text(max(val + 0.01, 0.02), bar.get_y() + bar.get_height() / 2,
                f"R² = {val:.4f}", va="center", fontsize=10, fontweight="bold")

    ax.set_xlim(-0.1, 1.15)

    ax.set_xlabel("R²")
    ax.legend()
    plt.tight_layout()
    plots.save(plt.gcf(), "r2_comparativo", "mlp")


def grafico_raciocinio_rede(hist, epocas_reais):
    """
    Gráfico 9 — Raciocínio da Rede Neural.

    Mostra COMO a rede mudou seu aprendizado ao longo do treino:
      - Painel superior: curva MAE com anotações nos pontos de inflexão
        (onde a taxa de melhora mudou significativamente).
      - Painel inferior: derivada suavizada da val_loss (taxa de mudança)
        — picos negativos = melhoras bruscas, platôs = estagnação.
      - Linhas verticais marcam cada redução do LR (ReduceLROnPlateau),
        indicando onde o modelo "mudou de estratégia".
    """
    val_loss   = np.array(hist.history["val_loss"])
    train_loss = np.array(hist.history["loss"])
    epocas     = np.arange(1, len(val_loss) + 1)

    # Derivada suavizada (janela 5) da val_loss — mede taxa de mudança
    def suavizar(arr, janela=5):
        return np.convolve(arr, np.ones(janela) / janela, mode="same")

    deriv = np.gradient(suavizar(val_loss))        # taxa de mudança do val_loss

    # Detecta inflexões: onde |deriv| > 1.5 * desvio-padrão
    limiar    = 1.5 * np.std(deriv)
    inflexoes = np.where(np.abs(deriv) > limiar)[0]
    # Agrupa inflexões em eventos (separa pontos muito próximos)
    eventos = []
    if len(inflexoes) > 0:
        grp = [inflexoes[0]]
        for i in inflexoes[1:]:
            if i - grp[-1] < 5:
                grp.append(i)
            else:
                eventos.append(int(np.mean(grp)))
                grp = [i]
        eventos.append(int(np.mean(grp)))
    eventos = eventos[:8]                          # Limita a 8 marcos

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor=PALETTE["fundo"],
                             gridspec_kw={"height_ratios": [2, 1]})


    # ── Painel superior: curvas de loss com marcos ────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor(PALETTE["fundo"])
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    ax1.plot(epocas, train_loss, color=PALETTE["azul"],    lw=1.5, label="MAE Treino",    alpha=0.7)
    ax1.plot(epocas, val_loss,   color=PALETTE["laranja"], lw=1.5, linestyle="--", label="MAE Validação")

    # Marca os eventos de inflexão
    cores_evento = plt.cm.Purples(np.linspace(0.4, 0.9, max(len(eventos), 1)))
    for k, ep in enumerate(eventos):
        ax1.axvline(ep + 1, color=cores_evento[k], lw=1.2, linestyle=":",
                    label=f"Marco {k+1} (época {ep+1})" if k < 4 else None)
        ax1.annotate(f"M{k+1}", xy=(ep + 1, val_loss[ep]),
                     xytext=(ep + 1 + 0.5, val_loss[ep] * 0.97),
                     fontsize=9, color=cores_evento[k], fontweight="bold")

    # Melhor época
    melhor = int(np.argmin(val_loss))
    ax1.scatter([melhor + 1], [val_loss[melhor]], color=PALETTE["verde"], s=70, zorder=5,
                label=f"Melhor val_loss (época {melhor+1})")
    ax1.set_ylabel("MAE")
    ax1.legend(loc="upper right")

    # ── Painel inferior: derivada (taxa de mudança) ───────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(PALETTE["fundo"])
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    ax2.plot(epocas, deriv, color=PALETTE["roxo"], lw=1.2, label="dMAE/dépoca (suavizado)")
    ax2.axhline(0, color=PALETTE["grade"], lw=1.0, linestyle="--")
    ax2.axhline( limiar, color=PALETTE["laranja"], lw=0.8, linestyle=":", alpha=0.6,
                 label=f"±Limiar (1.5σ = {limiar:.4f})")
    ax2.axhline(-limiar, color=PALETTE["laranja"], lw=0.8, linestyle=":", alpha=0.6)

    # Preenche regiões de melhora intensa (deriv muito negativo)
    ax2.fill_between(epocas, deriv, 0,
                     where=(deriv < -limiar),
                     alpha=0.25, color=PALETTE["verde"], label="Melhora intensa")
    # Preenche regiões de piora / estagnação
    ax2.fill_between(epocas, deriv, 0,
                     where=(deriv > limiar),
                     alpha=0.25, color=PALETTE["vermelho"], label="Piora / estagnação")

    # Marca marcos no painel inferior também
    for k, ep in enumerate(eventos):
        ax2.axvline(ep + 1, color=cores_evento[k], lw=1.0, linestyle=":")

    ax2.set_xlabel("Época"); ax2.set_ylabel("Taxa de Mudança (MAE)")

    ax2.legend(loc="upper right")

    plt.tight_layout()
    plots.save(plt.gcf(), "raciocinio_rede", "mlp")


# ─────────────────────────────────────────────
# 1. CARREGAMENTO E PRÉ-PROCESSAMENTO
# ─────────────────────────────────────────────
print("=" * 60)
print("  MLP — REDE NEURAL — ALTA PRECISÃO")
print("=" * 60)
print("\n[1/7] Carregando dados...")

try:
    df = pd.read_csv(CAMINHO_DADOS)
    print(f"     Dataset: {df.shape[0]} linhas x {df.shape[1]} colunas")
except FileNotFoundError:
    print(f"ERRO: Arquivo não encontrado em '{CAMINHO_DADOS}'.")
    raise SystemExit(1)

df = normalizar_colunas(df)

Y = df[COLUNA_ALVO].values.ravel()
X = df.drop(columns=[COLUNA_ALVO]).select_dtypes(include=[np.number])
feature_names = X.columns.tolist()
X = X.values
print(f"     Features: {len(feature_names)}")

Y_orig = Y.copy()
if LOG_ALVO:
    Y = np.log1p(Y)
    print(f"     Log-transform: [{Y_orig.min():.2f}, {Y_orig.max():.2f}] → "
          f"[{Y.min():.4f}, {Y.max():.4f}]")

# ─────────────────────────────────────────────
# 2. DIVISÃO E NORMALIZAÇÃO
# ─────────────────────────────────────────────
print("\n[2/7] Dividindo e normalizando dados...")

X_tv, X_teste, Y_tv, y_teste = train_test_split(
    X, Y, test_size=TEST_SIZE, random_state=SEED
)
X_treino, X_val, Y_treino, Y_val = train_test_split(
    X_tv, Y_tv, test_size=VAL_SIZE, random_state=SEED
)

# Scaler fit APENAS no treino — evita data leakage
scaler     = MinMaxScaler()
X_treino_n = scaler.fit_transform(X_treino)
X_val_n    = scaler.transform(X_val)
X_teste_n  = scaler.transform(X_teste)
X_tv_n     = scaler.transform(X_tv)

print(f"     Treino: {X_treino_n.shape[0]}  |  Val: {X_val_n.shape[0]}  |  Teste: {X_teste_n.shape[0]}")
print(f"     Features: {X_treino_n.shape[1]}")

if USE_WEIGHTS and THRESHOLD is not None:
    thr_efetivo = np.log1p(THRESHOLD) if LOG_ALVO else THRESHOLD
    sample_weights = np.where(Y_treino > thr_efetivo, W_MINOR, W_MAJOR).astype(np.float32)
    print(f"     Pesos amostrais: threshold={THRESHOLD} (log={thr_efetivo:.4f})")
else:
    sample_weights = None

# ─────────────────────────────────────────────
# 3. BUSCA DE HIPERPARÂMETROS
# ─────────────────────────────────────────────
print(f"\n[3/7] Buscando hiperparâmetros ({N_ITER_BUSCA} combinações)...")

tf.random.set_seed(SEED)
np.random.seed(SEED)

callbacks_busca = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=0),
]

melhor_mae_val  = np.inf
melhor_config   = {}
historico_busca = []

rng = np.random.RandomState(SEED)
combinacoes = [
    {
        "lr":         rng.choice(LEARNING_RATES),
        "batch_size": int(rng.choice(BATCH_SIZES)),
        "dropout":    float(rng.choice(DROPOUTS)),
    }
    for _ in range(N_ITER_BUSCA)
]

# Uma etapa da barra por combinacao testada.
with progress.bar(total=N_ITER_BUSCA) as _stage:
    for i, cfg in enumerate(combinacoes):
        modelo_tmp = construir_modelo(X_treino_n.shape[1],
                                      cfg["dropout"], cfg["lr"])
        hist = modelo_tmp.fit(
            X_treino_n, Y_treino,
            epochs=100,
            batch_size=cfg["batch_size"],
            validation_data=(X_val_n, Y_val),
            sample_weight=sample_weights,
            callbacks=callbacks_busca,
            verbose=0,
        )
        mae_val = min(hist.history["val_loss"])
        historico_busca.append({**cfg, "mae_val": mae_val})
        if mae_val < melhor_mae_val:
            melhor_mae_val = mae_val
            melhor_config  = cfg
        _stage.advance()

print(f"\n  Melhor config: {melhor_config}  |  MAE_val={melhor_mae_val:.4f}")

# ─────────────────────────────────────────────
# 4. TREINAMENTO FINAL
# ─────────────────────────────────────────────
print("\n[4/7] Treinando modelo final...")

# O ajuste final usa SOMENTE o conjunto de treino. A versao anterior treinava
# em treino+validacao (X_tv) e ainda passava (X_val_n, Y_val) como
# `validation_data`, o que quebrava duas coisas de uma vez:
#   1. a metrica de validacao virava erro de treino (otimista por construcao);
#   2. o EarlyStopping e o ReduceLROnPlateau monitoravam um `val_loss` medido
#      sobre dados ja treinados, entao o criterio de parada ficava cego ao
#      overfitting e o treino seguia alem do ponto util.
# Com treino e validacao separados, `val_loss` volta a medir generalizacao e os
# dois callbacks voltam a funcionar como pretendido.
modelo_final = construir_modelo(
    X_treino_n.shape[1], melhor_config["dropout"], melhor_config["lr"]
)
modelo_final.summary()

PATIENCE_FINAL = 25               # patience=25 para exploração mais ampla

callbacks_final = [
    EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE_FINAL,   # patience=25 conforme especificado
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=8,
        min_lr=1e-6,
        verbose=1,
    ),
]

# Uma etapa da barra por epoca. O total e o teto: o early stopping
# interrompe antes, entao esta barra costuma terminar incompleta.
with progress.bar(total=500) as _stage:
    historico = modelo_final.fit(
        X_treino_n, Y_treino,
        epochs=500,
        batch_size=melhor_config["batch_size"],
        validation_data=(X_val_n, Y_val),
        sample_weight=sample_weights,
        callbacks=[*callbacks_final, progress.keras_callback(_stage)],
        verbose=0,
    )

epocas_reais = len(historico.history["loss"])
print(f"\nTreino encerrado na época {epocas_reais}")

# ─────────────────────────────────────────────
# 5. AVALIAÇÃO
# ─────────────────────────────────────────────
print("\n[5/7] Avaliando o modelo...")

pred_val   = modelo_final.predict(X_val_n,   verbose=0).flatten()
pred_teste = modelo_final.predict(X_teste_n, verbose=0).flatten()
pred_treino = modelo_final.predict(X_treino_n, verbose=0).flatten()

if LOG_ALVO:
    pred_val    = np.expm1(pred_val)
    pred_teste  = np.expm1(pred_teste)
    pred_treino = np.expm1(pred_treino)
    Y_val_met   = np.expm1(Y_val)
    y_teste_met = np.expm1(y_teste)
    Y_treino_met = np.expm1(Y_treino)
    print("     Previsões revertidas para escala original.")
else:
    Y_val_met   = Y_val
    y_teste_met = y_teste
    Y_treino_met = Y_treino

met_val   = exibir_metricas(Y_val_met,   pred_val,   "Validação")
met_teste = exibir_metricas(y_teste_met, pred_teste, "Teste Final")

r2_treino = r2_score(Y_treino_met, pred_treino)
print(f"\n  R² Treino: {r2_treino:.4f}")
print(f"  Em média, o modelo erra ±{met_teste['mae']:.2f} no valor de CBR")

# Guarda o resultado do teste para o menu mostrar depois, sem treinar de novo.
scoreboard.record("mlp", "Rede Neural — modelo único", met_teste,
                  r2_treino_val=r2_treino)

residuos_val   = Y_val_met   - pred_val
residuos_teste = y_teste_met - pred_teste

# ─────────────────────────────────────────────
# 6. VISUALIZAÇÕES
# ─────────────────────────────────────────────
print("\n[6/7] Gerando gráficos...")

# ── Figura 1: Painel principal 2×3 ───────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

# A) Previsto vs Real — Validação
ax1 = fig.add_subplot(gs[0, 0])
lim = [min(Y_val_met.min(), pred_val.min()) * 0.95,
       max(Y_val_met.max(), pred_val.max()) * 1.05]
ax1.scatter(Y_val_met, pred_val, alpha=0.6, color=PALETTE["azul"],
            edgecolors="white", linewidths=0.4, s=50)
ax1.plot(lim, lim, "--", color=PALETTE["laranja"], lw=1.5, label="Ideal")
ax1.set_xlim(lim); ax1.set_ylim(lim)

ax1.set_xlabel("Real"); ax1.set_ylabel("Previsto")
ax1.legend()
ax1.text(0.05, 0.92, f"R² = {met_val['r2']:.4f}", transform=ax1.transAxes,
         fontsize=10, color=PALETTE["azul"], fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFFFFF", edgecolor="#E2E8F0", alpha=0.93))

# B) Previsto vs Real — Teste
ax2 = fig.add_subplot(gs[0, 1])
lim2 = [min(y_teste_met.min(), pred_teste.min()) * 0.95,
        max(y_teste_met.max(), pred_teste.max()) * 1.05]
ax2.scatter(y_teste_met, pred_teste, alpha=0.6, color=PALETTE["azul2"],
            edgecolors="white", linewidths=0.4, s=50)
ax2.plot(lim2, lim2, "--", color=PALETTE["laranja"], lw=1.5, label="Ideal")
ax2.set_xlim(lim2); ax2.set_ylim(lim2)

ax2.set_xlabel("Real"); ax2.set_ylabel("Previsto")
ax2.legend()
ax2.text(0.05, 0.92, f"R² = {met_teste['r2']:.4f}", transform=ax2.transAxes,
         fontsize=10, color=PALETTE["azul2"], fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFFFFF", edgecolor="#E2E8F0", alpha=0.93))

# C) Tabela de métricas
ax3 = fig.add_subplot(gs[0, 2])
ax3.axis("off")
table_data = [
    ["Métrica", "Validação", "Teste"],
    ["MSE",  f"{met_val['mse']:.4f}",  f"{met_teste['mse']:.4f}"],
    ["RMSE", f"{met_val['rmse']:.4f}", f"{met_teste['rmse']:.4f}"],
    ["MAE",  f"{met_val['mae']:.4f}",  f"{met_teste['mae']:.4f}"],
    ["R²",   f"{met_val['r2']:.4f}",   f"{met_teste['r2']:.4f}"],
]
tbl = ax3.table(cellText=table_data[1:], colLabels=table_data[0],
                cellLoc="center", loc="center", bbox=[0.05, 0.25, 0.9, 0.55])
tbl.auto_set_font_size(False); tbl.set_fontsize(11)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor(PALETTE["grade"])
    if r == 0:
        cell.set_facecolor(PALETTE["azul"])
        cell.set_text_props(color="white", fontweight="bold")
    else:
        cell.set_facecolor("#EFF6FF" if c != 0 else "#F1F5F9")


# D) Resíduos — Validação
ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(pred_val, residuos_val, alpha=0.6, color=PALETTE["azul"],
            edgecolors="white", linewidths=0.4, s=50)
ax4.axhline(0, color=PALETTE["laranja"], lw=1.5, linestyle="--")

ax4.set_xlabel("Previsto"); ax4.set_ylabel("Resíduo (Real − Previsto)")

# E) Resíduos — Teste
ax5 = fig.add_subplot(gs[1, 1])
ax5.scatter(pred_teste, residuos_teste, alpha=0.6, color=PALETTE["azul2"],
            edgecolors="white", linewidths=0.4, s=50)
ax5.axhline(0, color=PALETTE["laranja"], lw=1.5, linestyle="--")

ax5.set_xlabel("Previsto"); ax5.set_ylabel("Resíduo (Real − Previsto)")

# F) Histórico de Loss (MAE)
ax6 = fig.add_subplot(gs[1, 2])
ax6.plot(historico.history["loss"],     label="MAE Treino",    color=PALETTE["azul"],    lw=1.5)
ax6.plot(historico.history["val_loss"], label="MAE Validação", color=PALETTE["laranja"], lw=1.5, linestyle="--")

ax6.set_xlabel("Épocas"); ax6.set_ylabel("MAE")
ax6.legend()
plt.tight_layout(rect=[0, 0, 1, 0.97])

# ── Figura 2: Distribuição dos resíduos ──────────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5), facecolor=PALETTE["fundo"])

for ax, res, titulo, cor in zip(
    axes2, [residuos_val, residuos_teste], ["Validação", "Teste"],
    [PALETTE["azul"], PALETTE["azul2"]]
):
    ax.set_facecolor(PALETTE["fundo"])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    sns.histplot(res, kde=True, color=cor, ax=ax, alpha=0.7)
    ax.axvline(x=0, color=PALETTE["laranja"], linestyle="--", lw=1.5, label="Erro Zero")

    ax.set_xlabel("Erro (Real − Previsto)"); ax.set_ylabel("Frequência")
    ax.legend()
plt.tight_layout()

# ── Figura 3: Histórico de MSE ────────────────────────────────────────────────
# Só existe quando o histórico do Keras trouxe a métrica; permanece None
# caso contrário, para que o bloco de salvamento saiba que pular.
fig3 = None
if "mse" in historico.history:
    fig3, ax7 = plt.subplots(figsize=(12, 5), facecolor=PALETTE["fundo"])
    ax7.set_facecolor(PALETTE["fundo"])
    ax7.spines["top"].set_visible(False); ax7.spines["right"].set_visible(False)
    ax7.plot(historico.history["mse"],     label="MSE Treino",    color=PALETTE["azul"],    lw=1.5)
    ax7.plot(historico.history["val_mse"], label="MSE Validação", color=PALETTE["laranja"], lw=1.5, linestyle="--")

    ax7.set_xlabel("Épocas"); ax7.set_ylabel("MSE")
    ax7.legend()
    plt.tight_layout()

# ── Figura 4: Busca de hiperparâmetros ───────────────────────────────────────
df_busca = pd.DataFrame(historico_busca).sort_values("mae_val")
fig4, ax8 = plt.subplots(figsize=(12, 5), facecolor=PALETTE["fundo"])
ax8.set_facecolor(PALETTE["fundo"])
ax8.spines["top"].set_visible(False); ax8.spines["right"].set_visible(False)
cores_busca = [PALETTE["laranja"] if i == 0 else PALETTE["azul"] for i in range(len(df_busca))]
ax8.bar(range(len(df_busca)), df_busca["mae_val"], color=cores_busca, edgecolor="white", alpha=0.8)
ax8.axhline(df_busca["mae_val"].iloc[0], color=PALETTE["laranja"], lw=1.5,
            linestyle="--", label=f"Melhor MAE = {df_busca['mae_val'].iloc[0]:.4f}")

ax8.set_xlabel("Combinação (ranking)"); ax8.set_ylabel("MAE Validação")
ax8.legend()
plt.tight_layout()

# As figuras 1–4 são montadas acima e ficam abertas ao mesmo tempo. Elas são
# salvas aqui, antes de as funções seguintes trocarem a figura corrente —
# caso contrário só a última sobreviveria.
arquivos_gerados  = plots.save(fig,  "painel_performance",     "mlp")
arquivos_gerados += plots.save(fig2, "distribuicao_residuos",  "mlp")
if fig3 is not None:
    arquivos_gerados += plots.save(fig3, "historico_mse",      "mlp")
arquivos_gerados += plots.save(fig4, "busca_hiperparametros",  "mlp")

# ── Figuras 5–9: Gráficos novos ───────────────────────────────────────────────
grafico_data_leakage(historico)
grafico_outliers(residuos_val, residuos_teste)
grafico_early_stopping(historico, epocas_reais, PATIENCE_FINAL)
grafico_r2_comparativo(r2_treino, met_val["r2"], met_teste["r2"])
grafico_raciocinio_rede(historico, epocas_reais)

print(f"  Gráficos salvos em: {plots.figure_dir('mlp')}")

# ─────────────────────────────────────────────
# 7. SALVAMENTO
# ─────────────────────────────────────────────
print("\n[7/7] Salvando modelo e scaler...")
modelo_final.save(os.path.join(OUTPUT_DIR, "mlp_model.keras"))
dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))
print(f"  Modelo salvo → mlp_model.keras")
print(f"  Scaler salvo → scaler.joblib")
print(f"  Pasta        → {OUTPUT_DIR}")
print("\n" + "=" * 60)
print("  Processo concluído com sucesso!")
print("=" * 60)

# Veredito final: o que os números acima significam para quem vai usar o
# modelo. Fica por último de propósito — é a linha que alguém lê ao voltar
# ao terminal depois de um treino longo.
notas = []
if r2_treino < met_teste["r2"]:
    # Treino pior que teste não acontece por acaso: ou a divisão dos dados pôs
    # os casos fáceis no teste, ou o alvo foi transformado de um lado só. Vale
    # o aviso antes de alguém confiar no R² de teste.
    notas.append(f"R² de treino ({r2_treino:.4f}) abaixo do R² de "
                 "teste — verificar a divisão dos dados.")
metrics.report(met_teste, notes=notas)