"""
Modelo Random Forest Melhorado — Alta Performance
Meta: MSE < 0.780 no conjunto de teste.

Estrategias aplicadas:
  1. Feature Engineering — razoes e interacoes entre variaveis granulometricas
  2. Ensemble — VotingRegressor (RF + GradientBoosting + ExtraTrees)
  3. Busca extensa — 150 iteracoes, 10-fold CV, scoring=neg_MSE
  4. Retreino no conjunto treino+validacao apos tuning
  5. Metricas completas: MSE, RMSE, MAE, R²
  6. Alerta se MSE <= 1
  7. Graficos padronizados
"""

# ─────────────────────────────────────────────
# IMPORTAÇÕES
# ─────────────────────────────────────────────
import os                                        # Manipulação de caminhos e criação de pastas
import warnings                                  # Supressão de avisos não críticos do Python
import numpy as np                               # Operações numéricas e arrays multidimensionais
import pandas as pd                              # Leitura do CSV e manipulação tabular
import matplotlib.pyplot as plt                  # Geração de todos os gráficos
import matplotlib.gridspec as gridspec           # Layout avançado para múltiplos subplots

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
# normpath resolve o ".." corretamente em qualquer sistema operacional
CAMINHO_DADOS = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "dados_processados_1.csv"
))

COLUNA_ALVO   = "CBR "          # Nome exato da coluna alvo no CSV (atenção ao espaço no final)
SEED          = 42               # Semente global: garante splits, buscas e modelos reproduzíveis
TEST_SIZE     = 0.20             # 20% do dataset reservado para teste final (nunca visto no treino)
VAL_SIZE      = 0.15             # 15% do restante reservado para validação (separado do teste)
N_ITER_BUSCA  = 150              # Combinações testadas por modelo na busca — mais = melhor cobertura do espaço
CV_FOLDS      = 10               # 10-fold CV: estimativa mais estável que 5-fold em datasets menores
META_MSE      = 0.780            # Objetivo de performance: MSE abaixo deste valor no teste final

# Pasta onde ensemble e scaler serão salvos (criada automaticamente se não existir)
OUTPUT_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Modelo_salvo_RF"
))
os.makedirs(OUTPUT_DIR, exist_ok=True)          # exist_ok=True evita erro se pasta já existir

# Pesos amostrais (opcional):
# Quando USE_WEIGHTS=True, amostras com CBR acima de THRESHOLD recebem peso W_MINOR
# e as demais recebem W_MAJOR, forçando o modelo a prestar mais atenção nas regiões
# menos representadas no dataset (CBR muito alto, por exemplo)
USE_WEIGHTS = False              # False = todas as amostras têm influência igual
THRESHOLD   = None               # Valor de corte para separar amostras raras das comuns
W_MINOR     = 3.0                # Peso das amostras acima do threshold (raras → mais importantes)
W_MAJOR     = 1.0                # Peso das amostras abaixo do threshold (comuns → influência normal)

# ─────────────────────────────────────────────
# FUNÇÕES AUXILIARES
# ─────────────────────────────────────────────

def engenharia_features(df: pd.DataFrame, coluna_alvo: str) -> pd.DataFrame:
    """
    Cria novas features derivadas das variáveis granulométricas e índices físicos.

    Lógica de cada feature:
      - ratio_X_Y: razão entre peneiras adjacentes — captura o formato da curva
        granulométrica de forma relativa, independente dos valores absolutos
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
    df["ratio_9_25"]    = df["9.5mm"]   / (df["25.4mm"]  + eps)  # Fração retida entre 25.4mm e 9.5mm
    df["ratio_4_9"]     = df["4.8mm"]   / (df["9.5mm"]   + eps)  # Fração retida entre 9.5mm e 4.8mm
    df["ratio_2_4"]     = df["2.0mm"]   / (df["4.8mm"]   + eps)  # Fração retida entre 4.8mm e 2.0mm
    df["ratio_042_2"]   = df["0.42mm"]  / (df["2.0mm"]   + eps)  # Fração retida entre 2.0mm e 0.42mm
    df["ratio_076_042"] = df["0.076mm"] / (df["0.42mm"]  + eps)  # Fração retida entre 0.42mm e 0.076mm

    # Índice de atividade: quanto maior, mais plástico e menos resistente tende a ser o solo
    df["atividade"]     = df["LL "] - df["IP "]

    # Compacidade: solos mais densos com menor teor de umidade tendem a ter CBR maior
    df["compacidade"]   = df["Densidade máxima "] / (df["Umidade Ótima"] + eps)

    # Finos ao quadrado: amplifica a diferença entre solos com muito ou pouco material fino
    df["finos_sq"]      = df["0.076mm"] ** 2

    return df


def metricas(y_true: np.ndarray, y_pred: np.ndarray, nome: str) -> dict:
    """
    Calcula e imprime MSE, RMSE, MAE e R² para um conjunto de previsões.
    Emite alerta se MSE <= 1, indicando possível problema de escala ou data leakage.

    Args:
        y_true: Valores reais do alvo.
        y_pred: Valores previstos pelo modelo.
        nome:   Nome do conjunto avaliado (ex: "Validacao", "Teste Final").

    Returns:
        Dicionário com todas as métricas calculadas.
    """
    mse  = mean_squared_error(y_true, y_pred)   # Penaliza erros grandes mais do que erros pequenos
    rmse = np.sqrt(mse)                          # Raiz do MSE: na mesma unidade do alvo (%)
    mae  = mean_absolute_error(y_true, y_pred)  # Média dos erros absolutos — mais intuitivo para interpretação
    r2   = r2_score(y_true, y_pred)             # R²: 1.0 = perfeito, 0.0 = modelo constante

    print(f"\n  [{nome}]")
    print(f"    MSE  : {mse:.4f}")

    # MSE <= 1 é suspeito para CBR (que varia de ~2% a ~120%):
    # pode indicar que o alvo foi normalizado acidentalmente ou que há data leakage
    if mse <= 1:
        print(f"    ATENCAO: MSE={mse:.4f} <= 1 — valor suspeito. "
              "Verifique a escala do alvo ou possivel data leakage.")

    print(f"    RMSE : {rmse:.4f}")
    print(f"    MAE  : {mae:.4f}")
    print(f"    R²   : {r2:.4f}")
    return dict(nome=nome, mse=mse, rmse=rmse, mae=mae, r2=r2)


# ─────────────────────────────────────────────
# 1. CARREGAMENTO E FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("=" * 60)
print("  RANDOM FOREST — ALTA PERFORMANCE")
print(f"  META: MSE < {META_MSE}")
print("=" * 60)
print("\n[1/6] Carregando dados e aplicando feature engineering...")

df = pd.read_csv(CAMINHO_DADOS)                 # Lê o CSV e armazena como DataFrame
print(f"     Dataset original: {df.shape[0]} linhas x {df.shape[1]} colunas")

# Feature engineering aplicada antes de qualquer split para que todas as
# novas colunas estejam disponíveis em treino, validação e teste
df_eng = engenharia_features(df, COLUNA_ALVO)
print(f"     Dataset expandido: {df_eng.shape[0]} linhas x {df_eng.shape[1]} colunas")
print(f"     Novas features: {df_eng.shape[1] - df.shape[1]} adicionadas")

Y = df_eng[COLUNA_ALVO].values.ravel()         # Extrai o alvo como array 1D
X = df_eng.drop(columns=[COLUNA_ALVO])         # Remove o alvo; o restante são as features
feature_names = X.columns.tolist()             # Guarda os nomes das features para os gráficos de importância
X = X.values                                   # Converte para array NumPy (necessário para sklearn)

# ─────────────────────────────────────────────
# 2. DIVISÃO DOS DADOS
# ─────────────────────────────────────────────
print("\n[2/6] Dividindo dados...")

# Primeiro split: 20% reservados para o teste final
# Esses dados NUNCA são vistos durante a busca de hiperparâmetros nem durante o treino
X_tv, X_teste, Y_tv, y_teste = train_test_split(
    X, Y, test_size=TEST_SIZE, random_state=SEED
)

# Segundo split: dos 80% restantes, 15% para validação
# Usado para monitorar overfitting e comparar modelos antes de avaliar no teste
X_treino, X_val, Y_treino, Y_val = train_test_split(
    X_tv, Y_tv, test_size=VAL_SIZE, random_state=SEED
)

# Normalização MinMax — fit APENAS no treino para evitar data leakage
# Se o scaler fosse ajustado com todos os dados, os limites min/max
# do teste "vazariam" para o treino, inflando artificialmente as métricas
scaler     = MinMaxScaler()
X_treino_n = scaler.fit_transform(X_treino)    # Aprende min/max do treino e normaliza
X_val_n    = scaler.transform(X_val)           # Aplica a mesma escala do treino na validação
X_teste_n  = scaler.transform(X_teste)         # Aplica a mesma escala do treino no teste
X_tv_n     = scaler.transform(X_tv)            # Treino+validação normalizado para o retreino final

print(f"     Treino: {X_treino_n.shape[0]}  |  Validacao: {X_val_n.shape[0]}  |  Teste: {X_teste_n.shape[0]}")
print(f"     Features totais: {X_treino_n.shape[1]}")

# Pesos amostrais — opcionais, passados ao fit de cada modelo
if USE_WEIGHTS and THRESHOLD is not None:
    # Amostras com CBR acima do threshold recebem peso maior (W_MINOR),
    # forçando os modelos a reduzir o erro nessas regiões menos representadas
    sample_weights = np.where(Y_treino > THRESHOLD, W_MINOR, W_MAJOR).astype(np.float32)
    print(f"     Pesos amostrais ativos — threshold={THRESHOLD}")
else:
    sample_weights = None                       # None = todos os exemplos têm influência igual

# ─────────────────────────────────────────────
# 3. ESPAÇOS DE HIPERPARÂMETROS
# ─────────────────────────────────────────────
print("\n[3/6] Definindo espacos de hiperparametros...")

# ── Espaço do Random Forest ───────────────────────────────────────────────────
param_rf = {
    # Número de árvores: mais árvores = menor variância, mas diminui o retorno marginal
    # Prefixo "rf__" é exigido pelo Pipeline para mapear ao estimador correto
    "rf__n_estimators":      list(range(100, 801, 50)),

    # Profundidade máxima de cada árvore: None = cresce até a pureza total (pode overfit)
    "rf__max_depth":         list(range(3, 31, 2)) + [None],

    # Mínimo de amostras para dividir um nó interno: valores maiores evitam overfitting
    "rf__min_samples_split": list(range(2, 15)),

    # Mínimo de amostras em cada folha (nó terminal): controla o tamanho mínimo de cada regra
    "rf__min_samples_leaf":  list(range(1, 10)),

    # Fração de features consideradas em cada divisão:
    # "sqrt" e "log2" são heurísticas; floats permitem ajuste fino
    "rf__max_features":      ["sqrt", "log2"] + np.linspace(0.1, 1.0, 10).round(2).tolist(),

    # Fração das amostras usadas para treinar cada árvore (bootstrap sampling)
    # Valores menores aumentam a diversidade entre árvores, reduzindo correlação
    "rf__max_samples":       np.linspace(0.5, 1.0, 6).round(2).tolist(),
}

# ── Espaço do GradientBoosting ────────────────────────────────────────────────
# Cada árvore é treinada nos resíduos da anterior (boosting sequencial)
# Requer controle fino do learning_rate e subsample para evitar overfitting
param_gb = {
    # Número de árvores (rounds de boosting): mais árvores = mais correção, mas pode overfit
    "gb__n_estimators":      list(range(100, 601, 50)),

    # Profundidade de cada árvore de boosting: geralmente mais rasas que no RF
    "gb__max_depth":         list(range(2, 9)),

    # Taxa de aprendizado: quanto cada árvore contribui para a previsão final
    # LR menor = mais árvores necessárias, mas geralmente melhor generalização
    "gb__learning_rate":     [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2],

    # Mínimo de amostras para divisão de nó e para folha (regularização)
    "gb__min_samples_split": list(range(2, 10)),
    "gb__min_samples_leaf":  list(range(1, 8)),

    # Fração de amostras usadas para treinar cada árvore (sem reposição, estocástico)
    # Valores abaixo de 1.0 introduzem estocasticidade, melhorando a generalização
    "gb__subsample":         np.linspace(0.6, 1.0, 5).round(2).tolist(),

    # Fração de features candidatas em cada divisão
    "gb__max_features":      ["sqrt", "log2", None],
}

# ── Espaço do ExtraTrees ──────────────────────────────────────────────────────
# Splits são escolhidos completamente aleatórios (ao contrário do RF que busca o melhor split)
# Isso reduz variância e acelera o treino, ao custo de um leve aumento de viés
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
print(f"\n[4/6] Buscando melhores hiperparametros ({N_ITER_BUSCA} iter, {CV_FOLDS}-fold CV)...")

# KFold com shuffle=True: embaralha os dados antes de criar os folds
# Garante que cada fold seja representativo do dataset completo
kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

# Parâmetros extras para o fit da busca (pesos amostrais, se ativos)
fit_params_search = {}
if sample_weights is not None:
    fit_params_search["sample_weight"] = sample_weights

# Importação local do Pipeline para evitar conflito de namespace
from sklearn.pipeline import Pipeline as _Pipe

# ── Busca RF ──────────────────────────────────────────────────────────────────
print("  Buscando RF...")

# Pipeline com prefixo "rf": necessário para que o RandomizedSearchCV
# mapeie os parâmetros "rf__n_estimators" ao estimador RandomForestRegressor
pipe_rf   = _Pipe([("rf", RandomForestRegressor(random_state=SEED, n_jobs=-1))])
search_rf = RandomizedSearchCV(
    pipe_rf,
    param_rf,
    n_iter=N_ITER_BUSCA,                        # Número de combinações sorteadas do espaço
    scoring="neg_mean_squared_error",           # Sklearn maximiza; negativo do MSE = minimiza MSE
    cv=kf,                                      # Usa o KFold definido acima
    random_state=SEED,                          # Reprodutibilidade das combinações sorteadas
    n_jobs=-1,                                  # Paraleliza usando todos os núcleos da CPU
    verbose=0,                                  # Sem saída intermediária no terminal
)
search_rf.fit(X_treino_n, Y_treino)            # Executa a busca no conjunto de treino

# Remove o prefixo "rf__" dos parâmetros para usar diretamente no estimador
best_rf_params = {k.replace("rf__", ""): v for k, v in search_rf.best_params_.items()}
print(f"     Melhor MSE CV (RF): {-search_rf.best_score_:.4f}")  # best_score_ é negativo

# ── Busca GradientBoosting ────────────────────────────────────────────────────
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

# ── Busca ExtraTrees ──────────────────────────────────────────────────────────
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
print("\n[5/6] Treinando ensemble final...")

# Instancia cada modelo com os melhores hiperparâmetros encontrados na busca
# **best_XX_params descompacta o dicionário como argumentos nomeados para o construtor
rf_otimizado = RandomForestRegressor(**best_rf_params, random_state=SEED, n_jobs=-1)
gb_otimizado = GradientBoostingRegressor(**best_gb_params, random_state=SEED)
et_otimizado = ExtraTreesRegressor(**best_et_params,  random_state=SEED, n_jobs=-1)

# VotingRegressor: combina os 3 modelos pela média das previsões individuais
# "Wisdom of the crowd" — erros descorrelacionados entre modelos se cancelam,
# reduzindo a variância total sem aumentar o viés
ensemble = VotingRegressor(estimators=[
    ("rf", rf_otimizado),                       # Random Forest otimizado
    ("gb", gb_otimizado),                       # GradientBoosting otimizado
    ("et", et_otimizado),                       # ExtraTrees otimizado
])

# Prepara pesos para o conjunto treino+validação (se ativos)
fit_params_final = {}
if sample_weights is not None:
    # Recalcula pesos agora para o conjunto treino+validação combinado
    sw_tv = np.where(Y_tv > THRESHOLD, W_MINOR, W_MAJOR).astype(np.float32)
    fit_params_final["sample_weight"] = sw_tv

# Retreina o ensemble no conjunto treino+validação para maximizar os dados disponíveis
# Os hiperparâmetros já foram fixados na etapa de busca — não há data leakage
ensemble.fit(X_tv_n, Y_tv, **fit_params_final)
print("     Ensemble treinado em treino + validacao.")

# ─────────────────────────────────────────────
# 6. AVALIAÇÃO
# ─────────────────────────────────────────────
print("\n[6/6] Avaliando modelos...")

# Previsões do ensemble nos dois conjuntos de interesse
pred_val_ens   = ensemble.predict(X_val_n)      # Previsão no conjunto de validação
pred_teste_ens = ensemble.predict(X_teste_n)    # Previsão no conjunto de teste (avaliação final)

# Métricas do ensemble
met_val   = metricas(Y_val,   pred_val_ens,   "Validacao  — Ensemble")
met_teste = metricas(y_teste, pred_teste_ens, "Teste Final — Ensemble")

# Retreina os modelos individuais no conjunto completo (treino+val) para
# comparação justa com o ensemble no conjunto de teste
print("\n  --- Modelos Individuais (Teste) ---")
rf_otimizado.fit(X_tv_n, Y_tv)                 # RF individual retreinado
gb_otimizado.fit(X_tv_n, Y_tv)                 # GB individual retreinado
et_otimizado.fit(X_tv_n, Y_tv)                 # ET individual retreinado

met_rf = metricas(y_teste, rf_otimizado.predict(X_teste_n), "Random Forest")
met_gb = metricas(y_teste, gb_otimizado.predict(X_teste_n), "GradientBoosting")
met_et = metricas(y_teste, et_otimizado.predict(X_teste_n), "ExtraTrees")

# Verifica se o MSE do ensemble atingiu a meta definida nas configurações
print("\n" + "=" * 60)
if met_teste["mse"] < META_MSE:
    print(f"  META ATINGIDA! MSE = {met_teste['mse']:.4f} < {META_MSE}")
else:
    print(f"  MSE = {met_teste['mse']:.4f}  |  Meta: < {META_MSE}")
print("=" * 60)

# ─────────────────────────────────────────────
# VISUALIZAÇÕES
# ─────────────────────────────────────────────

# Paleta de cores padronizada — idêntica ao script MLP para consistência visual
PALETTE = {
    "azul":    "#2563EB",   # Conjunto de validação
    "verde":   "#16A34A",   # Conjunto de teste
    "laranja": "#EA580C",   # Linha de referência e destaques
    "roxo":    "#7C3AED",   # GradientBoosting no gráfico de barras
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

# ── Figura 1: Painel principal 2x3 ───────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.suptitle("Ensemble RF+GB+ET — Analise de Performance", fontsize=16, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)  # 2 linhas, 3 colunas

# A) Previsto vs Real — Validação
# Pontos próximos da diagonal laranja indicam previsões precisas
ax1 = fig.add_subplot(gs[0, 0])
lim = [min(Y_val.min(), pred_val_ens.min()) * 0.95,       # Limite inferior com margem de 5%
       max(Y_val.max(), pred_val_ens.max()) * 1.05]        # Limite superior com margem de 5%
ax1.scatter(Y_val, pred_val_ens, alpha=0.6, color=PALETTE["azul"],
            edgecolors="white", linewidths=0.4, s=50)
ax1.plot(lim, lim, "--", color=PALETTE["laranja"], lw=1.5, label="Ideal")  # Linha y=x (previsão perfeita)
ax1.set_xlim(lim); ax1.set_ylim(lim)
ax1.set_title("Previsto vs Real — Validacao", fontweight="bold")
ax1.set_xlabel("Real"); ax1.set_ylabel("Previsto")
ax1.legend(framealpha=0)
ax1.text(0.05, 0.92, f"R² = {met_val['r2']:.4f}", transform=ax1.transAxes,
         fontsize=10, color=PALETTE["azul"], fontweight="bold")

# B) Previsto vs Real — Teste
# Avaliação final: mede a capacidade de generalização para dados nunca vistos
ax2 = fig.add_subplot(gs[0, 1])
lim2 = [min(y_teste.min(), pred_teste_ens.min()) * 0.95,
        max(y_teste.max(), pred_teste_ens.max()) * 1.05]
ax2.scatter(y_teste, pred_teste_ens, alpha=0.6, color=PALETTE["verde"],
            edgecolors="white", linewidths=0.4, s=50)
ax2.plot(lim2, lim2, "--", color=PALETTE["laranja"], lw=1.5, label="Ideal")
ax2.set_xlim(lim2); ax2.set_ylim(lim2)
ax2.set_title("Previsto vs Real — Teste", fontweight="bold")
ax2.set_xlabel("Real"); ax2.set_ylabel("Previsto")
ax2.legend(framealpha=0)
ax2.text(0.05, 0.92, f"R² = {met_teste['r2']:.4f}", transform=ax2.transAxes,
         fontsize=10, color=PALETTE["verde"], fontweight="bold")

# Exibe o MSE final em verde se a meta foi atingida, laranja caso contrário
cor_meta = PALETTE["verde"] if met_teste["mse"] < META_MSE else PALETTE["laranja"]
ax2.text(0.05, 0.83, f"MSE = {met_teste['mse']:.4f}", transform=ax2.transAxes,
         fontsize=9, color=cor_meta, fontweight="bold")

# C) Tabela comparativa de métricas — resumo visual lado a lado
ax3 = fig.add_subplot(gs[0, 2])
ax3.axis("off")                                # Esconde os eixos; apenas a tabela será exibida
table_data = [
    ["Metrica", "Validacao", "Teste"],
    ["MSE",  f"{met_val['mse']:.4f}",  f"{met_teste['mse']:.4f}"],
    ["RMSE", f"{met_val['rmse']:.4f}", f"{met_teste['rmse']:.4f}"],
    ["MAE",  f"{met_val['mae']:.4f}",  f"{met_teste['mae']:.4f}"],
    ["R²",   f"{met_val['r2']:.4f}",   f"{met_teste['r2']:.4f}"],
]
tbl = ax3.table(cellText=table_data[1:], colLabels=table_data[0],
                cellLoc="center", loc="center", bbox=[0.05, 0.25, 0.9, 0.55])
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor(PALETTE["grade"])
    if r == 0:                                 # Cabeçalho: fundo azul, texto branco
        cell.set_facecolor(PALETTE["azul"])
        cell.set_text_props(color="white", fontweight="bold")
    else:                                      # Dados: fundo azul claro nas colunas numéricas
        cell.set_facecolor("#EFF6FF" if c != 0 else "#F1F5F9")
ax3.set_title("Comparativo de Metricas — Ensemble", fontweight="bold", pad=14)

# D) Resíduos — Validação
# Resíduo = Real - Previsto; distribuição em torno de 0 indica modelo sem viés
ax4 = fig.add_subplot(gs[1, 0])
residuos_val = Y_val - pred_val_ens            # Calcula os resíduos do conjunto de validação
ax4.scatter(pred_val_ens, residuos_val, alpha=0.6, color=PALETTE["azul"],
            edgecolors="white", linewidths=0.4, s=50)
ax4.axhline(0, color=PALETTE["laranja"], lw=1.5, linestyle="--")  # Linha de resíduo zero (ideal)
ax4.set_title("Residuos — Validacao", fontweight="bold")
ax4.set_xlabel("Previsto"); ax4.set_ylabel("Residuo (Real - Previsto)")

# E) Resíduos — Teste
# Padrões sistemáticos (funil, curva) indicam viés ou falta de ajuste do modelo
ax5 = fig.add_subplot(gs[1, 1])
residuos_teste = y_teste - pred_teste_ens      # Calcula os resíduos do conjunto de teste
ax5.scatter(pred_teste_ens, residuos_teste, alpha=0.6, color=PALETTE["verde"],
            edgecolors="white", linewidths=0.4, s=50)
ax5.axhline(0, color=PALETTE["laranja"], lw=1.5, linestyle="--")
ax5.set_title("Residuos — Teste", fontweight="bold")
ax5.set_xlabel("Previsto"); ax5.set_ylabel("Residuo (Real - Previsto)")

# F) Comparativo de MSE — modelos individuais vs ensemble
# Permite visualizar o ganho do ensemble sobre cada modelo sozinho
ax6 = fig.add_subplot(gs[1, 2])
nomes_mod   = ["Random\nForest", "Gradient\nBoosting", "Extra\nTrees", "Ensemble"]
mse_valores = [met_rf["mse"], met_gb["mse"], met_et["mse"], met_teste["mse"]]

# Ensemble fica verde escuro se meta atingida, laranja se não atingida
cores_bar = [
    PALETTE["azul"],                           # Random Forest
    PALETTE["roxo"],                           # GradientBoosting
    PALETTE["verde"],                          # ExtraTrees
    PALETTE["laranja"] if met_teste["mse"] >= META_MSE else "#15803D",  # Ensemble
]
bars = ax6.bar(nomes_mod, mse_valores, color=cores_bar, edgecolor="white", width=0.5)
ax6.axhline(META_MSE, color="red", lw=1.5, linestyle="--", label=f"Meta MSE={META_MSE}")

# Anota o valor de MSE em cima de cada barra
for bar, val in zip(bars, mse_valores):
    ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax6.set_title("MSE por Modelo — Teste Final", fontweight="bold")
ax6.set_ylabel("MSE")
ax6.legend(framealpha=0)

plt.tight_layout(rect=[0, 0, 1, 0.97])

# ── Figura 2: Importância de features ────────────────────────────────────────
# Média da importância Gini entre RF e ET (ambos têm feature_importances_)
# GradientBoosting também possui, mas RF e ET já representam bem os dois estilos
fig2, ax7 = plt.subplots(figsize=(12, 6), facecolor=PALETTE["fundo"])
ax7.set_facecolor(PALETTE["fundo"])
ax7.spines["top"].set_visible(False)
ax7.spines["right"].set_visible(False)

# Média das importâncias entre RF e ET: suaviza vieses individuais de cada modelo
importancias_media = (rf_otimizado.feature_importances_ + et_otimizado.feature_importances_) / 2

top_n = min(15, len(feature_names))            # Exibe no máximo 15 features (ou todas, se menos)
idx   = np.argsort(importancias_media)[-top_n:][::-1]  # Índices das top N features em ordem decrescente

# Gradiente de azul: mais escuro = mais importante
cores = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))[::-1]

ax7.barh(range(top_n), importancias_media[idx], color=cores, edgecolor="white")
ax7.set_yticks(range(top_n))
ax7.set_yticklabels([feature_names[i] for i in idx], fontsize=9)
ax7.invert_yaxis()                             # Coloca a mais importante no topo
ax7.set_title(f"Top {top_n} Features Mais Importantes (media RF+ET)", fontweight="bold")
ax7.set_xlabel("Importancia (Gini)")           # Gini: quanto cada feature reduz a impureza média
ax7.grid(color=PALETTE["grade"], linewidth=0.8)
plt.tight_layout()

plt.show()                                     # Exibe todas as figuras na tela

# ─────────────────────────────────────────────
# SALVAMENTO
# ─────────────────────────────────────────────

# Salva o ensemble completo (com os 3 modelos internos e seus pesos)
# O arquivo .joblib é mais eficiente que pickle para objetos NumPy/sklearn
dump(ensemble, os.path.join(OUTPUT_DIR, "rf_modelo_final.joblib"))

# Salva o scaler para garantir que novos dados sejam normalizados
# com exatamente os mesmos limites usados no treinamento
dump(scaler,   os.path.join(OUTPUT_DIR, "scaler.joblib"))

print(f"\n  Ensemble salvo -> rf_modelo_final.joblib")
print(f"  Scaler  salvo  -> scaler.joblib")
print(f"  Pasta          -> {OUTPUT_DIR}")

print("\n" + "=" * 60)
print("  Processo concluido com sucesso!")
print("=" * 60)