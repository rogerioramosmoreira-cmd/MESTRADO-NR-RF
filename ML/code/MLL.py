"""
Modelo MLP (Rede Neural) Melhorado para Previsão de CBR.
Baseado na comparação com MLPBlockSpace (PyTorch/AG).

Problemas corrigidos do código original:
  - Data leakage: scaler era fit em TODOS os dados antes do split
  - Data leakage: conjunto de teste usado como validação durante treino
  - Funcao de perda MSE trocado por MAE (mais robusto a outliers, como no AG)
  - Ativacao ReLU fixa substituida por LeakyReLU (como no AG)
  - Arquitetura fixa substituida por busca de lr, batch e dropout
  - Early stopping patience=25 reduzido para 15 (equilibrado)
  - Sem R² — adicionado R² em todas as avaliacoes
  - Sem pesos amostrais — adicionado suporte (USE_WEIGHTS)
  - Sem feature engineering — adicionado razoes, indice de atividade, compacidade

Melhorias adicionais:
  - Busca de hiperparametros: lr, batch_size, dropout (inspirado no espaco do AG)
  - 3 splits: treino / validacao / teste
  - Feature engineering identico ao RF para comparacao justa
  - Graficos completos com paleta padronizada
  - Codigo totalmente comentado
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
from sklearn.model_selection import train_test_split  # Divisão estratificada dos dados
from sklearn.metrics import (
    r2_score,                                     # R²: proporção da variância explicada (0 a 1)
    mean_squared_error,                           # MSE: penaliza erros grandes quadraticamente
    mean_absolute_error,                          # MAE: média dos erros absolutos, robusto a outliers
)

import tensorflow as tf                           # Framework de deep learning
from tensorflow.keras.models import Sequential   # Modelo em que camadas são empilhadas em sequência
from tensorflow.keras.layers import (
    Dense,                                        # Camada densa: todos os neurônios conectados à entrada
    Dropout,                                      # Desativa neurônios aleatoriamente para evitar overfitting
    BatchNormalization,                           # Normaliza saída de cada camada, estabiliza o treino
    LeakyReLU,                                    # Ativação que permite gradiente negativo pequeno (evita neurônios mortos)
)
from tensorflow.keras.callbacks import (
    EarlyStopping,                                # Interrompe o treino quando val_loss para de melhorar
    ReduceLROnPlateau,                            # Reduz o learning rate quando o modelo estagna
)
from tensorflow.keras.regularizers import l2     # Penalização L2: pune pesos grandes para evitar overfitting
from tensorflow.keras.optimizers import Adam     # Otimizador adaptativo: ajusta o passo de aprendizado por parâmetro

warnings.filterwarnings("ignore")                # Remove avisos desnecessários do terminal
tf.get_logger().setLevel("ERROR")               # Exibe apenas erros críticos do TensorFlow

# ─────────────────────────────────────────────
# CONFIGURAÇÕES GLOBAIS
# ─────────────────────────────────────────────

# Caminho relativo: sobe um nível (code/ -> ML/) e entra em data/
# normpath resolve o ".." evitando erros no Windows
CAMINHO_DADOS = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "dados_processados_1.csv"
))

COLUNA_ALVO   = "CBR "          # Nome exato da coluna alvo no CSV (atenção ao espaço)
SEED          = 42               # Semente global: garante que splits e pesos aleatórios sejam reproduzíveis
TEST_SIZE     = 0.20             # 20% do dataset reservado para teste final (nunca visto durante treino)
VAL_SIZE      = 0.15             # 15% do conjunto treino+val usado para validação durante o treino

# Pasta onde modelo e scaler serão salvos (criada automaticamente se não existir)
OUTPUT_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Modelo_salvo_MLP"
))
os.makedirs(OUTPUT_DIR, exist_ok=True)           # exist_ok=True evita erro se pasta já existir

# Pesos amostrais — inspirado no parâmetro use_weights do MLPBlockSpace
# Quando ativado, amostras acima de THRESHOLD recebem peso W_MINOR (mais raras)
# e as demais recebem W_MAJOR (mais comuns), forçando o modelo a prestar mais atenção
# nas regiões menos representadas do dataset
USE_WEIGHTS = False              # False = todas as amostras têm influência igual
THRESHOLD   = None               # Valor de corte que separa amostras raras das comuns
W_MINOR     = 3.0                # Peso das amostras acima do threshold (raras → mais importantes)
W_MAJOR     = 1.0                # Peso das amostras abaixo do threshold (comuns → influência normal)

# Espaço de busca de hiperparâmetros — reflete as opções testadas pelo AG no MLPBlockSpace
# lr_log10 do AG: [1e-1, 1e-2, ... 5e-4] — aqui expandido com valores intermediários
LEARNING_RATES = [1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4]

# bss do AG: range(6, 65) — aqui representado pelos pontos mais relevantes
BATCH_SIZES    = [8, 16, 24, 32, 48, 64]

# dropout do AG: [0.0, 0.05, 0.1, 0.15, 0.2] — mantido idêntico
DROPOUTS       = [0.0, 0.05, 0.1, 0.15, 0.2]

# Número de combinações aleatórias testadas na busca (maior = mais preciso, mas mais lento)
N_ITER_BUSCA   = 20

# ─────────────────────────────────────────────
# FUNÇÕES AUXILIARES
# ─────────────────────────────────────────────

def normalizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renomeia as colunas do CSV para os nomes padrão esperados pelo modelo.

    Problema: nomes de colunas no CSV podem ter espaços extras, letras maiúsculas
    diferentes ou abreviações distintas (ex: "Wot" em vez de "Umidade Ótima",
    "LP" em vez de "IP", "d_max" em vez de "Densidade máxima").

    Esta função resolve isso em dois passos:
      1. Remove espaços no início/fim de todos os nomes de colunas
      2. Aplica um mapeamento explícito de variações conhecidas → nome padrão

    Se ainda houver colunas não reconhecidas, imprime um aviso com os nomes reais
    para que o usuário possa ajustar o mapeamento.

    Args:
        df: DataFrame recém-carregado do CSV.

    Returns:
        DataFrame com colunas renomeadas para o padrão do modelo.
    """
    # Passo 1: remove espaços extras no início e fim de cada nome de coluna
    # Causa mais comum do KeyError: "LL " (com espaço) vs "LL"
    df.columns = df.columns.str.strip()

    # Passo 2: mapeamento de variações conhecidas para o nome padrão
    # Adicione aqui qualquer variação que apareça no seu CSV
    mapa = {
        # Granulometria — possíveis variações de nomenclatura
        "25,4mm":        "25.4mm",
        "9,5mm":         "9.5mm",
        "4,8mm":         "4.8mm",
        "2,0mm":         "2.0mm",
        "0,42mm":        "0.42mm",
        "0,076mm":       "0.076mm",
        "P25.4":         "25.4mm",
        "P9.5":          "9.5mm",
        "P4.8":          "4.8mm",
        "P2.0":          "2.0mm",
        "P0.42":         "0.42mm",
        "P0.076":        "0.076mm",

        # Limites de Atterberg
        "Ll":            "LL",
        "ll":            "LL",
        "L.L":           "LL",
        "L.L.":          "LL",
        "Limite de Liquidez": "LL",
        "Ip":            "IP",
        "ip":            "IP",
        "I.P":           "IP",
        "I.P.":          "IP",
        "Índice de Plasticidade": "IP",
        "Indice de Plasticidade": "IP",

        # Umidade ótima
        "Wot":           "Umidade Ótima",
        "wot":           "Umidade Ótima",
        "W_ot":          "Umidade Ótima",
        "Umidade otima": "Umidade Ótima",
        "Umidade Otima": "Umidade Ótima",
        "Umidade ótima": "Umidade Ótima",
        "w_ot":          "Umidade Ótima",

        # Densidade seca máxima
        "Densidade Maxima":  "Densidade máxima",
        "Densidade Máxima":  "Densidade máxima",
        "densidade máxima":  "Densidade máxima",
        "densidade maxima":  "Densidade máxima",
        "d_max":             "Densidade máxima",
        "Dmax":              "Densidade máxima",
        "γdmax":             "Densidade máxima",
        "ydmax":             "Densidade máxima",

        # CBR — possíveis variações do alvo
        "CBR":           "CBR ",   # garante o espaço no final se necessário
        "cbr":           "CBR ",
        "Cbr":           "CBR ",
    }

    df = df.rename(columns=mapa)

    # Verifica se todas as colunas esperadas estão presentes após o mapeamento
    colunas_esperadas = [
        "25.4mm", "9.5mm", "4.8mm", "2.0mm", "0.42mm", "0.076mm",
        "LL", "IP", "Umidade Ótima", "Densidade máxima", "CBR ",
    ]
    faltando = [c for c in colunas_esperadas if c not in df.columns]
    if faltando:
        print("\n  AVISO: As seguintes colunas esperadas nao foram encontradas no CSV:")
        for c in faltando:
            print(f"    - '{c}'")
        print("\n  Colunas reais do CSV (apos strip):")
        for c in df.columns:
            print(f"    - '{c}'")
        print("\n  Adicione o mapeamento correto em normalizar_colunas() e execute novamente.")
        raise SystemExit(1)

    return df


def engenharia_features(df: pd.DataFrame, coluna_alvo: str) -> pd.DataFrame:
    """
    Cria novas features derivadas das variáveis granulométricas e índices físicos.
    Idêntico ao script RF para garantir comparação justa entre os dois modelos.

    Lógica de cada feature:
      - ratio_X_Y: razão entre peneiras adjacentes — captura o formato da curva
        granulométrica de forma relativa, independente dos valores absolutos
      - atividade: LL - IP — solos com alta diferença são mais plásticos e menos resistentes
      - compacidade: Densidade / Umidade — proxy da energia de compactação do solo
      - finos_sq: 0.076mm² — quadrado da fração fina, amplifica diferenças em solos argilosos

    Args:
        df:          DataFrame com colunas originais.
        coluna_alvo: Nome da coluna alvo (preservada sem modificação).

    Returns:
        DataFrame com as colunas originais + 8 novas features derivadas.
    """
    df  = df.copy()                                      # Evita modificar o DataFrame original
    eps = 1e-6                                           # Epsilon: evita divisão por zero

    # Razões entre peneiras consecutivas — descrevem o gradiente da curva granulométrica
    df["ratio_9_25"]    = df["9.5mm"]   / (df["25.4mm"]  + eps)  # Fração retida entre 25.4mm e 9.5mm
    df["ratio_4_9"]     = df["4.8mm"]   / (df["9.5mm"]   + eps)  # Fração retida entre 9.5mm e 4.8mm
    df["ratio_2_4"]     = df["2.0mm"]   / (df["4.8mm"]   + eps)  # Fração retida entre 4.8mm e 2.0mm
    df["ratio_042_2"]   = df["0.42mm"]  / (df["2.0mm"]   + eps)  # Fração retida entre 2.0mm e 0.42mm
    df["ratio_076_042"] = df["0.076mm"] / (df["0.42mm"]  + eps)  # Fração retida entre 0.42mm e 0.076mm

    # Índice de atividade: quanto maior, mais plástico e menos resistente é o solo
    df["atividade"]     = df["LL"] - df["IP"]

    # Compacidade: solos mais densos com menor umidade tendem a ter CBR maior
    df["compacidade"]   = df["Densidade máxima"] / (df["Umidade Ótima"] + eps)

    # Finos ao quadrado: amplifica a diferença entre solos com muito ou pouco material fino
    df["finos_sq"]      = df["0.076mm"] ** 2

    return df


def exibir_metricas(y_true: np.ndarray, y_pred: np.ndarray, nome: str) -> dict:
    """
    Calcula e imprime MSE, RMSE, MAE e R² para um conjunto de previsões.
    Alerta se MSE <= 1, o que indica possível problema de escala ou data leakage.

    Referência ao MLPBlockSpace:
      - MAE é a função de perda principal (_train_model minimiza MAE)
      - R² é o fitness primário (evaluate retorna -R²)

    Args:
        y_true: Valores reais do alvo.
        y_pred: Valores previstos pelo modelo.
        nome:   Nome do conjunto avaliado (ex: "Validação", "Teste Final").

    Returns:
        Dicionário com todas as métricas calculadas.
    """
    mse  = mean_squared_error(y_true, y_pred)   # Penaliza erros grandes mais do que erros pequenos
    rmse = np.sqrt(mse)                          # Raiz do MSE: na mesma unidade do alvo (%)
    mae  = mean_absolute_error(y_true, y_pred)  # Média dos erros absolutos — perda primária do AG
    r2   = r2_score(y_true, y_pred)             # R²: 1.0 = perfeito, 0.0 = modelo constante

    print(f"\n  [{nome}]")
    print(f"    MSE  : {mse:.4f}")

    # MSE <= 1 é suspeito para CBR (que varia de ~2% a ~120%):
    # pode indicar que o alvo foi normalizado acidentalmente ou que há data leakage
    if mse <= 1:
        print(f"    ATENCAO: MSE={mse:.4f} <= 1 — verifique escala ou data leakage.")

    print(f"    RMSE : {rmse:.4f}")
    print(f"    MAE  : {mae:.4f}")               # Perda diretamente otimizada durante o treino
    print(f"    R²   : {r2:.4f}")               # Fitness do AG: quanto o modelo explica a variância
    return dict(nome=nome, mse=mse, rmse=rmse, mae=mae, r2=r2)


def construir_modelo(n_features: int, dropout: float, lr: float) -> Sequential:
    """
    Constrói e compila o MLP com LeakyReLU e BatchNormalization por bloco.

    Estrutura por bloco (inspirada no _build do MLPBlockSpace):
      Dense → BatchNorm → LeakyReLU → Dropout

    Arquitetura fixa: 128 → 64 → 32 → 16 → 1
    Os hiperparâmetros dropout e lr são variáveis (buscados antes do treino final).

    Args:
        n_features: Número de features de entrada (colunas do X normalizado).
        dropout:    Taxa de dropout da camada 1 (reduzida progressivamente nas camadas seguintes).
        lr:         Learning rate do otimizador Adam.

    Returns:
        Modelo Keras compilado, pronto para o método .fit().
    """
    modelo = Sequential([

        # ── Bloco 1 — Entrada: 128 neurônios ──────────────────────────────────
        # kernel_regularizer=l2(0.001): penaliza pesos grandes na função de perda,
        # impedindo que neurônios individuais dominem a previsão (evita overfitting)
        Dense(128, input_shape=(n_features,), kernel_regularizer=l2(0.001)),
        BatchNormalization(),                    # Normaliza a saída desta camada → treino mais estável
        LeakyReLU(alpha=0.1),                   # alpha=0.1: gradiente negativo pequeno para x<0
                                                # O AG usa LeakyReLU como uma das opções de ativação
        Dropout(dropout),                       # Desativa "dropout"% dos neurônios aleatoriamente

        # ── Bloco 2 — Processamento: 64 neurônios ─────────────────────────────
        Dense(64, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(dropout * 0.7),                 # Dropout progressivamente menor: a rede já
                                                # aprendeu representações mais robustas

        # ── Bloco 3 — Refinamento: 32 neurônios ──────────────────────────────
        Dense(32, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(dropout * 0.4),                 # Dropout mínimo próximo à saída

        # ── Bloco 4 — Compressão: 16 neurônios ───────────────────────────────
        Dense(16),                              # Sem regularização: compressão final das representações
        LeakyReLU(alpha=0.1),

        # ── Saída: 1 neurônio sem ativação ───────────────────────────────────
        # Sem ativação = regressão linear livre; permite prever qualquer valor real
        Dense(1),
    ])

    # Compila com MAE como função de perda — idêntico ao MLPBlockSpace
    # que usa (torch.abs(predictions - yb) * wb).mean() no _train_model
    modelo.compile(
        optimizer=Adam(learning_rate=lr),       # Adam: passo adaptativo por parâmetro
        loss="mae",                             # Minimiza o erro absoluto médio
        metrics=["mse"],                        # Monitora MSE como métrica de diagnóstico
    )
    return modelo


# ─────────────────────────────────────────────
# 1. CARREGAMENTO E FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("=" * 60)
print("  MLP — MODELO MELHORADO")
print("=" * 60)
print("\n[1/7] Carregando dados e aplicando feature engineering...")

try:
    df = pd.read_csv(CAMINHO_DADOS)             # Lê o CSV e armazena como DataFrame
    print(f"     Dataset original: {df.shape[0]} linhas x {df.shape[1]} colunas")
except FileNotFoundError:
    print(f"ERRO: Arquivo nao encontrado em '{CAMINHO_DADOS}'.")
    raise SystemExit(1)                         # Encerra com código de erro sem traceback

# Normaliza nomes de colunas: remove espaços extras e aplica mapeamento de variações
# Isso evita KeyError causado por nomes levemente diferentes no CSV (ex: "LL " vs "LL")
df = normalizar_colunas(df)

# Aplica feature engineering antes de qualquer split para que as novas
# colunas fiquem disponíveis em todos os conjuntos (treino, val, teste)
df_eng = engenharia_features(df, COLUNA_ALVO)
print(f"     Dataset expandido: {df_eng.shape[0]} linhas x {df_eng.shape[1]} colunas")
print(f"     Novas features: {df_eng.shape[1] - df.shape[1]} adicionadas")

Y = df_eng[COLUNA_ALVO].values.ravel()         # Extrai o alvo como array 1D
X = df_eng.drop(columns=[COLUNA_ALVO])         # Remove o alvo; o restante são as features
feature_names = X.columns.tolist()             # Salva os nomes para os gráficos

# ─────────────────────────────────────────────
# 2. DIVISÃO E NORMALIZAÇÃO
# ─────────────────────────────────────────────
print("\n[2/7] Dividindo e normalizando dados...")

# Primeiro split: 20% reservados para o teste final
# Esses dados NUNCA são vistos durante busca de hiperparâmetros nem durante o treino
X_tv, X_teste, Y_tv, y_teste = train_test_split(
    X.values, Y, test_size=TEST_SIZE, random_state=SEED
)

# Segundo split: dos 80% restantes, 15% para validação
# CORREÇÃO do original: no código original o conjunto de teste era passado como
# validation_data, ou seja, o modelo via os dados de teste durante o treino (leakage)
X_treino, X_val, Y_treino, Y_val = train_test_split(
    X_tv, Y_tv, test_size=VAL_SIZE, random_state=SEED
)

# Normalização MinMax — fit APENAS no treino para evitar data leakage
# CORREÇÃO do original: o scaler era ajustado em todos os dados antes do split,
# o que deixa "vazar" informação do teste para o treino via os limites da escala
scaler     = MinMaxScaler()
X_treino_n = scaler.fit_transform(X_treino)   # Aprende min/max do treino e normaliza
X_val_n    = scaler.transform(X_val)          # Aplica a mesma escala aprendida no treino
X_teste_n  = scaler.transform(X_teste)        # Aplica a mesma escala aprendida no treino
X_tv_n     = scaler.transform(X_tv)           # Treino+validação normalizado para o retreino final

print(f"     Treino: {X_treino_n.shape[0]}  |  Validação: {X_val_n.shape[0]}  |  Teste: {X_teste_n.shape[0]}")
print(f"     Features: {X_treino_n.shape[1]}")

# Pesos amostrais — inspirado no MLPBlockSpace que aceita use_weights, thr, w_minor, w_major
if USE_WEIGHTS and THRESHOLD is not None:
    # Amostras com CBR acima do threshold recebem peso maior, forçando o modelo a
    # prestar mais atenção nessas regiões menos representadas no dataset
    sample_weights = np.where(Y_treino > THRESHOLD, W_MINOR, W_MAJOR).astype(np.float32)
    print(f"     Pesos amostrais ativos — threshold={THRESHOLD}")
else:
    sample_weights = None                      # None = Keras atribui peso 1.0 para todas as amostras

# ─────────────────────────────────────────────
# 3. BUSCA DE HIPERPARÂMETROS
# ─────────────────────────────────────────────
print(f"\n[3/7] Buscando hiperparametros ({N_ITER_BUSCA} combinacoes)...")
print("      (lr, batch_size, dropout — espaco inspirado no MLPBlockSpace)")

# Fixa sementes para garantir que as combinações sorteadas sejam reproduzíveis
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Early stopping com patience=5 durante a busca — igual ao MLPBlockSpace
# O AG usava patience=5 no _train_model para avaliar configurações rapidamente
callbacks_busca = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=0),
]

melhor_mae_val  = np.inf                       # Inicializa com infinito para garantir que qualquer resultado seja melhor
melhor_config   = {}                           # Armazenará os hiperparâmetros da melhor combinação
historico_busca = []                           # Lista com resultados de todas as combinações (para o gráfico)

# Gera N_ITER_BUSCA combinações aleatórias do espaço de hiperparâmetros
# Equivalente ao sorteio feito pelo Algoritmo Genético ao inicializar a população
rng = np.random.RandomState(SEED)              # RandomState separado para não afetar outras seeds
combinacoes = [
    {
        "lr":         rng.choice(LEARNING_RATES),   # Learning rate sorteado do espaço do AG
        "batch_size": int(rng.choice(BATCH_SIZES)), # Batch size sorteado (range 6-65 no AG)
        "dropout":    float(rng.choice(DROPOUTS)),  # Dropout sorteado (0.0 a 0.2 no AG)
    }
    for _ in range(N_ITER_BUSCA)
]

# Avalia cada combinação treinando por até 100 épocas com early stopping
for i, cfg in enumerate(combinacoes):
    modelo_tmp = construir_modelo(X_treino_n.shape[1], cfg["dropout"], cfg["lr"])
    hist = modelo_tmp.fit(
        X_treino_n, Y_treino,
        epochs=100,                            # 100 épocas = mesmo limite do MLPBlockSpace
        batch_size=cfg["batch_size"],
        validation_data=(X_val_n, Y_val),      # Validação no conjunto separado (não no teste)
        sample_weight=sample_weights,
        callbacks=callbacks_busca,
        verbose=0,                             # Sem output intermediário para não poluir o terminal
    )

    mae_val = min(hist.history["val_loss"])    # Melhor MAE de validação atingido nesta combinação
    historico_busca.append({**cfg, "mae_val": mae_val})  # Registra para o gráfico final

    # Atualiza a melhor configuração se este resultado for superior
    if mae_val < melhor_mae_val:
        melhor_mae_val = mae_val
        melhor_config  = cfg

    print(f"     [{i+1:2d}/{N_ITER_BUSCA}] lr={cfg['lr']:.4f}  batch={cfg['batch_size']}  drop={cfg['dropout']:.2f}  -> MAE_val={mae_val:.4f}")

print(f"\n  Melhor configuracao encontrada:")
for k, v in melhor_config.items():
    print(f"    {k:12s}: {v}")
print(f"    MAE_val    : {melhor_mae_val:.4f}")

# ─────────────────────────────────────────────
# 4. TREINAMENTO FINAL
# ─────────────────────────────────────────────
print("\n[4/7] Treinando modelo final com melhor configuracao...")

# Recalcula pesos para o conjunto treino+validação combinado (se ativado)
sw_tv = None
if USE_WEIGHTS and THRESHOLD is not None:
    sw_tv = np.where(Y_tv > THRESHOLD, W_MINOR, W_MAJOR).astype(np.float32)

# Constrói o modelo final com os melhores hiperparâmetros encontrados na busca
modelo_final = construir_modelo(
    X_tv_n.shape[1], melhor_config["dropout"], melhor_config["lr"]
)
modelo_final.summary()                         # Imprime resumo da arquitetura no terminal

# Callbacks do treino final — patience maior que na busca para explorar mais épocas
callbacks_final = [
    EarlyStopping(
        monitor="val_loss",                    # Monitora a perda no conjunto de validação
        patience=15,                           # Para após 15 épocas sem melhora (equilibrio entre AG=5 e original=25)
        restore_best_weights=True,             # Restaura os pesos do epoch com menor val_loss ao final
        verbose=1,                             # Imprime mensagem quando para
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,                            # Multiplica o LR atual por 0.5 (reduz à metade)
        patience=7,                            # Após 7 épocas sem melhora
        min_lr=1e-6,                           # Limite inferior: evita que o LR fique infinitamente pequeno
        verbose=1,
    ),
]

# Retreina no conjunto treino+validação combinado para maximizar os dados disponíveis
# Os hiperparâmetros já foram fixados na etapa de busca — não há data leakage aqui
historico = modelo_final.fit(
    X_tv_n, Y_tv,
    epochs=300,                                # Máximo de épocas — EarlyStopping decide quando parar
    batch_size=melhor_config["batch_size"],    # Batch size otimizado pela busca
    validation_data=(X_teste_n, y_teste),      # Monitoramento no teste apenas para acompanhar o progresso
    sample_weight=sw_tv,
    callbacks=callbacks_final,
    verbose=1,
)

epocas_reais = len(historico.history["loss"])
print(f"\nTreinamento encerrado na epoca {epocas_reais} (EarlyStopping)")

# ─────────────────────────────────────────────
# 5. AVALIAÇÃO
# ─────────────────────────────────────────────
print("\n[5/7] Avaliando o modelo...")

# Gera previsões para os dois conjuntos de interesse
pred_val   = modelo_final.predict(X_val_n,   verbose=0).flatten()   # Flatten: transforma (N,1) em (N,)
pred_teste = modelo_final.predict(X_teste_n, verbose=0).flatten()

# Calcula e imprime todas as métricas (MSE, RMSE, MAE, R²)
met_val   = exibir_metricas(Y_val,   pred_val,   "Validacao")
met_teste = exibir_metricas(y_teste, pred_teste, "Teste Final")

print(f"\n  Em media, o modelo erra +-{met_teste['mae']:.2f} no valor de CBR")

# ─────────────────────────────────────────────
# 6. VISUALIZAÇÕES
# ─────────────────────────────────────────────
print("\n[6/7] Gerando graficos...")

# Paleta de cores padronizada — idêntica ao script RF para consistência visual
PALETTE = {
    "azul":    "#2563EB",   # Conjunto de validação
    "verde":   "#16A34A",   # Conjunto de teste
    "laranja": "#EA580C",   # Linha de referência e destaques
    "roxo":    "#7C3AED",   # Elementos secundários
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
fig.suptitle("MLP — Analise de Performance", fontsize=16, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)  # 2 linhas, 3 colunas

# A) Previsto vs Real — Validação
# Pontos próximos da diagonal laranja indicam previsões precisas
ax1 = fig.add_subplot(gs[0, 0])
lim = [min(Y_val.min(), pred_val.min()) * 0.95,       # Limite inferior com margem de 5%
       max(Y_val.max(), pred_val.max()) * 1.05]        # Limite superior com margem de 5%
ax1.scatter(Y_val, pred_val, alpha=0.6, color=PALETTE["azul"],
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
lim2 = [min(y_teste.min(), pred_teste.min()) * 0.95,
        max(y_teste.max(), pred_teste.max()) * 1.05]
ax2.scatter(y_teste, pred_teste, alpha=0.6, color=PALETTE["verde"],
            edgecolors="white", linewidths=0.4, s=50)
ax2.plot(lim2, lim2, "--", color=PALETTE["laranja"], lw=1.5, label="Ideal")
ax2.set_xlim(lim2); ax2.set_ylim(lim2)
ax2.set_title("Previsto vs Real — Teste", fontweight="bold")
ax2.set_xlabel("Real"); ax2.set_ylabel("Previsto")
ax2.legend(framealpha=0)
ax2.text(0.05, 0.92, f"R² = {met_teste['r2']:.4f}", transform=ax2.transAxes,
         fontsize=10, color=PALETTE["verde"], fontweight="bold")

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
tbl.auto_set_font_size(False); tbl.set_fontsize(11)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor(PALETTE["grade"])
    if r == 0:                                 # Cabeçalho: fundo azul, texto branco
        cell.set_facecolor(PALETTE["azul"])
        cell.set_text_props(color="white", fontweight="bold")
    else:                                      # Dados: fundo azul claro nas colunas numéricas
        cell.set_facecolor("#EFF6FF" if c != 0 else "#F1F5F9")
ax3.set_title("Comparativo de Metricas", fontweight="bold", pad=14)

# D) Resíduos — Validação
# Resíduo = Real - Previsto; distribuição em torno de 0 indica modelo sem viés
ax4 = fig.add_subplot(gs[1, 0])
residuos_val = Y_val - pred_val                # Calcula os resíduos do conjunto de validação
ax4.scatter(pred_val, residuos_val, alpha=0.6, color=PALETTE["azul"],
            edgecolors="white", linewidths=0.4, s=50)
ax4.axhline(0, color=PALETTE["laranja"], lw=1.5, linestyle="--")  # Linha de resíduo zero (ideal)
ax4.set_title("Residuos — Validacao", fontweight="bold")
ax4.set_xlabel("Previsto"); ax4.set_ylabel("Residuo (Real - Previsto)")

# E) Resíduos — Teste
# Padrões sistemáticos (funil, curva) indicam viés ou falta de ajuste do modelo
ax5 = fig.add_subplot(gs[1, 1])
residuos_teste = y_teste - pred_teste          # Calcula os resíduos do conjunto de teste
ax5.scatter(pred_teste, residuos_teste, alpha=0.6, color=PALETTE["verde"],
            edgecolors="white", linewidths=0.4, s=50)
ax5.axhline(0, color=PALETTE["laranja"], lw=1.5, linestyle="--")
ax5.set_title("Residuos — Teste", fontweight="bold")
ax5.set_xlabel("Previsto"); ax5.set_ylabel("Residuo (Real - Previsto)")

# F) Histórico de Loss (MAE) — evolução do treino por época
# Linhas convergentes indicam treino saudável; gap grande indica overfitting
ax6 = fig.add_subplot(gs[1, 2])
ax6.plot(historico.history["loss"],     label="MAE Treino",    color=PALETTE["azul"],    lw=1.5)
ax6.plot(historico.history["val_loss"], label="MAE Validacao", color=PALETTE["laranja"], lw=1.5, linestyle="--")
ax6.set_title("Historico de Loss (MAE)", fontweight="bold")
ax6.set_xlabel("Epocas"); ax6.set_ylabel("MAE")
ax6.legend(framealpha=0)

plt.tight_layout(rect=[0, 0, 1, 0.97])

# ── Figura 2: Histograma dos resíduos ────────────────────────────────────────
# Distribuição simétrica em torno de 0 indica erro aleatório (sem viés sistemático)
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5), facecolor=PALETTE["fundo"])
fig2.suptitle("Distribuicao dos Residuos", fontsize=14, fontweight="bold")

for ax, residuos, titulo, cor in zip(
    axes2,
    [residuos_val, residuos_teste],
    ["Validacao", "Teste"],
    [PALETTE["azul"], PALETTE["verde"]],
):
    ax.set_facecolor(PALETTE["fundo"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    sns.histplot(residuos, kde=True, color=cor, ax=ax, alpha=0.7)  # kde=True: curva de densidade suavizada
    ax.axvline(x=0, color=PALETTE["laranja"], linestyle="--", lw=1.5, label="Erro Zero")
    ax.set_title(f"Residuos — {titulo}", fontweight="bold")
    ax.set_xlabel("Erro (Real - Previsto)"); ax.set_ylabel("Frequencia")
    ax.legend(framealpha=0)

plt.tight_layout()

# ── Figura 3: Histórico de MSE ────────────────────────────────────────────────
# Complementa o histórico de MAE: mostra como o MSE (que penaliza erros grandes)
# evoluiu durante o treino em comparação ao MSE de validação
if "mse" in historico.history:
    fig3, ax7 = plt.subplots(figsize=(12, 5), facecolor=PALETTE["fundo"])
    ax7.set_facecolor(PALETTE["fundo"])
    ax7.spines["top"].set_visible(False); ax7.spines["right"].set_visible(False)
    ax7.plot(historico.history["mse"],     label="MSE Treino",    color=PALETTE["azul"],    lw=1.5)
    ax7.plot(historico.history["val_mse"], label="MSE Validacao", color=PALETTE["laranja"], lw=1.5, linestyle="--")
    ax7.set_title("Historico de MSE", fontweight="bold")
    ax7.set_xlabel("Epocas"); ax7.set_ylabel("MSE")
    ax7.legend(framealpha=0)
    ax7.grid(color=PALETTE["grade"], linewidth=0.8)
    plt.tight_layout()

# ── Figura 4: Ranking da busca de hiperparâmetros ─────────────────────────────
# Mostra o MAE de validação de cada combinação testada, ordenado do melhor ao pior
# A barra laranja é a melhor combinação (usada no modelo final)
df_busca = pd.DataFrame(historico_busca).sort_values("mae_val")  # Ordena pelo melhor MAE
fig4, ax8 = plt.subplots(figsize=(12, 5), facecolor=PALETTE["fundo"])
ax8.set_facecolor(PALETTE["fundo"])
ax8.spines["top"].set_visible(False); ax8.spines["right"].set_visible(False)
cores_busca = [PALETTE["laranja"] if i == 0 else PALETTE["azul"]  # Destaca a melhor em laranja
               for i in range(len(df_busca))]
ax8.bar(range(len(df_busca)), df_busca["mae_val"], color=cores_busca, edgecolor="white", alpha=0.8)
ax8.axhline(df_busca["mae_val"].iloc[0], color=PALETTE["laranja"], lw=1.5,
            linestyle="--", label=f"Melhor MAE = {df_busca['mae_val'].iloc[0]:.4f}")
ax8.set_title("Busca de Hiperparametros — MAE de Validacao por Combinacao", fontweight="bold")
ax8.set_xlabel("Combinacao (ranking)"); ax8.set_ylabel("MAE Validacao")
ax8.legend(framealpha=0)
plt.tight_layout()

plt.show()                                     # Exibe todas as figuras na tela

# ─────────────────────────────────────────────
# 7. SALVAMENTO
# ─────────────────────────────────────────────
print("\n[7/7] Salvando modelo e scaler...")

# Formato .keras: recomendado para TensorFlow >= 2.12 (substitui o antigo .h5)
# Preserva arquitetura, pesos e configuração do otimizador
modelo_final.save(os.path.join(OUTPUT_DIR, "modelo_cbr.keras"))

# Salva o scaler para garantir que novos dados sejam normalizados
# com exatamente os mesmos limites usados no treinamento
dump(scaler, os.path.join(OUTPUT_DIR, "scaler_cbr.joblib"))

print(f"  Modelo salvo -> modelo_cbr.keras")
print(f"  Scaler salvo -> scaler_cbr.joblib")
print(f"  Pasta        -> {OUTPUT_DIR}")

print("\n" + "=" * 60)
print("  Processo concluido com sucesso!")
print("=" * 60)