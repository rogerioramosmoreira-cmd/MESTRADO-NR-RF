"""
Sistema de Previsão de CBR — Random Forest
Lê metadados.json gerado pelo RANDOM_FOREST.py para replicar
automaticamente o cenário (D1/D2/D3/D4) e a engenharia de features
da Tabela 2 usados no treinamento.

Uso:
  1. Defina CENARIO para o mesmo cenário usado no treinamento
  2. Execute e informe os valores solicitados
"""

import os
import json
import numpy as np
import pandas as pd
from joblib import load

# ─────────────────────────────────────────────
# CONFIGURAÇÕES
# ─────────────────────────────────────────────

CENARIO = "D4"  # ← mesmo cenário usado no RANDOM_FOREST.py

# Pasta do modelo — deve coincidir com a pasta gerada pelo treinamento
PASTA_MODELOS = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), f"Modelo_salvo_RF_{CENARIO}"
))

ARQUIVO_MODELO    = os.path.join(PASTA_MODELOS, "rf_modelo_final.joblib")
ARQUIVO_SCALER    = os.path.join(PASTA_MODELOS, "scaler.joblib")
ARQUIVO_METADADOS = os.path.join(PASTA_MODELOS, "metadados.json")

# 10 features originais sempre coletadas do usuário
# (a engenharia reduz/deriva conforme o cenário)
NOMES_FEATURES_ORIGINAIS = [
    "25.4mm", "9.5mm", "4.8mm", "2.0mm", "0.42mm", "0.076mm",
    "LL", "IP", "Umidade Ótima", "Densidade máxima",
]

# Limites físicos por feature (baseados nas normas ABNT/DNIT)
LIMITES = {
    "25.4mm":           (0.0,    100.0),
    "9.5mm":            (0.0,    100.0),
    "4.8mm":            (0.0,    100.0),
    "2.0mm":            (0.0,    100.0),
    "0.42mm":           (0.0,    100.0),
    "0.076mm":          (0.0,    100.0),
    "LL":               (0.0,    100.0),
    "IP":               (0.0,     80.0),
    "Umidade Ótima":    (5.0,     40.0),
    "Densidade máxima": (1200.0, 2200.0),
}

# ─────────────────────────────────────────────
# FUNÇÕES
# ─────────────────────────────────────────────

def carregar_artefatos():
    """
    Carrega modelo, scaler e metadados do cenário.
    Os metadados contêm as features originais e derivadas usadas no treino,
    garantindo que a engenharia aplicada aqui seja idêntica à do treinamento.

    Returns:
        Tupla (modelo, scaler, metadados).
    """
    print(f"Carregando artefatos do cenário {CENARIO}...")
    for caminho in [ARQUIVO_MODELO, ARQUIVO_SCALER, ARQUIVO_METADADOS]:
        if not os.path.exists(caminho):
            print(f"\nERRO: Arquivo não encontrado — {caminho}")
            print(f"Execute RANDOM_FOREST.py com CENARIO='{CENARIO}' antes de usar este script.")
            raise SystemExit(1)

    modelo    = load(ARQUIVO_MODELO)
    scaler    = load(ARQUIVO_SCALER)
    with open(ARQUIVO_METADADOS, "r", encoding="utf-8") as f:
        metadados = json.load(f)

    print(f"  Cenário             : {metadados['cenario']}")
    print(f"  Features originais  : {metadados['features_cenario']}")
    print(f"  Features totais     : {metadados['feature_names']}")
    return modelo, scaler, metadados


def engenharia_features(valores_dict: dict, features_cenario: list) -> pd.DataFrame:
    """
    Aplica a engenharia de features da Tabela 2 com base no cenário do treino.

    Recebe os 10 valores originais em dicionário e gera as derivadas
    somente para as features presentes no cenário selecionado.

    Fórmulas (Tabela 2):
      ratio_9_25    = P9.5mm   / P25.4mm
      ratio_4_9     = P4.8mm   / P9.5mm
      ratio_2_4     = P2.0mm   / P4.8mm
      ratio_042_2   = P0.42mm  / P2.0mm
      ratio_076_042 = P0.076mm / P0.42mm
      atividade     = LL - LP  (LP = LL - IP)
      compactacao   = Densidade máxima / Umidade Ótima
      finos_sq      = (P0.076mm)²

    Args:
        valores_dict:     Dicionário com os 10 valores originais coletados.
        features_cenario: Lista de features originais do cenário (de metadados.json).

    Returns:
        DataFrame com 1 linha contendo as features originais do cenário + derivadas.
    """
    eps = 1e-6
    d   = {k: [v] for k, v in valores_dict.items()}  # Expande para DataFrame de 1 linha
    df  = pd.DataFrame(d)

    # ── Ratios granulométricas ────────────────────────────────────────────────
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
    if {"LL", "IP"}.issubset(features_cenario):
        LP = df["LL"] - df["IP"]             # LP = LL - IP
        df["atividade"] = df["LL"] - LP      # = IP (conforme Tabela 2: LL - LP)

    # ── Compactacao — ρdmax / Wot  (Tabela 2) ────────────────────────────────
    if {"Densidade máxima", "Umidade Ótima"}.issubset(features_cenario):
        df["compactacao"] = df["Densidade máxima"] / (df["Umidade Ótima"] + eps)

    # ── Finos ao quadrado — (P0.076mm)²  (Tabela 2) ──────────────────────────
    if "0.076mm" in features_cenario:
        df["finos_sq"] = df["0.076mm"] ** 2

    # Seleciona colunas na mesma ordem usada no treino (metadados["feature_names"])
    # garante que o scaler.transform receba exatamente a mesma estrutura
    colunas_finais = features_cenario + [
        c for c in df.columns if c not in NOMES_FEATURES_ORIGINAIS
    ]
    return df[colunas_finais]


def solicitar_valor(feature: str, minimo: float, maximo: float) -> float:
    """
    Solicita e valida um valor numérico dentro dos limites físicos.
    Aceita vírgula ou ponto como separador decimal.
    Só coleta features que fazem parte do cenário selecionado.
    """
    while True:
        try:
            entrada = input(f"  {feature:20s} [{minimo} – {maximo}]: ")
            valor   = float(entrada.replace(",", "."))
            if not (minimo <= valor <= maximo):
                print(f"  AVISO: Fora do intervalo ({minimo} – {maximo}). Tente novamente.")
                continue
            return valor
        except ValueError:
            print("  ERRO: Entrada inválida. Use número (ex: 12.5 ou 12,5).")


def prever_cbr(valores_dict: dict, modelo, scaler, metadados: dict) -> float:
    """
    Aplica a engenharia de features do cenário, normaliza e retorna o CBR previsto.

    Fluxo:
      1. Aplica engenharia de features da Tabela 2 (conforme cenário do treino)
      2. Reordena colunas conforme metadados["feature_names"] (ordem exata do scaler)
      3. Normaliza com o scaler salvo no treino
      4. Faz a previsão com o ensemble VotingRegressor

    Args:
        valores_dict: Dicionário com os 10 valores originais.
        modelo:       VotingRegressor carregado.
        scaler:       MinMaxScaler carregado.
        metadados:    Dicionário lido de metadados.json.

    Returns:
        CBR previsto como float (%).
    """
    features_cenario = metadados["features_cenario"]
    feature_names    = metadados["feature_names"]    # ordem exata usada no treino

    # Gera features originais do cenário + derivadas da Tabela 2
    df_entrada = engenharia_features(valores_dict, features_cenario)

    # Garante a ordem exata das colunas que o scaler espera
    df_entrada = df_entrada[feature_names]

    # Normaliza com os mesmos parâmetros do treino
    dados_norm = scaler.transform(df_entrada)

    # Previsão
    return float(modelo.predict(dados_norm)[0])


# ─────────────────────────────────────────────
# EXECUÇÃO PRINCIPAL
# ─────────────────────────────────────────────

print("=" * 55)
print(f"  PREVISÃO DE CBR — RANDOM FOREST | Cenário {CENARIO}")
print("=" * 55)

# 1. Carrega modelo, scaler e metadados
modelo_rf, scaler_rf, metadados = carregar_artefatos()

features_cenario = metadados["features_cenario"]

# 2. Coleta apenas as features do cenário (não solicita as ausentes)
print(f"\nInforme os dados da amostra de solo:")
print(f"  Cenário {CENARIO}: {len(features_cenario)} variáveis necessárias")
print(f"  (Aceita vírgula ou ponto como separador decimal)\n")

valores_dict = {}
for feature in NOMES_FEATURES_ORIGINAIS:
    if feature in features_cenario:              # solicita apenas features do cenário
        minimo, maximo = LIMITES[feature]
        valores_dict[feature] = solicitar_valor(feature, minimo, maximo)
    else:
        # Features fora do cenário ainda são necessárias para calcular derivadas
        # que dependem de peneiras presentes no cenário
        # Ex: ratio_076_042 exige 0.42mm, mesmo que 0.42mm não esteja no D1
        pass

# Garante que o dict tenha todos os valores necessários para a engenharia
# Para features ausentes do cenário mas necessárias para alguma derivada,
# usa 0.0 como valor neutro (não afetará o modelo — a derivada não é gerada)
for feature in NOMES_FEATURES_ORIGINAIS:
    if feature not in valores_dict:
        valores_dict[feature] = 0.0

# 3. Previsão
print("\nCalculando previsão...")
cbr_estimado = prever_cbr(valores_dict, modelo_rf, scaler_rf, metadados)

# 4. Resultado
print("\n" + "=" * 55)
print(f"  CBR PREVISTO (Random Forest | Cenário {CENARIO}): {cbr_estimado:.2f} %")

if cbr_estimado < 2:
    print("  AVISO: CBR muito baixo — verifique os dados inseridos.")
elif cbr_estimado > 100:
    print("  AVISO: CBR acima de 100% — valor atípico.")

print("=" * 55)