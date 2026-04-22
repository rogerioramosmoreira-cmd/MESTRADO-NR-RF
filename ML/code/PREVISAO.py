"""
Sistema de Previsão de CBR — Rede neural
Carrega o modelo e o scaler salvos pelo treinamento e estima o CBR
a partir dos dados inseridos pelo usuário.

Limites de entrada baseados na dissertação:
  
"""

import os                       # Manipulação de caminhos
import numpy as np              # Operações numéricas
import pandas as pd             # Criação do DataFrame para feature engineering
from joblib import load         # Carregamento do modelo e scaler serializados

# ─────────────────────────────────────────────
# CONFIGURAÇÕES
# ─────────────────────────────────────────────

# Pasta onde o treinamento salvou o modelo e o scaler (caminho relativo ao script)
PASTA_MODELOS  = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Modelo_salvo_RF"))

# Nomes dos arquivos gerados pelo script de treinamento
ARQUIVO_MODELO = os.path.join(PASTA_MODELOS, "rf_modelo_final.joblib")
ARQUIVO_SCALER = os.path.join(PASTA_MODELOS, "scaler.joblib")

# Features originais exigidas pelo modelo, na mesma ordem usada no treinamento
# As 8 features derivadas são calculadas automaticamente por engenharia_features()
NOMES_FEATURES = [
    "25.4mm", "9.5mm", "4.8mm", "2.0mm", "0.42mm", "0.076mm",
    "LL", "IP", "Umidade Ótima", "Densidade máxima",
]

# ─────────────────────────────────────────────────────────────────────────────
# LIMITES DE ENTRADA — baseados na dissertação (Moreira, 2025) e normas técnicas
# ─────────────────────────────────────────────────────────────────────────────
#
# A dissertação utiliza parâmetros de granulometria, Limites de Atterberg,
# umidade ótima e densidade seca máxima de solos brasileiros para pavimentação
# (DNIT, 2006; NBR 7182/86 — Ensaio de Compactação; NBR 6459/17 — Limite de
# Liquidez; NBR 7180/16 — Limite de Plasticidade).
#
# Critérios adotados para cada parâmetro:
#
#   GRANULOMETRIA (peneiras — porcentagem passante acumulada, %):
#     Cada peneira representa a fração do solo que passa por aquela abertura.
#     A curva granulométrica é acumulada: a peneira maior sempre tem valor >= peneira menor.
#     Intervalo técnico: 0.0 a 100.0% — válido para qualquer tipo de solo
#     (pedregulho, areia, silte, argila) conforme ABNT NBR 7181.
#
#   LIMITES DE ATTERBERG (LL e IP, %):
#     - LL (Limite de Liquidez): solo passa do estado plástico ao líquido.
#       Solos brasileiros lateríticos: tipicamente 20 a 80%.
#       Intervalo adotado: 0.0 a 100.0% — cobre solos não plásticos a argilas expansivas.
#     - IP (Índice de Plasticidade = LL - LP): mede a faixa plástica do solo.
#       Solos NP (não plásticos): IP = 0. Argilas expansivas: IP > 50%.
#       Intervalo adotado: 0.0 a 80.0% — conforme limites práticos de solos tropicais.
#
#   UMIDADE ÓTIMA (Wot, %):
#     Teor de umidade no ponto de máxima densidade seca (Proctor Normal, NBR 7182).
#     Solos granulares: ~6 a 12%. Solos argilosos tropicais: ~15 a 35%.
#     Intervalo adotado: 5.0 a 40.0% — cobre todos os solos citados na dissertação.
#
#   DENSIDADE SECA MÁXIMA (γdmax, kg/m³):
#     Peso específico seco máximo obtido no ensaio de compactação Proctor Normal.
#     Solos arenosos compactos: ~1900 a 2200 kg/m³.
#     Solos argilosos lateríticos: ~1400 a 1800 kg/m³.
#     Intervalo adotado: 1200.0 a 2200.0 kg/m³ — cobre toda a variação de solos
#     tropicais e lateríticos brasileiros.
#
# Formato: "nome_da_feature": (valor_mínimo, valor_máximo)
# ─────────────────────────────────────────────────────────────────────────────
LIMITES = {
    # ── Granulometria — porcentagem passante acumulada (%) ────────────────────
    # Abertura 25.4mm: separa pedregulho grosso de médio
    # Em solos finos (siltes/argilas) este valor é frequentemente 100%
    "25.4mm":           (0.0, 100.0),

    # Abertura 9.5mm: separa pedregulho médio de fino
    # Solos bem graduados de pavimentação: 60 a 100%
    "9.5mm":            (0.0, 100.0),

    # Abertura 4.8mm: peneira n°4 — separa pedregulho de areia
    # Referência para classificação HRB/USCS amplamente usada na dissertação
    "4.8mm":            (0.0, 100.0),

    # Abertura 2.0mm: peneira n°10 — início da fração areia
    "2.0mm":            (0.0, 100.0),

    # Abertura 0.42mm: peneira n°40 — separa areia grossa de fina
    # Importante para Limites de Atterberg (determinados no material < 0.42mm)
    "0.42mm":           (0.0, 100.0),

    # Abertura 0.076mm: peneira n°200 — separa areia fina de silte/argila
    # Fração fina (< 0.076mm) controla plasticidade e compressibilidade
    # Solos lateríticos brasileiros: frequentemente 20 a 80%
    "0.076mm":          (0.0, 100.0),

    # ── Limites de Atterberg ──────────────────────────────────────────────────
    # LL: solo passa do estado plástico ao líquido (NBR 6459)
    # Solos não plásticos: LL indefinido (excluídos da análise por convenção)
    # Solos lateríticos goianos (contexto da dissertação): 25 a 70%
    "LL":               (0.0, 100.0),

    # IP: faixa plástica do solo (LL - LP); IP=0 indica solo não plástico
    # Argilas de baixa plasticidade (CL/ML): IP 4 a 15%
    # Argilas de alta plasticidade (CH/MH): IP > 20%
    # Limite superior 80%: argilas expansivas extremas (montmorilonita)
    "IP":               (0.0,  80.0),

    # ── Parâmetros de Compactação (Proctor Normal — NBR 7182) ─────────────────
    # Umidade ótima: teor de umidade no pico da curva de compactação
    # Solos granulares (brita, areia): Wot 6 a 12%
    # Solos argilosos lateríticos (predominantes no Brasil Central): Wot 15 a 30%
    "Umidade Ótima":    (5.0,  40.0),

    # Densidade seca máxima: γdmax no ponto de compactação ótima
    # Solos pedregulhosos/arenosos: 1900 a 2200 kg/m³
    # Solos argilosos lateríticos: 1400 a 1800 kg/m³
    # Solos muito porosos (latossolos): pode chegar a 1200 kg/m³
    "Densidade máxima": (1200.0, 2200.0),
}

# ─────────────────────────────────────────────
# FUNÇÕES
# ─────────────────────────────────────────────

def engenharia_features(valores: list) -> pd.DataFrame:
    """
    Recebe os 10 valores originais e gera o DataFrame completo com as
    18 features (10 originais + 8 derivadas) esperadas pelo modelo RF.

    Esta função é IDÊNTICA à usada no treinamento. É obrigatório aplicá-la
    antes de normalizar os dados, pois o scaler foi ajustado nas 18 features.

    Features derivadas calculadas (mesma lógica do script de treinamento):
      - ratio_9_25:    9.5mm / 25.4mm  — gradiente granulométrico grosso
      - ratio_4_9:     4.8mm / 9.5mm   — gradiente granulométrico médio-grosso
      - ratio_2_4:     2.0mm / 4.8mm   — gradiente granulométrico médio
      - ratio_042_2:   0.42mm / 2.0mm  — gradiente granulométrico médio-fino
      - ratio_076_042: 0.076mm / 0.42mm — gradiente granulométrico fino
      - atividade:     LL - IP          — índice de atividade da argila
      - compacidade:   Densidade / Umidade — proxy de energia de compactação
      - finos_sq:      0.076mm²         — amplifica diferenças em solos argilosos

    Args:
        valores: Lista com os 10 valores originais na ordem de NOMES_FEATURES.

    Returns:
        DataFrame com 1 linha e 18 colunas (originais + derivadas).
    """
    eps = 1e-6  # Evita divisão por zero nas razões entre peneiras

    # Desempacota na ordem exata de NOMES_FEATURES
    p25, p9, p4, p2, p042, p076, ll, ip, umidade, densidade = valores

    # Monta dicionário com as 10 features originais
    dados = {
        "25.4mm":          p25,
        "9.5mm":           p9,
        "4.8mm":           p4,
        "2.0mm":           p2,
        "0.42mm":          p042,
        "0.076mm":         p076,
        "LL":              ll,
        "IP":              ip,
        "Umidade Ótima":   umidade,
        "Densidade máxima":densidade,
    }

    # Calcula as 8 features derivadas (idênticas ao treinamento)
    dados["ratio_9_25"]    = p9   / (p25  + eps)
    dados["ratio_4_9"]     = p4   / (p9   + eps)
    dados["ratio_2_4"]     = p2   / (p4   + eps)
    dados["ratio_042_2"]   = p042 / (p2   + eps)
    dados["ratio_076_042"] = p076 / (p042 + eps)
    dados["atividade"]     = ll   - ip
    dados["compacidade"]   = densidade / (umidade + eps)
    dados["finos_sq"]      = p076 ** 2

    return pd.DataFrame([dados])


def carregar_artefatos(arquivo_modelo: str, arquivo_scaler: str):
    """
    Carrega o ensemble Random Forest e o scaler do disco.

    Args:
        arquivo_modelo: Caminho para o arquivo .joblib do ensemble.
        arquivo_scaler: Caminho para o arquivo .joblib do scaler.

    Returns:
        Tupla (modelo, scaler) prontos para uso.

    Raises:
        SystemExit: Se os arquivos não forem encontrados ou ocorrer erro de leitura.
    """
    print("Carregando modelo e scaler...")
    try:
        modelo = load(arquivo_modelo)   # Carrega o VotingRegressor (RF+GB+ET)
        scaler = load(arquivo_scaler)   # Carrega o MinMaxScaler ajustado no treino
        print("Modelo e scaler carregados com sucesso.")
        return modelo, scaler
    except FileNotFoundError as e:
        print(f"\nERRO: Arquivo não encontrado — {e}")
        print(f"Verifique se o treinamento foi executado e se os arquivos estão em:\n  {PASTA_MODELOS}")
        raise SystemExit(1)
    except Exception as e:
        print(f"\nERRO inesperado ao carregar: {e}")
        raise SystemExit(1)


def solicitar_valor(feature: str, minimo: float, maximo: float) -> float:
    """
    Solicita um valor numérico ao usuário, validando tipo e intervalo físico.
    Repete o pedido até receber entrada válida dentro dos limites da dissertação.

    Args:
        feature: Nome da feature exibido no prompt.
        minimo:  Limite mínimo aceitável (baseado nas normas técnicas brasileiras).
        maximo:  Limite máximo aceitável.

    Returns:
        Valor numérico validado dentro do intervalo [minimo, maximo].
    """
    while True:
        try:
            entrada = input(f"  {feature:20s} [{minimo} – {maximo}]: ")
            valor   = float(entrada.replace(",", "."))  # Aceita vírgula como separador decimal

            # Verifica se o valor está dentro dos limites técnicos esperados
            if not (minimo <= valor <= maximo):
                print(f"  AVISO: Valor fora do intervalo esperado ({minimo} – {maximo}). Tente novamente.")
                continue

            return valor

        except ValueError:
            print("  ERRO: Entrada inválida. Digite um número (ex: 12.5 ou 12,5).")


def prever_cbr(valores: list, modelo, scaler) -> float:
    """
    Aplica feature engineering, normaliza os dados e retorna a previsão de CBR.

    Fluxo:
      1. Gera as 18 features (10 originais + 8 derivadas) via engenharia_features()
      2. Normaliza com o scaler ajustado durante o treinamento
      3. Faz a previsão com o ensemble RF+GB+ET carregado
      4. Retorna o valor escalar do CBR previsto

    Args:
        valores: Lista com os 10 valores originais na ordem de NOMES_FEATURES.
        modelo:  VotingRegressor (RF+GB+ET) carregado do arquivo .joblib.
        scaler:  Objeto MinMaxScaler carregado do arquivo .joblib.

    Returns:
        CBR previsto como float (valor em %).
    """
    # Passo 1: gera as features derivadas — OBRIGATÓRIO pois o modelo foi treinado com 18 features
    df_entrada = engenharia_features(valores)

    # Passo 2: normaliza com a mesma escala usada no treinamento
    dados_normalizados = scaler.transform(df_entrada)

    # Passo 3: faz a previsão e retorna como float simples
    previsao = modelo.predict(dados_normalizados)
    return float(previsao[0])


# ─────────────────────────────────────────────
# EXECUÇÃO PRINCIPAL
# ─────────────────────────────────────────────

print("=" * 50)
print("  PREVISÃO DE CBR — RANDOM FOREST")
print("=" * 50)

# 1. Carrega os artefatos salvos pelo treinamento
modelo_rf, scaler_rf = carregar_artefatos(ARQUIVO_MODELO, ARQUIVO_SCALER)

# 2. Coleta os dados da amostra com validação de cada campo
print("\nInforme os dados da amostra de solo:")
print(f"  (Aceita vírgula ou ponto como separador decimal)\n")

valores_amostra = []
for feature in NOMES_FEATURES:
    minimo, maximo = LIMITES[feature]
    valor = solicitar_valor(feature, minimo, maximo)
    valores_amostra.append(valor)

# 3. Realiza a previsão
print("\nCalculando previsão...")
cbr_estimado = prever_cbr(valores_amostra, modelo_rf, scaler_rf)

# 4. Exibe o resultado
print("\n" + "=" * 50)
print(f"  CBR PREVISTO (Random Forest): {cbr_estimado:.2f} %")

# Alerta se o CBR previsto for muito baixo ou muito alto (fora da faixa típica)
if cbr_estimado < 2:
    print("  AVISO: CBR muito baixo — verifique os dados inseridos.")
elif cbr_estimado > 100:
    print("  AVISO: CBR acima de 100% — valor atípico, verifique os dados.")

print("=" * 50)