import numpy as np
from keras.models import load_model
from joblib import load
import os


def prever_cbr(dados_brutos, modelo, scaler):
    dados_formados = np.array(dados_brutos).reshape(1, -1)
    dados_normalizados = scaler.transform(dados_formados)
    previsao = modelo.predict(dados_normalizados)
    return previsao[0][0]


print(">>>>>>>>>>>>Carregando modelos pré treinados<<<<<<<<<<<<")
caminho_modelo = 'R:/Arquivos/Codigos/MLL/ML/code/Modelo_salvo/modelo_cbr.keras'
caminho_scaler = 'R:/Arquivos/Codigos/MLL/ML/code/Modelo_salvo/scaler_cbr.joblib'

try:
    modelo_carregado = load_model(caminho_modelo)
    scaler_carregado = load(caminho_scaler)
    print("Artefatos carregados com sucesso.")
except Exception as e:
    print(f"ERRO: Não foi possível carregar os arquivos do modelo.")
    print(f"Verifique se os arquivos '{caminho_modelo}' e '{caminho_scaler}' existem na pasta 'saved_model/'.")
    exit()
    
nomes_das_features = ['ID', '25.4mm', '9.5mm', '4.8mm', '2.0mm', '0.42mm', '0.076mm', 'LL', 'IP', 'Umidade Ótima', 'Densidade máxima']

valores_da_amostra = []

print("\n--- Ferramenta de Previsão de CBR ---")
print("Por favor, informe os dados da nova amostra:")

for feature in nomes_das_features:
    while True:
        try:
            valor_str = input(f"- Valore para {feature}")
            valor_float = float(valor_str.replace(',','.'))
            valores_da_amostra.append(valor_float)
            break
        except ValueError:
            print("ERRO: Por favor, digite um valor numerico valido")
            
cbr_resultado = prever_cbr(valores_da_amostra, modelo_carregado, scaler_carregado)

print("\n"+"="*40)
print(f"Valor de CBR previsto é: {cbr_resultado}")
print("\n"+"="*40)

