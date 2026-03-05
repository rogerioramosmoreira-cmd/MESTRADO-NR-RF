import numpy as np
from joblib import load
import os

def prever_cbr_rf(dados_brutos, modelo, scaler):
    
    dados_formados = np.array(dados_brutos).reshape(1, -1)
    dados_normalizados = scaler.transform(dados_formados)
    
    previsao = modelo.predict(dados_normalizados)
    return previsao[0]


print("\n--- Sistema de Previsão de CBR (Random Forest) ---")

pasta_modelos = 'ML/code/Modelo_salvo_RF'
arquivo_modelo = os.path.join(pasta_modelos, 'modelo_rf_otimizado.joblib') 
arquivo_scaler = os.path.join(pasta_modelos, 'scaler_rf.joblib')     

try:
    print("Carregando modelo e scaler...")
    modelo_rf = load(arquivo_modelo)
    scaler_rf = load(arquivo_scaler)
    print("Sucesso! Modelo carregado.")
except FileNotFoundError:
    print(f"ERRO: Arquivos não encontrados na pasta '{pasta_modelos}'.")
    print("Verifique se você rodou o treinamento e se os nomes dos arquivos estão corretos.")
    exit()
except Exception as e:
    print(f"Erro ao carregar: {e}")
    exit()
    
nomes_das_features = ['25.4mm', '9.5mm', '4.8mm', '2.0mm', '0.42mm', '0.076mm', 'LL', 'IP', 'Umidade Ótima', 'Densidade máxima']
valores_da_amostra = []

print("\nPor favor, informe os dados da nova amostra de solo:")

for feature in nomes_das_features:
        while True:
            try:
                valor_str = input(f"  - Valor para {feature}: ")
                # Trata vírgula como ponto decimal
                valor_float = float(valor_str.replace(',', '.'))
                valores_da_amostra.append(valor_float)
                break
            except ValueError:
                print("  ERRO: Digite um número válido.")
                
print("\nCalculando...")
cbr_estimado = prever_cbr_rf(valores_da_amostra, modelo_rf, scaler_rf)
    
    # --- Resultado ---
print("\n" + "="*40)
print(f"  CBR PREVISTO (Random Forest): {cbr_estimado:.2f}%")
print("="*40)