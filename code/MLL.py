#Bibliotecas como modelos de Rede neural 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
#Bibliotecas para analise de dados 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ... (outras importações)

print("--- 1. Carregando e Limpando os Dados ---")

def limpar_e_converter_para_numerico(series):
    if pd.api.types.is_object_dtype(series):
        try:
            series_limpa = (
                series.str.replace('.', '', regex=False)
                      .str.replace(',', '.', regex=False)
                      .str.strip()
            )
            return pd.to_numeric(series_limpa, errors='raise')
        except (ValueError, AttributeError):
            return series
    return series

try:
    DF = pd.read_csv('ML/data/dados_processados_2.csv') 
except FileNotFoundError:
    print("ERRO: O arquivo 'dados_processados_2.csv' não foi encontrado.")
    exit()

DF.columns = DF.columns.str.strip()

if 'Unnamed: 0' in DF.columns:
    DF = DF.drop(columns=['Unnamed: 0'])

DF = DF.apply(limpar_e_converter_para_numerico)

print("\n--- Dados após limpeza e conversão ---")
DF.info()

print("\n--- 2. Preparando os Dados para o Modelo ---")

Coluna_alvo = "CBR"
DF.dropna(inplace=True)
DF.reset_index(drop=True, inplace=True)

if Coluna_alvo not in DF.columns:
    print(f"ERRO: A coluna alvo '{Coluna_alvo}' foi removida durante a limpeza.")
    exit()

Y = DF[Coluna_alvo]
X = DF.drop(columns=[Coluna_alvo])

colunas_de_texto_restantes = X.select_dtypes(include=['object']).columns
if not colunas_de_texto_restantes.empty:
    print(f"ERRO: As seguintes colunas não são numéricas: {list(colunas_de_texto_restantes)}")
    X = X.drop(columns=colunas_de_texto_restantes)
    print("Essas colunas foram removidas de X.")

escala = MinMaxScaler()
x_normalizado = escala.fit_transform(X)

# ... resto do seu código ...

X_treino, x_teste, Y_treino, y_teste = train_test_split(
    x_normalizado,
    Y,
    test_size= 0.2,
    random_state= 42

)

modelo = Sequential()

modelo.add(Dense(
    units=64,
    activation='relu',
    input_shape=(X_treino.shape[1],)
))

modelo.add(Dense(
    units=32,
    activation='relu'
))

modelo.add(Dense(
    units=16,
    activation='relu'
))

modelo.add(Dense(
    units=8,
    activation='relu'
))

modelo.add(Dense(
    units=4,
    activation='relu'
))

modelo.add(Dense(
    units=10
))

modelo.summary()

modelo.compile(
    optimizer='adam',
    loss='mean_squared_error'
)


historico = modelo.fit(
     X_treino,
     Y_treino,
     epochs=50,              
     batch_size=32,          
     validation_data=(x_teste, y_teste),
     verbose=1
)


resultados = modelo.evaluate(x_teste, y_teste, verbose=0)


print(f"Loss (Erro Quadrático Médio): {resultados}")


primeira_amostra_teste = x_teste[0]


amostra_para_prever = np.expand_dims(primeira_amostra_teste, axis=0)

previsao = modelo.predict(amostra_para_prever)

print(f"Valor Real (da primeira amostra de teste): {y_teste.iloc[0]}")
print(f"Valor Previsto pela Rede Neural: {previsao[0][0]}")

