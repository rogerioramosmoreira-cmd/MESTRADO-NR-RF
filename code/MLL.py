import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from joblib import dump

# -----
#PARTE 1:
print("--- 1. Carregando Dados Pré-Processados ---")
# Define o caminho para o arquivo que já foi limpo pelo outro script
caminho_dados_limpos = 'ML/data/dados_processados_5.csv' # IMPORTANTE: Verifique se o nome do arquivo está correto

try:
    DF = pd.read_csv(caminho_dados_limpos)
except FileNotFoundError:
    print(f"ERRO: O arquivo de dados limpos '{caminho_dados_limpos}' não foi encontrado.")
    print("Por favor, execute o script de limpeza primeiro para gerar este arquivo.")
    exit()


print("-----Iniciando o processo de dados-----\n")


Y = DF['CBR ']
X = DF.drop(columns=['CBR '])

escala = MinMaxScaler()

x_normalizado = escala.fit_transform(X)

X_treino, x_Teste, Y_treino, y_teste = train_test_split(
    x_normalizado, Y, test_size= 0.2, random_state= 42
)

#Parte 2:
print("-----Contruindo arquitetura do modelo-----\n")
modelo = Sequential()
modelo.add(Dense(units=64, activation='relu', input_shape=(X_treino.shape[1],)))
modelo.add(Dense(units=32, activation='relu'))
modelo.add(Dense(units=16, activation='relu'))
modelo.add(Dense(units=1, activation='relu'))
modelo.add(Dense(units=1))

modelo.summary()

#Parte 4:
print("-----Compilando modelo-----\n")
modelo.compile(optimizer='adam', loss='mean_squared_error')

#Parte 5:
print("-----Iniciando o treinamento-----\n")
historico = modelo.fit(
    X_treino, Y_treino,
    epochs=100,
    batch_size=32,
    validation_data=(X_treino, Y_treino),
    verbose=1
)

#Parte 6:
print("-----Validando o modelo-----\n")
resultados = modelo.evaluate(x_Teste, y_teste, verbose=0)
print(f"Loss final do modelo no conjunto do teste (MSE): {resultados}")

plt.figure(figsize=(10, 6))
plt.plot(historico.history['loss'], label='Perda de Treino')
plt.plot(historico.history['val_loss'], label='Perda de validação')
plt.title("Curvas de Aprendizagem")
plt.xlabel("Épocas")
plt.ylabel("Loss (Erro Quadrático Médio)")
plt.legend()
plt.grid(True)
plt.show

#Parte 7:
print("-----Salvando o modelo-----")
modelo.save('R:/Arquivos/Codigos/MLL/ML/code/Modelo_salvo/modelo_cbr.keras')
dump(escala, 'R:/Arquivos/Codigos/MLL/ML/code/Modelo_salvo/scaler_cbr.joblib')
print("Modelo e scaler foram salvos na pasta Modelo_salvo")