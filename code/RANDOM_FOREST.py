import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from joblib import dump
import os

print("--- Preparando Dados ---")

caminhos_dados_limpos = 'R:/Arquivos/Codigos/MLL/ML/data/dados_processados_1.csv'
DF = pd.read_csv(caminhos_dados_limpos)

coluna_alvo = 'CBR '
Y = DF[coluna_alvo]
X = DF.drop(columns=[coluna_alvo])

escala = MinMaxScaler()
x_normalizada = escala.fit_transform(X)

X_treino, x_teste, Y_treino, y_teste = train_test_split(
    x_normalizada, Y, test_size=0.2, random_state=42
)

print("\nDados prontos")

print("\n--- Treinamento do modelo ---")

rf_modelo = RandomForestRegressor(n_estimators= 100, random_state=42)

rf_modelo.fit(X_treino, Y_treino)

print("Treinamento concluido...")

print("\n--- Avaliando o Modelo ---")

privisao = rf_modelo.predict(x_teste)

erro_mse = mean_squared_error(y_teste, privisao )
erro_mae = mean_absolute_error(y_teste, privisao)


print(f"Erro Quadrático Médio (MSE): {erro_mse:.4f}")
print(f"Erro Absoluto Médio (MAE): {erro_mae:.4f}")

