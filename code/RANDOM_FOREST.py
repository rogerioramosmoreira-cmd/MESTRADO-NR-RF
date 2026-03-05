import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from joblib import dump
import os

print("--- Preparando Dados ---")

caminhos_dados_limpos = 'ML/data/dados_processados_2.csv'
DF = pd.read_csv(caminhos_dados_limpos)

coluna_alvo = 'CBR '
Y = DF[coluna_alvo]
X = DF.drop(columns=[coluna_alvo])

escala = MinMaxScaler()
x_normalizada = escala.fit_transform(X)

X_treino, x_teste, Y_treino, y_teste = train_test_split(
    x_normalizada, Y, test_size=0.2, random_state=42
)

print("/nDados prontos")

print("\n--- Treinamento do modelo ---")

rf_modelo = RandomForestRegressor(random_state=42)

para_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth':[None, 10, 20],
    'min_samples_split':[2,5]
}

print("Iniciando a busca pelos melhores hiperparâmetros (isso pode demorar)...")

grid_search = GridSearchCV(estimator=rf_modelo, param_grid= para_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)

grid_search.fit(X_treino, Y_treino)

print(f"Melhores parâmetros encontrados: {grid_search.best_params_}")

melhor_modelo = grid_search.best_estimator_

previsoes = melhor_modelo.predict(x_teste)

erro_mse = mean_squared_error(y_teste, previsoes)
erro_mae = mean_absolute_error(y_teste, previsoes)


print(f"Erro Quadrático Médio (MSE): {erro_mse:.4f}")
print(f"Erro Absoluto Médio (MAE): {erro_mae:.4f}")

print("\n--- Salvando o modelo vencedor ---")
pasta_salvamento = 'ML/code/Modelo_salvo_RF'
os.makedirs(pasta_salvamento, exist_ok=True)


dump(melhor_modelo, os.path.join(pasta_salvamento, 'modelo_rf_otimizado.joblib'))


dump(escala, os.path.join(pasta_salvamento, 'scaler_rf.joblib'))

print("Modelo e Scaler salvos com sucesso na pasta 'saved_model/'.")