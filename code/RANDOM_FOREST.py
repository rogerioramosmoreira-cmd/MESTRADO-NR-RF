import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree
from joblib import dump
import os

print("--- Preparando Dados ---")

# Referencia apenas o nome do arquivo, buscando na mesma pasta do script
pasta_script = os.path.dirname(os.path.abspath(__file__))
nome_arquivo = os.path.join('..', 'data', 'dados_processados_1.csv')
caminho_dados = os.path.join(pasta_script, nome_arquivo)

DF = pd.read_csv(caminho_dados)

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

rf_modelo = RandomForestRegressor(random_state=42)

para_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

print("Iniciando a busca pelos melhores hiperparâmetros (isso pode demorar)...")

grid_search = GridSearchCV(
    estimator=rf_modelo,
    param_grid=para_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

grid_search.fit(X_treino, Y_treino)

print(f"Melhores parâmetros encontrados: {grid_search.best_params_}")

melhor_modelo = grid_search.best_estimator_

previsoes = melhor_modelo.predict(x_teste)

erro_mse = mean_squared_error(y_teste, previsoes)
erro_mae = mean_absolute_error(y_teste, previsoes)

print(f"Erro Quadrático Médio (MSE): {erro_mse:.4f}")
print(f"Erro Absoluto Médio (MAE): {erro_mae:.4f}")

print("\n--- Salvando o modelo vencedor ---")
pasta_salvamento = os.path.join(pasta_script, 'Modelo_salvo_RF')
os.makedirs(pasta_salvamento, exist_ok=True)

# ===================================================================
# FASE 7: RELATÓRIO GRÁFICO E VISUALIZAÇÃO
# ===================================================================

print("\n--- Gerando Gráficos de Análise ---")

# 1. Gráfico de Dispersão (Real vs Previsto)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_teste, y=previsoes, color='green', alpha=0.6)
min_val = min(y_teste.min(), previsoes.min())
max_val = max(y_teste.max(), previsoes.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', lw=2, label='Ideal')
plt.title('Random Forest: Real vs. Previsto')
plt.xlabel('CBR Real')
plt.ylabel('CBR Previsto')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 2. Importância das Características (Feature Importance)
importancias = melhor_modelo.feature_importances_
nomes_features = X.columns

df_importancia = pd.DataFrame({'Feature': nomes_features, 'Importancia': importancias})
df_importancia = df_importancia.sort_values(by='Importancia', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importancia', y='Feature', data=df_importancia, palette='viridis')
plt.title('Importância das Características no Modelo')
plt.show()

# 3. Visualização de UMA Árvore da Floresta
primeira_arvore = melhor_modelo.estimators_[0]

plt.figure(figsize=(20, 10))
plot_tree(primeira_arvore,
          feature_names=X.columns,
          filled=True,
          rounded=True,
          max_depth=3,
          fontsize=10)
plt.title('Visualização de uma das Árvores da Floresta (Profundidade Limitada)')
plt.show()

print("\nGráficos gerados com sucesso!")

dump(melhor_modelo, os.path.join(pasta_salvamento, 'modelo_rf_otimizado.joblib'))
dump(escala, os.path.join(pasta_salvamento, 'scaler_rf.joblib'))

print("Modelo e Scaler salvos com sucesso na pasta 'Modelo_salvo_RF/'.")