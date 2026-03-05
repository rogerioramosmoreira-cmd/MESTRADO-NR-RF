import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from joblib import dump

# ===================================================================
# FUNÇÕES AUXILIARES
# ===================================================================

def gerar_relatorio_grafico_completo(historico, modelo, x_teste, y_teste):
    """
    Gera 3 tipos de gráficos para análise completa do modelo:
    1. Histórico de Treinamento (Loss e métricas).
    2. Dispersão: Valores Reais vs. Valores Previstos.
    3. Histograma dos Erros (Resíduos).
    """

    # --- 1. GRÁFICOS DO HISTÓRICO ---
    metricas = [key for key in historico.history.keys() if not key.startswith('val_')]

    for metrica in metricas:
        plt.figure(figsize=(10, 6))
        plt.plot(historico.history[metrica], label=f'{metrica} (Treino)')
        val_key = f'val_{metrica}'
        if val_key in historico.history:
            plt.plot(historico.history[val_key], label=f'{metrica} (Validação)', linestyle='--')
        plt.title(f'Evolução de: {metrica.upper()}')
        plt.xlabel('Épocas')
        plt.ylabel(metrica)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    # --- 2. GRÁFICO DE DISPERSÃO (REAL vs PREVISTO) ---
    previsoes = modelo.predict(x_teste).flatten()
    reais = y_teste.values

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=reais, y=previsoes, color='blue', alpha=0.6)
    min_val = min(min(reais), min(previsoes))
    max_val = max(max(reais), max(previsoes))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', lw=2, label='Previsão Perfeita')
    plt.title('Comparação: Valor Real vs. Valor Previsto')
    plt.xlabel('Valor Real (CBR)')
    plt.ylabel('Valor Previsto (CBR)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # --- 3. HISTOGRAMA DE RESÍDUOS ---
    erros = reais - previsoes
    plt.figure(figsize=(10, 6))
    sns.histplot(erros, kde=True, color='purple')
    plt.axvline(x=0, color='red', linestyle='--', label='Erro Zero')
    plt.title('Distribuição dos Erros (Resíduos)')
    plt.xlabel('Erro (Real - Previsto)')
    plt.ylabel('Frequência')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# ===================================================================
# PARTE 1: CARREGAMENTO DOS DADOS
# ===================================================================
print("--- 1. Carregando Dados Pré-Processados ---")

pasta_script = os.path.dirname(os.path.abspath(__file__))
caminho_dados = os.path.join(pasta_script, '..', 'data', 'dados_processados_1.csv')

try:
    DF = pd.read_csv(caminho_dados)
    print(f"Dataset carregado: {DF.shape[0]} linhas, {DF.shape[1]} colunas")
except FileNotFoundError:
    print(f"ERRO: Arquivo não encontrado em '{caminho_dados}'.")
    exit()


# ===================================================================
# PARTE 2: PREPARAÇÃO DOS DADOS
# ===================================================================
print("\n--- 2. Preparando os Dados ---")

coluna_alvo = 'CBR '
Y = DF[coluna_alvo]
X = DF.drop(columns=[coluna_alvo])

print(f"Variável alvo: '{coluna_alvo}'")
print(f"Estatísticas do alvo:\n{Y.describe()}\n")

# Normalização
escala = MinMaxScaler()
x_normalizado = escala.fit_transform(X)

# Divisão treino/teste
X_treino, x_teste, Y_treino, y_teste = train_test_split(
    x_normalizado, Y, test_size=0.2, random_state=42
)

print(f"Treino: {X_treino.shape[0]} amostras | Teste: {x_teste.shape[0]} amostras")


# ===================================================================
# PARTE 3: ARQUITETURA DO MODELO (CORRIGIDA E APRIMORADA)
# ===================================================================
print("\n--- 3. Construindo Arquitetura do Modelo ---")

n_features = X_treino.shape[1]

modelo = Sequential([
    # --- Camada 1: Entrada ---
    # BatchNormalization estabiliza e acelera o aprendizado
    Dense(128, activation='relu', input_shape=(n_features,), kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),   # Desativa 30% dos neurônios aleatoriamente → previne overfitting

    # --- Camada 2: Processamento profundo ---
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.2),

    # --- Camada 3: Refinamento ---
    Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.1),

    # --- Camada 4: Compressão final ---
    Dense(16, activation='relu'),

    # --- Camada de Saída ---
    # CORREÇÃO: apenas 1 neurônio, sem ativação (regressão linear livre)
    Dense(1)
])

modelo.summary()


# ===================================================================
# PARTE 4: COMPILAÇÃO (COM MÉTRICAS ADICIONAIS)
# ===================================================================
print("\n--- 4. Compilando o Modelo ---")

# Adam com learning rate menor = passos menores e mais precisos
otimizador = Adam(learning_rate=0.001)

modelo.compile(
    optimizer=otimizador,
    loss='mse',
    metrics=['mae']   # Acompanha MAE durante o treino para análise mais fácil
)


# ===================================================================
# PARTE 5: CALLBACKS (INTELIGÊNCIA DURANTE O TREINO)
# ===================================================================

callbacks = [
    # Para automaticamente quando o modelo para de melhorar
    # Salva os melhores pesos e restaura ao final
    EarlyStopping(
        monitor='val_loss',
        patience=25,          # Aguarda 25 épocas sem melhora antes de parar
        restore_best_weights=True,
        verbose=1
    ),

    # Reduz o learning rate quando o modelo estagna
    # Permite ajustes mais finos no final do treino
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,           # Reduz o LR pela metade
        patience=10,          # Após 10 épocas sem melhora
        min_lr=1e-6,
        verbose=1
    )
]


# ===================================================================
# PARTE 6: TREINAMENTO
# ===================================================================
print("\n--- 5. Iniciando o Treinamento ---")

historico = modelo.fit(
    X_treino, Y_treino,
    epochs=300,               # Máximo de épocas — EarlyStopping decide quando parar
    batch_size=32,
    validation_data=(x_teste, y_teste),
    callbacks=callbacks,
    verbose=1
)

epocas_reais = len(historico.history['loss'])
print(f"\nTreinamento encerrado na época {epocas_reais} (EarlyStopping)")


# ===================================================================
# PARTE 7: AVALIAÇÃO
# ===================================================================
print("\n--- 6. Avaliando o Modelo ---")

resultados = modelo.evaluate(x_teste, y_teste, verbose=0)
mse_final = resultados[0]
mae_final = resultados[1]
rmse_final = np.sqrt(mse_final)

print(f"\n{'='*40}")
print(f"  MSE  (Erro Quadrático Médio): {mse_final:.4f}")
print(f"  RMSE (Raiz do MSE):           {rmse_final:.4f}")
print(f"  MAE  (Erro Absoluto Médio):   {mae_final:.4f}")
print(f"{'='*40}")
print(f"\n  Interpretação: Em média, o modelo erra ±{mae_final:.2f} no valor de CBR")


# ===================================================================
# PARTE 8: VISUALIZAÇÃO
# ===================================================================
print("\n--- 7. Gerando Relatório Gráfico ---")
gerar_relatorio_grafico_completo(historico, modelo, x_teste, y_teste)


# ===================================================================
# PARTE 9: SALVAMENTO
# ===================================================================
print("\n--- 8. Salvando o Modelo ---")

pasta_salvamento = os.path.join(pasta_script, 'Modelo_salvo')
os.makedirs(pasta_salvamento, exist_ok=True)

modelo.save(os.path.join(pasta_salvamento, 'modelo_cbr.keras'))
dump(escala, os.path.join(pasta_salvamento, 'scaler_cbr.joblib'))

print(f"Modelo e scaler salvos em: {pasta_salvamento}")