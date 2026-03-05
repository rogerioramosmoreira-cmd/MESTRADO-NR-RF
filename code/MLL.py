# Importação das bibliotecas necessárias
import pandas as pd  # Usada para manipulação e análise de dados em tabelas (DataFrames)
import numpy as np   # Usada para operações matemáticas e arrays
from sklearn.preprocessing import MinMaxScaler  # Ferramenta para normalizar os dados (colocar na mesma escala)
from sklearn.model_selection import train_test_split  # Função para dividir os dados em treino e teste
import tensorflow as tf  # Biblioteca principal de Deep Learning
from keras.models import Sequential  # Tipo de modelo onde as camadas são empilhadas em sequência
from keras.layers import Dense  # Tipo de camada onde todos os neurônios são conectados
import matplotlib.pyplot as plt  # Usada para gerar gráficos
from joblib import dump  # Usada para salvar arquivos auxiliares (como o scaler)
import seaborn as sns


def gerar_relatorio_grafico_completo(historico, modelo, x_teste, y_teste):
    """
    Gera 3 tipos de gráficos para análise completa do modelo:
    1. Histórico de Treinamento (para todas as métricas disponíveis).
    2. Dispersão: Valores Reais vs. Valores Previstos.
    3. Histograma dos Erros (Resíduos).
    """
    
    # --- 1. GRÁFICOS DO HISTÓRICO (LOSS E MÉTRICAS) ---
    # Descobre quais métricas existem no histórico (ex: 'loss', 'mse', 'mae')
    # Filtra apenas as métricas de treino (sem o prefixo 'val_')
    metricas = [key for key in historico.history.keys() if not key.startswith('val_')]

    for metrica in metricas:
        plt.figure(figsize=(10, 6))
        
        # Plota dados de Treino
        plt.plot(historico.history[metrica], label=f'{metrica} (Treino)')
        
        # Plota dados de Validação (se existirem)
        val_key = f'val_{metrica}'
        if val_key in historico.history:
            plt.plot(historico.history[val_key], label=f'{metrica} (Validação)', linestyle='--')
            
        plt.title(f'Evolução de: {metrica.upper()}')
        plt.xlabel('Épocas')
        plt.ylabel(metrica)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    # --- PREPARAÇÃO PARA GRÁFICOS DE PREVISÃO ---
    # Faz previsões com os dados de teste para comparar
    previsoes = modelo.predict(x_teste).flatten() # .flatten() transforma em array 1D
    reais = y_teste.values # Garante que estamos usando os valores puros
    
    
    # --- 2. GRÁFICO DE DISPERSÃO (REAL vs PREVISTO) ---
    plt.figure(figsize=(10, 6))
    
    # Plota os pontos (cada ponto é uma amostra)
    sns.scatterplot(x=reais, y=previsoes, color='blue', alpha=0.6)
    
    # Cria uma linha diagonal perfeita (Ideal)
    # Se o ponto estiver na linha, a previsão foi perfeita
    min_val = min(min(reais), min(previsoes))
    max_val = max(max(reais), max(previsoes))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', lw=2, label='Previsão Perfeita')
    
    plt.title('Comparação: Valor Real vs. Valor Previsto')
    plt.xlabel('Valor Real (CBR)')
    plt.ylabel('Valor Previsto (CBR)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


    # --- 3. HISTOGRAMA DE RESÍDUOS (ERROS) ---
    # Calcula a diferença (erro) para cada amostra
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

# --- COMO USAR ESTA FUNÇÃO NO SEU CODIGO MLL.PY ---
# Coloque esta linha no final do seu script MLL.py, depois do treinamento:
# gerar_relatorio_grafico_completo(historico, modelo, x_teste, y_teste)
# -----
# PARTE 1: Carregamento e Preparação dos Dados
print("--- 1. Carregando Dados Pré-Processados ---")

# Define o caminho para o arquivo CSV que contém os dados já limpos
caminho_dados_limpos = 'R:/Arquivos/Codigos/MLL/ML/data/dados_processados_1.csv' # IMPORTANTE: Verifique se o nome do arquivo está correto

# Bloco try-except para garantir que o arquivo existe antes de tentar ler
try:
    # Lê o arquivo CSV e carrega para dentro de um DataFrame do Pandas chamado DF
    DF = pd.read_csv(caminho_dados_limpos)
except FileNotFoundError:
    # Se o arquivo não for encontrado, imprime mensagem de erro e encerra o programa
    print(f"ERRO: O arquivo de dados limpos '{caminho_dados_limpos}' não foi encontrado.")
    print("Por favor, execute o script de limpeza primeiro para gerar este arquivo.")
    exit()

print("-----Iniciando o processo de dados-----\n")

# Separação das variáveis:
# Y: Variável Alvo (Target). É o valor de 'CBR ' que queremos que a rede aprenda a prever.
Y = DF['CBR '] 

# X: Variáveis de Entrada (Features). São todos os dados da tabela, exceto a coluna 'CBR '.
# O modelo usará esses dados para tentar calcular o Y.
X = DF.drop(columns=['CBR '])

# Inicializa o normalizador MinMaxScaler, que ajustará os dados para um intervalo entre 0 e 1
escala = MinMaxScaler()

# Ajusta o normalizador aos dados X e transforma os valores para a nova escala
x_normalizado = escala.fit_transform(X)

# Divide os dados em dois grupos: Treinamento (para ensinar o modelo) e Teste (para validar o aprendizado)
# test_size=0.2: 20% dos dados vão para teste, 80% para treino.
# random_state=42: Garante que a divisão seja sempre a mesma a cada execução (reprodutibilidade).
X_treino, x_Teste, Y_treino, y_teste = train_test_split(
    x_normalizado, Y, test_size= 0.2, random_state= 42 
)

# Parte 2: Arquitetura da Rede Neural
print("-----Contruindo arquitetura do modelo-----\n")

# Inicializa um modelo sequencial (uma pilha linear de camadas)
modelo = Sequential()

# Adiciona a 1ª Camada (Entrada + Oculta):
# - 32 neurônios.
# - Ativação 'relu' (zera negativos, mantém positivos).
# - input_shape: Define quantas colunas de dados entram na rede (baseado no X_treino).
modelo.add(Dense(units=32, activation='relu', input_shape=(X_treino.shape[1],)))

# Adiciona a 2ª Camada Oculta:
# - 16 neurônios com ativação 'relu'.
modelo.add(Dense(units=16, activation='relu'))

# Adiciona a 3ª Camada Oculta:
# - 1 neurônio com ativação 'relu'.
modelo.add(Dense(units=1, activation='relu'))

# Adiciona a Camada de Saída:
# - 1 neurônio (pois queremos prever um único valor contínuo, o CBR).
# - Sem função de ativação (linear), permitindo prever qualquer valor numérico.
modelo.add(Dense(units=1))

# Exibe no console um resumo da estrutura da rede (camadas, formato de saída e parâmetros)
modelo.summary()

# Parte 4: Compilação
print("-----Compilando modelo-----\n")

# Configura o modelo para treinamento:
# - optimizer='adam': Algoritmo eficiente para ajustar os pesos da rede.
# - loss='mean_squared_error': A métrica de erro que o modelo tentará diminuir (Erro Quadrático Médio).
modelo.compile(optimizer='adam', loss='mean_squared_error')

# Parte 5: Treinamento
print("-----Iniciando o treinamento-----\n")

# Inicia o processo de aprendizado
historico = modelo.fit(
    X_treino, Y_treino,  # Dados usados para treinar
    epochs=100,          # Número de vezes que o modelo passará por todos os dados
    batch_size=32,       # Quantas amostras processar antes de atualizar os pesos
    # Dados usados para validar o modelo durante o treino (Nota: aqui está usando X de treino e Y de teste)
    validation_data=(x_Teste, y_teste), 
    verbose=1            # Mostra a barra de progresso
)

# Parte 6: Avaliação e Visualização
print("-----Validando o modelo-----\n")

# Avalia o modelo final usando os dados de teste reservados
print("\n--- Gerando Relatório Gráfico Completo ---")
gerar_relatorio_grafico_completo(historico, modelo, x_Teste, y_teste)
# Parte 7: Salvamento
print("-----Salvando o modelo-----")

# Salva o modelo treinado (arquitetura + pesos) no formato Keras
modelo.save('ML/code/Modelo_salvo/modelo_cbr.keras')

# Salva o objeto 'scaler' para que possamos normalizar novos dados da mesma forma no futuro
dump(escala, 'ML/code/Modelo_salvo/scaler_cbr.joblib')

print("Modelo e scaler foram salvos na pasta Modelo_salvo")