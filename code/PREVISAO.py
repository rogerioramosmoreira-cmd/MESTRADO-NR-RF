# Importa a biblioteca numpy, usada para operações matemáticas eficientes e manipulação de arrays.
import numpy as np
# Importa a função load_model da biblioteca Keras, usada para carregar modelos de rede neural previamente salvos (formato .keras ou .h5).
from keras.models import load_model
# Importa a função load da biblioteca joblib, usada para carregar objetos Python salvos, como o nosso escalonador (scaler).
from joblib import load
# Importa a biblioteca os, que fornece funções para interagir com o sistema operacional (como verificar caminhos de arquivos), embora não seja explicitamente usada nas chamadas de função abaixo.
import os


# Função para realizar a previsão do valor de CBR.
# Recebe os dados brutos de entrada, o modelo treinado e o escalonador ajustado.
def prever_cbr(dados_brutos, modelo, scaler):
    # Converte a lista de dados de entrada em um array numpy e redimensiona para o formato (1, n_features).
    # O modelo espera um lote de dados (mesmo que seja de tamanho 1), por isso o reshape(1, -1).
    dados_formados = np.array(dados_brutos).reshape(1, -1)
    
    # Aplica a mesma transformação de escala usada no treinamento (MinMaxScaler) aos novos dados.
    # Isso garante que os dados de entrada estejam na mesma faixa de valores que o modelo aprendeu.
    dados_normalizados = scaler.transform(dados_formados)
    
    # Faz a previsão usando o modelo carregado e os dados normalizados.
    # O resultado é um array bidimensional com a previsão.
    previsao = modelo.predict(dados_normalizados)
    
    # Retorna o valor previsto. Como a previsão é um array [[valor]], acessamos o elemento [0][0] para obter o número escalar.
    return previsao[0][0]


print(">>>>>>>>>>>>Carregando modelos pré treinados<<<<<<<<<<<<")
# Define os caminhos absolutos para os arquivos do modelo treinado e do escalonador.
# É importante que esses caminhos estejam corretos e apontem para os arquivos gerados pelo script de treinamento.
caminho_modelo = 'R:/Arquivos/Codigos/MLL/ML/code/Modelo_salvo/modelo_cbr.keras'
caminho_scaler = 'R:/Arquivos/Codigos/MLL/ML/code/Modelo_salvo/scaler_cbr.joblib'

# Tenta carregar o modelo e o escalonador.
try:
    modelo_carregado = load_model(caminho_modelo) # Carrega o modelo da rede neural.
    scaler_carregado = load(caminho_scaler)     # Carrega o objeto scaler.
    print("Artefatos carregados com sucesso.")
except Exception as e:
    # Se ocorrer algum erro (ex: arquivo não encontrado), exibe uma mensagem de erro e encerra o programa.
    print(f"ERRO: Não foi possível carregar os arquivos do modelo.")
    print(f"Verifique se os arquivos '{caminho_modelo}' e '{caminho_scaler}' existem na pasta 'saved_model/'.")
    exit()
    
# Lista com os nomes das características (features) que o modelo espera como entrada.
# A ordem deve ser EXATAMENTE a mesma usada durante o treinamento do modelo.
nomes_das_features = ['25.4mm', '9.5mm', '4.8mm', '2.0mm', '0.42mm', '0.076mm', 'LL', 'IP', 'Umidade Ótima', 'Densidade máxima']

# Lista vazia para armazenar os valores que o usuário irá digitar.
valores_da_amostra = []

print("\n--- Ferramenta de Previsão de CBR ---")
print("Por favor, informe os dados da nova amostra:")

# Loop para solicitar ao usuário o valor de cada característica definida na lista 'nomes_das_features'.
for feature in nomes_das_features:
    while True:
        try:
            # Solicita a entrada do usuário para a característica atual.
            valor_str = input(f"- Valore para {feature}: ")
            
            # Tenta converter a entrada (string) para um número decimal (float).
            # O .replace(',', '.') permite que o usuário use vírgula como separador decimal, comum em português.
            valor_float = float(valor_str.replace(',','.'))
            
            # Se a conversão for bem-sucedida, adiciona o valor à lista de amostras.
            valores_da_amostra.append(valor_float)
            
            # Sai do loop while e passa para a próxima característica.
            break
        except ValueError:
            # Se a conversão falhar (o usuário digitou texto ou formato inválido), exibe erro e pede novamente.
            print("ERRO: Por favor, digite um valor numerico valido")
            
# Chama a função de previsão com os valores coletados, o modelo e o scaler.
cbr_resultado = prever_cbr(valores_da_amostra, modelo_carregado, scaler_carregado)

print("\n" + "="*40)
# Aplica a formatação .2f para arredondar para duas casas decimais
# Exibe o resultado final da previsão do CBR.
print(f"\nValor de CBR previsto é: {cbr_resultado:.2f}") 
print("="*40) 