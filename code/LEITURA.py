# Importa a biblioteca Pandas para manipulação e análise de dados tabulares
import pandas as pd
# Importa a biblioteca NumPy para operações numéricas eficientes e tratamento de valores nulos (NaN)
import numpy as np
# Importa a biblioteca OS para interagir com o sistema operacional (manipular caminhos e arquivos)
import os

# ================================================================================================================
# FUNÇÃO 1: Apenas escolhe o caminho do arquivo
# Objetivo: Listar arquivos CSV em uma pasta, permitir que o usuário escolha um e retornar o caminho completo.
def Verificar_CSV(caminho_pasta):
    numeracao = 0 # Contador para numerar as opções de arquivos no menu
    arquivos_csv_encontrados = [] # Lista para armazenar os nomes dos arquivos CSV encontrados

    try:
        # Tenta listar todos os arquivos dentro da pasta especificada
        for arquivo in os.listdir(caminho_pasta):
            # Verifica se o arquivo atual termina com a extensão '.csv'
            if arquivo.endswith('.csv'):
                # Adiciona o nome do arquivo à lista de encontrados
                arquivos_csv_encontrados.append(arquivo)
                # Incrementa o contador para a numeração do menu
                numeracao += 1
                # Imprime o arquivo encontrado com seu número correspondente
                print(f'!!!!!!Arquivos encotrado!!!!!! \n Nome dos arquivos: {numeracao} - {arquivo}')
        
        # Se a lista de arquivos encontrados estiver vazia (nenhum CSV encontrado)
        if not arquivos_csv_encontrados:
            print("Não foi possivel encotrar nenhum arquivo no formato CSV\n")
            return None # Retorna None para indicar que nenhum arquivo foi selecionado

        # Loop infinito para garantir que o usuário faça uma escolha válida
        while True:
            try:
                print("\nEscolha um arquivo que deseja usar!!")
                # Solicita ao usuário o número do arquivo desejado
                escolha_arquivo_str = input("Opção de arquivo: ")
                # Tenta converter a entrada do usuário para um número inteiro
                escolha_arquivo_int = int(escolha_arquivo_str)

                # Verifica se o número escolhido está dentro do intervalo válido (1 até o total de arquivos)
                if 1 <= escolha_arquivo_int <= numeracao:
                    # Seleciona o nome do arquivo da lista usando o índice correto (número escolhido - 1)
                    nome_arquivo_escolhido = arquivos_csv_encontrados[escolha_arquivo_int - 1]
                    # Constrói o caminho completo do arquivo combinando a pasta e o nome do arquivo
                    caminho_completo = os.path.join(caminho_pasta, nome_arquivo_escolhido)
                    print(f"\nArquivo escolhido foi: {nome_arquivo_escolhido}")
                    # Retorna o caminho completo do arquivo selecionado
                    return caminho_completo
                else:
                    # Mensagem de erro se o número estiver fora do intervalo
                    print(f"Opção inválida! Por favor, escolha um número entre 1 e {numeracao}.")
            except ValueError:
                # Mensagem de erro se a entrada não for um número válido
                print("Entrada inválida! Por favor, digite um número.")
    except FileNotFoundError:
        # Tratamento de erro caso a pasta especificada no caminho_pasta não exista
        print(f"ERRO: O caminho '{caminho_pasta}' não foi encontrado.")
        return None

# ================================================================================================================
# FUNÇÃO 2: Limpa um DataFrame que é passado para ela
# Objetivo: Realizar limpeza de dados (conversão de tipos, remoção de caracteres indesejados, tratamento de outliers e nulos).
def Limpeza(df_entrada):
    print("\n--- 1. Iniciando limpeza e conversão de tipos ---")
    
    # Itera sobre cada coluna do DataFrame
    for coluna in df_entrada.columns: 
        # Verifica se a coluna é do tipo 'object' (texto), pois apenas estas precisam de limpeza de string
        if pd.api.types.is_object_dtype(df_entrada[coluna]): 
            try:
                # Aplica uma série de transformações na coluna de texto para limpá-la:
                coluna_limpa = (
                    df_entrada[coluna]
                    .str.replace(' ', '', regex=False)    # Remove todos os espaços em branco
                    .str.replace('.', '', regex=False)    # Remove pontos (que podem ser separadores de milhar)
                    .str.replace(',', '.', regex=False)   # Substitui vírgulas por pontos (para o formato decimal padrão)
                    .str.strip()                          # Remove espaços em branco nas extremidades (redundante com o primeiro replace, mas seguro)
                ) 
                # Tenta converter a coluna limpa para numérico. 'errors=raise' fará com que falhe se ainda houver texto inválido.
                df_entrada[coluna] = pd.to_numeric(coluna_limpa, errors='raise') 
            except (ValueError, AttributeError):
                # Se a conversão falhar (ex: coluna de texto legítimo), ignora e mantém a coluna como está
                pass
    print("Conversão de tipos concluída.")
    
    # Parte 3: Verifica se há numeros fora da curva (outliers)
    print("\n--- 2. Verificando valores fora da curva (outliers) ---")
    
    # Lista de colunas específicas que devem ser verificadas quanto a limites de valor
    colunas_para_verificar = ['25.4mm', '9.5mm', '4.8mm', '2.0mm', '0.42mm', '0.076mm', 'LL', 'IP', 'Umidade Ótima', 'Densidade máxima', 'CBR']
    
    for coluna in colunas_para_verificar:
        # Verifica se a coluna existe no DataFrame e se ela é do tipo numérico antes de comparar
        if coluna in df_entrada.columns and pd.api.types.is_numeric_dtype(df_entrada[coluna]): 
            # Cria uma máscara booleana onde True indica valores fora do intervalo aceitável (> 3000 ou < 1)
            condicao_outlier = (df_entrada[coluna] > 3000) | (df_entrada[coluna] < 1) 
            # Substitui os valores identificados como outliers por NaN (Not a Number)
            df_entrada[coluna] = df_entrada[coluna].mask(condicao_outlier, np.nan) 
    print("Verificação de outliers concluída.")
    
    print("\n--- 3. Removendo linhas com dados ausentes ---")
    
    # Conta o número de linhas antes da remoção
    linhas_antes = len(df_entrada) 
    
    # Remove todas as linhas que contêm pelo menos um valor nulo (NaN)
    # inplace=True modifica o DataFrame original diretamente
    df_entrada.dropna(inplace=True)
    
    # Conta o número de linhas após a remoção
    linhas_depois = len(df_entrada)
    
    # Informa ao usuário quantas linhas foram removidas, se houver
    if linhas_antes > linhas_depois:
        print(f"{linhas_antes - linhas_depois} linhas com valores nulos ou outliers foram removidas.")
    else:
        print("Nenhuma linha com valores nulos foi encontrada.")
        
    # Reorganiza o índice do DataFrame para ser sequencial (0, 1, 2...) após a remoção de linhas
    df_entrada.reset_index(drop=True, inplace=True)
    
    # Retorna o DataFrame limpo e processado
    return df_entrada


# ================================================================================================================
# FLUXO PRINCIPAL DO PROGRAMA
# ================================================================================================================

# 1. Chama a função Verificar_CSV para obter o CAMINHO do arquivo selecionado pelo usuário
caminho_do_arquivo_escolhido = Verificar_CSV('R:/Arquivos/Codigos/MLL/ML/data')

# 2. VERIFICA se um caminho foi realmente retornado (usuário escolheu um arquivo válido) antes de prosseguir
if caminho_do_arquivo_escolhido:
    
    # 3. CARREGA o DataFrame usando o caminho escolhido
    print(f"\nCarregando dados de: {caminho_do_arquivo_escolhido}")
    # Lê o CSV. Não especificamos dtype aqui para permitir que o pandas infira, mas a limpeza cuidará de tipos mistos.
    DF = pd.read_csv(caminho_do_arquivo_escolhido)
    
    # 4. PASSA o DataFrame carregado para a função de limpeza
    # A função retornará o DataFrame modificado
    DF_limpo = Limpeza(DF)
    
    # Definições para o salvamento do novo arquivo
    Caminho_principal = "R:/Arquivos/Codigos/MLL/ML/data"
    Nome_base = "dados_processados_"
    extensao = '.csv'

    # --- ETAPA 1: CONTAR OS ARQUIVOS QUE JÁ EXISTEM PARA GERAR NOME ÚNICO ---
    contador = 0
    try:
        # Garante que a pasta de destino exista antes de tentar listar seu conteúdo
        os.makedirs(Caminho_principal, exist_ok=True)
        
        # Lista o conteúdo da pasta de destino
        conteudo_da_pasta = os.listdir(Caminho_principal)
        
        # Loop com o único propósito de contar arquivos que seguem o padrão de nomenclatura
        for nome_do_item in conteudo_da_pasta:
            # Verifica se o arquivo começa com o nome base e termina com a extensão csv
            if nome_do_item.startswith(Nome_base) and nome_do_item.endswith(extensao):
                contador += 1 # Incrementa o contador para cada arquivo encontrado
                
    except Exception as e:
        # Captura erros ao tentar acessar a pasta (ex: permissão negada)
        print(f"ERRO ao verificar a pasta de destino: {e}")
        exit() # Para a execução se houver um erro de pasta

    # --- ETAPA 2: CALCULAR O NOVO NOME (AGORA FORA DO LOOP) ---

    # O número do novo arquivo será a quantidade existente (contador) + 1
    proximo_numero = contador + 1

    # Monta o novo nome do arquivo combinando base, número e extensão
    novo_nome_de_arquivo = f"{Nome_base}{proximo_numero}{extensao}"
    # Cria o caminho completo onde o novo arquivo será salvo
    caminho_completo_novo_arquivo = os.path.join(Caminho_principal, novo_nome_de_arquivo)

    # --- ETAPA 3: SALVAR O ARQUIVO ---
    try:
        print(f"\nSalvando o DataFrame limpo como: '{caminho_completo_novo_arquivo}'")
        
        # Salva o DataFrame limpo no disco.
        # decimal="." garante que o separador decimal seja ponto (padrão internacional/Python).
        # index=False evita salvar a coluna de índice do pandas no arquivo.
        DF_limpo.to_csv(caminho_completo_novo_arquivo, decimal=".", index=False)
        
        print("Arquivo salvo com sucesso!")

    except Exception as e:
        # Captura erros durante o salvamento (ex: permissão de escrita, disco cheio)
        print(f"ERRO FATAL ao salvar o arquivo: {e}")


    # Exibe uma amostra final dos dados para confirmação
    print("\n\n--- Processo Finalizado ---")
    print("Amostra do DataFrame limpo:")
    print(DF_limpo.head())

else:
    # Caso a função Verificar_CSV retorne None (nenhum arquivo escolhido ou pasta não encontrada)
    print(f"Caminho não encontrado {caminho_do_arquivo_escolhido}")