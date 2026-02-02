import pandas as pd
import numpy as np
import os

# ================================================================================================================
# FUNÇÃO 1: Apenas escolhe o caminho do arquivo
def Verificar_CSV(caminho_pasta):
    numeracao = 0
    arquivos_csv_encontrados = []
    try:
        for arquivo in os.listdir(caminho_pasta):
            if arquivo.endswith('.csv'):
                arquivos_csv_encontrados.append(arquivo)
                numeracao += 1
                print(f'!!!!!!Arquivos encotrado!!!!!! \n Nome dos arquivos: {numeracao} - {arquivo}')
        
        if not arquivos_csv_encontrados:
            print("Não foi possivel encotrar nenhum arquivo no formato CSV\n")
            return None

        while True:
            try:
                print("\nEscolha um arquivo que deseja usar!!")
                escolha_arquivo_str = input("Opção de arquivo: ")
                escolha_arquivo_int = int(escolha_arquivo_str)

                if 1 <= escolha_arquivo_int <= numeracao:
                    nome_arquivo_escolhido = arquivos_csv_encontrados[escolha_arquivo_int - 1]
                    caminho_completo = os.path.join(caminho_pasta, nome_arquivo_escolhido)
                    print(f"\nArquivo escolhido foi: {nome_arquivo_escolhido}")
                    return caminho_completo
                else:
                    print(f"Opção inválida! Por favor, escolha um número entre 1 e {numeracao}.")
            except ValueError:
                print("Entrada inválida! Por favor, digite um número.")
    except FileNotFoundError:
        print(f"ERRO: O caminho '{caminho_pasta}' não foi encontrado.")
        return None

# ================================================================================================================
# FUNÇÃO 2: Limpa um DataFrame que é passado para ela
def Limpeza(df_entrada):
    print("\n--- 1. Iniciando limpeza e conversão de tipos ---")
    for coluna in df_entrada.columns: #Loop que percorre todas as colunas 
        if pd.api.types.is_object_dtype(df_entrada[coluna]): #If que verifica se as colunas tem um dado em formato objeto
            try:
                coluna_limpa = (
                    df_entrada[coluna]
                    .str.replace(' ', '', regex=False)
                    .str.replace('.', '', regex=False)
                    .str.replace(',', '.', regex=False)
                    .str.strip()
                ) #Troca os pontos por virgulas e virgulas por pontos 
                df_entrada[coluna] = pd.to_numeric(coluna_limpa, errors='raise') #Garante que todas as colunas sejam em formato numerico
            except (ValueError, AttributeError):
                pass
    print("Conversão de tipos concluída.")
    #Parte 3: Verifica se há a numeros fora da curva 
    print("\n--- 2. Verificando valores fora da curva (outliers) ---")
    colunas_para_verificar = ['25.4mm', '9.5mm', '4.8mm', '2.0mm', '0.42mm', '0.076mm', 'LL', 'IP', 'Umidade Ótima', 'Densidade máxima', 'CBR']
    for coluna in colunas_para_verificar:
        if coluna in df_entrada.columns and pd.api.types.is_numeric_dtype(df_entrada[coluna]): #Loops percorre colunas e linhas 
            condicao_outlier = (df_entrada[coluna] > 3000) | (df_entrada[coluna] < 1) #Verifica se a celula tem algum numero fora da curva
            df_entrada[coluna] = df_entrada[coluna].mask(condicao_outlier, np.nan) # Caso tenha um numero fora da curva, ele apaga 
    print("Verificação de outliers concluída.")
    
    print("\n--- 3. Removendo linhas com dados ausentes ---")
    linhas_antes = len(df_entrada) # Conta e armazena a quantidade de linhas 
    df_entrada.dropna(inplace=True)
    linhas_depois = len(df_entrada)
    if linhas_antes > linhas_depois:
        print(f"{linhas_antes - linhas_depois} linhas com valores nulos ou outliers foram removidas.")
    else:
        print("Nenhuma linha com valores nulos foi encontrada.")
    df_entrada.reset_index(drop=True, inplace=True)
    
    return df_entrada


# ================================================================================================================
# FLUXO PRINCIPAL DO PROGRAMA
# ================================================================================================================

# 1. Chama a função para obter o CAMINHO do arquivo
caminho_do_arquivo_escolhido = Verificar_CSV('R:/Arquivos/Codigos/MLL/ML/data')

# 2. VERIFICA se um caminho foi realmente retornado antes de prosseguir
if caminho_do_arquivo_escolhido:
    
    # 3. CARREGA o DataFrame usando o caminho escolhido
    print(f"\nCarregando dados de: {caminho_do_arquivo_escolhido}")
    DF = pd.read_csv(caminho_do_arquivo_escolhido)
    
    # 4. PASSA o DataFrame carregado para a função de limpeza
    DF_limpo = Limpeza(DF)
    
    # ... (seu código até a linha DF_limpo = Limpeza(DF)) ...
    
    Caminho_principal = "R:/Arquivos/Codigos/MLL/ML/data"
    Nome_base = "dados_processados_"
    extensao = '.csv'

    # --- ETAPA 1: CONTAR OS ARQUIVOS QUE JÁ EXISTEM ---
    contador = 0
    try:
        os.makedirs(Caminho_principal, exist_ok=True)
        conteudo_da_pasta = os.listdir(Caminho_principal)
        
        # O único trabalho deste loop é contar
        for nome_do_item in conteudo_da_pasta:
            if nome_do_item.startswith(Nome_base) and nome_do_item.endswith(extensao):
                contador += 1
                
    except Exception as e:
        print(f"ERRO ao verificar a pasta de destino: {e}")
        exit() # Para a execução se houver um erro de pasta

    # --- ETAPA 2: CALCULAR O NOVO NOME (AGORA FORA DO LOOP) ---

    # O número do novo arquivo é a quantidade existente (contador) + 1
    proximo_numero = contador + 1

    # Monta o novo nome e o caminho completo.
    # Essas variáveis agora são criadas DEPOIS da contagem e sempre existirão.
    novo_nome_de_arquivo = f"{Nome_base}{proximo_numero}{extensao}"
    caminho_completo_novo_arquivo = os.path.join(Caminho_principal, novo_nome_de_arquivo)

    # --- ETAPA 3: SALVAR O ARQUIVO ---
    try:
        print(f"\nSalvando o DataFrame limpo como: '{caminho_completo_novo_arquivo}'")
        
        DF_limpo.to_csv(caminho_completo_novo_arquivo, decimal=".", index=False)
        
        print("Arquivo salvo com sucesso!")

    except Exception as e:
        print(f"ERRO FATAL ao salvar o arquivo: {e}")

# ... (seu bloco 'else' continua aqui)
    
    print("\n\n--- Processo Finalizado ---")
    print("Amostra do DataFrame limpo:")
    print(DF_limpo.head())
else:
    print(f"Caminho não encontrado {caminho_do_arquivo_escolhido}")
    