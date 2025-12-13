#Bibliotecas necessárias para leitura e manipulação de dados
#Melhora codigo
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
#================================================================================================================
#Função verifica se a arquivos em csv e procura arquivos 
def caminhos_csv(caminho_pasta):
    nome_arquivos = None

    for arquivo in os.listdir(caminho_pasta):
        if arquivo.endswith(".csv"):
            nome_arquivos = arquivo
            print("Arquivos encontados!!!")
            break
    if nome_arquivos != None:
        caminho_completo = os.path.join(caminho_pasta, nome_arquivos)
        dados = pd.read_csv(caminho_completo)
        return dados
    else: print("Não nenhum arquivo **csv**")
    return dados
    
DF = caminhos_csv('ML/data')
Pasta_original = 'R:/Arquivos/Codigos/MLL/ML'

#================================================================================================================

#Verifica valores Nulos e os substitui
if(DF.isnull() == 0).all().all():
        if(DF.isnull().values.all()):
            DF.dropna(inplace=True)
            print("Foram encontrados valores nulos no DataFrame e removidos")
else:
        print("Não há valores nulos no DataFrame")

#================================================================================================================
#Verifica valores fora da curva 
for Linha in DF.index:
        for coluna in DF.columns:
            valor_celula = DF.loc[Linha, coluna]
            if valor_celula > 3000 or valor_celula < 1:
                DF.loc[valor_celula, coluna] = np.nan
                print("A valores fora da curva")
            else:
                print("Valore está dentro da curva") 
                break
          

#================================================================================================================
#Substitui os virgulas por pontos 
for colunas_1 in DF.columns:
        if DF[colunas_1].dtype == 'object' or DF[colunas_1].dtype == 'numeric' :
            try:
                DF.to_csv('dados execel - Table 1 - dados execel - Table 1.csv.csv',  decimal='.', index=True)
                DF[colunas_1] = DF[colunas_1].str.replace(',', '.', regex=False)
                Escrever = pd.to_numeric(DF[colunas_1], errors='coerce')
                print(Escrever.head())
            except Exception as e:
                print(f"Não foi possivel subistituir as , por ., talvez já exista esse valor: {e}")
                break
#================================================================================================================
#Enviar arquivo para pasta data 

pasta_destino = 'ML/data'

arquivo_para_mover = 'dados execel - Table 1 - dados execel - Table 1.csv.csv'

caminho_origem = os.path.join('.' , arquivo_para_mover )


if os.path.isfile(caminho_origem):
    print(" -> Arquivo encontrado!")
    
    try:
      
        os.makedirs(pasta_destino, exist_ok=True)
        
        
        arquivos_no_destino = [f for f in os.listdir(pasta_destino) if os.path.isfile(os.path.join(pasta_destino, f))]
        quantidade_existente = len(arquivos_no_destino)
        
        print(f" -> A pasta de destino '{pasta_destino}' já contém {quantidade_existente} arquivos.")
        
        novo_nome = f'dados_processados_{quantidade_existente + 1}.csv'
        

        caminho_destino_completo = os.path.join(pasta_destino, novo_nome)
        
        print(f" -> O arquivo será movido e renomeado para '{caminho_destino_completo}'")
        
  
        shutil.move(caminho_origem, caminho_destino_completo)
        
        print(f"\nSUCESSO! Arquivo movido e renomeado.")

    except Exception as e:
        print(f"ERRO: Ocorreu um erro durante o processo de movimentação: {e}")

else:
    print(f"AVISO: O arquivo '{arquivo_para_mover}' não foi encontrado na pasta principal. Nenhuma ação foi tomada.")
          
teste = 'ML/data/dados_processados_2.csv'


