#Bibliotecas necessárias para leitura e manipulação de dados

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

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
            else:print("Valore está dentro da curva")
            break

#================================================================================================================
#Substitui os virgulas por pontos 
for colunas_1 in DF.columns:
        if DF[colunas_1].dtype == 'object' or DF[colunas_1].dtype == 'numeric' :
            try:
               #DF.to_csv('dados execel - Table 1 - dados execel - Table 1.csv.csv', decimal='.', index=True, regex=True)
                DF[colunas_1] = DF[colunas_1].str.replace(',', '.', regex=False)
                Escrever = pd.to_numeric(DF[colunas_1], errors='coerce')
                print(Escrever.head())
            except Exception as e:
                print("Não foi possivel subistituir as , por ., talvez já exista esse valor: {e}")



