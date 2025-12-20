#Bibliotecas necessárias para leitura e manipulação de dados
#Melhora codigo
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
#================================================================================================================
#Modificação total do codigo de limpeza
'''''
print("--- 1. Carregando e Limpando os Dados ---")

def limpar_e_converter_para_numerico(series):
    if pd.api.types.is_object_dtype(series):
        try:
            series_limpa = (
                series.str.replace('.', '', regex=False)
                      .str.replace(',', '.', regex=False)
                      .str.strip()
            )
            return pd.to_numeric(series_limpa, errors='raise')
        except (ValueError, AttributeError):
            return series
    return series

try:
    DF = pd.read_csv('ML/data/dados_processados_2.csv') 
except FileNotFoundError:
    print("ERRO: O arquivo 'dados_processados_2.csv' não foi encontrado.")
    exit()

DF.columns = DF.columns.str.strip()

if 'Unnamed: 0' in DF.columns:
    DF = DF.drop(columns=['Unnamed: 0'])

DF = DF.apply(limpar_e_converter_para_numerico)

print("\n--- Dados após limpeza e conversão ---")
DF.info()

print("\n--- 2. Preparando os Dados para o Modelo ---")

Coluna_alvo = "CBR"
DF.dropna(inplace=True)
DF.reset_index(drop=True, inplace=True)

if Coluna_alvo not in DF.columns:
    print(f"ERRO: A coluna alvo '{Coluna_alvo}' foi removida durante a limpeza.")
    exit()
'''
def Verificar_CSV(caminho_pasta):
    caminho_Arquivo = None
    numeracao = 0
    arquivos_csv_encontrados = []

    try:
        for arquivo in os.listdir(caminho_pasta):
            if arquivo.endswith('.csv'):
                arquivos_csv_encontrados.append(arquivo)
                numeracao += 1
                print(f'>>>>>>>>>>>>>>>>>>>>>>!!!!Arquivos encotrado!!!!<<<<<<<<<<<<<<<<< \n Nome dos arquivos: {numeracao} - {arquivo}')
        
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

DF = Verificar_CSV('R:/Arquivos/Codigos/MLL/ML/data')

