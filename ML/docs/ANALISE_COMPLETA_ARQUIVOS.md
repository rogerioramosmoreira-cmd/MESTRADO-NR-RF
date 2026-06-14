# ANÁLISE COMPLETA DOS ARQUIVOS PYTHON — Projeto ML/CBR

**Data:** 6 de maio de 2026  
**Projeto:** Previsão de CBR (California Bearing Ratio) em solos brasileiros  
**Tecnologias:** Python, scikit-learn, TensorFlow/Keras, matplotlib

---

## 📋 ÍNDICE RÁPIDO

1. [LEITURA.py](#1-leiturapy)
2. [MLL.py](#2-mllpy)
3. [PREVISAO.py](#3-previsaopy)
4. [PREVISAO_RF.py](#4-previsao_rfpy)
5. [RANDOM_FLOREST_Dn.py](#5-random_florest_dnpy)
6. [RANDOM_FLOREST_Separado.py](#6-random_florest_separadopy)
7. [RANDOM_FOREST.py](#7-random_forestpy)
8. [RANDOM_FOREST_EN.py](#8-random_forest_enpy)

---

## 1. LEITURA.py

### 📝 Nome do Arquivo
**LEITURA.py**

### 🎯 Propósito Principal
Script de **pré-processamento e limpeza de dados** que carrega um arquivo CSV, remove outliers, trata valores nulos e converte tipos de dados. Salva o arquivo limpo com nomenclatura sequencial (dados_processados_1.csv, dados_processados_2.csv, etc.).

### 🏗️ Estrutura Geral

**Funções principais:**
- `Verificar_CSV(caminho_pasta)` — Menu interativo para seleção de arquivo CSV
- `Limpeza(df_entrada)` — Processa e limpa o DataFrame em 3 etapas

**Fluxo principal:**
1. Lista arquivos CSV na pasta `/data`
2. Usuário escolhe um arquivo interativamente
3. Carrega o CSV com `pd.read_csv()`
4. Aplica limpeza (3 etapas):
   - Remove espaços e converte strings em números
   - Identifica e remove outliers (valores < 1 ou > 3000)
   - Remove linhas com dados faltantes (NaN)
5. Salva arquivo limpo com nome único

### 📥 Entrada e 📤 Saída

**Entrada:**
- Arquivo CSV selecionado do usuário (qualquer formato com dados tabulares)
- Localização: `R:/Arquivos/Codigos/MLL/ML/data`

**Saída:**
- Arquivo CSV limpo salvo como `dados_processados_N.csv`
- DataFrame em memória `DF_limpo` (exibido como amostra)
- Relatório de limpeza no console (linhas removidas, conversões realizadas)

### 🔧 Principais Algoritmos/Técnicas

| Técnica | Descrição |
|---------|-----------|
| **Limpeza de String** | Remove espaços, pontos, substitui vírgulas por pontos |
| **Conversão de Tipo** | `pd.to_numeric()` com validação de erros |
| **Detecção de Outliers** | Critério fixo (< 1 ou > 3000) baseado em conhecimento do domínio |
| **Remoção de NaN** | `dropna()` — remove linhas com qualquer valor faltante |
| **Reset de Índice** | `reset_index(drop=True)` — reindexação sequencial |

---

## 2. MLL.py

### 📝 Nome do Arquivo
**MLL.py**

### 🎯 Propósito Principal
Script de **treinamento de modelo MLP (Rede Neural)** para previsão de CBR com foco em **alta precisão**, incluindo regularização avançada, log-transform no alvo, pesos amostrais e busca de hiperparâmetros com 30 iterações usando `RandomizedSearchCV`.

### 🏗️ Estrutura Geral

**Componentes principais:**
- **Importações:** TensorFlow/Keras, scikit-learn, matplotlib, pandas, numpy
- **Configurações globais:** Caminhos, seeds, hiperparâmetros de busca, cores
- **Funções auxiliares:**
  - `normalizar_colunas(df)` — Padroniza nomes de colunas
  - `exibir_metricas(y_true, y_pred, nome)` — Calcula e exibe MSE, RMSE, MAE, R²
  - `construir_modelo(n_features, dropout, lr)` — Cria modelo MLP sequencial
  
- **Funções de gráfico (9 gráficos):**
  1. Painel principal (Previsto vs Real, Resíduos, Histórico MAE)
  2. Distribuição dos resíduos (histograma + KDE)
  3. Histórico de MSE
  4. Busca de hiperparâmetros
  5. Verificação de Data Leakage
  6. Outliers nos resíduos
  7. Detalhamento do Early Stopping
  8. R² comparativo (treino, validação, teste)
  9. Raciocínio da Rede Neural

**Fluxo principal:**
1. Carrega dados de `dados_processados_1.csv`
2. Normaliza nomes de colunas
3. Aplica log-transform no alvo (log1p)
4. Split treino/validação/teste (20% test, 15% val)
5. Normaliza features com MinMaxScaler
6. Busca de hiperparâmetros: 30 iterações, 5-fold CV
7. Treina modelo MLP com Early Stopping
8. Avalia e gera 9 gráficos

### 📥 Entrada e 📤 Saída

**Entrada:**
- `dados_processados_1.csv` (esperado com colunas: 25.4mm, 9.5mm, 4.8mm, 2.0mm, 0.42mm, 0.076mm, LL, IP, Umidade Ótima, Densidade máxima, CBR)

**Saída:**
- Modelos e artifacts salvos em `Modelo_salvo_MLP/`:
  - `modelo_cbr.keras` — Modelo neural treinado
  - `scaler_cbr.joblib` — MinMaxScaler ajustado no treino
- 9 gráficos matplotlib exibidos na tela
- Métricas impressas no console

### 🔧 Principais Algoritmos/Técnicas

| Técnica | Descrição |
|---------|-----------|
| **Arquitetura MLP** | 128 → 64 → 32 → 16 → 1 neurônios por camada |
| **Ativação** | LeakyReLU (α=0.1) em camadas intermediárias; sem ativação na saída (regressão) |
| **Regularização** | BatchNormalization, Dropout, L2 (0.001) |
| **Função de Perda** | MAE (Mean Absolute Error) |
| **Otimizador** | Adam com learning rate adaptativo |
| **Early Stopping** | Patience=25 épocas, monitora val_loss |
| **ReduceLROnPlateau** | Reduz LR quando modelo estagna |
| **Log-transform** | log1p(CBR) → comprime assimetria → expm1(pred) |
| **Pesos Amostrais** | CBR > 20% recebe W=3.0, senão W=1.0 |
| **Busca HP** | RandomizedSearchCV: 30 iterações, 5-fold CV, scoring=MSE |

---

## 3. PREVISAO.py

### 📝 Nome do Arquivo
**PREVISAO.py**

### 🎯 Propósito Principal
Script **interativo de previsão** que carrega o modelo MLP treinado (`modelo_cbr.keras`) e o scaler (`scaler_cbr.joblib`), coleta dados de entrada do usuário (10 features de solo) e retorna a previsão de CBR com validação de limites.

### 🏗️ Estrutura Geral

**Estrutura modular:**
- **Configurações:** Caminhos para modelo/scaler, nomes de features, limites de entrada baseados nas normas ABNT/DNIT
- **Funções:**
  - `engenharia_features(valores)` — Calcula 8 features derivadas a partir de 10 originais
  - `carregar_artefatos(arquivo_modelo, arquivo_scaler)` — Carrega modelo e normalizador
  - `solicitar_valor(feature, minimo, maximo)` — Entrada interativa com validação
  - `prever_cbr(valores, modelo, scaler)` — Aplica feature engineering, normaliza e prediz

**Fluxo principal:**
1. Exibe menu com título e instruções
2. Carrega modelo MLP e scaler
3. Solicita 10 valores de entrada (com validação de intervalo)
4. Calcula 8 features derivadas
5. Normaliza com o scaler do treinamento
6. Faz previsão com o modelo MLP
7. Exibe resultado formatado com alertas

### 📥 Entrada e 📤 Saída

**Entrada:**
- 10 valores numéricos inseridos pelo usuário (via input/console):
  - Granulometria: 25.4mm, 9.5mm, 4.8mm, 2.0mm, 0.42mm, 0.076mm (%)
  - Limites de Atterberg: LL, IP (%)
  - Compactação: Umidade Ótima (%), Densidade máxima (kg/m³)

**Saída:**
- CBR previsto em % (valor escalar)
- Alertas se CBR < 2% ou > 100%
- Mensagens de validação

### 🔧 Principais Algoritmos/Técnicas

| Técnica | Descrição |
|---------|-----------|
| **Feature Engineering** | ratio_X_Y (gradientes granulométricos), atividade (LL-IP), compacidade, finos_sq |
| **Normalização** | MinMaxScaler ajustado no treinamento (escala [0, 1]) |
| **Previsão** | Modelo MLP → expm1(pred) para reverter log-transform |
| **Validação** | Limites por feature baseados em dissertação + normas técnicas |

---

## 4. PREVISAO_RF.py

### 📝 Nome do Arquivo
**PREVISAO_RF.py**

### 🎯 Propósito Principal
Script **interativo de previsão com Random Forest** (ensemble RF+GB+ET). Carrega modelo e scaler treinados, coleta 10 features do usuário, calcula 8 features derivadas e retorna previsão de CBR. Mais simples que PREVISAO.py mas usa ensemble em vez de rede neural.

### 🏗️ Estrutura Geral

**Estrutura:**
- **Configurações:** Caminhos, nomes de features, limites de entrada (idênticos a PREVISAO.py)
- **Funções:**
  - `engenharia_features(valores)` — Calcula 8 derivadas (mesmo que PREVISAO.py)
  - `carregar_artefatos(arquivo_modelo, arquivo_scaler)` — Carrega VotingRegressor + scaler
  - `solicitar_valor(feature, minimo, maximo)` — Entrada com validação
  - `prever_cbr(valores, modelo, scaler)` — Feature eng. → normalização → previsão

**Fluxo:**
1. Carrega modelo ensemble (VotingRegressor) e scaler
2. Solicita 10 valores ao usuário
3. Aplica feature engineering (8 derivadas)
4. Normaliza dados
5. Faz previsão com ensemble
6. Exibe resultado

### 📥 Entrada e 📤 Saída

**Entrada:**
- 10 valores numéricos de entrada (idênticos a PREVISAO.py)

**Saída:**
- CBR previsto (%) via ensemble RF+GB+ET
- Alertas se CBR < 2% ou > 100%

### 🔧 Principais Algoritmos/Técnicas

| Técnica | Descrição |
|---------|-----------|
| **Ensemble** | VotingRegressor combinando RF + GradientBoosting + ExtraTrees (média simples) |
| **Feature Engineering** | ratio_X_Y, atividade, compacidade, finos_sq (idêntico a PREVISAO.py) |
| **Normalização** | MinMaxScaler [0, 1] |

---

## 5. RANDOM_FLOREST_Dn.py

### 📝 Nome do Arquivo
**RANDOM_FLOREST_Dn.py**

### 🎯 Propósito Principal
Script de **treinamento de Random Forest com divisão por faixas de CBR** (D1 a D5 representando quintis). Treina um modelo independente para cada faixa de CBR, realiza busca de hiperparâmetros isolada por grupo e avalia qual faixa é mais desafiadora para o modelo.

### 🏗️ Estrutura Geral

**Estrutura modular:**
- **Configurações globais:** Paths, seeds, split ratios (25% teste, 20% val), espaço HP
- **Funções:**
  - `metricas(y_true, y_pred, nome)` — Calcula e exibe MSE, RMSE, MAE, R²
  - `treinar_quintil(rotulo, mask, X_full, Y_orig, Y_log)` — Pipeline completo para um quintil
  - Funções de gráfico para cada quintil (previsto vs real, resíduos, busca HP, importância)
  
- **Gráficos:**
  - Distribuição do CBR com fronteiras dos quintis
  - Previsto vs Real por quintil (val + teste)
  - Resíduos por quintil
  - Busca de HP por quintil (top 10 configurações)
  - Importância de features por quintil
  - Comparativo final (MSE, R², MAE, scatter global)
  - Heatmap de importância comparativa (features vs quintis)
  - Fluxograma conceitual

**Fluxo principal:**
1. Carrega dados de `dados_processados_1.csv`
2. Calcula percentis 0, 20, 40, 60, 80, 100 do CBR
3. Cria 5 máscaras (D1 a D5) para quintis
4. Para cada quintil:
   - Split independente treino/val/teste
   - Normaliza com MinMaxScaler
   - Busca de HP com RandomizedSearchCV (80 iterações, 5-fold CV)
   - Treina RF final com melhores HP
   - Avalia e salva modelo + scaler
5. Gera 8 gráficos comparativos

### 📥 Entrada e 📤 Saída

**Entrada:**
- `dados_processados_1.csv` com 10 features + CBR

**Saída:**
- Pasta `Modelo_salvo_RF_Quintis/` com subpastas D1, D2, D3, D4, D5
- Em cada subpasta:
  - `rf_modelo.joblib` — Modelo RF treinado
  - `scaler.joblib` — MinMaxScaler do grupo
- 8 gráficos matplotlib
- Relatório no console (resumo por quintil, melhores HP)

### 🔧 Principais Algoritmos/Técnicas

| Técnica | Descrição |
|---------|-----------|
| **Divisão em Quintis** | Percentis 0, 20, 40, 60, 80, 100 do CBR |
| **Split Independente** | 25% teste, 20% validação por grupo |
| **Busca HP** | RandomizedSearchCV: 80 iterações, 5-fold CV, scoring=-MSE |
| **Espaço HP RF** | n_est [50-500], max_depth [2-20], min_split [2-12], etc. |
| **Pesos Amostrais** | CBR > mediana do grupo recebe W=2.0, senão W=1.0 |
| **Log-transform** | log1p(CBR) durante treinamento; expm1() na avaliação |
| **Normalização** | MinMaxScaler independente por quintil |

---

## 6. RANDOM_FLOREST_Separado.py

### 📝 Nome do Arquivo
**RANDOM_FLOREST_Separado.py** (812 linhas)

### 🎯 Propósito Principal
Script **avançado com análise detalhada por quintis**, incluindo busca de melhor combinação de features (1-2 features mais preditivas por quintil), comparativo detalhado de importância entre groups e visualizações complexas com gridspec. Expande `RANDOM_FLOREST_Dn.py` com análise exploratória de combinações de features.

### 🏗️ Estrutura Geral (812 linhas)

**Seções principais:**

1. **Cabeçalho e Imports** (linhas 1-50)
   - Docstring explicando objetivo (divisão D1-D5)
   - Imports de numpy, pandas, matplotlib, scikit-learn

2. **Configurações Globais** (linhas 51-120)
   - Caminho de dados, nomes de features, cores, seeds
   - Constantes: TEST_SIZE=0.25, VAL_SIZE=0.20, CV_FOLDS=5
   - N_ITER_BUSCA=80, MAX_COMB_SIZE=2, TOP_N_COMB=5

3. **Funções Auxiliares** (linhas 121-200)
   - `metricas()` — Calcula MSE, RMSE, MAE, R²
   - `encontrar_melhor_combinacao_features()` — Avalia todas as combos de 1-2 features
   - `treinar_quintil()` — Pipeline completo com feature combination search

4. **Funções de Gráfico** (linhas 201-600)
   - `grafico_distribuicao_quintis()` — Histograma com faixas dos quintis
   - `grafico_previsto_vs_real_grupo()` — 2×1 (val + teste)
   - `grafico_residuos_grupo()` — Scatter de resíduos
   - `grafico_busca_grupo()` — Evolução da busca de HP
   - `grafico_importancia_grupo()` — Barras de importância (Gini)
   - `grafico_melhor_combinacao()` — Combo features + scatter 2D/1D
   - `grafico_comparativo_final()` — Painel 3×3 com todos os quintis (MSE, R², MAE, scatter global, tabela resumo)
   - `grafico_importancia_comparativa()` — Heatmap de importância (quintis × features)
   - `fluxograma_arvore_aleatoria()` — Fluxograma conceitual do RF

5. **Fluxo Principal** (linhas 601-812)
   - Carrega dados e calcula quintis
   - Loop sobre 5 grupos: treina RF + encontra melhores combinações de features
   - Para cada grupo: gera 5 gráficos (previsto vs real, resíduos, busca HP, importância, melhor combo)
   - Gera 3 gráficos comparativos (final, importância comparativa, fluxograma)
   - Imprime resumo final no console

### 📥 Entrada e 📤 Saída

**Entrada:**
- `dados_processados_1.csv`

**Saída:**
- `Modelo_salvo_RF_Quintis/D1/`, `D2/`, ..., `D5/`
  - Cada pasta contém: `rf_modelo.joblib`, `scaler.joblib`
- **19+ gráficos** exibidos ao longo da execução:
  - 5 gráficos por quintil × 5 quintis = 25 gráficos por quintil
  - Realmente: distribuição quintis (1) + 5 gráficos × 5 quintis (25) + 3 comparativos (3) = 29 gráficos
- Relatório detalhado no console com resumo final

### 🔧 Principais Algoritmos/Técnicas

| Técnica | Descrição |
|---------|-----------|
| **Busca de Melhor Combo** | itertools.combinations: avalia todas as 1-2 feature subsets, treina RF rápido em cada, ordena por MSE validação |
| **Quintis (D1-D5)** | Percentis 0-20-40-60-80-100 do CBR |
| **Split Independente** | 25% teste, 20% val por grupo |
| **Busca HP** | RandomizedSearchCV: 80 iter, 5-fold CV |
| **Feature Combo Eval** | RF rápido (120 est, max_depth=10) em cada subconjunto |
| **Visualização Heatmap** | Matriz (5 quintis × 10 features) com importância Gini |
| **Tabela Resumo** | matplotlib.table mostrando HP, MSE CV, MSE teste, R², MAE, melhor combo por quintil |

---

## 7. RANDOM_FOREST.py

### 📝 Nome do Arquivo
**RANDOM_FOREST.py**

### 🎯 Propósito Principal
Script de **treinamento de modelo Random Forest único** com busca extensiva de hiperparâmetros (150 iterações, 10-fold CV), retreino no conjunto treino+validação, múltiplas métricas e visualizações de raciocínio do modelo. Objetivo: MSE < 0.780 no teste.

### 🏗️ Estrutura Geral

**Componentes:**
- **Configurações:** Paths, seeds (42), split ratios (20% teste, 15% val), espaço HP extenso
- **Funções auxiliares:**
  - `metricas(y_true, y_pred, nome)` — Calcula e exibe MSE, RMSE, MAE, R²
  
- **Funções de gráfico (8 gráficos):**
  1. Previsto vs Real — Validação
  2. Previsto vs Real — Teste
  3. Tabela comparativa de métricas
  4. Resíduos — Validação
  5. Resíduos — Teste
  6. Importância das features (Gini)
  7. Raciocínio do RF (evolução da busca de HP)
  8. Convergência OOB (erro by número de árvores)

**Fluxo:**
1. Carrega `dados_processados_1.csv`
2. Normaliza nomes de colunas
3. Split treino/val/teste (20% + 15%)
4. Normaliza features com MinMaxScaler
5. Busca de HP: 150 iterações, 10-fold CV
6. Treina RF final com melhores HP no treino+val combinados
7. Avalia e gera 8 gráficos
8. Salva modelo + scaler + metadados.json

### 📥 Entrada e 📤 Saída

**Entrada:**
- `dados_processados_1.csv`

**Saída:**
- `Modelo_salvo_RF/`:
  - `rf_modelo_final.joblib`
  - `scaler.joblib`
  - `metadados.json` (feature names, cenário, etc.)
- 8 gráficos matplotlib
- Relatório console com métricas e HP finais

### 🔧 Principais Algoritmos/Técnicas

| Técnica | Descrição |
|---------|-----------|
| **Arquitetura RF** | Múltiplas árvores de decisão paralelas sem poda |
| **Busca HP** | RandomizedSearchCV: 150 iterações, 10-fold CV, scoring=-MSE |
| **Espaço HP** | n_est [50-500], max_depth [2-20], min_split [2-12], max_features [sqrt, log2, 0.3-1.0], max_samples [0.6-1.0] |
| **Log-transform** | log1p(CBR) durante treino; expm1(pred) na avaliação |
| **Pesos Amostrais** | CBR > 25% recebe W=3.0, senão W=1.0 |
| **Normalização** | MinMaxScaler [0, 1] |
| **OOB Score** | Out-of-Bag error para estimar convergência |

---

## 8. RANDOM_FOREST_EN.py

### 📝 Nome do Arquivo
**RANDOM_FOREST_EN.py**

### 🎯 Propósito Principal
Script de **treinamento de ensemble (VotingRegressor)** combinando 3 modelos: Random Forest, Gradient Boosting e Extra Trees. Realiza busca de hiperparâmetros individual por modelo, depois combina previsões por média simples. Objetivo: MSE < 0.780 no teste com performance mais robusta através do ensemble.

### 🏗️ Estrutura Geral

**Componentes:**
- **Configurações:** Paths, seeds, split ratios (20% + 15%), espaço HP por modelo
- **Funções auxiliares:**
  - `engenharia_features(df, coluna_alvo)` — Calcula 8 features derivadas
  - `metricas(y_true, y_pred, nome)` — MSE, RMSE, MAE, R²
  
- **Funções de gráfico (9 gráficos):**
  1. Previsto vs Real — Validação
  2. Previsto vs Real — Teste
  3. Tabela de métricas
  4. Resíduos — Validação
  5. Resíduos — Teste
  6. Comparativo de MSE (RF vs GB vs ET vs Ensemble)
  7. Importância das features (média RF + ET)
  8. (Não especificado no código lido, mas estrutura sugere mais gráficos)

**Fluxo:**
1. Carrega dados e aplica feature engineering
2. Split treino/val/teste
3. Normaliza com MinMaxScaler
4. Busca de HP para RF (150 iterações)
5. Busca de HP para GB (150 iterações)
6. Busca de HP para ET (150 iterações)
7. Cria VotingRegressor com 3 modelos
8. Treina ensemble no treino+val
9. Avalia e gera gráficos
10. Salva ensemble + scaler

### 📥 Entrada e 📤 Saída

**Entrada:**
- `dados_processados_1.csv` com 10 features originais

**Saída:**
- `Modelo_salvo_RF/`:
  - `rf_modelo_final.joblib` (VotingRegressor)
  - `scaler.joblib`
  - Metadados com feature names
- 9+ gráficos
- Relatório console com métricas de cada modelo e ensemble

### 🔧 Principais Algoritmos/Técnicas

| Técnica | Descrição |
|---------|-----------|
| **Feature Engineering** | ratio_X_Y (5), atividade, compacidade, finos_sq (total 18 features) |
| **Random Forest** | Busca HP: 150 iter, 10-fold CV |
| **Gradient Boosting** | Boosting sequencial corrigindo erros (busca HP: 150 iter) |
| **Extra Trees** | RF com splits aleatórios (busca HP: 150 iter) |
| **VotingRegressor** | Média simples das 3 previsões |
| **Ensemble Advantage** | Reduz variância e viés, mais robusto que modelo único |
| **Normalização** | MinMaxScaler global [0, 1] |
| **Log-transform** | log1p(CBR) no treino; expm1(pred) na avaliação |

---

## 📊 TABELA COMPARATIVA DOS ARQUIVOS

| Arquivo | Tipo | Entrada | Saída | Modelo | Objetivo |
|---------|------|---------|-------|--------|----------|
| **LEITURA.py** | Pré-processamento | CSV bruto | CSV limpo | — | Limpar dados |
| **MLL.py** | Treinamento | dados_processados_1.csv | Modelo MLP + scaler | MLP (Keras) | MSE < 0.780 |
| **PREVISAO.py** | Predição | Entrada do usuário (10 valores) | CBR previsto (%) | MLP carregado | Predizer CBR |
| **PREVISAO_RF.py** | Predição | Entrada do usuário (10 valores) | CBR previsto (%) | Ensemble RF+GB+ET | Predizer CBR |
| **RANDOM_FLOREST_Dn.py** | Treinamento | dados_processados_1.csv | 5 modelos RF (D1-D5) + gráficos | RF × 5 quintis | Comparar por faixa CBR |
| **RANDOM_FLOREST_Separado.py** | Treinamento | dados_processados_1.csv | 5 modelos RF + 29 gráficos | RF × 5 + feature combo search | Análise detalhada D1-D5 |
| **RANDOM_FOREST.py** | Treinamento | dados_processados_1.csv | Modelo RF final + scaler | RF único | MSE < 0.780 |
| **RANDOM_FOREST_EN.py** | Treinamento | dados_processados_1.csv | VotingRegressor + scaler | RF+GB+ET ensemble | MSE < 0.780 (ensemble) |

---

## 🔄 FLUXO DE EXECUÇÃO RECOMENDADO

```
1. LEITURA.py
   ↓ (gera dados_processados_1.csv)
2. Escolher uma rota:
   ┌─ RANDOM_FOREST.py (único modelo RF)
   ├─ MLL.py (modelo MLP)
   ├─ RANDOM_FOREST_EN.py (ensemble)
   ├─ RANDOM_FLOREST_Dn.py (análise por quintis)
   └─ RANDOM_FLOREST_Separado.py (análise detalhada por quintis)
3. Após treinamento de qualquer modelo:
   ├─ PREVISAO.py (se modelo é MLP)
   └─ PREVISAO_RF.py (se modelo é RF/ensemble)
```

---

## 🎓 RESUMO TÉCNICO

### Tecnologias Utilizadas
- **Python 3.10+**
- **pandas** — Manipulação tabular de dados
- **numpy** — Operações numéricas e arrays
- **scikit-learn** — ML (RandomForest, GradientBoosting, ExtraTrees, VotingRegressor, MinMaxScaler)
- **TensorFlow/Keras** — Deep learning (MLP, BatchNorm, Dropout, LeakyReLU)
- **matplotlib + seaborn** — Visualização de dados
- **joblib** — Serialização de modelos

### Principais Conceitos
- **Log-transform:** Comprime assimetria à direita da distribuição do CBR
- **Pesos Amostrais:** Aumenta relevância de amostras raras (alto CBR)
- **Feature Engineering:** Ratios granulométricos, atividade, compacidade
- **Busca de Hiperparâmetros:** RandomizedSearchCV com 10-fold CV
- **Early Stopping:** Previne overfitting em treino neural
- **Ensemble:** Combina múltiplos modelos para predição mais robusta
- **Quintis (D1-D5):** Divisão do dataset por faixa de CBR para análise estratificada

### Limites Técnicos (baseados em normas ABNT/DNIT)
- **Granulometria:** 0-100% (porcentagem passante acumulada)
- **LL, IP:** 0-100%, 0-80% (respectivamente)
- **Umidade Ótima:** 5-40%
- **Densidade máxima:** 1200-2200 kg/m³

---

## 📁 ESTRUTURA DE ARQUIVOS GERADA

```
Modelo_salvo_MLP/
  ├─ modelo_cbr.keras
  └─ scaler_cbr.joblib

Modelo_salvo_RF/
  ├─ rf_modelo_final.joblib
  ├─ scaler.joblib
  └─ metadados.json

Modelo_salvo_RF_Quintis/
  ├─ D1/
  │   ├─ rf_modelo.joblib
  │   └─ scaler.joblib
  ├─ D2/, D3/, D4/, D5/
  │   (estrutura idêntica)
  └─ ...
```

---

**Documentação completada em 6 de maio de 2026**
