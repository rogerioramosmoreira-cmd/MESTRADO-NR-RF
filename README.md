# Previsão de CBR por Machine Learning

<<<<<<< HEAD
Dissertação de Mestrado — PUC Goiás

Pipeline de Machine Learning para previsão do **CBR (California Bearing Ratio)** de solos a partir de variáveis granulométricas, de plasticidade e de compactação Proctor. Inclui modelos de Árvore Aleatória (Random Forest) e Rede Neural (MLP), comparação entre conjuntos de variáveis, e dashboard interativo em Streamlit.

---

## Início rápido

```bash
# 1. Criar o ambiente virtual
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux/macOS

# 2. Instalar as dependências
pip install -r requirements.txt

# 3. Conferir se está tudo instalado
python src/main.py --check

# 4. Abrir o menu
python src/main.py
```

No Windows, `iniciar.bat` faz os passos 3 e 4 em uma janela de terminal já configurada para UTF-8.

### Rodando de uma IDE

Da IDE, execute **`src/main.py`** — e não `run.py`, que abre uma janela de console separada.

O ponto que decide se funciona bem é **onde a saída aparece**. O menu lê teclas com `input()` e a barra de progresso reescreve a linha; os dois precisam de um terminal de verdade:

| IDE | O que usar |
|---|---|
| **VS Code** | `Run and Debug` → *Menu principal* (já configurado em `.vscode/launch.json`). Ou botão ▶ com **"Run Python File in Terminal"** — nunca o Debug Console. |
| **PyCharm** | `Run` normal já funciona. Em *Edit Configurations*, marque **Emulate terminal in output console**. |
| **Spyder / Jupyter** | O menu interativo não funciona bem. Use a forma sem menu: `python src/main.py --run 7`. |

Se a barra aparecer como linhas soltas de 10% em vez de uma barra que anda, o painel da IDE não está se declarando terminal. Force:

```
MLL_BAR=1
```

`MLL_BAR=0` faz o contrário — sempre marcos, útil quando você redireciona a saída para um arquivo de log.

---

## Variáveis de entrada

| Variável | Significado | Grupo |
|---|---|---|
| 25.4mm (IG) | Pedregulho grosso | Granulometria |
| 9.5mm (EXP) | Pedregulho médio/fino | Granulometria |
| 4.8mm (D3) | Peneira nº 4 — limite areia/pedregulho | Granulometria |
| 2.0mm (D4) | Peneira nº 10 — areia grossa | Granulometria |
| 0.42mm (D5) | Peneira nº 40 — areia fina | Granulometria |
| 0.076mm (D6) | Peneira nº 200 — silte/argila | Granulometria |
| LL | Limite de Liquidez | Plasticidade |
| IP | Índice de Plasticidade | Plasticidade |
| Umidade Ótima | Umidade ótima Proctor | Compactação |
| Densidade máxima | Massa específica seca máxima Proctor | Compactação |

Alvo: **CBR (%)**. Dataset: 498 amostras completas.

---

## Conjuntos de variáveis (C1–C5)

Os scripts de comparação treinam um modelo independente por conjunto, todos sobre a **mesma divisão treino/validação/teste** (`random_state=42`). Uma diferença entre eles é diferença de informação, não de sorte no split.

| Conjunto | Variáveis | Nº |
|---|---|---|
| **C1** | Granulometria (6 peneiras) | 6 |
| **C2** | Plasticidade (LL, IP) | 2 |
| **C3** | Compactação (Umidade Ótima, Densidade máxima) | 2 |
| **C4** | Granulometria + Plasticidade | 8 |
| **C5** | Todas | 10 |

A pergunta prática: quais ensaios de laboratório realmente precisam ser feitos para prever o CBR.

---

## Estrutura do projeto

```
mll/
├── run.py                       # Abre o pipeline em uma nova janela de terminal
├── iniciar.bat                  # Atalho Windows para o mesmo
├── requirements.txt
├── data/
│   ├── raw/cbr_raw.csv          # Exportação original da planilha
│   └── processed/cbr_dataset.csv
├── models/                      # Artefatos treinados (.joblib / .keras)
│   ├── random_forest/
│   ├── random_forest_en/
│   ├── random_forest_ensemble/
│   ├── random_forest_quintis/   # D1 … D5
│   ├── mlp/
│   └── subsets/                 # random_forest/ e mlp/, cada um com c1 … c5
├── reports/
│   └── figures/                 # Todos os gráficos salvos automaticamente
├── docs/
└── src/
    ├── main.py                  # Menu e launcher
    ├── dashboard.py             # Dashboard Streamlit
    ├── core/                    # Infraestrutura compartilhada
    │   ├── paths.py             # Resolução central de caminhos
    │   ├── console.py           # Interface de terminal (rich)
    │   ├── dependencies.py      # Verificação de bibliotecas
    │   ├── data.py              # Carregamento, normalização, conjuntos C1–C5
    │   ├── ingest.py            # XLSX → CSV, limpeza, montagem do dataset
    │   ├── metrics.py           # MSE, RMSE, MAE, MAPE, R² e o veredito
    │   ├── scoreboard.py        # Precisão da última execução de cada modelo
    │   ├── plots.py             # Estilo, remoção de títulos, salvamento
    │   └── subset_report.py     # Gráficos e persistência da comparação C1–C5
    ├── models/
    │   ├── random_forest.py             # RF padrão (PT)
    │   ├── random_forest_en.py          # RF alta performance (EN)
    │   ├── random_forest_ensemble.py    # VotingRegressor RF + GB + ET
    │   ├── random_forest_quintis.py     # Um RF por faixa de CBR (D1–D5)
    │   ├── random_forest_quintis_full.py# Quintis + combinações de features
    │   ├── mlp.py                       # Rede neural (modelo único)
    │   ├── subsets_rf.py                # Árvore Aleatória em C1–C5
    │   ├── subsets_mlp.py               # Rede Neural em C1–C5
    │   └── subsets_comparison.py        # RF vs MLP, conjunto a conjunto
    ├── predict/
    │   ├── predict_rf.py                # Previsão com o RF salvo
    │   └── predict_rf_quintis.py        # Previsão com um modelo de quintil
    ├── prepare/
    │   └── build_dataset.py             # Planilha/CSV bruto → dataset de treino
    └── explore/
        └── inspect_data.py              # Limpeza manual antiga — fora do menu,
                                         # substituída por prepare/build_dataset.py
=======
Dissertação de Mestrado — PUC Goiás 2025

Pipeline de Machine Learning para previsão do **CBR (California Bearing Ratio)** de solos a partir de variáveis granulométricas e de compactação Proctor. Inclui modelos de Random Forest e Rede Neural MLP, com dashboard interativo em Streamlit.

---

## Visão Geral

| Componente | Descrição |
|---|---|
| Alvo | CBR (%) — suporte de solo para pavimentação |
| Meta de desempenho | MSE < 0,780 no conjunto de teste |
| Modelos | Random Forest (PT e EN), RF por Quintis D1–D5, MLP |
| Interface | Menu CLI (`main.py`) + Dashboard Streamlit |

---

## Variáveis de Entrada

| Variável | Significado |
|---|---|
| 25.4mm (IG) | Pedregulho grosso |
| 9.5mm (EXP) | Pedregulho médio/fino |
| 4.8mm (D3) | Peneira nº 4 — limite areia/pedregulho |
| 2.0mm (D4) | Peneira nº 10 — areia grossa |
| 0.42mm (D5) | Peneira nº 40 — areia fina |
| 0.076mm (D6) | Peneira nº 200 — silte/argila |
| LL | Limite de Liquidez |
| IP | Índice de Plasticidade |
| Umidade Ótima | Umidade ótima Proctor |
| Densidade máxima | Massa específica seca máxima Proctor |

---

## Estrutura do Projeto

```
ML/
├── data/
│   ├── raw/                        # Dados brutos originais
│   └── processed/
│       └── dados_processados_1.csv # Dataset de entrada do pipeline
├── models/
│   ├── rf/                         # Modelo RF (Português)
│   ├── rf_en/                      # Modelo RF (English)
│   ├── rf_quintis/                 # Modelos RF por quintis
│   └── mlp/                        # Rede Neural MLP
├── src/
│   ├── main.py                     # Ponto de entrada — menu interativo
│   ├── RANDOM_FOREST.py            # RF padrão (Português)
│   ├── RANDOM_FOREST_EN.py         # RF padrão (English) — alto desempenho
│   ├── RANDOM_FLOREST_Dn.py        # RF por quintis D1–D5 (simples)
│   ├── RANDOM_FLOREST_Separado.py  # RF por quintis D1–D5 (completo + combos)
│   ├── MLL.py                      # Rede Neural MLP (TensorFlow/Keras)
│   ├── dashboard.py                # Dashboard Streamlit
│   ├── PREVISAO.py                 # Previsão com modelo MLP salvo
│   ├── PREVISAO_RF.py              # Previsão com modelo RF salvo
│   └── LEITURA.py                  # Leitura e inspeção de dados
└── docs/
    ├── ANALISE_COMPLETA_ARQUIVOS.md
    └── DOCUMENTACAO_COMPLETA_ML.docx
```

---

## Instalação

**Pré-requisito:** Python 3.13+

```bash
# Criar e ativar ambiente virtual
python -m venv ML
ML\Scripts\activate          # Windows
# source ML/bin/activate     # Linux/macOS

# Instalar dependências
pip install -r requirements.txt
>>>>>>> 5be1326de65eba6a8127ed3262a0b20ebf5729b0
```

---

## Uso

<<<<<<< HEAD
### Menu interativo

```bash
python src/main.py
```

Uma tela só: cabeçalho com o estado do ambiente, a precisão dos modelos já treinados, e o catálogo numerado — modelos (1–9), preparação dos dados (10), grupos (11–14), diagnóstico de bibliotecas (15).

### Preparação dos dados (item 10)

Largue a planilha em `data/raw` e mande treinar: a conversão é automática. Cada `.xlsx` vira CSV e o dataset é remontado sem ninguém precisar abrir o Excel.

```bash
python src/prepare/build_dataset.py                  # converte tudo e regrava
python src/prepare/build_dataset.py --arquivo X.csv  # usa um arquivo específico
python src/prepare/build_dataset.py --limites        # descarta valores implausíveis
python src/prepare/build_dataset.py --manter         # não sobrescreve o dataset atual
```

Com mais de um CSV em `data/raw`, ele lista os candidatos e pergunta qual usar; ENTER fica com o modificado por último, que é quase sempre o que acabou de ser convertido. Sem terminal interativo, segue no mais recente em vez de travar esperando resposta.

`--limites` descarta valores fora de `[1, 3000]` — a checagem de plausibilidade que existia na limpeza manual antiga. É opcional de propósito: o dataset atual tem um IP de 0,57, baixo mas possível em solo praticamente não plástico, e apagar essa linha por conta própria seria decidir pelo usuário. Cada valor descartado é contado por coluna.

A limpeza (`core/ingest.py`) faz, nesta ordem:

1. **acha a linha de cabeçalho de verdade.** Planilha de laboratório quase nunca começa por ela: vêm antes uma linha em branco e uma faixa de títulos mesclados (`Granulometria % Passante nas Peneiras`). Lida de cima para baixo, a planilha inteira viraria `Unnamed: 0`, `Unnamed: 1`… A linha escolhida é a que tem mais nomes de coluna conhecidos, com no mínimo três
2. **completa os nomes em branco** olhando a linha de baixo — é lá que mora a designação da primeira peneira (`#1"` = 25.4mm) — e, se ainda faltar, a linha de cima, onde estão os títulos mesclados. A linha de designações é descartada junto: é rótulo, não amostra
3. **tira a unidade do nome** — `"LL (%)"` vira `LL`, `"Densidade máxima (Kg/m³)"` vira `Densidade máxima`
4. **vírgula decimal vira ponto** — `"100,00"` vira `100.0`
5. **cabeçalhos** perdem espaços sobrando (`"CBR "`) e ganham o nome canônico de `core/data.py`
6. **linhas e colunas** inteiramente vazias somem
7. **linhas incompletas** (sem alguma variável ou sem CBR) somem, e a contagem do descarte é informada

Coluna que sobrou sem nome vira `coluna_<n>` e **fica** no CSV convertido, com um aviso de quantos valores ela tem: costuma ser um número solto digitado fora da tabela, e apagar dado de planilha sem avisar é pior do que manter uma coluna estranha. Ela não entra no dataset de treino.

Quando o dataset muda de tamanho, o aviso é explícito — a precisão registrada de cada modelo foi medida no dataset anterior e só volta a valer depois de treinar de novo.

Planilha de uma aba vira `<nome>.csv`; com várias abas, cada uma vira `<nome>__<aba>.csv`. Planilha já convertida e sem alteração desde então é pulada, então a checagem no início de cada treino não custa nada.

O dataset processado só é regravado por conta própria quando **não existe** — o arquivo em `data/processed` costuma ser o que foi conferido à mão, e sobrescrevê-lo a cada treino desfaria essa conferência em silêncio. Para regravar de propósito, use o item 11 do menu.

Exige `openpyxl` (já em `requirements.txt`, grupo de dependências `excel`).

### Precisão dos modelos

O menu abre mostrando o que cada modelo entregou no conjunto de teste — R², MAE, RMSE, MSE e MAPE — com a data do treino:

```
┌─────────────────────────┬──────────┬──────────┬───────────┬───────────────┐
│  Modelo                 │      R²  │     MAE  │      MSE  │     Treinado  │
├─────────────────────────┼──────────┼──────────┼───────────┼───────────────┤
│  1. RF — modelo único   │  0.8051  │  4.9773  │  58.3051  │  19/08 10:45  │
│  4. RF — quintis D1–D5  │  0.9235  │  2.6181  │  28.7283  │  19/08 10:48  │
└─────────────────────────┴──────────┴──────────┴───────────┴───────────────┘
```

Cada treino grava o próprio resultado em `reports/metrics/<script>.json` (`core/scoreboard.py`), então a tabela é sempre a última execução real de cada modelo — nada é recalculado ao abrir o menu. Em terminal com menos de 104 colunas, RMSE e MAPE saem da tabela para os nomes não quebrarem. Um `—` na célula significa que aquele script não mede aquela métrica: os quintis e a rede neural, por exemplo, não calculam MAPE.

### Veredito

Todo treino termina com a classificação do modelo, e a mesma classificação aparece na coluna **Situação** do menu:

```
╭──────────────── AVALIAÇÃO DO MODELO ────────────────╮
│  ADEQUADO                                           │
│  previsão utilizável; confirmar os casos críticos    │
│  em ensaio                                          │
│                                                     │
│  R² 0.8051 — o modelo explica 81% da variação do    │
│  CBR em dados novos                                 │
│  Erro médio (MAE): ±4.98 no valor de CBR            │
│  MSE 58.3051  |  RMSE 7.6358  |  MAPE 31.44%        │
╰─────────────────────────────────────────────────────╯
```

A régua é o R² do conjunto de teste:

| R² | Classificação | Leitura |
|---|---|---|
| ≥ 0.90 | **EXCELENTE** | previsão confiável; serve para dimensionamento |
| ≥ 0.80 | **ADEQUADO** | previsão utilizável; confirmar os casos críticos em ensaio |
| ≥ 0.70 | **BOA** | serve para estimativa preliminar, não para decisão final |
| ≥ 0.50 | **RUIM** | erra demais para substituir o ensaio; use só como indicativo |
| < 0.50 | **INUTILIZÁVEL** | não explica o CBR; não use para previsão |

Os cortes são uma escolha do projeto, não uma verdade estatística — um R² de 0,75 que seria fraco em laboratório é razoável para CBR de campo, onde o próprio ensaio tem dispersão alta. Para mexer na régua, edite `GRADE_THRESHOLDS` em `core/metrics.py`; a tabela do veredito, a coluna do menu e a legenda saem todas de lá.

O painel ainda acrescenta ressalvas quando os números pedem: MAPE acima de 30% (o modelo erra proporcionalmente mais nos CBR baixos) ou R² de treino abaixo do de teste (sinal de problema na divisão dos dados).

### Execução direta, sem menu

```bash
python src/main.py --run 7          # um item do catálogo
python src/main.py --run subsets    # grupo: subsets_rf + subsets_mlp + comparação
python src/main.py --run all        # todos os modelos
python src/main.py --check          # só o diagnóstico de bibliotecas
```

Grupos disponíveis: `all`, `rf`, `neural`, `subsets`.

### Scripts individuais

Cada script roda sozinho e verifica suas próprias dependências antes de começar:

```bash
python src/models/subsets_rf.py
python src/models/subsets_mlp.py
python src/models/subsets_comparison.py
```

`subsets_comparison.py` treina automaticamente o que ainda não tiver resultado salvo. Use `--train` para forçar o retreino dos dois.

### Dashboard

```bash
streamlit run src/dashboard.py
```

---

## Gráficos

Todos os gráficos são **salvos em `reports/figures/`**, organizados por modelo:

```
reports/figures/
├── random_forest/
├── random_forest_en/
├── random_forest_ensemble/
├── random_forest_quintis/
├── mlp/
└── subsets/
    ├── random_forest/{c1..c5}/     # previsto vs real, resíduos, importância
    ├── random_forest/              # comparativos entre os 5 conjuntos
    ├── mlp/{c1..c5}/               # previsto vs real, resíduos, curva de aprendizado
    ├── mlp/                        # comparativos entre os 5 conjuntos
    └── comparison/                 # Árvore Aleatória vs Rede Neural
```

**Os gráficos não têm título.** A informação fica nas legendas, nos rótulos dos eixos e nas anotações, porque a legenda da figura na dissertação é que cumpre o papel do título. A regra é estrutural: `core/plots.save()` remove qualquer título antes de gravar, então um `set_title` esquecido em uma edição futura não vaza para a figura publicada.

### Variáveis de ambiente

| Variável | Padrão | Efeito |
|---|---|---|
| `MLL_SHOW` | `1` | Abre as figuras na tela no fim do script, além de salvá-las. `0` usa o backend Agg — não abre janelas, só salva |
| `MLL_SHOW_LIMIT` | `12` | Teto de janelas abertas de uma vez; as excedentes ficam só em disco |
| `MLL_FIG_FORMATS` | `png` | Lista separada por vírgula, ex. `png,pdf` |
| `MLL_FIG_DPI` | `200` | Resolução das figuras salvas |
| `MLL_BAR` | detecta | `1` força a barra animada, `0` força marcos de 10% |
| `MLL_FAST` | `0` | `1` reduz as buscas e grava em pastas `_rapido` |

---

## Barras de progresso

Todo treino demorado mostra em que ponto está — uma linha numerada com a etapa e uma barra abaixo dela:

```
[1/5] C1 — busca de hiperparâmetros
[|||||||||||||||||||         ]  70%
```

Onde o total é conhecido de antemão:

| Script | Etapa | Total |
|---|---|---|
| `subsets_rf.py` | busca de hiperparâmetros | `60 × 5 folds` = 300 ajustes |
| `subsets_mlp.py` | busca | 12 candidatos |
| `subsets_mlp.py` | treino final | até 400 épocas |
| `random_forest_ensemble.py` | busca de cada estimador | `150 × 10 folds` = 1500 ajustes |

A contagem dos ajustes do `scikit-learn` vem do `BatchCompletionCallBack` do joblib, substituído temporariamente por `core/progress.joblib_stage`. É um detalhe interno do joblib, então qualquer falha ao instalar o gancho apenas desiste da barra — a busca roda igual.

A barra do treino final da rede neural **normalmente termina incompleta** (30%, 40%). Não é falha: o total é o teto de épocas, e o early stopping interrompe assim que a validação estaciona.

A barra é escrita direto em `stdout` com retorno de carro, sem biblioteca de terminal. Quando a saída não é um terminal — redirecionada para arquivo, lida por outro processo — ela é impressa em marcos de 10%, uma linha por marco, para o log continuar legível.

---

## Verificação de bibliotecas e erros

**A verificação é automática.** Ao abrir o projeto — `python src/main.py`, `run.py` ou `iniciar.bat` — a checagem roda antes de o menu aparecer, sem que o usuário precise pedir.

Quando está tudo instalado, resume em uma linha:

```
──────────── Verificação automática de bibliotecas ────────────
  ✓ 9 bibliotecas verificadas, nenhuma ausente.
      joblib 1.5.3, matplotlib 3.11.1, numpy 2.5.2, pandas 3.0.5, ...
```

Quando falta alguma coisa, mostra **o que parou de funcionar** e o comando exato de instalação, e marca no catálogo os itens indisponíveis:

```
  ✗ 2 biblioteca(s) ausente(s): streamlit, tensorflow
      grupo 'neural' incompleto — falta tensorflow
      grupo 'dashboard' incompleto — falta streamlit

  •  6. Rede Neural — modelo único   — indisponível
```

O menu não é bloqueado: faltar TensorFlow inutiliza apenas os modelos de rede neural, e travar tudo por causa disso impediria de rodar o que continua funcionando. A exceção é o grupo `core` — sem ele, nada roda, e a mensagem diz isso.

A opção 3 do menu reexecuta o diagnóstico detalhado, versão por versão, para depois de instalar o que faltava.

`core/dependencies.py` resolve cada módulo com `importlib.util.find_spec`, sem importá-lo — checar o TensorFlow não custa os segundos de carregamento dele.

Além da checagem de abertura, cada script declara o grupo de que precisa (`core`, `neural`, `plots`, `dashboard`) e refaz a verificação ao iniciar, para o caso de ser executado direto, fora do menu.

Códigos de saída:

| Código | Significado |
|---|---|
| 0 | Concluído |
| 1 | Erro durante a execução |
| 2 | Bibliotecas ausentes |
| 130 | Interrompido pelo usuário |

Qualquer exceção não tratada é enfileirada por `console.guard()` em um painel de erro com o tipo, a mensagem e o traceback — na mesma linguagem visual do resto da execução.

---

## Convenções de código

- **Comentários e docstrings em português.** Vale para todo o projeto.
- **Exceção única:** `src/models/random_forest_en.py` é mantido em inglês de
  propósito — ele é a variante em inglês do `random_forest.py`. Os dois têm os
  mesmos hiperparâmetros e as mesmas dez funções; a diferença é só o idioma.
  Rodar os dois treina modelos equivalentes, em `models/random_forest/` e
  `models/random_forest_en/`. **Não são resultados independentes**, e não devem
  ser citados como se fossem.
- Nomes de arquivos, pastas e identificadores em inglês, `snake_case`.
- Rótulos de eixo, legendas e saída de terminal em português.
=======
### Menu CLI (todos os modelos)

```bash
cd ML/src
python main.py
```

Opções disponíveis:
- **Modo Separado** — executa um modelo por vez
- **Modo Juntos** — executa grupo de modelos em sequência

### Dashboard Streamlit

```bash
cd ML/src
streamlit run dashboard.py
```

Acesso em `http://localhost:8501` — visualização de métricas, previsão interativa e gráficos de desempenho.
>>>>>>> 5be1326de65eba6a8127ed3262a0b20ebf5729b0

---

## Modelos

<<<<<<< HEAD
### Árvore Aleatória — padrão (`random_forest.py`, `random_forest_en.py`)
Busca aleatória de hiperparâmetros, log-transform no alvo (`log1p`/`expm1`), pesos amostrais para CBR alto, retreino em treino + validação. Gráficos de curva de busca, convergência OOB e importância de features.

### Árvore Aleatória — ensemble (`random_forest_ensemble.py`)
`VotingRegressor` combinando três estratégias de árvore com erros pouco correlacionados:

- **RandomForest** — média de árvores profundas sobre amostras bootstrap
- **GradientBoosting** — cada árvore corrige o resíduo da anterior
- **ExtraTrees** — cortes em limiares aleatórios, descorrelaciona o erro dos outros dois

Cada um é ajustado separadamente antes de entrar no ensemble. Inclui 8 features derivadas (razões entre peneiras, atividade, compacidade, finos²). Um gráfico compara o MSE de cada membro contra o do ensemble — se o ensemble não ficar abaixo de todos, a média não está agregando nada.

### Árvore Aleatória — quintis (`random_forest_quintis.py`, `..._full.py`)
Cinco modelos independentes, um por faixa de CBR (D1–D5). A versão `_full` também testa combinações de features. Cada modelo grava `metadata.json` com o contrato que o previsor precisa.

### Rede Neural (`mlp.py`)
MLP 128 → 64 → 32 → 16 → 1, com BatchNorm, LeakyReLU e dropout decrescente em direção à saída. Early stopping com `patience=25` e redução de learning rate em platô.

### Comparação C1–C5 (`subsets_rf.py`, `subsets_mlp.py`, `subsets_comparison.py`)
Os dois algoritmos acima, treinados uma vez por conjunto de variáveis, com a mesma semente de divisão. Resultados salvos em `models/subsets/<modelo>/results.json` e `predictions.npz`, para que a comparação possa ser refeita sem retreinar.
=======
### Random Forest (RANDOM_FOREST_EN.py)
- Busca aleatória: 150 iterações, 10-fold CV
- Log-transform no alvo (`log1p` / `expm1`)
- Sample weights para amostras raras (CBR > 25%)
- Retreino em treino + validação após tuning
- Gráficos: curva de aprendizado, convergência OOB, importância de features

### Random Forest por Quintis (RANDOM_FLOREST_Dn.py / Separado)
- 5 modelos independentes por faixas de CBR (D1–D5)
- Versão completa inclui combinações entre quintis

### MLP — Rede Neural (MLL.py)
- Implementado com TensorFlow/Keras
- Entrada: 10 features normalizadas

---

## Dependências Principais

| Biblioteca | Versão |
|---|---|
| scikit-learn | 1.7.2 |
| TensorFlow / Keras | 2.20.0 / 3.12.0 |
| numpy | 2.2.6 |
| pandas | 2.3.3 |
| matplotlib | 3.10.7 |
| seaborn | 0.13.2 |
| streamlit | — |
| joblib | 1.5.2 |
>>>>>>> 5be1326de65eba6a8127ed3262a0b20ebf5729b0

---

## Referência

<<<<<<< HEAD
> Dissertação de Mestrado — Previsão de CBR por Machine Learning
> PUC Goiás
=======
> Dissertação de Mestrado — Previsão de CBR por Machine Learning  
> PUC Goiás, 2025
>>>>>>> 5be1326de65eba6a8127ed3262a0b20ebf5729b0
