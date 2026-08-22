"""
Entrada de dados: planilha em, dataset limpo fora.

Os dados chegam como planilha do Excel ou como CSV exportado dela, e nenhum dos
dois está pronto para uso: os números vêm com vírgula decimal e entre aspas
(`"100,00"`), os cabeçalhos vêm com espaço sobrando (`"CBR "`) e as linhas de
total ou de anotação no fim da planilha entram como linhas vazias.

Este módulo faz esse caminho inteiro — converte planilha para CSV, limpa e
grava `data/processed/cbr_dataset.csv` — para que ninguém precise abrir o Excel
e "Salvar como CSV" na mão antes de treinar. A conversão é automática: basta
largar o `.xlsx` em `data/raw`.

O que a limpeza faz, em ordem:

1. `,` decimal vira `.` — `"100,00"` vira `100.0`
2. cabeçalhos perdem espaços sobrando e ganham o nome canônico de `core/data.py`
3. linhas e colunas inteiramente vazias somem
4. linhas sem alguma variável ou sem o alvo somem, porque não treinam nada
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from core import console, data, paths

# Formatos de planilha que o pandas lê com o openpyxl. O `.xls` antigo exige o
# `xlrd`, que é outra dependência e outro formato de arquivo — fica de fora até
# alguém precisar de verdade.
SPREADSHEET_SUFFIXES = (".xlsx", ".xlsm")

# Sufixo do CSV gerado a partir de uma aba, quando a planilha tem mais de uma.
SHEET_SEPARATOR = "__"

# Unidade no fim do nome da coluna: "LL (%)", "Densidade máxima (Kg/m³)".
UNIT_SUFFIX = re.compile(r"\s*\([^)]*\)\s*$")

# Designação das peneiras, que na planilha original aparece em uma linha
# separada logo abaixo do cabeçalho — e é o único lugar onde a primeira peneira
# (25.4mm) está nomeada.
#
# Fica aqui, e não em `data.COLUMN_ALIASES`, porque `"4"` e `"10"` são nomes
# perigosos demais para valer em qualquer arquivo: só são consultados ao
# preencher um nome de coluna que ficou vazio.
SIEVE_ALIASES = {
    '#1"': "25.4mm", "#1": "25.4mm", '1"': "25.4mm",
    '#3/8"': "9.5mm", '3/8"': "9.5mm",
    "#4": "4.8mm", "4": "4.8mm",
    "#10": "2.0mm", "10": "2.0mm",
    "#40": "0.42mm", "40": "0.42mm",
    "#200": "0.076mm", "200": "0.076mm",
}

# Até onde procurar a linha de cabeçalho, e quantos nomes conhecidos ela precisa
# ter para ser aceita como tal. Três evita que uma linha de dados com valores
# coincidentes seja promovida a cabeçalho por acidente.
HEADER_SEARCH_ROWS = 15
MIN_HEADER_MATCHES = 3

CANONICAL_NAMES = frozenset(data.FEATURES + [data.TARGET])

# Faixa de plausibilidade das medidas do ensaio, herdada da limpeza manual que
# existia em `explore/inspect_data.py`. Um valor acima de 3000 ou abaixo de 1
# quase sempre é erro de digitação — quase, e é por isso que o filtro é opcional
# (`--limites`): o dataset atual tem um IP de 0,57, que é baixo mas possível em
# solo praticamente não plástico. Apagar essa linha sozinho seria decidir pelo
# usuário.
PLAUSIBLE_MIN = 1.0
PLAUSIBLE_MAX = 3000.0


class IngestError(RuntimeError):
    """Levantado quando uma planilha não pode ser lida ou convertida."""


# ── Leitura ──────────────────────────────────────────────────────────────────

def read_spreadsheet(source: Path) -> dict[str, pd.DataFrame]:
    """
    Lê todas as abas de uma planilha, no nome da aba -> DataFrame.

    O `openpyxl` é importado aqui dentro, e não no topo do módulo, para que o
    projeto continue abrindo sem ele: quem só treina com o CSV já pronto não
    precisa da biblioteca, e uma importação no topo faria o menu inteiro quebrar
    por causa de um caminho de código que ninguém usou.
    """
    try:
        import openpyxl  # noqa: F401 - só para dar erro nomeado antes do pandas
    except ImportError as exc:
        raise IngestError(
            "Ler planilha .xlsx exige a biblioteca 'openpyxl'.\n"
            "Instale com:  pip install openpyxl"
        ) from exc

    try:
        # `header=None`: quem decide onde está o cabeçalho é `promote_header`,
        # porque a primeira linha da planilha costuma ser em branco ou um título
        # mesclado — e o pandas a aceitaria como nome das colunas.
        sheets = pd.read_excel(source, sheet_name=None, dtype=str, header=None)
    except Exception as exc:  # noqa: BLE001 - pandas levanta muitos tipos aqui
        raise IngestError(f"Falha ao ler a planilha '{source.name}': {exc}") from exc

    if not sheets:
        raise IngestError(f"A planilha '{source.name}' não tem nenhuma aba.")
    return sheets


def read_table(source: Path) -> pd.DataFrame:
    """Lê um CSV ou a primeira aba de uma planilha, como texto puro."""
    if source.suffix.lower() in SPREADSHEET_SUFFIXES:
        return next(iter(read_spreadsheet(source).values()))

    try:
        frame = pd.read_csv(source, dtype=str)
        # CSV exportado de uma planilha bagunçada carrega a bagunça junto. Se a
        # primeira linha não parece cabeçalho, relê sem cabeçalho nenhum e
        # deixa `promote_header` procurar o certo.
        if known_names(frame.columns) < MIN_HEADER_MATCHES:
            frame = pd.read_csv(source, dtype=str, header=None)
        return frame
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise IngestError(f"Falha ao ler '{source.name}': {exc}") from exc


# ── Cabeçalho ────────────────────────────────────────────────────────────────

def canonical(value) -> str:
    """
    Nome canônico de uma célula de cabeçalho, ou string vazia se não houver.

    Tira espaços sobrando, remove a unidade entre parênteses (`"LL (%)"` vira
    `"LL"`) e aplica o mapa de variações de `core/data.py`.
    """
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in ("nan", "none", ""):
        return ""
    text = UNIT_SUFFIX.sub("", text).strip()
    return data.COLUMN_ALIASES.get(text, text)


def canonical_label(value) -> str:
    """Como `canonical`, mais as designações de peneira (`#1"` vira 25.4mm)."""
    name = canonical(value)
    return SIEVE_ALIASES.get(name, name)


def known_names(values) -> int:
    """Quantos nomes de uma sequência são colunas conhecidas do dataset."""
    return sum(1 for value in values if canonical(value) in CANONICAL_NAMES)


def promote_header(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Encontra a linha de cabeçalho de verdade e descarta o que vier antes dela.

    Planilha de laboratório quase nunca começa pelo cabeçalho: vêm primeiro uma
    linha em branco e uma faixa de títulos mesclados ("Granulometria % Passante
    nas Peneiras"), e só então os nomes das colunas. Lida de cima para baixo, a
    planilha inteira vira `Unnamed: 0`, `Unnamed: 1`...

    Também completa os nomes que ficaram em branco, olhando a linha de baixo — é
    lá que mora a designação da primeira peneira (`#1"`) — e, se ainda faltar, a
    linha de cima, onde estão os títulos mesclados. A linha de designações é
    descartada junto: ela é rótulo, não amostra.
    """
    if known_names(frame.columns) >= MIN_HEADER_MATCHES:
        return frame

    limit = min(len(frame), HEADER_SEARCH_ROWS)
    header_index, best = None, 0
    for index in range(limit):
        score = known_names(frame.iloc[index])
        if score > best:
            header_index, best = index, score

    if header_index is None or best < MIN_HEADER_MATCHES:
        # Sem cabeçalho reconhecível não há o que promover; quem chamou reporta
        # as colunas ausentes, que é uma mensagem mais útil do que um palpite.
        return frame

    names = [canonical(value) for value in frame.iloc[header_index]]

    used_next = False
    if header_index + 1 < len(frame):
        below = [canonical_label(value) for value in frame.iloc[header_index + 1]]
        for position, name in enumerate(names):
            if not name and below[position]:
                names[position] = below[position]
                used_next = True

    if header_index > 0:
        above = [canonical(value) for value in frame.iloc[header_index - 1]]
        for position, name in enumerate(names):
            if not name and above[position]:
                names[position] = above[position]

    names = [name or f"coluna_{position}" for position, name in enumerate(names)]

    first_row = header_index + (2 if used_next else 1)
    promoted = frame.iloc[first_row:].copy()
    promoted.columns = names
    return promoted.reset_index(drop=True)


# ── Limpeza ──────────────────────────────────────────────────────────────────

def to_number(column: pd.Series) -> pd.Series:
    """
    Converte uma coluna de texto para número, tratando a vírgula decimal.

    Devolve a coluna original quando a conversão falha em tudo: uma coluna de
    texto legítima (um código de amostra, por exemplo) não deve virar uma coluna
    de `NaN` só porque passou por aqui.
    """
    if not (column.dtype == object or isinstance(column.dtype, pd.StringDtype)):
        return column

    cleaned = (column.astype(str)
               .str.strip()
               .str.replace(" ", "", regex=False)
               .str.replace(",", ".", regex=False))
    converted = pd.to_numeric(cleaned, errors="coerce")

    # Tudo virou NaN onde havia texto: a coluna não era numérica.
    if converted.notna().sum() == 0:
        return column
    return converted


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Aplica a limpeza inteira a um DataFrame recém-lido."""
    frame = promote_header(frame)
    frame = frame.dropna(axis=1, how="all").dropna(axis=0, how="all")
    frame = data.normalise_columns(frame)

    # Colunas repetidas aparecem quando a planilha tem duas com o mesmo nome
    # depois de normalizado (`"CBR "` e `"CBR"`). Fica a primeira: a segunda
    # costuma ser uma coluna de conferência, vazia.
    frame = frame.loc[:, ~frame.columns.duplicated()]

    for name in frame.columns:
        frame[name] = to_number(frame[name])

    return frame.dropna(axis=0, how="all")


# ── Conversão de planilha para CSV ───────────────────────────────────────────

def csv_destination(source: Path, sheet: str | None = None) -> Path:
    """Onde o CSV de uma planilha (ou de uma aba dela) é gravado."""
    stem = source.stem if sheet is None else f"{source.stem}{SHEET_SEPARATOR}{sheet}"
    # Nome de aba vira nome de arquivo: o Excel aceita caracteres que o sistema
    # de arquivos recusa.
    stem = "".join("_" if character in '\\/:*?"<>|' else character
                   for character in stem).strip()
    return source.with_name(f"{stem}.csv")


def is_current(source: Path, destination: Path) -> bool:
    """True quando o CSV já existe e é mais novo que a planilha."""
    return (destination.exists()
            and destination.stat().st_mtime >= source.stat().st_mtime)


def convert_spreadsheet(source: Path, *, force: bool = False) -> list[Path]:
    """
    Converte uma planilha em CSV — um por aba, quando há mais de uma.

    Planilha de uma aba só vira `<nome>.csv`; com várias abas, cada uma vira
    `<nome>__<aba>.csv`. Sem essa distinção, converter uma planilha de cinco
    abas gravaria cinco vezes no mesmo arquivo e sobraria só a última.
    """
    sheets = read_spreadsheet(source)
    single = len(sheets) == 1

    written: list[Path] = []
    for sheet, frame in sheets.items():
        destination = csv_destination(source, None if single else sheet)

        if not force and is_current(source, destination):
            console.detail(f"{destination.name} já está atualizado")
            continue

        cleaned = clean(frame)
        if cleaned.empty:
            console.warn(f"Aba '{sheet}' de {source.name} está vazia — ignorada.")
            continue

        cleaned.to_csv(destination, index=False, encoding="utf-8")
        console.ok(f"{source.name}"
                   f"{'' if single else f' [{sheet}]'} → {destination.name} "
                   f"({len(cleaned)} linha(s), {len(cleaned.columns)} coluna(s))")

        # Coluna sem nome costuma ser um valor solto digitado fora da tabela.
        # Fica no CSV — apagar dado de planilha sem avisar é pior do que manter
        # uma coluna estranha —, mas é anunciada para ninguém descobrir depois.
        unnamed = [name for name in cleaned.columns if str(name).startswith("coluna_")]
        for name in unnamed:
            console.detail(f"coluna sem nome mantida: {name} "
                           f"({cleaned[name].notna().sum()} valor(es))")

        written.append(destination)

    return written


def convert_folder(folder: Path | None = None, *, force: bool = False) -> list[Path]:
    """
    Converte todas as planilhas de uma pasta (por padrão, `data/raw`).

    Planilha já convertida e sem alteração desde então é pulada, para que rodar
    isto no início de todo treino não custe nada.
    """
    folder = Path(folder) if folder is not None else paths.RAW_DIR
    if not folder.exists():
        return []

    sources = sorted(path for path in folder.iterdir()
                     if path.suffix.lower() in SPREADSHEET_SUFFIXES
                     and not path.name.startswith("~$"))  # temporário do Excel

    written: list[Path] = []
    for source in sources:
        try:
            written.extend(convert_spreadsheet(source, force=force))
        except IngestError as exc:
            # Uma planilha ilegível não pode impedir as outras de serem
            # convertidas, nem derrubar o treino que chamou isto de passagem.
            console.warn(str(exc))

    return written


# ── Montagem do dataset ──────────────────────────────────────────────────────

def list_sources(folder: Path | None = None) -> list[Path]:
    """CSV candidatos a virar o dataset, do modificado mais recentemente ao mais antigo."""
    folder = Path(folder) if folder is not None else paths.RAW_DIR
    if not folder.exists():
        return []
    return sorted(folder.glob("*.csv"),
                  key=lambda path: path.stat().st_mtime, reverse=True)


def pick_source(folder: Path | None = None) -> Path:
    """
    Escolhe qual CSV de `data/raw` vira o dataset.

    Com mais de um candidato, vence o modificado por último — é o que acabou de
    ser exportado ou convertido. A escolha é anunciada por quem chama, para que
    ninguém treine em cima do arquivo errado sem perceber.
    """
    candidates = list_sources(folder)
    if not candidates:
        raise IngestError(
            f"Nenhum CSV ou planilha utilizável em "
            f"'{folder if folder is not None else paths.RAW_DIR}'.\n"
            f"Coloque a planilha (.xlsx) ou o CSV exportado nessa pasta."
        )
    return candidates[0]


def out_of_range(frame: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    """Quantos valores de cada coluna caem fora da faixa de plausibilidade."""
    counts = {}
    for name in columns:
        if name not in frame.columns:
            continue
        column = pd.to_numeric(frame[name], errors="coerce")
        outside = int(((column < PLAUSIBLE_MIN) | (column > PLAUSIBLE_MAX)).sum())
        if outside:
            counts[name] = outside
    return counts


def build_dataset(source: Path | None = None, *, force: bool = False,
                  limits: bool = False) -> Path:
    """
    Converte, limpa e grava `data/processed/cbr_dataset.csv`.

    Sem `force`, um dataset já existente e mais novo que a origem é mantido: o
    arquivo processado costuma ser o que foi conferido à mão, e regravá-lo por
    conta própria a cada treino desfaria essa conferência em silêncio.
    """
    convert_folder(force=force)

    origin = Path(source) if source is not None else pick_source()
    destination = paths.DATASET

    if not force and is_current(origin, destination):
        console.detail(f"{destination.name} já está atualizado "
                       f"(origem: {origin.name})")
        return destination

    frame = clean(read_table(origin))

    expected = data.FEATURES + [data.TARGET]
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise IngestError(
            f"Faltam colunas em '{origin.name}': {missing}\n"
            f"Colunas encontradas: {list(frame.columns)}\n"
            f"Se o cabeçalho mudou, acrescente a variação em COLUMN_ALIASES "
            f"(core/data.py)."
        )

    frame = frame[expected]

    if limits:
        # Fora da faixa vira ausente, e a linha cai no `dropna` logo abaixo — é
        # o mesmo efeito da limpeza manual antiga, agora contado item a item em
        # vez de silencioso.
        outside = out_of_range(frame, expected)
        if outside:
            for name, count in outside.items():
                console.detail(f"{name}: {count} valor(es) fora de "
                               f"[{PLAUSIBLE_MIN:g}, {PLAUSIBLE_MAX:g}]")
            numeric = frame.apply(pd.to_numeric, errors="coerce")
            frame = frame.mask((numeric < PLAUSIBLE_MIN) | (numeric > PLAUSIBLE_MAX))
        else:
            console.detail("Nenhum valor fora da faixa de plausibilidade.")

    before = len(frame)
    frame = frame.dropna(subset=expected)
    if frame.empty:
        raise IngestError(f"'{origin.name}' não tem nenhuma linha completa.")

    previous = None
    if destination.exists():
        try:
            previous = len(pd.read_csv(destination))
        except Exception:  # noqa: BLE001 - contagem é só para o aviso
            previous = None

    paths.ensure(destination.parent)
    frame.to_csv(destination, index=False, encoding="utf-8")

    console.ok(f"{origin.name} → {destination.name} "
               f"({len(frame)} linha(s) completas de {before})")

    # Dataset com outro tamanho é outro dataset: a precisão registrada de cada
    # modelo foi medida no anterior e deixa de valer até o próximo treino.
    if previous is not None and previous != len(frame):
        console.warn(f"O dataset mudou de {previous} para {len(frame)} amostras — "
                     f"a precisão registrada dos modelos é do dataset anterior. "
                     f"Treine de novo para atualizá-la.")
    if before > len(frame):
        console.detail(f"{before - len(frame)} linha(s) descartada(s) por terem "
                       f"variável ou CBR em branco.")

    return destination
