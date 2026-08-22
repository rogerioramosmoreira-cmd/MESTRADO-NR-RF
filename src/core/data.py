"""
Carregamento do dataset, normalização de colunas e definição dos conjuntos.

O CSV é escrito à mão e seus cabeçalhos variam entre exportações — espaços no
fim (`"CBR "`), vírgula decimal (`"0,42mm"`), variações de acento
(`"Umidade Otima"`). Antes, cada script carregava seu próprio mapa de renomes;
agora todos compartilham este, então uma nova variação de cabeçalho é corrigida
em um lugar só.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from core import paths

# ── Esquema canônico ─────────────────────────────────────────────────────────

TARGET = "CBR"

GRANULOMETRY = ["25.4mm", "9.5mm", "4.8mm", "2.0mm", "0.42mm", "0.076mm"]
PLASTICITY = ["LL", "IP"]
COMPACTION = ["Umidade Ótima", "Densidade máxima"]

FEATURES = GRANULOMETRY + PLASTICITY + COMPACTION

# Rótulos de exibição para eixos e legendas dos gráficos.
LABELS = {
    "25.4mm": "25.4mm (IG)",
    "9.5mm": "9.5mm (EXP)",
    "4.8mm": "4.8mm (D3)",
    "2.0mm": "2.0mm (D4)",
    "0.42mm": "0.42mm (D5)",
    "0.076mm": "0.076mm (D6)",
    "LL": "LL",
    "IP": "IP",
    "Umidade Ótima": "Umidade Ótima",
    "Densidade máxima": "Densidade Máxima Seca",
}

# Toda grafia de cabeçalho já vista nas exportações, mapeada para o nome
# canônico.
COLUMN_ALIASES = {
    "25,4mm": "25.4mm", "9,5mm": "9.5mm", "4,8mm": "4.8mm",
    "2,0mm": "2.0mm", "0,42mm": "0.42mm", "0,076mm": "0.076mm",
    "P25.4": "25.4mm", "P9.5": "9.5mm", "P4.8": "4.8mm",
    "P2.0": "2.0mm", "P0.42": "0.42mm", "P0.076": "0.076mm",
    "Ll": "LL", "ll": "LL", "L.L": "LL", "L.L.": "LL",
    "Ip": "IP", "ip": "IP", "I.P": "IP", "I.P.": "IP",
    "Wot": "Umidade Ótima", "wot": "Umidade Ótima", "W_ot": "Umidade Ótima",
    "Umidade otima": "Umidade Ótima", "Umidade Otima": "Umidade Ótima",
    "Umidade ótima": "Umidade Ótima",
    "Densidade Maxima": "Densidade máxima", "Densidade Máxima": "Densidade máxima",
    "Densidade maxima": "Densidade máxima",
    "d_max": "Densidade máxima", "Dmax": "Densidade máxima",
    "cbr": "CBR", "Cbr": "CBR",
}


# ── Conjuntos de variáveis (C1–C5) ───────────────────────────────────────────

@dataclass(frozen=True)
class Subset:
    """Um grupo de variáveis comparado contra os demais."""

    key: str
    name: str
    description: str
    features: tuple[str, ...]

    @property
    def labels(self) -> list[str]:
        return [LABELS[feature] for feature in self.features]

    def __len__(self) -> int:
        return len(self.features)


SUBSETS: dict[str, Subset] = {
    "c1": Subset(
        key="c1",
        name="C1 — Granulometria",
        description="Seis peneiras da curva granulométrica",
        features=tuple(GRANULOMETRY),
    ),
    "c2": Subset(
        key="c2",
        name="C2 — Plasticidade",
        description="Limite de Liquidez e Índice de Plasticidade",
        features=tuple(PLASTICITY),
    ),
    "c3": Subset(
        key="c3",
        name="C3 — Compactação",
        description="Umidade ótima e densidade máxima seca (Proctor)",
        features=tuple(COMPACTION),
    ),
    "c4": Subset(
        key="c4",
        name="C4 — Granulometria + Plasticidade",
        description="Curva granulométrica somada a LL e IP",
        features=tuple(GRANULOMETRY + PLASTICITY),
    ),
    "c5": Subset(
        key="c5",
        name="C5 — Todos",
        description="Todas as variáveis de entrada disponíveis",
        features=tuple(FEATURES),
    ),
}

SUBSET_ORDER = ["c1", "c2", "c3", "c4", "c5"]


# ── Carregamento ─────────────────────────────────────────────────────────────

class DatasetError(RuntimeError):
    """Levantado quando o dataset está ausente, ilegível ou fora do esquema."""


def normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove espaços dos cabeçalhos e aplica o mapa de variações."""
    frame = frame.copy()
    frame.columns = frame.columns.str.strip()
    return frame.rename(columns=COLUMN_ALIASES)


def load(path: Path | None = None) -> pd.DataFrame:
    """
    Lê o dataset processado e o devolve com os nomes canônicos de coluna.

    Levanta `DatasetError` com a causa concreta — arquivo ausente, ilegível ou
    colunas faltando — em vez de deixar um erro cru do pandas vazar.

    Com o dataset ausente, tenta montá-lo a partir do que houver em `data/raw`
    (planilha `.xlsx` inclusive) antes de desistir: largar a planilha na pasta e
    mandar treinar é o caminho que o usuário espera que funcione.
    """
    source = Path(path) if path is not None else paths.DATASET

    if not source.exists() and path is None:
        # A importação é local porque `ingest` importa este módulo; no topo, os
        # dois ficariam esperando um pelo outro.
        from core import ingest

        try:
            source = ingest.build_dataset()
        except ingest.IngestError as exc:
            raise DatasetError(
                f"Dataset não encontrado em '{source}', e não foi possível "
                f"montá-lo a partir de {paths.RAW_DIR}:\n{exc}"
            ) from exc

    if not source.exists():
        raise DatasetError(
            f"Dataset não encontrado em '{source}'.\n"
            f"Verifique se o arquivo existe em {paths.PROCESSED_DIR}."
        )

    try:
        frame = pd.read_csv(source)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise DatasetError(f"Falha ao ler '{source}': {exc}") from exc

    frame = normalise_columns(frame)

    expected = FEATURES + [TARGET]
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise DatasetError(
            f"Colunas ausentes no dataset: {missing}\n"
            f"Colunas encontradas: {list(frame.columns)}\n"
            f"Se o cabeçalho mudou, adicione a variação em COLUMN_ALIASES "
            f"({Path(__file__).name})."
        )

    frame = frame[expected]

    # Uma linha sem alguma variável ou sem o alvo não treina nem avalia nada.
    # Descartar em silêncio mudaria a contagem de amostras sem avisar, então
    # quem chamou fica sabendo pelo tamanho do DataFrame devolvido.
    frame = frame.dropna(subset=expected)

    if frame.empty:
        raise DatasetError(f"Dataset '{source}' não tem nenhuma linha completa.")

    return frame


def matrices(frame: pd.DataFrame, subset: Subset | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Separa um DataFrame carregado na matriz de variáveis X e no vetor alvo y.

    Quando `subset` é informado, só as colunas daquele conjunto são devolvidas —
    é isso que faz a comparação C1–C5 ser um caminho de código só, em vez de
    cinco.
    """
    columns = list(subset.features) if subset is not None else FEATURES
    features = frame[columns].to_numpy(dtype=float)
    target = frame[TARGET].to_numpy(dtype=float).ravel()
    return features, target
