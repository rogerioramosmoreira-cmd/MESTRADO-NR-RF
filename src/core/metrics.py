"""
Métricas de regressão.

Uma implementação só, compartilhada por todos os modelos, para que a métrica
reportada da Árvore Aleatória seja calculada exatamente como a da Rede Neural e
as duas sejam de fato comparáveis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from core import console

# Protege o denominador do MAPE. O CBR é uma grandeza positiva, mas um valor
# exatamente zero nos dados produziria uma porcentagem infinita.
_EPSILON = 1e-6


@dataclass(frozen=True)
class Scores:
    """Métricas de um modelo sobre uma divisão do dataset."""

    name: str
    mse: float
    rmse: float
    mae: float
    mape: float
    r2: float
    extra: dict = field(default_factory=dict)

    def as_row(self) -> list[str]:
        return [
            self.name,
            f"{self.mse:.4f}",
            f"{self.rmse:.4f}",
            f"{self.mae:.4f}",
            f"{self.mape:.2f}%",
            f"{self.r2:.4f}",
        ]

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "mse": self.mse,
            "rmse": self.rmse,
            "mae": self.mae,
            "mape": self.mape,
            "r2": self.r2,
            **self.extra,
        }


HEADERS = ["Conjunto", "MSE", "RMSE", "MAE", "MAPE", "R²"]


def evaluate(y_true, y_pred, name: str) -> Scores:
    """Calcula MSE, RMSE, MAE, MAPE e R² para um conjunto de previsões."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Formatos incompatíveis em '{name}': "
            f"y_true={y_true.shape} vs y_pred={y_pred.shape}"
        )
    if y_true.size == 0:
        raise ValueError(f"Conjunto '{name}' está vazio — nada a avaliar.")

    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + _EPSILON))) * 100)

    return Scores(
        name=name,
        mse=mse,
        rmse=float(np.sqrt(mse)),
        mae=mae,
        mape=mape,
        r2=float(r2_score(y_true, y_pred)),
    )


def show(scores: Sequence[Scores], headers: Sequence[str] = HEADERS) -> None:
    """Renderiza um ou mais conjuntos de métricas como tabela no terminal."""
    console.metrics_table([score.as_row() for score in scores], headers=headers)


# ── Veredito ─────────────────────────────────────────────────────────────────
# Classificação do modelo pelo R² do conjunto de teste — a fração da variação
# do CBR que o modelo explica em dados que ele nunca viu.
#
# Os cortes ficam todos aqui, em um lugar só, porque são uma escolha do projeto
# e não uma verdade estatística: um R² de 0,75 que seria fraco em laboratório é
# bom para CBR de campo, onde o próprio ensaio tem dispersão alta. Mexer na
# régua é mexer nesta tupla, e mais nada.
GRADE_THRESHOLDS = (
    (0.90, "EXCELENTE", "ok"),
    (0.80, "ADEQUADO", "ok"),
    (0.70, "BOA", "warn"),
    (0.50, "RUIM", "warn"),
)
WORST_GRADE = ("INUTILIZÁVEL", "error")

GRADE_MEANINGS = {
    "EXCELENTE": "previsão confiável; serve para dimensionamento",
    "ADEQUADO": "previsão utilizável; confirmar os casos críticos em ensaio",
    "BOA": "serve para estimativa preliminar, não para decisão final",
    "RUIM": "erra demais para substituir o ensaio; use só como indicativo",
    "INUTILIZÁVEL": "não explica o CBR; não use para previsão",
}


@dataclass(frozen=True)
class Grade:
    """A classificação de um modelo e o que ela significa."""

    label: str
    state: str
    r2: float

    @property
    def meaning(self) -> str:
        return GRADE_MEANINGS.get(self.label, "")


def grade(r2: float) -> Grade:
    """Classifica um modelo pelo R² de teste."""
    value = float(r2)
    for threshold, label, state in GRADE_THRESHOLDS:
        if value >= threshold:
            return Grade(label, state, value)
    label, state = WORST_GRADE
    return Grade(label, state, value)


def _values(source) -> dict:
    """Aceita tanto um `Scores` quanto o `dict` das funções `metricas()`."""
    return source.as_dict() if hasattr(source, "as_dict") else dict(source)


def report(source, *, title: str = "AVALIAÇÃO DO MODELO",
           notes: Sequence[str] = ()) -> Grade:
    """
    Fecha um treino com o veredito: a classificação do modelo e o porquê dela.

    O bloco de métricas que cada script já imprime responde *quanto* o modelo
    errou; esta função responde a pergunta que vem logo depois — *isso é bom o
    bastante?* — para que a resposta não dependa de o leitor saber de cor que
    R² 0,74 é razoável para CBR.

    `notes` recebe as ressalvas que só o script chamador conhece, como um R² de
    treino incoerente com o de teste.
    """
    values = _values(source)
    r2 = float(values.get("r2", float("nan")))
    verdict = grade(r2) if r2 == r2 else Grade(*WORST_GRADE, r2=float("nan"))

    lines = [f"R² {r2:.4f} — o modelo explica {max(r2, 0.0) * 100:.0f}% "
             "da variação do CBR em dados novos"]

    mae = values.get("mae")
    if isinstance(mae, (int, float)):
        lines.append(f"Erro médio (MAE): ±{mae:.2f} no valor de CBR")

    numbers = [f"{name.upper()} {values[name]:.4f}"
               for name in ("mse", "rmse") if isinstance(values.get(name), (int, float))]
    mape = values.get("mape")
    if isinstance(mape, (int, float)):
        numbers.append(f"MAPE {mape:.2f}%")
    if numbers:
        lines.append("  |  ".join(numbers))

    automatic = []
    if isinstance(mape, (int, float)) and mape > 30:
        automatic.append(f"Erro percentual alto ({mape:.0f}%): o modelo erra "
                         "proporcionalmente mais nos CBR baixos.")

    remarks = [*automatic, *notes]
    if remarks:
        lines.append("")
        lines.extend(remarks)

    console.verdict_panel(title, verdict.label, verdict.meaning, lines,
                          verdict.state)
    return verdict
