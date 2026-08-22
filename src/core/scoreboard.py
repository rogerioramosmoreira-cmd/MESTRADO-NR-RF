"""
Placar de precisão dos modelos.

Um treino imprime MSE, RMSE, MAE, MAPE e R² no terminal e o número some junto
com a janela. Aqui cada modelo grava o resultado da sua última execução em
`reports/metrics/<modelo>.json`, e o menu lê todos de uma vez — para que a
pergunta "qual modelo está melhor hoje?" seja respondida sem treinar nada de
novo.

Um arquivo por modelo, em vez de um placar único: dois treinos rodando ao mesmo
tempo (o que o item "executar um grupo" faz em sequência, mas nada impede em
duas janelas) escreveriam no mesmo arquivo e um perderia o resultado do outro.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from core import paths

# Chaves de métrica reconhecidas, na ordem em que aparecem na tela.
METRIC_FIELDS = ("r2", "mae", "rmse", "mse", "mape")


@dataclass(frozen=True)
class Record:
    """A precisão da última execução de um modelo."""

    key: str
    label: str
    split: str
    recorded_at: datetime | None
    values: dict[str, float]
    extra: dict[str, Any]

    def value(self, field: str) -> float | None:
        raw = self.values.get(field)
        return float(raw) if isinstance(raw, (int, float)) else None

    @property
    def age_text(self) -> str:
        """Quando o treino rodou, em formato curto para a tabela do menu."""
        if self.recorded_at is None:
            return "—"
        return self.recorded_at.strftime("%d/%m %H:%M")


def _path(key: str) -> Path:
    return paths.METRICS_DIR / f"{key}.json"


def _as_values(source: Any) -> dict[str, float]:
    """
    Extrai as métricas de um `metrics.Scores` ou de um dicionário solto.

    Os scripts mais antigos devolvem `dict(nome=..., mse=..., r2=...)` das suas
    próprias funções `metricas()`, e os mais novos devolvem `Scores`. Aceitar os
    dois evita reescrever oito scripts de treino só para gravar um número.
    """
    if hasattr(source, "as_dict"):
        source = source.as_dict()
    if not isinstance(source, dict):
        raise TypeError(f"Métricas em formato não suportado: {type(source)!r}")

    values: dict[str, float] = {}
    for field in METRIC_FIELDS:
        raw = source.get(field)
        # NaN e infinito entram quando a faixa não tinha amostras suficientes.
        # Ficam de fora: `json.dumps` os escreve como `NaN`, que não é JSON
        # válido, e na tela um "nan" diz menos que a ausência do número.
        if isinstance(raw, (int, float)) and math.isfinite(raw):
            values[field] = float(raw)
    return values


def record(key: str, label: str, source: Any, *, split: str = "teste",
           **extra: Any) -> Path | None:
    """
    Grava a precisão de um modelo, substituindo o resultado anterior dele.

    Nunca levanta exceção: um treino que levou minutos não pode ser perdido
    porque a gravação de um relatório acessório falhou. Devolve o caminho
    escrito, ou None quando não deu.
    """
    try:
        values = _as_values(source)
        if not values:
            return None

        paths.METRICS_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "key": key,
            "label": label,
            "split": split,
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
            **values,
            **{name: _jsonable(value) for name, value in extra.items()},
        }
        destination = _path(key)
        destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        return destination
    except Exception:  # noqa: BLE001 - relatório acessório, nunca fatal
        return None


def _jsonable(value: Any) -> Any:
    """Converte escalares do numpy, que o `json.dumps` não serializa."""
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            return str(value)
    return value


def load(key: str) -> Record | None:
    """Lê o placar de um modelo; devolve None se ele nunca foi treinado."""
    destination = _path(key)
    if not destination.exists():
        return None

    try:
        payload = json.loads(destination.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # Arquivo truncado por um treino interrompido no meio da gravação. Vale
        # como "sem resultado", e não como motivo para o menu não abrir.
        return None

    stamp = payload.get("recorded_at")
    try:
        recorded_at = datetime.fromisoformat(stamp) if stamp else None
    except ValueError:
        recorded_at = None

    known = {"key", "label", "split", "recorded_at", *METRIC_FIELDS}
    return Record(
        key=payload.get("key", key),
        label=payload.get("label", key),
        split=payload.get("split", "teste"),
        recorded_at=recorded_at,
        values={field: payload[field] for field in METRIC_FIELDS if field in payload},
        extra={name: value for name, value in payload.items() if name not in known},
    )


def load_all(keys: list[str] | None = None) -> dict[str, Record]:
    """
    Lê o placar de vários modelos de uma vez.

    Sem `keys`, varre a pasta inteira — assim um modelo novo aparece no menu sem
    ninguém precisar registrá-lo em uma lista.
    """
    if keys is None:
        if not paths.METRICS_DIR.exists():
            return {}
        keys = sorted(path.stem for path in paths.METRICS_DIR.glob("*.json"))

    found = {}
    for key in keys:
        entry = load(key)
        if entry is not None:
            found[key] = entry
    return found
