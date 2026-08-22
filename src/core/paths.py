"""
Resolução central de caminhos.

Todo script resolve seus caminhos a partir daqui, em vez de montar cadeias de
`os.path.join(__file__, "..", "..")` na mão. Assim, mover um script de pasta
não quebra mais os caminhos que ele usa.
"""

from pathlib import Path

from core import runtime

# src/core/paths.py -> core -> src -> raiz do repositório
ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = ROOT / "src"

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DATASET = PROCESSED_DIR / "cbr_dataset.csv"

# Em modo rápido (MLL_FAST=1) tudo o que é gerado ganha um sufixo, para que uma
# execução de teste nunca sobrescreva o resultado de uma execução completa.
_SUFFIX = runtime.SUFFIX

MODELS_DIR = ROOT / f"models{_SUFFIX}"
RF_DIR = MODELS_DIR / "random_forest"
RF_EN_DIR = MODELS_DIR / "random_forest_en"
RF_ENSEMBLE_DIR = MODELS_DIR / "random_forest_ensemble"
RF_QUINTIS_DIR = MODELS_DIR / "random_forest_quintis"
MLP_DIR = MODELS_DIR / "mlp"
SUBSETS_DIR = MODELS_DIR / "subsets"

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / f"figures{_SUFFIX}"

# Métricas da última execução de cada modelo, um JSON por modelo. É o que
# permite ao menu mostrar a precisão sem ter de treinar nada de novo.
METRICS_DIR = REPORTS_DIR / f"metrics{_SUFFIX}"

REQUIREMENTS = ROOT / "requirements.txt"


def ensure(*directories: Path) -> None:
    """Cria cada pasta (e as pastas-pai) caso ainda não exista."""
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
