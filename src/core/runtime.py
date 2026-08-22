r"""
Modo rápido, para desenvolvimento.

Os orçamentos de busca do projeto são grandes de propósito — o ensemble faz
`3 x 150 x 10 = 4500` ajustes e leva perto de uma hora. Isso é adequado para o
resultado final, e péssimo para quem só quer conferir se uma alteração no código
não quebrou nada.

Com `MLL_FAST=1`, cada script troca seu orçamento pelo equivalente reduzido e
termina em poucos minutos. Os valores padrão não mudam: quem não define a
variável continua obtendo exatamente os mesmos números de antes.

    MLL_FAST=1 python src/models/subsets_rf.py       (Linux/macOS)
    $env:MLL_FAST=1; python src\models\subsets_rf.py  (PowerShell)

Os modelos gerados nesse modo NÃO servem para a dissertação — a busca cobre uma
fração do espaço de hiperparâmetros. Por isso cada execução avisa em destaque
quando o modo está ativo.
"""

from __future__ import annotations

import os

FAST = os.environ.get("MLL_FAST", "0").strip().lower() in {"1", "true", "sim"}

# Sufixo aplicado às pastas de saída em modo rápido. Sem ele, um teste de
# trinta segundos gravaria por cima dos modelos e das métricas de uma execução
# completa de uma hora — e a perda passaria despercebida, porque os arquivos
# continuariam existindo, só que com números piores.
SUFFIX = "_rapido" if FAST else ""


def budget(full: int, fast: int) -> int:
    """Devolve o orçamento reduzido em modo rápido, e o cheio caso contrário."""
    return fast if FAST else full


def announce() -> None:
    """Avisa, no início do script, que os números não são definitivos."""
    if not FAST:
        return

    # Importado aqui, e não no topo, para que `core.paths` possa usar este
    # módulo sem arrastar o console junto.
    from core import console

    console.warn("MODO RÁPIDO ativo (MLL_FAST=1) — busca reduzida.")
    console.detail("Os modelos gerados servem para testar o código, não para "
                   "citar resultados. Remova MLL_FAST para a execução final.")
    console.detail(f"Saída isolada nas pastas com sufixo '{SUFFIX}' — os "
                   f"resultados da execução completa não são tocados.")
