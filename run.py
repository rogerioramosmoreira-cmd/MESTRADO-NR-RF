"""
Ponto de entrada do projeto — abre o pipeline em uma janela de terminal.

Dar duplo clique em um `.py` no Windows roda o script em um console que fecha no
instante em que ele termina, então qualquer erro fica ilegível. Este script abre
um console que permanece aberto, roda o menu dentro dele e informa qual
interpretador foi usado.

Execução:  python run.py
           iniciar.bat        (Windows — atalho para o mesmo)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from core import console  # noqa: E402
import main as launcher  # noqa: E402


@console.guard("Falha ao abrir a janela do terminal.")
def start() -> None:
    console.banner("PREVISÃO DE CBR", f"Interpretador: {sys.executable}")

    if not console.RICH_AVAILABLE:
        console.warn("A biblioteca 'rich' não está instalada — a interface será "
                     "exibida em modo texto simples.")
        console.detail(f"{sys.executable} -m pip install rich")

    if launcher.relaunch_in_new_window():
        console.ok("Janela de terminal aberta. Esta pode ser fechada.")
        return

    # Nenhuma janela separada foi aberta, então roda o menu aqui mesmo em vez de
    # deixar o usuário sem nada.
    console.info("Executando nesta janela.")
    launcher.main()


if __name__ == "__main__":
    start()
