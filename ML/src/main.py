"""
CBR Soil Prediction — Machine Learning Pipeline
Dissertação de Mestrado | PUC Goiás 2025
"""

import os
import sys
import subprocess

CODE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELOS = {
    "1": ("RF Padrão — Modelo Único (Português)",   "RANDOM_FOREST.py"),
    "2": ("RF Standard — Single Model (English)",    "RANDOM_FOREST_EN.py"),
    "3": ("RF Quintis D1–D5 (Simples)",              "RANDOM_FLOREST_Dn.py"),
    "4": ("RF Quintis D1–D5 (Completo + Combos)",    "RANDOM_FLOREST_Separado.py"),
    "5": ("MLP — Rede Neural",                       "MLL.py"),
}

GRUPOS = {
    "1": list(MODELOS.keys()),
    "2": ["1", "2"],
    "3": ["3", "4"],
    "4": ["5"],
}


def executar(chave: str) -> bool:
    desc, filename = MODELOS[chave]
    caminho = os.path.join(CODE_DIR, filename)
    if not os.path.exists(caminho):
        print(f"  ERRO: arquivo não encontrado — {caminho}")
        return False
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    resultado = subprocess.run([sys.executable, caminho], cwd=CODE_DIR)
    ok = resultado.returncode == 0
    if not ok:
        print(f"  AVISO: {filename} encerrou com código {resultado.returncode}")
    return ok


def modo_separado():
    print("\n  Escolha o modelo:\n")
    for k, (desc, _) in MODELOS.items():
        print(f"  {k}. {desc}")
    print("  0. Voltar")

    op = input("\n  Opção: ").strip()
    if op in MODELOS:
        executar(op)
    elif op != "0":
        print("  Opção inválida.")


def modo_juntos():
    print("\n  Executar grupo:\n")
    print("  1. Todos os modelos")
    print("  2. Apenas Random Forest (PT + EN)")
    print("  3. Apenas Quintis D1–D5")
    print("  4. Apenas MLP — Rede Neural")
    print("  0. Voltar")

    op = input("\n  Grupo: ").strip()
    if op == "0":
        return
    if op not in GRUPOS:
        print("  Opção inválida.")
        return

    for k in GRUPOS[op]:
        ok = executar(k)
        if not ok:
            continuar = input("  Continuar mesmo assim? [s/N]: ").strip().lower()
            if continuar != "s":
                print("  Execução interrompida.")
                break


def main():
    while True:
        print("\n" + "=" * 60)
        print("  PREVISÃO DE CBR — Machine Learning")
        print("  Dissertação de Mestrado | PUC Goiás 2025")
        print("=" * 60)
        print("\n  1. Modo Separado  — executar um modelo por vez")
        print("  2. Modo Juntos    — executar múltiplos modelos")
        print("  0. Sair")

        op = input("\n  Opção: ").strip()
        if op == "0":
            print("\n  Encerrando.\n")
            break
        elif op == "1":
            modo_separado()
        elif op == "2":
            modo_juntos()
        else:
            print("  Opção inválida.")


if __name__ == "__main__":
    main()
