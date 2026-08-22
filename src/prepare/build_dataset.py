"""
Preparação dos dados — planilha ou CSV bruto vira o dataset de treino.

Um caminho só para tudo o que acontece antes do treino: converte as planilhas
`.xlsx` de `data/raw` em CSV, limpa o resultado (cabeçalho em linha errada,
vírgula decimal, unidade no nome da coluna, linhas vazias) e regrava
`data/processed/cbr_dataset.csv`.

Antes eram dois itens de menu — um convertia a planilha, outro limpava um CSV
escolhido à mão — e a divisão não correspondia a nada: quem larga uma planilha
na pasta quer o dataset pronto, e quem quer escolher o arquivo quer escolher
antes da mesma limpeza. Aqui a escolha é uma pergunta, não outro programa.

Execução:
    python src/prepare/build_dataset.py                 # converte tudo e regrava
    python src/prepare/build_dataset.py --arquivo X.csv # usa um arquivo específico
    python src/prepare/build_dataset.py --limites       # descarta valores implausíveis
    python src/prepare/build_dataset.py --manter        # não sobrescreve o dataset atual
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import console, dependencies, ingest, paths  # noqa: E402

dependencies.require("excel")


def choose_source(candidates: list[Path]) -> Path:
    """
    Pergunta qual arquivo usar quando há mais de um candidato.

    O padrão é o modificado por último — normalmente o que acabou de ser
    convertido —, então ENTER resolve o caso comum. Sem terminal interativo
    (execução automática, redirecionamento), segue no padrão sem perguntar, em
    vez de travar esperando uma resposta que ninguém vai digitar.
    """
    if len(candidates) == 1:
        return candidates[0]

    console.section("Arquivos disponíveis em data/raw")
    for position, path in enumerate(candidates, start=1):
        stamp = datetime.fromtimestamp(path.stat().st_mtime).strftime("%d/%m %H:%M")
        marker = "  (mais recente)" if position == 1 else ""
        console.info(f"{position}. {path.name}")
        console.detail(f"modificado em {stamp}{marker}")

    if not sys.stdin.isatty():
        console.detail(f"Sem terminal interativo — usando {candidates[0].name}.")
        return candidates[0]

    while True:
        try:
            answer = input(f"\n  Arquivo [1-{len(candidates)}, ENTER = 1]: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.info("Usando o mais recente.")
            return candidates[0]

        if not answer:
            return candidates[0]
        if answer.isdigit() and 1 <= int(answer) <= len(candidates):
            return candidates[int(answer) - 1]
        console.warn("Opção inválida.")


@console.guard("Falha ao preparar os dados.")
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Converte planilhas, limpa e monta o dataset de treino."
    )
    parser.add_argument(
        "--arquivo", metavar="CAMINHO",
        help="Usa este arquivo como origem, sem perguntar.",
    )
    parser.add_argument(
        "--limites", action="store_true",
        help=f"Descarta valores fora de [{ingest.PLAUSIBLE_MIN:g}, "
             f"{ingest.PLAUSIBLE_MAX:g}] — provável erro de digitação.",
    )
    parser.add_argument(
        "--manter", action="store_true",
        help="Mantém o dataset atual se ele já estiver mais novo que a origem.",
    )
    arguments = parser.parse_args()
    force = not arguments.manter

    console.banner("PREPARAÇÃO DOS DADOS", str(paths.RAW_DIR))

    # 1. Planilhas viram CSV.
    console.section("Planilhas")
    spreadsheets = sorted(path.name for path in paths.RAW_DIR.glob("*")
                          if path.suffix.lower() in ingest.SPREADSHEET_SUFFIXES
                          and not path.name.startswith("~$"))
    if spreadsheets:
        for name in spreadsheets:
            console.info(name)
    else:
        console.detail("Nenhuma planilha .xlsx — seguindo com os CSV da pasta.")

    console.section("Conversão")
    ingest.convert_folder(force=force)

    # 2. Escolha da origem.
    if arguments.arquivo:
        origin = Path(arguments.arquivo)
        if not origin.exists():
            console.error(f"Arquivo não encontrado: {origin}")
            sys.exit(1)
    else:
        candidates = ingest.list_sources()
        if not candidates:
            console.error(f"Nenhum CSV utilizável em {paths.RAW_DIR}.")
            console.detail("Coloque a planilha (.xlsx) ou o CSV exportado nessa pasta.")
            sys.exit(1)
        origin = choose_source(candidates)

    # 3. Limpeza e gravação.
    console.section("Limpeza")
    console.info(f"Origem: {origin.name}")
    try:
        destination = ingest.build_dataset(origin, force=force,
                                           limits=arguments.limites)
    except ingest.IngestError as exc:
        console.error(str(exc))
        sys.exit(1)

    frame = pd.read_csv(destination)
    console.result_panel(
        "DATASET PRONTO",
        [
            f"{len(frame)} amostra(s), {len(frame.columns)} coluna(s)",
            f"Origem:  {origin.name}",
            f"Arquivo: {destination}",
            "",
            "Já pode treinar: python src/main.py",
        ],
    )


if __name__ == "__main__":
    main()
