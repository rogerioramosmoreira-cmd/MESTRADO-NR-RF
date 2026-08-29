"""
Previsão de CBR — menu e executor do pipeline.

Ponto de entrada único para todos os modelos do projeto. Sua função é deixar o
estado do sistema óbvio à primeira vista: qual modelo está rodando, há quanto
tempo, se terminou e — quando não terminou — por quê.

Os scripts filhos herdam este terminal em vez de terem a saída redirecionada,
para que a renderização de progresso deles continue intacta. Este executor
apenas emoldura cada execução e reporta o resultado.

Execução:  python src/main.py
           python src/main.py --check         (só o diagnóstico de bibliotecas)
           python src/main.py --run subsets   (executa um item ou grupo sem menu)
           python src/main.py --new-window    (abre uma nova janela de terminal)
"""

from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import console, dependencies, metrics, paths, scoreboard  # noqa: E402

SRC = Path(__file__).resolve().parent

# Códigos de saída usados pelos scripts filhos, e o que cada um significa para
# o usuário.
EXIT_MEANINGS = {
    0: "concluído",
    1: "erro durante a execução",
    2: "bibliotecas ausentes",
    126: "script não pôde ser iniciado",
    127: "script não encontrado",
    130: "interrompido pelo usuário",
}


@dataclass(frozen=True)
class Task:
    """Um script executável do catálogo."""

    key: str
    name: str
    description: str
    script: Path
    group: str
    # Grupo de dependências declarado em core/dependencies.GROUPS. É o que
    # permite ao menu avisar, antes de o usuário escolher, que um item vai
    # falhar por falta de biblioteca.
    requires: str = "core"

    def exists(self) -> bool:
        return self.script.exists()


TASKS: list[Task] = [
    Task("1", "Árvore Aleatória — modelo único",
         "Random Forest padrão sobre as 10 variáveis",
         SRC / "models" / "random_forest.py", "Modelos individuais"),
    Task("2", "Árvore Aleatória — alta performance (EN)",
         "Busca de 150 iterações, 10-fold CV, meta R² >= 0.80",
         SRC / "models" / "random_forest_en.py", "Modelos individuais"),
    Task("3", "Árvore Aleatória — ensemble",
         "VotingRegressor: RF + Gradient Boosting + Extra Trees",
         SRC / "models" / "random_forest_ensemble.py", "Modelos individuais"),
    Task("4", "Árvore Aleatória — quintis D1–D5",
         "Um modelo por faixa de CBR",
         SRC / "models" / "random_forest_quintis.py", "Modelos individuais"),
    Task("5", "Árvore Aleatória — quintis completo",
         "Quintis D1–D5 com combinações entre faixas",
         SRC / "models" / "random_forest_quintis_full.py", "Modelos individuais"),
    Task("6", "Rede Neural — modelo único",
         "MLP 128→64→32→16→1 sobre as 10 variáveis",
         SRC / "models" / "mlp.py", "Modelos individuais", requires="neural"),

    Task("7", "Conjuntos C1–C5 — Árvore Aleatória",
         "Um Random Forest por conjunto de variáveis",
         SRC / "models" / "subsets_rf.py", "Comparação de conjuntos"),
    Task("8", "Conjuntos C1–C5 — Rede Neural",
         "Um MLP por conjunto de variáveis",
         SRC / "models" / "subsets_mlp.py", "Comparação de conjuntos",
         requires="neural"),
    Task("9", "Conjuntos C1–C5 — comparação entre modelos",
         "Árvore Aleatória vs Rede Neural, conjunto a conjunto",
         SRC / "models" / "subsets_comparison.py", "Comparação de conjuntos"),

    Task("10", "Preparar dados — planilha ou CSV vira dataset",
         "Converte os .xlsx de data/raw, limpa e regrava o dataset de treino",
         SRC / "prepare" / "build_dataset.py", "Ferramentas", requires="excel"),
]

TASKS_BY_KEY = {task.key: task for task in TASKS}

# Grupos nomeados, usados pelo `--run <nome>` e pelo menu "executar grupo".
BATCHES: dict[str, tuple[str, list[str]]] = {
    "all": ("Todos os modelos", [task.key for task in TASKS if task.group != "Ferramentas"]),
    "rf": ("Todas as Árvores Aleatórias", ["1", "2", "3", "4", "5"]),
    "neural": ("Todas as Redes Neurais", ["6", "8"]),
    "subsets": ("Conjuntos C1–C5 completo", ["7", "8", "9"]),
}


@dataclass
class Outcome:
    """O que aconteceu com uma tarefa em uma execução."""

    task: Task
    code: int
    seconds: float

    @property
    def succeeded(self) -> bool:
        return self.code == 0

    @property
    def label(self) -> str:
        return EXIT_MEANINGS.get(self.code, f"código {self.code}")


def child_environment() -> dict:
    """
    Ambiente dos scripts filhos.

    Força UTF-8 na saída porque o console legado do Windows usa cp1252 por
    padrão e levantaria `UnicodeEncodeError` na saída acentuada em português e
    nos caracteres de desenho de caixa que os scripts imprimem.
    """
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    environment.setdefault("PYTHONUTF8", "1")
    return environment


def format_duration(seconds: float) -> str:
    """Tempo decorrido em formato legível."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}min {remainder}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}min"


def run_task(task: Task) -> Outcome:
    """Executa um script, emoldurado por um cabeçalho e uma linha de resultado."""
    console.section(task.name)
    console.detail(task.description)

    if not task.exists():
        console.error(f"Script não encontrado: {task.script}")
        return Outcome(task, code=127, seconds=0.0)

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [sys.executable, str(task.script)],
            cwd=str(paths.ROOT),
            env=child_environment(),
        )
        code = completed.returncode
    except OSError as exc:
        console.error(f"Não foi possível iniciar '{task.script.name}': {exc}")
        return Outcome(task, code=126, seconds=time.perf_counter() - started)

    elapsed = time.perf_counter() - started
    outcome = Outcome(task, code=code, seconds=elapsed)

    if outcome.succeeded:
        console.ok(f"{task.name} — concluído em {format_duration(elapsed)}")
    elif code == 2:
        console.error(f"{task.name} — bibliotecas ausentes; instale-as antes de repetir")
    elif code == 130:
        console.warn(f"{task.name} — interrompido pelo usuário")
    else:
        console.error(f"{task.name} — falhou ({outcome.label}) após "
                      f"{format_duration(elapsed)}")

    return outcome


def run_many(keys: list[str]) -> list[Outcome]:
    """
    Executa um grupo, seguindo em frente mesmo depois de uma falha.

    Parar no primeiro erro jogaria fora os modelos que vêm depois dele, cada um
    levando minutos para treinar. O resumo final informa exatamente quais
    quebraram.
    """
    outcomes: list[Outcome] = []
    for position, key in enumerate(keys, start=1):
        task = TASKS_BY_KEY.get(key)
        if task is None:
            console.warn(f"Item desconhecido: {key}")
            continue
        console.info(f"[{position}/{len(keys)}] iniciando")
        outcomes.append(run_task(task))
    return outcomes


def figures_folder() -> str:
    try:
        return str(paths.FIGURES_DIR.relative_to(paths.ROOT))
    except ValueError:
        return str(paths.FIGURES_DIR)


def show_summary(outcomes: list[Outcome]) -> None:
    """Tabela final: o que rodou, como terminou e quanto tempo levou."""
    if not outcomes:
        return

    rows = [
        [outcome.task.name,
         "OK" if outcome.succeeded else outcome.label.upper(),
         format_duration(outcome.seconds)]
        for outcome in outcomes
    ]
    console.section("Resumo da execução")
    console.metrics_table(rows, headers=["Modelo", "Situação", "Tempo"])

    failed = [outcome for outcome in outcomes if not outcome.succeeded]
    total = format_duration(sum(outcome.seconds for outcome in outcomes))

    if failed:
        console.result_panel(
            "EXECUÇÃO CONCLUÍDA COM FALHAS",
            [f"{len(outcomes) - len(failed)} de {len(outcomes)} concluíram.",
             "Falharam: " + ", ".join(outcome.task.name for outcome in failed),
             f"Tempo total: {total}"],
            success=False,
        )
    else:
        console.result_panel(
            "EXECUÇÃO CONCLUÍDA",
            [f"{len(outcomes)} modelo(s) executado(s) sem erro.",
             f"Tempo total: {total}",
             f"Gráficos em {figures_folder()}"],
        )


# ── Menu ─────────────────────────────────────────────────────────────────────

# Teclas do menu. Os modelos ficam com 1–10, herdados de `TASKS`; os grupos e o
# diagnóstico seguem a mesma numeração, para que a tela inteira seja uma lista
# só de números — sem submenu e sem tecla especial para decorar.
BATCH_KEYS = {str(index): name
              for index, name in enumerate(BATCHES, start=len(TASKS) + 1)}
DIAGNOSTIC_KEY = str(len(TASKS) + len(BATCHES) + 1)


def library_state(broken: dict[str, list[str]]) -> tuple[str, str]:
    """Resumo do ambiente em uma linha, para o cabeçalho do menu."""
    if not broken:
        return "OK", "ok"
    missing = {package for packages in broken.values() for package in packages}
    return f"atenção ({len(missing)} ausente(s))", "warn"


def catalogue_rows(broken: dict[str, list[str]]) -> list[tuple[str, str, str]]:
    """
    Monta a tabela do menu: modelos, grupos e ambiente em uma lista contínua.

    Itens que já se sabe que vão falhar — script ausente ou biblioteca faltando —
    são marcados aqui, antes da escolha, e não depois de o usuário esperar um
    treino começar.
    """
    rows: list[tuple[str, str, str]] = []

    def section(title: str) -> None:
        # Linha vazia antes do título: sem ela os blocos encostam uns nos
        # outros e a tabela vira um bloco único de 15 linhas.
        if rows:
            rows.append(("", "", ""))
        rows.append(("", title, ""))

    current_group = None
    for task in TASKS:
        if task.group != current_group:
            current_group = task.group
            section(current_group.upper())

        if not task.exists():
            name = f"{task.name}   — script ausente"
        elif task.requires in broken:
            name = f"{task.name}   — indisponível"
        else:
            name = task.name
        rows.append((task.key, name, task.description))

    section("GRUPOS")
    for key, batch in BATCH_KEYS.items():
        label, keys = BATCHES[batch]
        rows.append((key, label, f"{len(keys)} modelo(s)  •  --run {batch}"))

    section("AMBIENTE")
    rows.append((DIAGNOSTIC_KEY, "Rever bibliotecas",
                 "diagnóstico detalhado, versão por versão"))
    rows.append(("0", "Sair", ""))
    return rows


# Nomes curtos para a tabela de precisão. Os do catálogo têm até 45 caracteres
# e, ao lado de seis colunas numéricas, quebrariam em seis linhas cada.
SHORT_NAMES = {
    "random_forest": "RF — modelo único",
    "random_forest_en": "RF — alta performance",
    "random_forest_ensemble": "RF — ensemble",
    "random_forest_quintis": "RF — quintis D1–D5",
    "random_forest_quintis_full": "RF — quintis completo",
    "mlp": "Rede Neural",
    "subsets_rf": "C1–C5 — RF",
    "subsets_mlp": "C1–C5 — Rede Neural",
}

# Largura a partir da qual cabem também RMSE e MAPE. Abaixo dela a tabela fica
# com o essencial: R², MAE e MSE.
WIDE_SCREEN = 104


def scoreboard_table() -> tuple[list[str], list[list[str]]]:
    """
    Precisão da última execução de cada modelo, na ordem do catálogo.

    A chave do placar é o nome do arquivo do script (`random_forest.py` ->
    `random_forest`), então um modelo novo aparece aqui sozinho, sem precisar
    ser registrado em uma segunda lista que alguém esqueceria de atualizar.
    """
    detailed = console.width() >= WIDE_SCREEN

    columns = [("r2", "R²", 4, ""), ("mae", "MAE", 4, "")]
    if detailed:
        columns.append(("rmse", "RMSE", 4, ""))
    columns.append(("mse", "MSE", 4, ""))
    if detailed:
        columns.append(("mape", "MAPE", 2, "%"))

    headers = ["Modelo", *(label for _, label, _, _ in columns), "Situação"]
    if detailed:
        headers.append("Treinado")

    records = scoreboard.load_all([task.script.stem for task in TASKS])

    rows: list[list[str]] = []
    for task in TASKS:
        entry = records.get(task.script.stem)
        if entry is None:
            continue

        cells = []
        for field, _, digits, suffix in columns:
            value = entry.value(field)
            cells.append("—" if value is None else f"{value:.{digits}f}{suffix}")

        # Mesma régua do veredito impresso no fim do treino, vinda do mesmo
        # lugar: duas tabelas de corte separadas divergiriam na primeira vez que
        # alguém ajustasse uma delas.
        r2 = entry.value("r2")
        situation = "—" if r2 is None else metrics.grade(r2).label

        name = SHORT_NAMES.get(task.script.stem, task.name)
        row = [f"{task.key}. {name}", *cells, situation]
        if detailed:
            row.append(entry.age_text)
        rows.append(row)

    return headers, rows


def show_scoreboard() -> None:
    """Tabela de precisão — o que cada modelo entregou no conjunto de teste."""
    console.section("Precisão dos modelos — conjunto de teste")
    headers, rows = scoreboard_table()

    if not rows:
        console.detail("Nenhum modelo treinado ainda. A precisão aparece aqui "
                       "assim que o primeiro treino terminar.")
        return

    console.metrics_table(rows, headers=headers)
    console.detail("RF = Árvore Aleatória.  R² mais alto é melhor; "
                   "os erros (MAE, MSE) mais baixos.")
    # A régua sai da própria tabela de cortes, e não de um texto fixo: assim
    # mexer nos limites não deixa a legenda mentindo.
    scale = ",  ".join(f"≥{threshold:.2f} {label.lower()}"
                       for threshold, label, _ in metrics.GRADE_THRESHOLDS)
    console.detail(f"Situação pelo R²: {scale},  abaixo disso "
                   f"{metrics.WORST_GRADE[0].lower()}.")


def show_screen(broken: dict[str, list[str]]) -> None:
    """
    Desenha a tela do menu: cabeçalho de estado, precisão e catálogo completo.

    Tudo cabe em uma tela só, de propósito: com submenu, escolher um grupo
    custava duas decisões e escondia metade do catálogo atrás da primeira.

    A precisão vem antes do catálogo porque é ela que informa a escolha — saber
    que a Rede Neural está em R² 0,62 é o que faz alguém escolher treinar outra
    coisa.
    """
    status, state = library_state(broken)
    now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")

    console.menu_panel(
        "PREVISÃO DE CBR — MENU PRINCIPAL",
        "Aplicação de ML para Predição do CBR de Solos",
        f"{now}   •   Python {sys.version.split()[0]}   •   Bibliotecas: ",
        status,
        state,
    )
    show_scoreboard()
    console.menu_table(catalogue_rows(broken))


def ask(prompt: str) -> str | None:
    """Lê uma resposta do menu; devolve None quando o usuário aborta."""
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return None


def menu(broken: dict[str, list[str]]) -> None:
    while True:
        show_screen(broken)

        choice = ask("\n  Escolha uma opção: ")
        if choice is None or choice == "0":
            console.info("Encerrando.")
            return

        if choice in TASKS_BY_KEY:
            show_summary([run_task(TASKS_BY_KEY[choice])])
        elif choice in BATCH_KEYS:
            label, keys = BATCHES[BATCH_KEYS[choice]]
            console.section(label)
            show_summary(run_many(keys))
        elif choice == DIAGNOSTIC_KEY:
            # Reavalia em vez de reusar o resultado da abertura: o usuário pode
            # ter instalado o que faltava em outra janela sem fechar o menu.
            dependencies.check_all()
            broken = dependencies.broken_groups()
        else:
            console.warn("Opção inválida.")


# ── Nova janela de terminal ──────────────────────────────────────────────────

def relaunch_in_new_window() -> bool:
    """
    Reabre este menu em uma janela de terminal nova. Devolve True se conseguiu.

    Usado por `run.py`, para que iniciar o projeto dê ao usuário um terminal de
    verdade em vez de uma janela que pisca e fecha. A variável de guarda
    `MLL_NEW_WINDOW` impede o novo processo de abrir mais uma janela.
    """
    environment = child_environment()
    environment["MLL_NEW_WINDOW"] = "1"
    command = [sys.executable, str(Path(__file__).resolve())]

    if sys.platform == "win32":
        try:
            subprocess.Popen(
                command, cwd=str(paths.ROOT), env=environment,
                creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
            )
            return True
        except OSError as exc:
            console.warn(f"Não foi possível abrir nova janela: {exc}")
            return False

    # No Linux/macOS não existe um emulador de terminal garantido; tenta os mais
    # comuns e, se nenhum existir, roda na própria janela em vez de falhar.
    for terminal, arguments in (
        ("x-terminal-emulator", ["-e"]),
        ("gnome-terminal", ["--"]),
        ("konsole", ["-e"]),
        ("xterm", ["-e"]),
    ):
        try:
            subprocess.Popen([terminal, *arguments, *command],
                             cwd=str(paths.ROOT), env=environment)
            return True
        except (FileNotFoundError, OSError):
            continue

    console.warn("Nenhum emulador de terminal encontrado — executando nesta janela.")
    return False


@console.guard("Falha no menu principal.")
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Executa os modelos de previsão de CBR."
    )
    parser.add_argument("--check", action="store_true",
                        help="Verifica as bibliotecas instaladas e sai.")
    parser.add_argument("--run", metavar="ALVO",
                        help="Executa um item (1-10) ou um grupo "
                             f"({', '.join(BATCHES)}) sem abrir o menu.")
    parser.add_argument("--new-window", action="store_true",
                        help="Abre o menu em uma nova janela de terminal.")
    arguments = parser.parse_args()

    if arguments.new_window and os.environ.get("MLL_NEW_WINDOW") != "1":
        if relaunch_in_new_window():
            return

    if arguments.check:
        console.banner("DIAGNÓSTICO DE BIBLIOTECAS", str(paths.ROOT))
        sys.exit(0 if dependencies.check_all() else 2)

    # Só no caminho não interativo: o menu abre com o próprio painel de
    # cabeçalho, e um banner logo acima dele daria duas molduras seguidas
    # dizendo a mesma coisa.
    if arguments.run:
        console.banner(
            "PREVISÃO DE CBR — MACHINE LEARNING",
            f"{paths.ROOT}\n{sys.executable}",
        )

    # Verificação automática na abertura do projeto. Roda em todos os caminhos —
    # menu interativo e `--run` — para que ninguém descubra uma biblioteca
    # ausente só depois de esperar um treino começar.
    broken = dependencies.startup_check()

    if arguments.run:
        target = arguments.run.strip().lower()
        if target in BATCHES:
            label, keys = BATCHES[target]
            console.section(label)
            outcomes = run_many(keys)
        elif target in TASKS_BY_KEY:
            outcomes = [run_task(TASKS_BY_KEY[target])]
        else:
            console.error(f"Alvo desconhecido: {arguments.run}")
            console.detail(f"Itens: {', '.join(TASKS_BY_KEY)}")
            console.detail(f"Grupos: {', '.join(BATCHES)}")
            sys.exit(2)

        show_summary(outcomes)
        sys.exit(0 if all(outcome.succeeded for outcome in outcomes) else 1)

    menu(broken)

    # Uma janela aberta por `run.py` fecha no instante em que isto retorna,
    # levando o resumo junto. Segura a janela até o usuário ler a saída.
    if os.environ.get("MLL_NEW_WINDOW") == "1":
        try:
            input("\n  Pressione ENTER para fechar...")
        except (EOFError, KeyboardInterrupt):
            pass


if __name__ == "__main__":
    main()
