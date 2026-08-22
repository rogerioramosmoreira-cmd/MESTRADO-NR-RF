"""
Interface de terminal do pipeline.

Dá a todos os scripts o mesmo vocabulário visual mínimo — cabeçalho, etapas
numeradas, indicadores de progresso, tabelas de métricas e linhas claramente
marcadas de sucesso/aviso/erro — para que, a qualquer momento, fique óbvio se o
sistema está *carregando*, *treinando*, *concluído* ou *quebrado*.

Usa `rich` quando disponível. Se `rich` não estiver instalado, o módulo cai para
`print` simples — porque é este arquivo que reporta a ausência do `rich`, e ele
jamais pode ser a peça que quebra por causa dela.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Iterable, Iterator, Sequence

# ── Identidade visual ────────────────────────────────────────────────────────
# Mantida em sincronia com a paleta do matplotlib em core/plots.py, para que o
# terminal e os gráficos sejam lidos como um sistema só.
ACCENT = "#2563EB"       # azul primário  — títulos, trabalho em andamento
ACCENT_SOFT = "#60A5FA"  # azul claro     — valores secundários
SUCCESS = "#16A34A"      # verde          — concluído
WARNING = "#EA580C"      # laranja        — degradado, mas recuperável
DANGER = "#DC2626"       # vermelho       — falhou
MUTED = "#94A3B8"        # cinza-azulado  — texto de apoio

# Os símbolos têm alternativas em ASCII puro para o caminho sem `rich`: o
# console legado do Windows (cp1252) levanta UnicodeEncodeError em caracteres
# de desenho de caixa.
_MARK_OK = "OK"
_MARK_WARN = "!!"
_MARK_ERR = "XX"
_MARK_INFO = "--"

try:
    from rich import box
    from rich.console import Console as _RichConsole
    from rich.markup import escape as _rich_escape
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:  # o próprio `rich` é uma dependência que talvez falte
    RICH_AVAILABLE = False


def _safe(text) -> str:
    """
    Neutraliza a marcação do rich em texto vindo de fora.

    O rich lê `[qualquer coisa]` como tag de estilo e descarta o trecho quando
    ele não corresponde a um estilo conhecido. Sem escapar, marcadores legítimos
    como `[script ausente]` ou `[falta pandas]` desaparecem da tela — o texto
    some em silêncio, sem erro nenhum.
    """
    text = str(text)
    return _rich_escape(text) if RICH_AVAILABLE else text


class _PlainConsole:
    """Substituto mínimo usado quando o `rich` não está instalado."""

    def print(self, *args, **kwargs) -> None:
        kwargs.pop("style", None)
        kwargs.pop("justify", None)
        kwargs.pop("highlight", None)
        text = " ".join(str(a) for a in args)
        # Remove a marcação do rich para que a saída simples não imprima
        # literalmente as tags [bold].
        cleaned = _strip_markup(text)
        try:
            print(cleaned, **kwargs)
        except UnicodeEncodeError:
            print(cleaned.encode("ascii", "replace").decode("ascii"), **kwargs)

    def rule(self, title: str = "", **_kwargs) -> None:
        self.print(f"\n--- {_strip_markup(title)} " + "-" * max(0, 50 - len(title)))


def _strip_markup(text: str) -> str:
    """Remove as tags `[estilo]` para que a marcação nunca vaze na saída."""
    out, depth = [], 0
    for char in text:
        if char == "[":
            depth += 1
        elif char == "]" and depth:
            depth -= 1
        elif not depth:
            out.append(char)
    return "".join(out)


console = _RichConsole(highlight=False) if RICH_AVAILABLE else _PlainConsole()


# ── Elementos estruturais ────────────────────────────────────────────────────

def banner(title: str, subtitle: str = "") -> None:
    """Bloco de abertura de um script — o que está prestes a rodar."""
    if RICH_AVAILABLE:
        body = Text(title, style=f"bold {ACCENT}")
        if subtitle:
            body.append(f"\n{subtitle}", style=MUTED)
        console.print(Panel(body, border_style=ACCENT, padding=(1, 3)))
    else:
        console.print("\n" + "=" * 62)
        console.print(f"  {title}")
        if subtitle:
            console.print(f"  {subtitle}")
        console.print("=" * 62)


def section(title: str) -> None:
    """Divisor horizontal entre as fases de um script."""
    if RICH_AVAILABLE:
        console.rule(Text(title, style=f"bold {ACCENT}"), style=MUTED)
    else:
        console.rule(title)


def step(index: int, total: int, description: str) -> None:
    """Marcador numerado de progresso, ex.: `[3/7] Treinando modelo final`."""
    if RICH_AVAILABLE:
        console.print(f"[{ACCENT}]\\[{index}/{total}][/] [bold]{_safe(description)}[/]")
    else:
        console.print(f"[{index}/{total}] {description}")


def detail(message: str) -> None:
    """Linha de apoio sob uma etapa — recuada e em tom discreto."""
    if RICH_AVAILABLE:
        console.print(f"      [{MUTED}]{_safe(message)}[/]")
    else:
        console.print(f"      {message}")


def ok(message: str) -> None:
    _status_line(SUCCESS, "✓", _MARK_OK, message)


def warn(message: str) -> None:
    _status_line(WARNING, "▲", _MARK_WARN, message)


def error(message: str) -> None:
    _status_line(DANGER, "✗", _MARK_ERR, message)


def info(message: str) -> None:
    _status_line(ACCENT_SOFT, "•", _MARK_INFO, message)


def _status_line(color: str, glyph: str, fallback: str, message: str) -> None:
    if RICH_AVAILABLE:
        console.print(f"  [{color}]{glyph}[/] {_safe(message)}")
    else:
        console.print(f"  {fallback} {message}")


@contextmanager
def working(message: str) -> Iterator[None]:
    """
    Envolve uma operação demorada com um indicador de atividade.

    O indicador é desligado no `finally`, para que uma exceção no meio do treino
    não deixe o terminal preso em um cursor girando com a tela alternativa
    ligada.
    """
    if not RICH_AVAILABLE:
        console.print(f"  ... {message}")
        yield
        return

    status = console.status(f"[{ACCENT}]{message}[/]", spinner="dots")
    status.start()
    try:
        yield
    finally:
        status.stop()


def width() -> int:
    """
    Largura útil do terminal.

    Serve para quem monta tabela decidir quantas colunas cabem: sem isso, uma
    tabela de sete colunas em uma janela de 80 quebra cada nome em seis linhas
    e fica ilegível.
    """
    return int(getattr(console, "width", 0) or 80)


def metrics_table(rows: Sequence[Sequence[str]], headers: Sequence[str]) -> None:
    """Renderiza uma comparação de métricas como tabela alinhada."""
    if RICH_AVAILABLE:
        table = Table(show_header=True, header_style=f"bold {ACCENT}",
                      border_style=MUTED, padding=(0, 2))
        for position, header in enumerate(headers):
            table.add_column(_safe(header),
                             justify="left" if position == 0 else "right")
        for row in rows:
            table.add_row(*[_safe(cell) for cell in row])
        console.print(table)
    else:
        widths = [max(len(str(r[i])) for r in [headers, *rows]) for i in range(len(headers))]
        console.print("  " + "  ".join(str(h).ljust(w) for h, w in zip(headers, widths)))
        for row in rows:
            console.print("  " + "  ".join(str(c).ljust(w) for c, w in zip(row, widths)))


# ── Menu ───────────────────────────────────────────────────────────────────────

STATE_COLORS = {"ok": SUCCESS, "warn": WARNING, "error": DANGER}


def menu_panel(title: str, heading: str, subtitle: str = "",
               status: str = "", state: str = "ok") -> None:
    """
    Cabeçalho emoldurado do menu.

    Responde de uma vez o que o projeto é, quando está sendo executado, sob qual
    interpretador e se o ambiente está íntegro — para que a decisão do usuário
    não dependa de rolar a tela para trás.
    """
    color = STATE_COLORS.get(state, MUTED)

    if RICH_AVAILABLE:
        body = Text(heading, style="bold")
        if subtitle:
            body.append(f"\n{subtitle}", style=MUTED)
        if status:
            body.append(status, style=f"bold {color}")
        console.print(Panel(
            body,
            title=f"[bold {ACCENT}]{_safe(title)}[/]",
            border_style=ACCENT,
            box=box.ROUNDED,
            padding=(1, 3),
        ))
    else:
        console.print("\n" + "=" * 62)
        console.print(f"  {title}")
        console.print(f"  {heading}")
        if subtitle or status:
            console.print(f"  {subtitle}{status}")
        console.print("=" * 62)


def menu_table(rows: Sequence[Sequence[str]]) -> None:
    """
    Catálogo de opções como tabela alinhada.

    Cada linha é `(tecla, nome, detalhe)`. Uma tecla vazia marca um título de
    seção, e não uma opção escolhível: sem essa distinção o usuário digitaria
    o nome de um grupo esperando executá-lo.
    """
    if RICH_AVAILABLE:
        table = Table(show_header=False, box=box.SIMPLE, pad_edge=False)
        table.add_column("Opção", style=f"bold {ACCENT}", width=4, justify="center")
        table.add_column("Item")
        table.add_column("Detalhe", style=MUTED)
        for key, name, note in rows:
            if not key:
                table.add_row("", Text(name, style=f"bold {MUTED}"), "")
            elif key == "0":
                table.add_row(key, Text(name, style=f"bold {DANGER}"), _safe(note))
            else:
                table.add_row(key, _safe(name), _safe(note))
        console.print(table)
    else:
        for key, name, note in rows:
            if not key:
                # A linha separadora vazia não tem o que imprimir aqui: sem
                # tabela, o `\n` do título já dá o espaçamento sozinho.
                if name:
                    console.print(f"\n  {name}")
                continue
            console.print(f"  {key:>3}. {name}")
            if note:
                console.print(f"       {note}")


def result_panel(title: str, lines: Iterable[str], success: bool = True) -> None:
    """Bloco de encerramento de um script — o resultado, emoldurado."""
    color = SUCCESS if success else DANGER
    body = "\n".join(lines)
    if RICH_AVAILABLE:
        console.print(Panel(_safe(body), title=_safe(title),
                            border_style=color, padding=(1, 3)))
    else:
        console.print("\n" + "=" * 62)
        console.print(f"  {title}")
        for line in body.splitlines():
            console.print(f"  {line}")
        console.print("=" * 62)


def verdict_panel(title: str, verdict: str, meaning: str,
                  lines: Iterable[str], state: str = "ok") -> None:
    """
    Bloco de veredito — a classificação do modelo, em destaque, e o porquê.

    A classificação vai sozinha na primeira linha, em corpo grande e na cor do
    estado: é a única linha que alguém lê quando volta ao terminal depois de um
    treino de meia hora, e ela não pode estar disputando espaço com números.
    """
    color = STATE_COLORS.get(state, MUTED)
    body = "\n".join(lines)

    if RICH_AVAILABLE:
        text = Text(verdict, style=f"bold {color}")
        if meaning:
            text.append(f"\n{meaning}", style=MUTED)
        if body:
            text.append(f"\n\n{_strip_markup(body)}")
        console.print(Panel(text, title=f"[bold {color}]{_safe(title)}[/]",
                            border_style=color, box=box.ROUNDED, padding=(1, 3)))
    else:
        console.print("\n" + "=" * 62)
        console.print(f"  {title}: {verdict}")
        if meaning:
            console.print(f"  {meaning}")
        for line in body.splitlines():
            console.print(f"  {line}")
        console.print("=" * 62)


def failure(exc: BaseException, context: str = "") -> None:
    """
    Reporta uma exceção não tratada como um bloco emoldurado e legível.

    Imprime primeiro o tipo e a mensagem da exceção — a parte que diz ao usuário
    o que fazer — e só depois o traceback, para que a linha acionável não fique
    enterrada sob os quadros de pilha.
    """
    import traceback

    header = f"{type(exc).__name__}: {exc}"
    trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    if RICH_AVAILABLE:
        body = Text(header, style=f"bold {DANGER}")
        if context:
            body.append(f"\n\n{context}", style=MUTED)
        console.print(Panel(body, title="ERRO", border_style=DANGER, padding=(1, 3)))
        # O traceback carrega colchetes com frequência (índices, repr de listas);
        # sem escapar, trechos dele sumiriam da tela.
        console.print(f"[{MUTED}]{_safe(trace)}[/]")
    else:
        console.print("\n" + "!" * 62)
        console.print(f"  ERRO — {header}")
        if context:
            console.print(f"  {context}")
        console.print("!" * 62)
        console.print(trace)


def guard(context: str = ""):
    """
    Decorador que transforma qualquer exceção não tratada em um bloco de erro
    emoldurado e em um código de saída diferente de zero, no lugar de um
    traceback cru despejado no terminal.

    Usado como invólucro do ponto de entrada de todo script executável, para que
    o usuário sempre veja *o que* falhou na mesma linguagem visual de uma
    execução bem-sucedida.
    """
    def decorator(function):
        def wrapper(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except SystemExit:
                # Saída deliberada — uma verificação de dependências que falhou,
                # um `--check` que terminou. Repassada intacta para que o código
                # de saída sobreviva e não seja reportada como quebra.
                raise
            except KeyboardInterrupt:
                warn("Execução interrompida pelo usuário.")
                sys.exit(130)
            except BaseException as exc:  # noqa: BLE001 - reporte de topo
                failure(exc, context)
                sys.exit(1)
        wrapper.__name__ = getattr(function, "__name__", "wrapper")
        wrapper.__doc__ = function.__doc__
        return wrapper
    return decorator
