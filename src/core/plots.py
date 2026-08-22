"""
Estilo e persistência dos gráficos.

Duas regras do projeto são garantidas aqui, em vez de ficarem a cargo de cada
script:

1. **Os gráficos não têm título.** Legendas, rótulos de eixo e anotações dentro
   da figura fazem a explicação, porque estas figuras vão para uma dissertação
   onde a legenda da figura é que cumpre o papel do título. `save()` remove
   qualquer título de eixo e o supertítulo antes de gravar, então um `set_title`
   esquecido em uma edição futura não vaza para a figura publicada.

2. **Todo gráfico é gravado em disco.** Antes, os 78 gráficos existiam apenas
   como janelas de `plt.show()` e se perdiam ao fechar. A exibição na tela
   continua acontecendo — `show_all()` a faz no fim do script —, mas agora é
   um extra sobre o arquivo gravado, e não o único lugar onde a figura existe.

A informação que antes vivia no título (contagens, melhor época, MSE atingido)
pertence à legenda — `note()` acrescenta uma entrada de legenda sem marcador
visual, para que o valor sobreviva à remoção do título.
"""

from __future__ import annotations

import atexit
import os
from pathlib import Path

import matplotlib

# A escolha do backend precisa acontecer antes de o pyplot ser importado.
# Execuções em lote usam o Agg, para que gerar dezenas de figuras nunca abra uma
# janela que trava o processo.
SHOW_FIGURES = os.environ.get("MLL_SHOW", "1") != "0"
if not SHOW_FIGURES:
    matplotlib.use("Agg")

# Teto de janelas abertas de uma vez. A comparação de conjuntos gera mais de
# vinte figuras; abrir todas travaria a máquina, e as excedentes já estão em
# disco de qualquer forma.
SHOW_LIMIT = int(os.environ.get("MLL_SHOW_LIMIT", "12"))

import matplotlib.pyplot as plt  # noqa: E402 - precisa vir depois do backend

from core import console, paths  # noqa: E402

# ── Paleta ───────────────────────────────────────────────────────────────────
# Espelha core/console.py, para que a saída do terminal e os gráficos
# compartilhem uma identidade só.
PALETTE = {
    "blue": "#2563EB",
    "blue_light": "#60A5FA",
    "orange": "#EA580C",
    "green": "#16A34A",
    "red": "#DC2626",
    "purple": "#7C3AED",
    "amber": "#F59E0B",
    "background": "#F8FAFC",
    "grid": "#E2E8F0",
    "ink": "#111827",
}

# Cores ordenadas da comparação C1–C5, uma por conjunto. Escolhidas para
# continuarem distinguíveis em impressão preto e branco, que é onde uma
# dissertação pode acabar.
SUBSET_COLORS = {
    "c1": "#2563EB",
    "c2": "#EA580C",
    "c3": "#16A34A",
    "c4": "#7C3AED",
    "c5": "#0F172A",
}

MODEL_COLORS = {
    "random_forest": "#2563EB",
    "mlp": "#EA580C",
}

FIGURE_FORMATS = tuple(
    fmt.strip() for fmt in os.environ.get("MLL_FIG_FORMATS", "png").split(",") if fmt.strip()
)
FIGURE_DPI = int(os.environ.get("MLL_FIG_DPI", "200"))


def apply_style() -> None:
    """Instala os padrões de matplotlib usados em todo o projeto."""
    plt.rcParams.update({
        "figure.facecolor": PALETTE["background"],
        "axes.facecolor": PALETTE["background"],
        "savefig.facecolor": PALETTE["background"],
        "axes.grid": True,
        "axes.axisbelow": True,   # grade atrás das barras, não cortando-as
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.8,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 0,      # os títulos são removidos; deixa-os invisíveis
        "legend.framealpha": 0.0,
        "figure.autolayout": False,
    })


apply_style()


# ── Auxiliares de eixos ──────────────────────────────────────────────────────

def new_axes(figsize=(10, 5)):
    """Cria um par figura/eixo já estilizado."""
    figure, axes = plt.subplots(figsize=figsize, facecolor=PALETTE["background"])
    _style_axes(axes)
    return figure, axes


def new_grid(rows: int, columns: int, figsize=(14, 5), **kwargs):
    """Cria uma grade de subgráficos já estilizada."""
    figure, axes = plt.subplots(rows, columns, figsize=figsize,
                                facecolor=PALETTE["background"], **kwargs)
    flat = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for single in flat:
        _style_axes(single)
    return figure, axes


def _style_axes(axes) -> None:
    axes.set_facecolor(PALETTE["background"])
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)


def note(axes, text: str, color: str | None = None) -> None:
    """
    Acrescenta uma entrada de legenda que carrega informação mas não desenha
    marcador nenhum.

    É assim que um valor que antes ficava no título do gráfico (uma contagem de
    outliers, o MSE atingido, a época de parada) continua visível depois que os
    títulos foram removidos.
    """
    axes.plot([], [], linestyle="none", marker="none",
              label=text, color=color or PALETTE["ink"])


def legend(axes, **kwargs):
    """Desenha a legenda com os padrões do projeto, se houver o que desenhar."""
    handles, labels = axes.get_legend_handles_labels()
    if not handles:
        return None
    options = {"framealpha": 0.0, "fontsize": 9}
    options.update(kwargs)
    return axes.legend(handles, labels, **options)


# ── Supressão de títulos ─────────────────────────────────────────────────────

def strip_titles(figure) -> None:
    """
    Limpa todos os títulos de uma figura — os de cada eixo e o supertítulo.

    Chamada automaticamente por `save()`. Tornar a regra estrutural significa
    que um gráfico não consegue voltar a ter título por acidente.
    """
    for axes in figure.get_axes():
        if axes.get_title():
            axes.set_title("")
        for position in ("left", "center", "right"):
            if axes.get_title(loc=position):
                axes.set_title("", loc=position)

    # O `suptitle` fica guardado em `figure._suptitle`; não existe getter
    # público, então o atributo é consultado de forma defensiva e seu texto é
    # esvaziado quando presente.
    suptitle = getattr(figure, "_suptitle", None)
    if suptitle is not None:
        suptitle.set_text("")


# ── Persistência ─────────────────────────────────────────────────────────────

# Figuras já gravadas e ainda abertas, esperando `show_all()`.
_pending: list = []


def figure_dir(*parts: str) -> Path:
    """Resolve (e cria) uma subpasta dentro de reports/figures."""
    target = paths.FIGURES_DIR.joinpath(*parts)
    target.mkdir(parents=True, exist_ok=True)
    return target


def save(figure, name: str, *subfolders: str, close: bool = True) -> list[Path]:
    """
    Remove os títulos, grava a figura em reports/figures/<subpastas>/<nome> e
    devolve os caminhos escritos.

    Com a exibição ligada, a figura fica aberta até o fim do script e é mostrada
    por `show_all()`. Fechá-la aqui, como se fazia antes, criava e destruía a
    janela do backend Tk sem que o laço de eventos dele jamais rodasse — o que
    fazia o coletor de lixo derrubar `RuntimeError: main thread is not in main
    loop` no fim de toda execução.

    Com a exibição desligada (`MLL_SHOW=0`, backend Agg), a figura é fechada na
    hora, a menos que `close=False`: um treino que produz vinte gráficos
    manteria todos em memória e dispararia o aviso de figuras abertas do
    matplotlib.
    """
    strip_titles(figure)
    figure.tight_layout()

    target = figure_dir(*subfolders)
    written: list[Path] = []
    for extension in FIGURE_FORMATS:
        destination = target / f"{name}.{extension}"
        figure.savefig(destination, dpi=FIGURE_DPI, bbox_inches="tight",
                       facecolor=figure.get_facecolor())
        written.append(destination)

    if SHOW_FIGURES:
        _pending.append(figure)
    elif close:
        plt.close(figure)

    return written


def show_all() -> None:
    """
    Mostra na tela, de uma vez, tudo o que foi gravado nesta execução.

    Chamada automaticamente ao fim do script (via `atexit`), para que nenhum dos
    dez scripts de modelo precise lembrar de fazê-lo. `plt.show()` bloqueia até
    o usuário fechar as janelas — é o laço de eventos do backend rodando, o
    mesmo cuja ausência quebrava o encerramento.
    """
    global _pending

    figures, _pending = _pending, []
    if not SHOW_FIGURES or not figures:
        return

    excess = figures[SHOW_LIMIT:]
    for figure in excess:
        plt.close(figure)

    shown = figures[:SHOW_LIMIT]
    if not shown:
        return

    console.info(f"{len(shown)} gráfico(s) na tela — feche as janelas para continuar.")
    if excess:
        console.detail(f"Outros {len(excess)} ficaram só em disco "
                       f"(teto MLL_SHOW_LIMIT={SHOW_LIMIT}).")

    try:
        plt.show()
    except Exception as exc:  # noqa: BLE001 - exibir é acessório, nunca fatal
        # Sem servidor gráfico (SSH, container) abrir janela falha. Os arquivos
        # já estão gravados, então isto é um aviso, não uma falha do treino.
        console.warn(f"Não foi possível exibir os gráficos: {exc}")
    finally:
        plt.close("all")


atexit.register(show_all)


def relative(path: Path) -> str:
    """Formata um caminho gravado em relação à raiz do repositório, para log."""
    try:
        return str(path.relative_to(paths.ROOT))
    except ValueError:
        return str(path)
