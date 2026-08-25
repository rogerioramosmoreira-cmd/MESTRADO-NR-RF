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

    Use com parcimônia: cada nota engorda a caixa de legenda, que é desenhada
    *dentro* dos eixos. Para texto descritivo longo — a definição de cada
    conjunto, por exemplo — prefira `caption()`, que escreve fora da área de
    dados e portanto não tem como cobrir barra nenhuma.
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


# ── Legendas e texto fora da área de dados ───────────────────────────────────
#
# A regra "sem títulos" empurra toda a explicação para dentro da figura, e o
# jeito mais fácil de fazer isso — acumular `note()` e chamar `legend()` — cria
# uma caixa que cresce com o número de conjuntos e acaba deitada sobre as
# barras. As funções abaixo dão o mesmo texto sem disputar espaço com o dado.

def legend_below(figure, axes, ncol: int = 3, y: float = 0.02, **kwargs):
    """
    Legenda das séries numa faixa abaixo da figura.

    Fora dos eixos, então é impossível cobrir dado. `ncol` espalha as entradas
    na horizontal em vez de empilhá-las numa coluna alta.

    A âncora é a borda **inferior** (`loc="lower center"`): assim `y` é o piso
    da caixa e ela cresce para cima conforme o número de linhas. Ancorada pelo
    topo, cada linha a mais desceria sobre o que estivesse no rodapé.
    """
    handles, labels = axes.get_legend_handles_labels()
    if not handles:
        return None
    options = {
        "loc": "lower center",
        "bbox_to_anchor": (0.5, y),
        "ncol": ncol,
        "frameon": False,
        "fontsize": 9,
    }
    options.update(kwargs)
    return figure.legend(handles, labels, **options)


def legend_right(figure, axes, **kwargs):
    """Legenda das séries numa coluna à direita da figura."""
    handles, labels = axes.get_legend_handles_labels()
    if not handles:
        return None
    options = {
        "loc": "center left",
        "bbox_to_anchor": (1.0, 0.5),
        "frameon": False,
        "fontsize": 9,
    }
    options.update(kwargs)
    return figure.legend(handles, labels, **options)


def caption(figure, lines: list[str], fontsize: int = 8) -> None:
    """
    Bloco de texto explicativo no rodapé da figura.

    É onde vai a descrição dos conjuntos e o destaque do melhor resultado —
    informação que pertence à figura mas não é uma série de dados, e que dentro
    dos eixos vira uma tarja por cima do gráfico.
    """
    if not lines:
        return
    figure.text(0.01, 0.005, "\n".join(lines), ha="left", va="bottom",
                fontsize=fontsize, color=PALETTE["ink"], linespacing=1.5)


def reserve_bottom(figure, lines_count: int, base: float = 0.10) -> None:
    """Reserva altura no rodapé proporcional ao número de linhas do `caption`."""
    figure.subplots_adjust(bottom=min(base + 0.028 * lines_count, 0.45))


# ── Espaço para rótulos de valor ─────────────────────────────────────────────

def headroom(axes, top: float = 0.18, bottom: float = 0.0) -> None:
    """
    Abre folga acima (e opcionalmente abaixo) dos dados.

    Rótulo de valor desenhado no topo da barra ocupa altura que o autoscale do
    matplotlib não conhece — sem esta folga ele encosta na borda do eixo ou
    some por baixo da legenda.
    """
    low, high = axes.get_ylim()
    span = (high - low) or 1.0
    axes.set_ylim(low - span * bottom, high + span * top)


def value_labels(axes, bars, values, fmt: str = "{:.4f}",
                 fontsize: int = 9, fontweight: str = "bold",
                 clip: bool = False) -> None:
    """
    Escreve o valor de cada barra do lado de fora dela.

    Barra negativa recebe o rótulo abaixo, positiva acima: colado à barra, o
    texto acompanha o sentido do dado em vez de flutuar no meio do gráfico.

    `clip=True` descarta o rótulo que cair fora dos limites do eixo. É o que um
    eixo quebrado precisa: as mesmas barras são desenhadas nos dois painéis, e
    sem recorte o rótulo de um valor distante fica pendurado longe da área do
    painel — e como `savefig` grava com `bbox_inches="tight"`, a figura inteira
    se estica para incluí-lo.
    """
    # O afastamento sai da altura visível do eixo, não da amplitude dos valores.
    # Num eixo quebrado os dois painéis compartilham a mesma lista de valores
    # mas têm escalas diferentes: medir pelos valores daria a ambos o mesmo
    # afastamento em unidades de dado, que no painel estreito é a altura toda.
    low, high = axes.get_ylim()
    span = (high - low) or 1.0
    for bar, value in zip(bars, values):
        value = float(value)
        offset = span * 0.02
        if value >= 0:
            y_position, valign = bar.get_height() + offset, "bottom"
        else:
            y_position, valign = bar.get_height() - offset, "top"
        text = axes.text(bar.get_x() + bar.get_width() / 2, y_position,
                         fmt.format(value), ha="center", va=valign,
                         fontsize=fontsize, fontweight=fontweight)
        if clip:
            text.set_clip_on(True)
            text.set_clip_box(axes.bbox)


# ── Escala robusta a valor discrepante ───────────────────────────────────────
#
# Uma rede neural que diverge em um conjunto produz R² = −12 enquanto os outros
# quatro ficam entre 0,3 e 0,7. Num eixo linear único, esse valor sozinho achata
# os outros quatro contra o zero e o gráfico deixa de comparar o que interessa.

def find_scale_break(values, legible_fraction: float = 0.15) -> tuple[float, float] | None:
    """
    Decide se um valor discrepante torna os demais ilegíveis num eixo único.

    Devolve `(fim_do_grupo_de_baixo, início_do_grupo_de_cima)` quando vale a
    pena quebrar o eixo, ou None quando uma escala só já serve.

    O critério não é "existe dispersão" — isso é verdade em quase todo conjunto
    de métricas e levaria a quebrar eixos perfeitamente legíveis. O critério é
    **quanta altura do eixo as barras do grupo maior chegam a ocupar**. Como
    barras partem do zero, a altura de cada uma é o próprio valor; se a mais
    alta do grupo maior não alcança `legible_fraction` da altura do eixo, todas
    elas viram uma faixa rasteira e a comparação entre conjuntos — o motivo do
    gráfico existir — se perde.

    Exemplos reais deste projeto:

    - R² da rede neural, [−12,11  0,35  0,65  0,62  0,58]: a maior barra do
      grupo útil alcança 5% da altura do eixo. Quebra.
    - MAE da floresta, [6,02  10,78  5,79  6,01  4,87]: alcança 56%. As cinco
      barras são distinguíveis a olho, não quebra.
    - MSE da floresta, [117,6  269,9  82,9  116,2  68,5]: alcança 44%. Idem.
    """
    ordered = sorted(float(value) for value in values)
    if len(ordered) < 3:
        return None

    # Barras partem do zero, então ele faz parte do eixo mesmo sem ser um valor.
    axis_span = max(0.0, ordered[-1]) - min(0.0, ordered[0])
    if axis_span <= 0:
        return None

    gaps = [(ordered[i + 1] - ordered[i], i) for i in range(len(ordered) - 1)]
    largest, index = max(gaps)
    if largest <= 0:
        return None

    lower_group = ordered[: index + 1]
    upper_group = ordered[index + 1:]

    # Um discrepante é minoria. Metade dos valores de cada lado é bimodalidade —
    # informação legítima do experimento, que a quebra esconderia.
    minority = min(len(lower_group), len(upper_group))
    if minority > len(ordered) // 3:
        return None

    major = lower_group if len(lower_group) >= len(upper_group) else upper_group
    # Altura máxima que uma barra do grupo maior atinge, medida a partir do zero.
    reach = max(abs(value) for value in major)
    if reach >= axis_span * legible_fraction:
        return None

    return ordered[index], ordered[index + 1]


def bar_panels(values, figsize=(10, 6), broken_figsize=(10, 7)):
    """
    Monta a figura de barras já decidindo entre eixo único e eixo quebrado.

    Devolve `(figura, painéis, aplicar_limites)`. O chamador desenha as mesmas
    barras em cada painel de `painéis` e **depois** chama `aplicar_limites()`.

    A ordem não é negociável: `set_ylim` desliga o autoscale, então limites
    aplicados antes das barras congelam o eixo no intervalo (0, 1) de um eixo
    vazio. As barras saem inteiras da área visível e os rótulos vão parar a
    milhares de unidades de distância — que o `bbox_inches="tight"` do savefig
    inclui, gravando uma figura de dezenas de milhares de pixels de altura.

    O recorte dos rótulos (`value_labels(..., clip=True)`) continua correto com
    os limites definidos depois: o recorte é avaliado no desenho, não na
    criação do texto.

    Concentrar a decisão aqui é o que faz o gráfico de um modelo e o de
    comparação entre modelos se comportarem igual diante do mesmo dado — antes,
    cada um decidia a própria escala e só um deles lidava com discrepantes.
    """
    values = [float(value) for value in values]
    break_at = find_scale_break(values)

    if break_at is None:
        figure, axes = new_axes(figsize=figsize)

        def apply_limits() -> None:
            headroom(axes, top=0.16, bottom=0.10 if min(values) < 0 else 0.0)

        return figure, [axes], apply_limits

    figure = plt.figure(figsize=broken_figsize, facecolor=PALETTE["background"])
    upper, lower, low_end, high_start = broken_bar_axes(figure, break_at)

    low_group = [value for value in values if value <= low_end]
    high_group = [value for value in values if value >= high_start]
    # As barras partem do zero, então o painel que mostra o grupo maior precisa
    # conter o zero para que elas tenham base. O painel do grupo discrepante
    # mostra só a faixa dele: se também contivesse o zero, redesenharia as
    # barras do outro grupo inteiras e duplicaria cada rótulo.
    discrepant_is_high = len(high_group) <= len(low_group)

    def apply_limits() -> None:
        if discrepant_is_high:
            # Discrepante no alto: painel de cima só a faixa dele, painel de
            # baixo do zero até o topo do grupo maior.
            high_span = (max(high_group) - min(high_group)) or abs(high_start) or 1.0
            upper.set_ylim(min(high_group) - high_span * 0.35,
                           max(high_group) + high_span * 0.45)
            low_span = (max(low_group) - min(0.0, min(low_group))) or 1.0
            lower.set_ylim(min(0.0, min(low_group)) - low_span * 0.05,
                           max(low_group) + low_span * 0.30)
        else:
            # Discrepante embaixo (tipicamente um R² muito negativo): painel de
            # cima do zero até o topo, painel de baixo só a faixa dele.
            high_span = (max(high_group) - min(0.0, min(high_group))) or 1.0
            upper.set_ylim(min(0.0, min(high_group)) - high_span * 0.05,
                           max(high_group) + high_span * 0.30)
            low_span = (max(low_group) - min(low_group)) or abs(low_end) or 1.0
            lower.set_ylim(min(low_group) - low_span * 0.55,
                           max(low_group) + low_span * 0.55)

    return figure, [upper, lower], apply_limits


def broken_bar_axes(figure, break_at: tuple[float, float], margin: float = 0.12):
    """
    Divide a figura em dois eixos que compartilham o eixo x e mostram faixas
    diferentes do eixo y, com as marcas diagonais que sinalizam a interrupção.

    Devolve `(eixo_de_cima, eixo_de_baixo)`. As mesmas barras são desenhadas nos
    dois; cada um recorta a sua faixa.
    """
    grid = figure.add_gridspec(2, 1, height_ratios=[2.4, 1], hspace=0.08)
    upper = figure.add_subplot(grid[0])
    lower = figure.add_subplot(grid[1], sharex=upper)
    for axes in (upper, lower):
        _style_axes(axes)

    low_end, high_start = break_at
    upper.spines["bottom"].set_visible(False)
    lower.spines["top"].set_visible(False)
    upper.tick_params(labelbottom=False, bottom=False)

    # Marcas diagonais na altura do corte — a convenção que avisa o leitor de
    # que o eixo foi interrompido e as alturas não são comparáveis entre os
    # dois painéis.
    kwargs = {
        "marker": [(-1, -margin * 8), (1, margin * 8)],
        "markersize": 9,
        "linestyle": "none",
        "color": PALETTE["ink"],
        "mec": PALETTE["ink"],
        "mew": 1.2,
        "clip_on": False,
    }
    upper.plot([0, 1], [0, 0], transform=upper.transAxes, **kwargs)
    lower.plot([0, 1], [1, 1], transform=lower.transAxes, **kwargs)

    return upper, lower, low_end, high_start


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


def save(figure, name: str, *subfolders: str, close: bool = True,
         tight: bool = True) -> list[Path]:
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

    `tight=False` preserva o espaçamento montado à mão. É o que uma figura com
    `caption()` no rodapé ou legenda fora dos eixos precisa: o `tight_layout`
    recalcula as margens olhando só para os eixos e engole a faixa reservada.
    """
    strip_titles(figure)
    if tight:
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
