"""
Verificação do comportamento de escala e layout dos gráficos.

Roda sem pytest — `python tests/test_plots.py` — porque o projeto não tem
dependência de teste instalada e não vale a pena adicionar uma só para isto.
Com pytest disponível, ele também coleta e roda os mesmos `test_*`.

O que estes testes travam é a decisão que quebrou os gráficos dos conjuntos:
quando um valor discrepante deve interromper o eixo e quando não deve. Errar
para mais é tão ruim quanto errar para menos — um eixo quebrado sem motivo é
mais difícil de ler que um eixo contínuo.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")   # sem janela: isto roda em terminal e em CI

from PIL import Image  # noqa: E402

from core.plots import (  # noqa: E402
    bar_panels,
    find_scale_break,
    headroom,
    new_axes,
    value_labels,
)

# Valores reais medidos neste projeto, para que o teste falhe se a calibração
# deixar de servir para os dados que ele precisa atender.
R2_MLP = [-12.1141, 0.3470, 0.6467, 0.6233, 0.5796]
MSE_MLP = [5577.081, 277.719, 150.236, 160.182, 178.797]
MSE_RF = [117.5982, 269.9437, 82.8674, 116.1905, 68.5481]
R2_RF = [0.723, 0.365, 0.805, 0.727, 0.839]
MAE_RF = [6.5, 9.8, 5.4, 6.4, 4.9]


# ── Detecção de escala ───────────────────────────────────────────────────────

def test_r2_divergente_quebra_o_eixo():
    """R² de −12 contra valores em torno de 0,6: sem quebra, os quatro bons
    ficam achatados contra o zero e não dá para compará-los."""
    assert find_scale_break(R2_MLP) is not None


def test_mse_divergente_quebra_o_eixo():
    assert find_scale_break(MSE_MLP) is not None


def test_metricas_bem_distribuidas_nao_quebram():
    """Estes três estavam legíveis antes da mudança e precisam continuar em
    eixo contínuo — quebrar sem necessidade prejudica a leitura."""
    assert find_scale_break(MSE_RF) is None
    assert find_scale_break(R2_RF) is None
    assert find_scale_break(MAE_RF) is None


def test_bimodal_nao_quebra():
    """Metade dos valores de cada lado é resultado do experimento, não um
    discrepante. A quebra esconderia a distância entre os dois grupos."""
    assert find_scale_break([1.0, 1.1, 1.2, 90.0, 91.0, 92.0]) is None


def test_casos_degenerados_nao_quebram():
    assert find_scale_break([3.0, 3.0, 3.0, 3.0]) is None
    assert find_scale_break([0.0, 0.0, 0.0]) is None
    assert find_scale_break([1.0, 500.0]) is None       # poucos pontos
    assert find_scale_break([]) is None


# ── Montagem dos painéis ─────────────────────────────────────────────────────

def _panel_covers(axes, value: float) -> bool:
    low, high = axes.get_ylim()
    return low <= value <= high


def test_painel_unico_quando_nao_ha_discrepante():
    figure, panels, apply_limits = bar_panels(MSE_RF)
    try:
        assert len(panels) == 1
        panels[0].bar(range(len(MSE_RF)), MSE_RF)
        apply_limits()
        for value in MSE_RF:
            assert _panel_covers(panels[0], value)
    finally:
        matplotlib.pyplot.close(figure)


def test_limites_aplicados_depois_das_barras_cobrem_os_valores():
    """`set_ylim` desliga o autoscale.

    Aplicados antes das barras, os limites congelam o eixo no (0, 1) de um eixo
    vazio: as barras saem da área visível, os rótulos vão parar a milhares de
    unidades de distância e o `bbox_inches="tight"` grava uma figura de dezenas
    de milhares de pixels. Aconteceu duas vezes; este teste fecha a porta.
    """
    for values in (MSE_RF, MAE_RF, R2_RF, MSE_MLP, R2_MLP):
        figure, panels, apply_limits = bar_panels(values)
        try:
            for axes in panels:
                axes.bar(range(len(values)), values)
            apply_limits()

            # Todo valor precisa estar visível em pelo menos um painel.
            for value in values:
                assert any(_panel_covers(axes, value) for axes in panels), (
                    f"valor {value} fora de todos os painéis em {values}"
                )
        finally:
            matplotlib.pyplot.close(figure)


def test_figura_gravada_tem_altura_plausivel(tmp_path=None):
    """Verificação de ponta: grava a figura e confere as dimensões do arquivo.

    É o sintoma que o usuário vê. Os testes de limites acima explicam a causa;
    este garante que o arquivo final não saia deformado por nenhuma outra.
    """
    import tempfile

    destino = Path(tmp_path or tempfile.mkdtemp()) / "figura.png"
    for values in (MSE_RF, MSE_MLP, R2_MLP, MAE_RF):
        figure, panels, apply_limits = bar_panels(values)
        try:
            for axes in panels:
                bars = axes.bar(range(len(values)), values)
                value_labels(axes, bars, values, clip=len(panels) > 1)
            apply_limits()
            figure.savefig(destino, dpi=100, bbox_inches="tight")
        finally:
            matplotlib.pyplot.close(figure)

        with Image.open(destino) as imagem:
            largura, altura = imagem.size
        assert altura < 3000, f"figura com {altura}px de altura para {values}"
        assert largura < 3000, f"figura com {largura}px de largura para {values}"


def test_discrepante_alto_fica_sozinho_no_painel_de_cima():
    """O painel do discrepante não pode conter o zero: contendo, ele redesenha
    as barras do outro grupo inteiras e duplica cada rótulo de valor."""
    figure, panels, apply_limits = bar_panels(MSE_MLP)
    try:
        assert len(panels) == 2
        upper, lower = panels
        apply_limits()

        assert _panel_covers(upper, 5577.081)
        assert not _panel_covers(upper, 0.0)
        assert not _panel_covers(upper, 277.719)

        assert _panel_covers(lower, 0.0)
        for value in [277.719, 150.236, 160.182, 178.797]:
            assert _panel_covers(lower, value)
        assert not _panel_covers(lower, 5577.081)
    finally:
        matplotlib.pyplot.close(figure)


def test_discrepante_baixo_fica_sozinho_no_painel_de_baixo():
    """Caso do R² muito negativo: espelho do anterior."""
    figure, panels, apply_limits = bar_panels(R2_MLP)
    try:
        assert len(panels) == 2
        upper, lower = panels
        apply_limits()

        assert _panel_covers(lower, -12.1141)
        assert not _panel_covers(lower, 0.0)

        assert _panel_covers(upper, 0.0)
        for value in [0.3470, 0.6467, 0.6233, 0.5796]:
            assert _panel_covers(upper, value)
        assert not _panel_covers(upper, -12.1141)
    finally:
        matplotlib.pyplot.close(figure)


# ── Espaço para rótulos ──────────────────────────────────────────────────────

def test_headroom_abre_espaco_acima():
    figure, axes = new_axes()
    try:
        axes.bar(range(3), [1.0, 2.0, 3.0])
        _, before = axes.get_ylim()
        headroom(axes, top=0.20)
        _, after = axes.get_ylim()
        assert after > before
    finally:
        matplotlib.pyplot.close(figure)


def test_rotulo_recortado_nao_estica_a_figura():
    """Rótulo fora dos limites do eixo, sem recorte, entra no cálculo do
    `bbox_inches="tight"` do savefig e estica a figura inteira para incluí-lo —
    foi assim que uma figura saiu com 13 mil pixels de altura."""
    values = [1.0, 2.0, 900.0]
    figure, axes = new_axes()
    try:
        bars = axes.bar(range(len(values)), values)
        axes.set_ylim(0, 3)
        value_labels(axes, bars, values, clip=True)

        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        axes_box = axes.get_window_extent(renderer)
        for text in axes.texts:
            box = text.get_window_extent(renderer)
            # Cada rótulo visível precisa caber na altura da área de dados.
            if box.y1 > axes_box.y1 + 1:
                assert not text.get_visible() or text.get_clip_on()
    finally:
        matplotlib.pyplot.close(figure)


def test_rotulo_acompanha_o_sinal_da_barra():
    """Rótulo de barra negativa vai abaixo dela, de positiva vai acima."""
    values = [5.0, -5.0]
    figure, axes = new_axes()
    try:
        bars = axes.bar(range(len(values)), values)
        value_labels(axes, bars, values)
        positive, negative = axes.texts
        assert positive.get_position()[1] > 5.0
        assert negative.get_position()[1] < -5.0
    finally:
        matplotlib.pyplot.close(figure)


# ── Execução direta ──────────────────────────────────────────────────────────

def main() -> int:
    testes = [(nome, funcao) for nome, funcao in sorted(globals().items())
              if nome.startswith("test_") and callable(funcao)]

    falhas = 0
    for nome, funcao in testes:
        try:
            funcao()
        except AssertionError as erro:
            falhas += 1
            print(f"FALHOU  {nome}\n        {erro or 'assert'}")
        except Exception as erro:  # noqa: BLE001 - relatar, não interromper
            falhas += 1
            print(f"ERRO    {nome}\n        {type(erro).__name__}: {erro}")
        else:
            print(f"ok      {nome}")

    print(f"\n{len(testes) - falhas}/{len(testes)} testes passaram")
    return 1 if falhas else 0


if __name__ == "__main__":
    raise SystemExit(main())
