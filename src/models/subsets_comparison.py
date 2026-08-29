"""
Árvore Aleatória vs Rede Neural nos cinco conjuntos de variáveis (C1–C5).

Lê os resultados gravados por `subsets_rf.py` e `subsets_mlp.py` e coloca os
dois modelos lado a lado em cada conjunto. Como os dois scripts dividem os dados
com a mesma semente, uma diferença nos gráficos daqui é diferença entre os
modelos — não entre as linhas que cada um recebeu por sorteio.

Execução:  python src/models/subsets_comparison.py
           python src/models/subsets_comparison.py --train   (treina antes)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import console, dependencies  # noqa: E402

dependencies.require("core")

import numpy as np  # noqa: E402

from core import paths, plots  # noqa: E402
from core.data import SUBSET_ORDER, SUBSETS  # noqa: E402
from core.plots import MODEL_COLORS, PALETTE, SUBSET_COLORS  # noqa: E402
from core.subset_report import MODEL_LABELS, load_results  # noqa: E402

MODELS = ["random_forest", "mlp"]

TRAINERS = {
    "random_forest": Path(__file__).with_name("subsets_rf.py"),
    "mlp": Path(__file__).with_name("subsets_mlp.py"),
}

OUTPUT_SUBFOLDERS = ("subsets", "comparison")


def _metric(payload: dict, subset_key: str, metric: str) -> float:
    return float(payload["subsets"][subset_key]["test"][metric])


def _available_subsets(loaded: dict[str, dict]) -> list[str]:
    """
    Conjuntos presentes em todos os modelos carregados, na ordem canônica.

    A proteção contra entrada vazia importa: `all()` sobre nenhum modelo é True
    para qualquer chave, o que reportaria os cinco conjuntos como disponíveis
    quando nada foi carregado.
    """
    if not loaded:
        return []
    return [
        key for key in SUBSET_ORDER
        if all(key in payload["subsets"] for payload in loaded.values())
    ]


# ── Gráficos ─────────────────────────────────────────────────────────────────

def chart_metric_by_model(loaded: dict[str, dict], keys: list[str],
                          metric: str, axis_label: str,
                          lower_is_better: bool) -> list[Path]:
    """
    Barras agrupadas: um grupo por conjunto, uma barra por modelo.

    Passa pelo mesmo `bar_panels` dos gráficos de um modelo só: quando a rede
    neural diverge em um conjunto, o MSE dela chega a 5577 contra 68–278 de
    todo o resto, e num eixo único as outras nove barras viram uma faixa
    rasteira onde não se compara mais nada.
    """
    positions = np.arange(len(keys))
    bar_width = 0.8 / len(loaded)
    all_values = [_metric(payload, key, metric)
                  for payload in loaded.values() for key in keys]

    figure, panels, apply_limits = plots.bar_panels(
        all_values, figsize=(12, 6), broken_figsize=(12, 7.5))
    broken = len(panels) > 1

    for axes in panels:
        for index, (model, payload) in enumerate(loaded.items()):
            offset = (index - (len(loaded) - 1) / 2) * bar_width
            values = [_metric(payload, key, metric) for key in keys]
            bars = axes.bar(positions + offset, values, width=bar_width * 0.9,
                            color=MODEL_COLORS[model], edgecolor="white",
                            alpha=0.9, label=MODEL_LABELS[model])
            plots.value_labels(axes, bars, values, fmt="{:.3f}",
                               fontsize=9.5, fontweight="normal", clip=broken)

        axes.set_xticks(positions)
        # Sigla no tick, nome completo no rodapé: rótulo longo e inclinado no
        # eixo x colide com o vizinho e come altura da área de dados.
        axes.set_xticklabels([key.upper() for key in keys], fontsize=10)
        if any(value < 0 for value in all_values):
            axes.axhline(0, lw=1.0, color=PALETTE["ink"], alpha=0.6)

    # Depois das barras, nunca antes — ver a nota em `plots.bar_panels`.
    apply_limits()

    if broken:
        figure.supylabel(axis_label, fontsize=11, x=0.03)
        lines_extra = ["Eixo interrompido: as alturas não são comparáveis "
                       "entre os dois painéis."]
    else:
        panels[0].set_ylabel(axis_label)
        lines_extra = []

    # Legenda dos modelos só uma vez, mesmo com dois painéis.
    plots.legend(panels[0], loc="upper right")

    winners = []
    for key in keys:
        scores = {model: _metric(payload, key, metric)
                  for model, payload in loaded.items()}
        winner = min(scores, key=scores.get) if lower_is_better else max(scores, key=scores.get)
        winners.append(f"{key.upper()}: {MODEL_LABELS[winner]}")

    lines = [f"{SUBSETS[key].name}: {SUBSETS[key].description}" for key in keys]
    lines.append("Vencedor por conjunto — " + " | ".join(winners))
    lines += lines_extra
    plots.reserve_bottom(figure, len(lines))
    plots.caption(figure, lines)

    return plots.save(figure, f"modelos_{metric}", *OUTPUT_SUBFOLDERS,
                      tight=False)


def chart_model_difference(loaded: dict[str, dict], keys: list[str]) -> list[Path]:
    """
    Diferença de MSE com sinal, por conjunto: negativo significa que a floresta
    venceu; positivo, que a rede venceu.

    Um gráfico só que responde "a complexidade extra da rede neural compra
    alguma coisa neste dataset?", conjunto a conjunto.
    """
    if not {"random_forest", "mlp"} <= loaded.keys():
        return []

    figure, axes = plots.new_axes(figsize=(11, 5.5))

    differences = [
        _metric(loaded["random_forest"], key, "mse") - _metric(loaded["mlp"], key, "mse")
        for key in keys
    ]
    colors = [MODEL_COLORS["random_forest"] if value < 0 else MODEL_COLORS["mlp"]
              for value in differences]

    bars = axes.bar(range(len(keys)), differences, color=colors,
                    edgecolor="white", alpha=0.9)
    axes.axhline(0, lw=1.4, color=PALETTE["ink"], linestyle="--",
                 label="Empate entre os modelos")

    span = max(abs(value) for value in differences) or 1.0
    plots.value_labels(axes, bars, differences, fmt="{:+.3f}")

    axes.set_xticks(range(len(keys)))
    axes.set_xticklabels([key.upper() for key in keys], fontsize=10)
    axes.set_ylabel("MSE(Árvore Aleatória) − MSE(Rede Neural)")

    # Folga acima e abaixo para os rótulos de valor, que ficam do lado de fora
    # da barra e encostariam na borda do eixo.
    axes.set_ylim(min(differences) - span * 0.25, max(differences) + span * 0.32)
    plots.legend(axes, loc="upper right")

    lines = [f"{SUBSETS[key].name}: {SUBSETS[key].description}" for key in keys]
    lines.append("Barra negativa: Árvore Aleatória tem menor erro   |   "
                 "Barra positiva: Rede Neural tem menor erro")
    plots.reserve_bottom(figure, len(lines))
    plots.caption(figure, lines)

    return plots.save(figure, "modelos_diferenca_mse", *OUTPUT_SUBFOLDERS,
                      tight=False)


def chart_paired_predictions(loaded: dict[str, dict],
                             arrays: dict[str, dict[str, np.ndarray]],
                             keys: list[str]) -> list[Path]:
    """Um painel por conjunto, com as previsões dos dois modelos no mesmo eixo."""
    columns = min(len(keys), 3)
    rows = int(np.ceil(len(keys) / columns))
    figure, grid = plots.new_grid(rows, columns, figsize=(5.2 * columns, 4.8 * rows))
    panels = np.atleast_1d(grid).ravel()

    for panel, key in zip(panels, keys):
        stacked = []
        for model in loaded:
            stacked.append(arrays[model][f"{key}_y_test"])
            stacked.append(arrays[model][f"{key}_pred_test"])
        every_value = np.concatenate(stacked)
        limits = [float(every_value.min() * 0.95), float(every_value.max() * 1.05)]

        for model, payload in loaded.items():
            panel.scatter(arrays[model][f"{key}_y_test"],
                          arrays[model][f"{key}_pred_test"],
                          alpha=0.55, s=34, color=MODEL_COLORS[model],
                          edgecolors="white", linewidths=0.3,
                          label=f"{MODEL_LABELS[model]} — R² "
                                f"{_metric(payload, key, 'r2'):.3f}")

        panel.plot(limits, limits, "--", lw=1.3, color=PALETTE["orange"],
                   label="Previsão ideal (1:1)")
        panel.set_xlim(limits)
        panel.set_ylim(limits)
        panel.set_xlabel("CBR medido (%)")
        panel.set_ylabel("CBR previsto (%)")
        # Identificação do painel fora da área de dados, acima do eixo: dentro
        # da legenda ela empurrava a caixa sobre a nuvem de pontos.
        panel.text(0.0, 1.02, SUBSETS[key].name, transform=panel.transAxes,
                   fontsize=10, fontweight="bold", color=SUBSET_COLORS[key],
                   ha="left", va="bottom",
                   bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFFFFF", edgecolor="#E2E8F0", alpha=0.93))
        plots.legend(panel, loc="lower right")

    for unused in panels[len(keys):]:
        unused.axis("off")

    lines = [f"{SUBSETS[key].name}: {SUBSETS[key].description}" for key in keys]
    figure.subplots_adjust(bottom=0.08 + 0.030 * len(lines), hspace=0.38,
                           wspace=0.26, top=0.95)
    plots.caption(figure, lines)

    return plots.save(figure, "modelos_previsto_vs_real", *OUTPUT_SUBFOLDERS,
                      tight=False)


def chart_summary_table(loaded: dict[str, dict], keys: list[str]) -> list[Path]:
    """Tabela de resultados em forma de figura, pronta para a dissertação."""
    figure, axes = plots.new_axes(figsize=(11, 1.1 + 0.5 * len(keys) * len(loaded)))
    axes.axis("off")
    axes.grid(False)

    header = ["Conjunto", "Modelo", "MSE", "RMSE", "MAE", "R²"]
    body = []
    for key in keys:
        for model, payload in loaded.items():
            entry = payload["subsets"][key]["test"]
            body.append([
                SUBSETS[key].name,
                MODEL_LABELS[model],
                f"{entry['mse']:.4f}",
                f"{entry['rmse']:.4f}",
                f"{entry['mae']:.4f}",
                f"{entry['r2']:.4f}",
            ])

    table = axes.table(cellText=body, colLabels=header, cellLoc="center",
                       loc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor(PALETTE["grid"])
        if row == 0:
            cell.set_facecolor(PALETTE["blue"])
            cell.set_text_props(color="white", fontweight="bold")
        else:
            model_name = body[row - 1][1]
            tint = "#EFF6FF" if model_name == MODEL_LABELS["random_forest"] else "#FFF7ED"
            cell.set_facecolor(tint if column != 0 else "#F1F5F9")

    return plots.save(figure, "modelos_tabela_resumo", *OUTPUT_SUBFOLDERS)


# ── Orquestração ─────────────────────────────────────────────────────────────

def train_missing(models: list[str]) -> None:
    """Executa o script de treino de cada modelo sem resultados salvos."""
    for model in models:
        script = TRAINERS[model]
        console.section(f"Treinando {MODEL_LABELS[model]} — {script.name}")
        completed = subprocess.run([sys.executable, str(script)])
        if completed.returncode != 0:
            raise RuntimeError(
                f"'{script.name}' terminou com código {completed.returncode}. "
                f"A comparação precisa dos dois modelos treinados."
            )


@console.guard("Falha ao comparar os modelos por conjunto.")
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara Árvore Aleatória e Rede Neural nos conjuntos C1–C5."
    )
    parser.add_argument("--train", action="store_true",
                        help="Treina os dois modelos antes de comparar.")
    arguments = parser.parse_args()

    console.banner(
        "COMPARAÇÃO ENTRE MODELOS — CONJUNTOS C1–C5",
        "Árvore Aleatória vs Rede Neural sobre a mesma divisão de dados",
    )

    console.step(1, 3, "Localizando resultados salvos")
    if arguments.train:
        train_missing(MODELS)

    loaded: dict[str, dict] = {}
    arrays: dict[str, dict[str, np.ndarray]] = {}
    missing: list[str] = []

    for model in MODELS:
        found = load_results(model)
        if found is None:
            missing.append(model)
            continue
        loaded[model], arrays[model] = found
        console.ok(f"{MODEL_LABELS[model]} — resultados encontrados")

    if missing:
        if not arguments.train:
            console.warn(
                "Sem resultados para: "
                + ", ".join(MODEL_LABELS[model] for model in missing)
            )
            train_missing(missing)

        # Alcançado nos dois caminhos: com --train os modelos acabaram de ser
        # treinados e têm de ter produzido resultados; sem ele, foram treinados
        # logo acima. Um modelo que ainda falte aqui é falha dura, não algo a
        # contornar — seguir adiante reduziria em silêncio uma comparação de
        # dois modelos a um gráfico de um só.
        for model in missing:
            found = load_results(model)
            if found is None:
                raise RuntimeError(
                    f"'{TRAINERS[model].name}' rodou mas não gravou resultados em "
                    f"{paths.SUBSETS_DIR / model}."
                )
            loaded[model], arrays[model] = found

    keys = _available_subsets(loaded)
    if not keys:
        raise RuntimeError(
            "Nenhum conjunto em comum entre os modelos treinados. "
            "Execute subsets_rf.py e subsets_mlp.py novamente."
        )

    console.step(2, 3, f"Comparando {len(keys)} conjunto(s) entre {len(loaded)} modelo(s)")
    rows = []
    for key in keys:
        for model, payload in loaded.items():
            entry = payload["subsets"][key]["test"]
            rows.append([
                SUBSETS[key].name, MODEL_LABELS[model],
                f"{entry['mse']:.4f}", f"{entry['rmse']:.4f}",
                f"{entry['mae']:.4f}", f"{entry['r2']:.4f}",
            ])
    console.metrics_table(rows, headers=["Conjunto", "Modelo", "MSE", "RMSE", "MAE", "R²"])

    console.step(3, 3, "Gerando gráficos comparativos")
    written: list[Path] = []
    written += chart_metric_by_model(loaded, keys, "mse", "MSE", lower_is_better=True)
    written += chart_metric_by_model(loaded, keys, "r2", "R²", lower_is_better=False)
    written += chart_metric_by_model(loaded, keys, "mae", "MAE", lower_is_better=True)
    written += chart_model_difference(loaded, keys)
    written += chart_paired_predictions(loaded, arrays, keys)
    written += chart_summary_table(loaded, keys)

    best_model, best_key, best_mse = None, None, float("inf")
    for model, payload in loaded.items():
        for key in keys:
            value = _metric(payload, key, "mse")
            if value < best_mse:
                best_model, best_key, best_mse = model, key, value

    console.result_panel(
        "COMPARAÇÃO CONCLUÍDA",
        [
            f"Melhor combinação: {MODEL_LABELS[best_model]} em {SUBSETS[best_key].name}",
            f"MSE {best_mse:.4f} — {SUBSETS[best_key].description}",
            "",
            f"{len(written)} arquivo(s) em "
            f"{plots.relative(paths.FIGURES_DIR.joinpath(*OUTPUT_SUBFOLDERS))}",
        ],
    )


if __name__ == "__main__":
    main()
