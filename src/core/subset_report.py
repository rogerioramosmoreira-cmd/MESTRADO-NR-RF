"""
Relatórios compartilhados dos experimentos com os conjuntos C1–C5.

Tanto a Árvore Aleatória quanto a Rede Neural treinam os mesmos cinco conjuntos
e respondem à mesma pergunta — *qual grupo de variáveis do solo carrega o sinal
do CBR?* — então o contêiner de resultados, o formato de persistência e todos os
gráficos moram aqui uma vez só, e os dois scripts de modelo ficam enxutos.

Deliberadamente livre de imports do TensorFlow: o experimento da Árvore
Aleatória precisa rodar em uma máquina onde o TensorFlow não esteja instalado.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core import paths, plots
from core.data import SUBSET_ORDER, SUBSETS, Subset
from core.metrics import Scores
from core.plots import PALETTE, SUBSET_COLORS

# Nomes legíveis dos modelos, usados nas legendas.
MODEL_LABELS = {
    "random_forest": "Árvore Aleatória",
    "mlp": "Rede Neural",
}


@dataclass
class SubsetResult:
    """Tudo o que um par (modelo, conjunto) produziu."""

    model: str
    subset_key: str
    validation: Scores
    test: Scores
    y_validation: np.ndarray
    prediction_validation: np.ndarray
    y_test: np.ndarray
    prediction_test: np.ndarray
    parameters: dict = field(default_factory=dict)
    importances: np.ndarray | None = None
    history: dict | None = None

    @property
    def subset(self) -> Subset:
        return SUBSETS[self.subset_key]

    @property
    def residuals_test(self) -> np.ndarray:
        return self.y_test - self.prediction_test

    @property
    def residuals_validation(self) -> np.ndarray:
        return self.y_validation - self.prediction_validation


# ── Persistência ─────────────────────────────────────────────────────────────

def _model_dir(model: str) -> Path:
    directory = paths.SUBSETS_DIR / model
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_results(model: str, results: list[SubsetResult]) -> Path:
    """
    Grava as métricas em JSON e os vetores de previsão em um NPZ comprimido.

    O script de comparação entre modelos lê esses arquivos de volta em vez de
    retreinar, então `subsets_comparison.py` roda em segundos sobre o que foi
    treinado por último.
    """
    directory = _model_dir(model)

    payload = {
        "model": model,
        "subsets": {
            result.subset_key: {
                "name": result.subset.name,
                "description": result.subset.description,
                "n_features": len(result.subset),
                "features": list(result.subset.features),
                "validation": result.validation.as_dict(),
                "test": result.test.as_dict(),
                "parameters": _jsonable(result.parameters),
            }
            for result in results
        },
    }

    metrics_path = directory / "results.json"
    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                            encoding="utf-8")

    arrays: dict[str, np.ndarray] = {}
    for result in results:
        arrays[f"{result.subset_key}_y_test"] = result.y_test
        arrays[f"{result.subset_key}_pred_test"] = result.prediction_test
        arrays[f"{result.subset_key}_y_val"] = result.y_validation
        arrays[f"{result.subset_key}_pred_val"] = result.prediction_validation
        if result.importances is not None:
            arrays[f"{result.subset_key}_importances"] = result.importances
    np.savez_compressed(directory / "predictions.npz", **arrays)

    return metrics_path


def load_results(model: str) -> tuple[dict, dict[str, np.ndarray]] | None:
    """Lê de volta o que `save_results` gravou, ou None se nunca rodou."""
    directory = paths.SUBSETS_DIR / model
    metrics_path = directory / "results.json"
    arrays_path = directory / "predictions.npz"

    if not metrics_path.exists() or not arrays_path.exists():
        return None

    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        with np.load(arrays_path) as archive:
            arrays = {key: archive[key] for key in archive.files}
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        raise RuntimeError(
            f"Resultados de '{model}' estão corrompidos em '{directory}': {exc}\n"
            f"Execute o treinamento novamente para regravá-los."
        ) from exc

    return payload, arrays


def _jsonable(value):
    """Converte escalares/arrays do numpy para o `json.dumps` não engasgar."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


# ── Gráficos por conjunto ────────────────────────────────────────────────────

def chart_predicted_vs_actual(result: SubsetResult) -> list[Path]:
    """Dispersão do CBR previsto contra o medido, com a reta ideal 1:1."""
    figure, axes = plots.new_axes(figsize=(7, 6))
    color = SUBSET_COLORS[result.subset_key]

    limits = _shared_limits(result.y_test, result.prediction_test)
    axes.scatter(result.y_test, result.prediction_test, alpha=0.65, s=50,
                 color=color, edgecolors="white", linewidths=0.4,
                 label=f"{result.subset.name} — teste")
    axes.plot(limits, limits, "--", lw=1.5, color=PALETTE["orange"],
              label="Previsão ideal (1:1)")
    axes.set_xlim(limits)
    axes.set_ylim(limits)
    axes.set_xlabel("CBR medido (%)")
    axes.set_ylabel("CBR previsto (%)")

    plots.legend(axes, loc="upper left")

    # Métricas no rodapé: dentro dos eixos, este canto é justamente onde caem os
    # pontos de um modelo que superestima, e a caixa passava por cima deles.
    plots.reserve_bottom(figure, 2)
    plots.caption(figure, [
        f"{result.subset.name}: {result.subset.description} "
        f"({len(result.subset)} variável(is) de entrada)",
        f"R² = {result.test.r2:.4f}   |   MSE = {result.test.mse:.4f}   |   "
        f"RMSE = {result.test.rmse:.4f}   |   MAE = {result.test.mae:.4f}",
    ])

    return plots.save(figure, "previsto_vs_real",
                      "subsets", result.model, result.subset_key, tight=False)


def chart_residuals(result: SubsetResult) -> list[Path]:
    """Dispersão dos resíduos contra a previsão, com a referência de erro zero."""
    figure, axes = plots.new_axes(figsize=(8, 5))
    color = SUBSET_COLORS[result.subset_key]
    residuals = result.residuals_test

    axes.scatter(result.prediction_test, residuals, alpha=0.65, s=50,
                 color=color, edgecolors="white", linewidths=0.4,
                 label=f"{result.subset.name} — teste")
    axes.axhline(0, lw=1.5, linestyle="--", color=PALETTE["orange"],
                 label="Erro zero")
    axes.set_xlabel("CBR previsto (%)")
    axes.set_ylabel("Resíduo (medido − previsto)")

    plots.legend(axes, loc="upper right")

    plots.reserve_bottom(figure, 2)
    plots.caption(figure, [
        f"{result.subset.name}: {result.subset.description}",
        f"Desvio padrão = {np.std(residuals):.3f}   |   "
        f"Viés médio = {np.mean(residuals):+.3f}",
    ])

    return plots.save(figure, "residuos", "subsets", result.model,
                      result.subset_key, tight=False)


def chart_feature_importance(result: SubsetResult) -> list[Path] | None:
    """Importância Gini por variável de entrada. Só para a Árvore Aleatória."""
    if result.importances is None:
        return None

    figure, axes = plots.new_axes(figsize=(10, max(3.0, 0.55 * len(result.subset))))
    order = np.argsort(result.importances)[::-1]
    labels = [result.subset.labels[index] for index in order]
    values = result.importances[order]

    shades = plots.plt.cm.Blues(np.linspace(0.9, 0.4, len(values)))
    axes.barh(range(len(values)), values, color=shades, edgecolor="white",
              label=f"{result.subset.name} — importância Gini")
    axes.set_yticks(range(len(values)))
    axes.set_yticklabels(labels, fontsize=10)
    axes.invert_yaxis()
    axes.set_xlabel("Importância (Gini)")
    # Folga à direita: a barra mais longa encostava na borda e o valor colado
    # nela ficava sem espaço.
    axes.set_xlim(0, float(values.max()) * 1.18)
    plots.legend(axes, loc="lower right")

    plots.reserve_bottom(figure, 2)
    plots.caption(figure, [
        f"{result.subset.name}: {result.subset.description}",
        f"Mais influente: {labels[0]} ({values[0]:.3f})",
    ])

    return plots.save(figure, "importancia_features",
                      "subsets", result.model, result.subset_key, tight=False)


def chart_learning_curve(result: SubsetResult) -> list[Path] | None:
    """Perda de treino e de validação por época. Só para a Rede Neural."""
    if not result.history:
        return None

    loss = np.asarray(result.history.get("loss", []), dtype=float)
    val_loss = np.asarray(result.history.get("val_loss", []), dtype=float)
    if loss.size == 0:
        return None

    figure, axes = plots.new_axes(figsize=(10, 5))
    epochs = np.arange(1, loss.size + 1)
    color = SUBSET_COLORS[result.subset_key]

    axes.plot(epochs, loss, lw=1.6, color=color,
              label=f"{result.subset.name} — treino")
    if val_loss.size:
        axes.plot(epochs, val_loss, lw=1.6, linestyle="--", color=PALETTE["orange"],
                  label=f"{result.subset.name} — validação")
        best = int(np.argmin(val_loss))
        axes.scatter([best + 1], [val_loss[best]], s=80, zorder=5,
                     color=PALETTE["green"],
                     label=f"Melhor época: {best + 1} (MAE {val_loss[best]:.4f})")

    axes.set_xlabel("Época")
    axes.set_ylabel("MAE (escala de treino)")
    plots.legend(axes, loc="upper right")

    return plots.save(figure, "curva_aprendizado",
                      "subsets", result.model, result.subset_key)


# ── Gráficos entre conjuntos (um modelo, cinco conjuntos) ────────────────────

def chart_metric_by_subset(results: list[SubsetResult], metric: str,
                           axis_label: str, lower_is_better: bool) -> list[Path]:
    """
    Barras de uma métrica nos cinco conjuntos, com o melhor destacado.

    Duas coisas que este gráfico precisa resolver e antes não resolvia:

    1. A descrição dos cinco conjuntos ia para a legenda, dentro dos eixos, e a
       caixa resultante cobria as barras. Agora vai para o rodapé da figura.
    2. Uma rede neural que diverge em um conjunto produz R² = −12 contra 0,6 dos
       outros. Num eixo linear único, esse valor achata os outros quatro contra
       o zero; quando a dispersão é essa, o eixo é quebrado.
    """
    ordered = _ordered(results)
    values = [float(getattr(result.test, metric)) for result in ordered]
    best = int(np.argmin(values) if lower_is_better else np.argmax(values))
    colors = [SUBSET_COLORS[result.subset_key] for result in ordered]
    # Só a sigla nos ticks; o nome completo está no rodapé. Rótulo longo e
    # inclinado no eixo x é o outro jeito de dois textos se atropelarem.
    ticks = [result.subset.key.upper() for result in ordered]

    figure, panels, apply_limits = plots.bar_panels(values)
    broken = len(panels) > 1

    for axes in panels:
        bars = axes.bar(range(len(ordered)), values, color=colors,
                        edgecolor="white", alpha=0.9)
        bars[best].set_edgecolor(PALETTE["ink"])
        bars[best].set_linewidth(2.0)
        plots.value_labels(axes, bars, values, clip=broken)
        axes.set_xticks(range(len(ordered)))
        axes.set_xticklabels(ticks, fontsize=10)

    # Depois das barras, nunca antes — ver a nota em `plots.bar_panels`.
    apply_limits()

    if broken:
        figure.supylabel(axis_label, fontsize=11, x=0.03)
    else:
        panels[0].set_ylabel(axis_label)

    lines = [f"{result.subset.name}: {result.subset.description}"
             for result in ordered]
    lines.append(f"Melhor: {ordered[best].subset.name} "
                 f"({axis_label} = {values[best]:.4f})")
    if broken:
        lines.append("Eixo interrompido: as alturas não são comparáveis entre "
                     "os dois painéis.")

    plots.reserve_bottom(figure, len(lines))
    plots.caption(figure, lines)

    model = ordered[0].model
    return plots.save(figure, f"comparativo_{metric}", "subsets", model,
                      tight=False)


def chart_all_metrics(results: list[SubsetResult]) -> list[Path]:
    """
    Uma métrica por painel, cada uma com a sua escala.

    Antes MSE, RMSE e MAE dividiam um eixo. Como o MSE é quadrático, ele chega
    a milhares enquanto RMSE e MAE ficam nas dezenas — e as barras dessas duas
    sumiam contra o chão do gráfico. Não era escolha de estilo: duas das quatro
    métricas eram ilegíveis.

    Com um painel por métrica, cada uma ocupa a própria faixa e as quatro ficam
    comparáveis entre conjuntos, que é a pergunta que o gráfico responde.
    """
    ordered = _ordered(results)
    metrics = [
        ("mse", "MSE — menor é melhor"),
        ("rmse", "RMSE (CBR %) — menor é melhor"),
        ("mae", "MAE (CBR %) — menor é melhor"),
        ("r2", "R² — maior é melhor"),
    ]

    figure, grid = plots.new_grid(2, 2, figsize=(13, 8.5))
    panels = np.atleast_1d(grid).ravel()
    ticks = [result.subset.key.upper() for result in ordered]
    colors = [SUBSET_COLORS[result.subset_key] for result in ordered]

    for axes, (metric, axis_label) in zip(panels, metrics):
        values = [float(getattr(result.test, metric)) for result in ordered]
        bars = axes.bar(range(len(ordered)), values, color=colors,
                        edgecolor="white", alpha=0.9)

        # Rótulo por barra tornaria o painel pequeno ilegível; o valor exato
        # está na tabela-resumo. Aqui interessa a forma da comparação.
        axes.set_xticks(range(len(ordered)))
        axes.set_xticklabels(ticks, fontsize=10)
        axes.set_ylabel(axis_label, fontsize=10)
        axes.tick_params(labelsize=8)

        if min(values) < 0:
            axes.axhline(0, lw=1.0, color=PALETTE["ink"], alpha=0.6)
        plots.headroom(axes, top=0.12, bottom=0.08 if min(values) < 0 else 0.0)

        # Uma série por conjunto, registrada uma vez só para a legenda comum.
        if metric == metrics[0][0]:
            for bar, result in zip(bars, ordered):
                bar.set_label(f"{result.subset.key.upper()} — "
                              f"{result.subset.description} "
                              f"({len(result.subset)} var.)")

    lines = [f"{result.subset.name}: {result.subset.description}"
             for result in ordered]
    figure.subplots_adjust(bottom=0.10 + 0.028 * len(lines), hspace=0.35,
                           wspace=0.28)
    plots.caption(figure, lines)

    return plots.save(figure, "comparativo_todas_metricas",
                      "subsets", ordered[0].model, tight=False)


def chart_residual_spread(results: list[SubsetResult]) -> list[Path]:
    """Boxplot dos resíduos de teste — qual conjunto espalha menos o erro."""
    ordered = _ordered(results)
    figure, axes = plots.new_axes(figsize=(11, 5.5))

    data = [result.residuals_test for result in ordered]
    box = axes.boxplot(data, patch_artist=True, widths=0.55,
                       medianprops={"color": PALETTE["ink"], "linewidth": 2})
    for patch, result in zip(box["boxes"], ordered):
        patch.set_facecolor(SUBSET_COLORS[result.subset_key])
        patch.set_alpha(0.45)
        patch.set_edgecolor(SUBSET_COLORS[result.subset_key])

    axes.axhline(0, lw=1.4, linestyle="--", color=PALETTE["orange"],
                 label="Erro zero")
    axes.set_xticks(range(1, len(ordered) + 1))
    axes.set_xticklabels([result.subset.key.upper() for result in ordered],
                         fontsize=10)
    axes.set_ylabel("Resíduo (medido − previsto)")
    plots.legend(axes, loc="upper right")

    # O desvio padrão de cada conjunto é o número que se lê deste gráfico, mas
    # cinco linhas dele na legenda cobriam justamente as caixas que descrevem.
    lines = [f"{result.subset.name}: σ = {np.std(result.residuals_test):.3f} "
             f"— {result.subset.description}" for result in ordered]
    plots.reserve_bottom(figure, len(lines))
    plots.caption(figure, lines)

    return plots.save(figure, "comparativo_residuos", "subsets",
                      ordered[0].model, tight=False)


def chart_overlaid_predictions(results: list[SubsetResult]) -> list[Path]:
    """
    Os cinco conjuntos sobrepostos no mesmo plano previsto × medido.

    A legenda vai para baixo da figura. Num gráfico de dispersão não existe
    canto garantidamente livre — a nuvem muda de forma a cada treino, e um
    `loc` fixo escolhido na mão acerta hoje e cobre pontos amanhã.
    """
    ordered = _ordered(results)
    figure, axes = plots.new_axes(figsize=(8.5, 8))

    every_value = np.concatenate(
        [np.concatenate([r.y_test, r.prediction_test]) for r in ordered]
    )
    limits = [every_value.min() * 0.95, every_value.max() * 1.05]

    for result in ordered:
        axes.scatter(result.y_test, result.prediction_test, alpha=0.55, s=38,
                     color=SUBSET_COLORS[result.subset_key],
                     edgecolors="white", linewidths=0.3,
                     label=f"{result.subset.key.upper()} — "
                           f"R² {result.test.r2:.3f}")

    axes.plot(limits, limits, "--", lw=1.5, color=PALETTE["orange"],
              label="Previsão ideal (1:1)")
    axes.set_xlim(limits)
    axes.set_ylim(limits)
    axes.set_xlabel("CBR medido (%)")
    axes.set_ylabel("CBR previsto (%)")

    # Rodapé empilhado, de baixo para cima: descrição dos conjuntos, legenda das
    # séries, e só então a área de dados. Cada faixa tem a sua altura reservada.
    lines = [f"{result.subset.name}: {result.subset.description}"
             for result in ordered]
    caption_height = 0.022 * len(lines)
    figure.subplots_adjust(bottom=caption_height + 0.16)
    plots.caption(figure, lines)
    plots.legend_below(figure, axes, ncol=3, y=caption_height + 0.03)

    return plots.save(figure, "comparativo_previsto_vs_real",
                      "subsets", ordered[0].model, tight=False)


def chart_subset_ranking(results: list[SubsetResult]) -> list[Path]:
    """
    Ordena os conjuntos pelo MSE de teste e anota quanto cada um custa em
    número de variáveis de entrada — o trade-off prático para um laboratório
    decidir quais ensaios realmente executar em uma amostra de solo.
    """
    ordered = sorted(_ordered(results), key=lambda result: result.test.mse)
    figure, axes = plots.new_axes(figsize=(11, 5.5))

    labels = [result.subset.name for result in ordered]
    values = [result.test.mse for result in ordered]
    colors = [SUBSET_COLORS[result.subset_key] for result in ordered]

    bars = axes.barh(range(len(ordered)), values, height=0.62, color=colors,
                     edgecolor="white", alpha=0.9)
    axes.set_yticks(range(len(ordered)))
    axes.set_yticklabels(labels, fontsize=10)
    axes.invert_yaxis()

    # Um conjunto em que o modelo diverge tem MSE duas ordens de grandeza acima
    # dos demais, e em escala linear as outras quatro barras encostam no eixo.
    # Aqui a escala logarítmica serve bem: o que este gráfico comunica é a
    # ordem — que a escala preserva — e o valor exato, que está escrito na
    # barra. O MSE é sempre positivo, então o log é sempre definido.
    logarithmic = plots.find_scale_break(values) is not None
    if logarithmic:
        axes.set_xscale("log")
        axes.set_xlim(min(values) * 0.55, max(values) * 4.0)
    else:
        # Folga à direita para os rótulos de valor, que de outro modo seriam
        # cortados pela borda do eixo na barra mais longa.
        axes.set_xlim(0, max(values) * 1.32)

    axes.set_xlabel("MSE no conjunto de teste (menor é melhor)"
                    + (" — escala logarítmica" if logarithmic else ""))

    for bar, result in zip(bars, ordered):
        # Em escala logarítmica o afastamento precisa ser multiplicativo: uma
        # folga fixa que funciona na barra longa some sob a barra curta.
        width = bar.get_width()
        offset = width * 0.10 if logarithmic else max(values) * 0.015
        axes.text(width + offset, bar.get_y() + bar.get_height() / 2,
                  f"{result.test.mse:.4f}  ({len(result.subset)} var.)",
                  va="center", fontsize=9.5, fontweight="bold")

    lines = [f"Melhor conjunto: {ordered[0].subset.name} "
             f"— {ordered[0].subset.description}",
             f"Pior conjunto: {ordered[-1].subset.name} "
             f"— {ordered[-1].subset.description}"]
    if logarithmic:
        lines.append("Escala logarítmica: os comprimentos não são "
                     "proporcionais entre si.")
    plots.reserve_bottom(figure, len(lines))
    plots.caption(figure, lines)

    return plots.save(figure, "ranking_conjuntos", "subsets",
                      ordered[0].model, tight=False)


def render_model_charts(results: list[SubsetResult]) -> list[Path]:
    """Produz todos os gráficos de um modelo nos seus cinco conjuntos."""
    written: list[Path] = []

    for result in results:
        written += chart_predicted_vs_actual(result)
        written += chart_residuals(result)
        written += chart_feature_importance(result) or []
        written += chart_learning_curve(result) or []

    written += chart_metric_by_subset(results, "mse", "MSE", lower_is_better=True)
    written += chart_metric_by_subset(results, "r2", "R²", lower_is_better=False)
    written += chart_metric_by_subset(results, "mae", "MAE", lower_is_better=True)
    written += chart_all_metrics(results)
    written += chart_residual_spread(results)
    written += chart_overlaid_predictions(results)
    written += chart_subset_ranking(results)

    return written


def _ordered(results: list[SubsetResult]) -> list[SubsetResult]:
    """Coloca os resultados na ordem canônica C1..C5, venham como vierem."""
    by_key = {result.subset_key: result for result in results}
    return [by_key[key] for key in SUBSET_ORDER if key in by_key]


def _shared_limits(*arrays) -> list[float]:
    """Limites de eixo comuns, para que a reta 1:1 fique de fato a 45 graus."""
    stacked = np.concatenate([np.asarray(array, dtype=float).ravel() for array in arrays])
    return [float(stacked.min() * 0.95), float(stacked.max() * 1.05)]
