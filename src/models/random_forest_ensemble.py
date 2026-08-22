"""
Ensemble de Árvore Aleatória — VotingRegressor (RF + GradientBoosting + ExtraTrees).

Recuperado do histórico do projeto (commit 16fb7b8), onde havia sido substituído
pelo script de floresta única. Volta como módulo próprio, e não como
substituição, para que as duas abordagens fiquem disponíveis para trabalhos
futuros.

Por que três estimadores e não um:
  - RandomForest      — faz a média de árvores profundas treinadas em amostras
                        bootstrap; variância baixa, mas todas as árvores usam o
                        mesmo critério de corte
  - GradientBoosting  — cada árvore corrige o resíduo da anterior; captura
                        estrutura que a floresta dilui na média, ao custo de
                        risco de sobreajuste
  - ExtraTrees        — corta em limiares aleatórios; descorrelaciona os erros
                        que RF e GB compartilham

Cada um é ajustado separadamente e depois combinado pelo `VotingRegressor`, que
faz a média das previsões. A média ajuda exatamente quando os erros dos membros
não são correlacionados, que é a razão de escolher três estratégias de árvore
diferentes.

Estratégias herdadas do original:
  1. Engenharia de features — 8 razões e índices granulométricos derivados
  2. Busca aleatória independente por estimador
  3. Transformação logarítmica no alvo (log1p / expm1)
  4. Pesos amostrais para as amostras raras de CBR alto
  5. Retreino em treino + validação após o ajuste

Execução:  python src/models/random_forest_ensemble.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import console, dependencies  # noqa: E402

dependencies.require("core")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import dump  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split  # noqa: E402
from sklearn.preprocessing import MinMaxScaler  # noqa: E402

from core import (  # noqa: E402
    data, metrics, paths, plots, progress, runtime, scoreboard,
)
from core.plots import PALETTE  # noqa: E402

SEED = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.15

# Busca herdada do código original: 150 iterações e 10 dobras, por estimador —
# 4500 ajustes no total, perto de uma hora de execução.
# O segundo valor vale apenas com MLL_FAST=1 — ver core/runtime.py.
SEARCH_ITERATIONS = runtime.budget(150, 15)
CV_FOLDS = runtime.budget(10, 3)

# ATENÇÃO — a meta de 0.780 vem do código original e está em escala ambígua.
# As métricas abaixo são calculadas na escala real do CBR (%), onde o MSE
# medido fica na casa das dezenas; 0.780 só é atingível na escala log1p, que é
# onde a busca de hiperparâmetros reporta seu MSE de validação cruzada.
# Comparar as duas faz a execução anunciar "meta não atingida" sempre.
# Confirme qual escala a dissertação adota antes de citar este número.
MSE_GOAL = 0.780
MSE_GOAL_SCALE = "log1p"  # "log1p" ou "cbr" — define contra o que a meta é medida

LOG_TARGET = True
WEIGHT_THRESHOLD = 25.0
WEIGHT_RARE = 3.0
WEIGHT_COMMON = 1.0

# Protege todos os denominadores das razões abaixo. As colunas de peneira são
# porcentagens que chegam legitimamente a 0.0 em um solo grosso, então a divisão
# não é segura sem isso.
EPSILON = 1e-6

FIGURE_FOLDER = ("random_forest_ensemble",)

ESTIMATOR_LABELS = {
    "rf": "Random Forest",
    "gb": "Gradient Boosting",
    "et": "Extra Trees",
}

ESTIMATOR_COLORS = {
    "rf": PALETTE["blue"],
    "gb": PALETTE["orange"],
    "et": PALETTE["blue_light"],
    "ensemble": PALETTE["purple"],
}

SEARCH_SPACES = {
    "rf": {
        "n_estimators": list(range(100, 801, 50)),
        "max_depth": list(range(3, 31, 2)) + [None],
        "min_samples_split": list(range(2, 15)),
        "min_samples_leaf": list(range(1, 10)),
        "max_features": ["sqrt", "log2"] + np.linspace(0.1, 1.0, 10).round(2).tolist(),
        "max_samples": np.linspace(0.5, 1.0, 6).round(2).tolist(),
    },
    "gb": {
        "n_estimators": list(range(100, 601, 50)),
        "max_depth": list(range(2, 9)),
        "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2],
        "min_samples_split": list(range(2, 10)),
        "min_samples_leaf": list(range(1, 8)),
        "subsample": np.linspace(0.6, 1.0, 5).round(2).tolist(),
        "max_features": ["sqrt", "log2", None],
    },
    "et": {
        "n_estimators": list(range(100, 601, 50)),
        "max_depth": list(range(3, 31, 2)) + [None],
        "min_samples_split": list(range(2, 15)),
        "min_samples_leaf": list(range(1, 10)),
        "max_features": ["sqrt", "log2"] + np.linspace(0.1, 1.0, 10).round(2).tolist(),
    },
}


def build_estimator(key: str, parameters: dict):
    """Instancia um estimador ajustado a partir da sua chave."""
    if key == "rf":
        return RandomForestRegressor(**parameters, random_state=SEED, n_jobs=-1)
    if key == "gb":
        return GradientBoostingRegressor(**parameters, random_state=SEED)
    if key == "et":
        return ExtraTreesRegressor(**parameters, random_state=SEED, n_jobs=-1)
    raise KeyError(f"Estimador desconhecido: {key!r}")


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Acrescenta oito variáveis derivadas às dez medidas.

      ratio_*      — cada peneira dividida pela imediatamente superior. Descreve
                     o *formato* da curva granulométrica de forma independente
                     das porcentagens absolutas, que é o que distingue um solo
                     bem graduado de um uniforme.
      atividade    — LL − IP, ou seja, o limite de plasticidade. Quanto maior,
                     mais larga a faixa plástica e, em geral, menor a capacidade
                     de suporte.
      compacidade  — densidade seca máxima sobre umidade ótima. Uma medida
                     indireta de quanta energia de compactação o solo converte
                     em densidade.
      finos_sq     — a fração de finos ao quadrado, amplificando a diferença
                     entre solos argilosos e granulares.
    """
    frame = frame.copy()

    frame["ratio_9_25"] = frame["9.5mm"] / (frame["25.4mm"] + EPSILON)
    frame["ratio_4_9"] = frame["4.8mm"] / (frame["9.5mm"] + EPSILON)
    frame["ratio_2_4"] = frame["2.0mm"] / (frame["4.8mm"] + EPSILON)
    frame["ratio_042_2"] = frame["0.42mm"] / (frame["2.0mm"] + EPSILON)
    frame["ratio_076_042"] = frame["0.076mm"] / (frame["0.42mm"] + EPSILON)

    frame["atividade"] = frame["LL"] - frame["IP"]
    frame["compacidade"] = frame["Densidade máxima"] / (frame["Umidade Ótima"] + EPSILON)
    frame["finos_sq"] = frame["0.076mm"] ** 2

    return frame


def sample_weights(target: np.ndarray) -> np.ndarray:
    """Dá às amostras raras de CBR alto mais peso que às comuns."""
    threshold = np.log1p(WEIGHT_THRESHOLD) if LOG_TARGET else WEIGHT_THRESHOLD
    return np.where(target > threshold, WEIGHT_RARE, WEIGHT_COMMON).astype(np.float32)


# ── Gráficos ─────────────────────────────────────────────────────────────────

def chart_predicted_vs_actual(y_true, prediction, scores, split: str, color: str):
    figure, axes = plots.new_axes(figsize=(7, 6))
    stacked = np.concatenate([y_true, prediction])
    limits = [float(stacked.min() * 0.95), float(stacked.max() * 1.05)]

    axes.scatter(y_true, prediction, alpha=0.65, s=50, color=color,
                 edgecolors="white", linewidths=0.4,
                 label=f"Ensemble — {split}")
    axes.plot(limits, limits, "--", lw=1.5, color=PALETTE["orange"],
              label="Previsão ideal (1:1)")
    axes.set_xlim(limits)
    axes.set_ylim(limits)
    axes.set_xlabel("CBR medido (%)")
    axes.set_ylabel("CBR previsto (%)")

    plots.note(axes, f"R² = {scores.r2:.4f}", color)
    plots.note(axes, f"MSE = {scores.mse:.4f}", color)
    plots.legend(axes, loc="upper left")

    return plots.save(figure, f"previsto_vs_real_{split}", *FIGURE_FOLDER)


def chart_residuals(y_true, prediction, split: str, color: str):
    figure, axes = plots.new_axes(figsize=(8, 5))
    residuals = y_true - prediction

    axes.scatter(prediction, residuals, alpha=0.65, s=50, color=color,
                 edgecolors="white", linewidths=0.4, label=f"Ensemble — {split}")
    axes.axhline(0, lw=1.5, linestyle="--", color=PALETTE["orange"], label="Erro zero")
    axes.set_xlabel("CBR previsto (%)")
    axes.set_ylabel("Resíduo (medido − previsto)")

    plots.note(axes, f"Desvio padrão = {np.std(residuals):.3f}", color)
    plots.note(axes, f"Viés médio = {np.mean(residuals):+.3f}", color)
    plots.legend(axes, loc="upper right")

    return plots.save(figure, f"residuos_{split}", *FIGURE_FOLDER)


def chart_member_comparison(member_scores: dict, ensemble_scores):
    """
    Barras do MSE de teste de cada membro do ensemble contra o do próprio
    ensemble.

    É este gráfico que justifica o ensemble existir: se a barra combinada não
    ficar abaixo de todos os membros, a média não está comprando nada.
    """
    figure, axes = plots.new_axes(figsize=(8, 5.5))

    keys = list(member_scores) + ["ensemble"]
    labels = [ESTIMATOR_LABELS.get(key, "Ensemble") for key in keys]
    values = [member_scores[key].mse for key in member_scores] + [ensemble_scores.mse]

    bars = axes.bar(labels, values, width=0.55, edgecolor="white",
                    color=[ESTIMATOR_COLORS[key] for key in keys])

    # Sem linha de meta aqui: este eixo está na escala real do CBR e MSE_GOAL
    # está em log1p. Desenhá-la colocaria a referência colada no zero, dando a
    # impressão de que todos os modelos falharam por ordens de grandeza.

    for bar, value in zip(bars, values):
        axes.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
                  f"{value:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    axes.set_ylabel("MSE no conjunto de teste")

    # Todas as barras têm alturas parecidas, então não existe canto vazio para a
    # legenda. Abrir uma faixa de 30% acima da barra mais alta cria esse espaço
    # sem truncar o eixo — truncá-lo exageraria diferenças de poucas unidades.
    axes.set_ylim(0, max(values) * 1.30)

    best = min(range(len(values)), key=lambda index: values[index])
    plots.note(axes, f"Menor MSE: {labels[best]} ({values[best]:.4f})", PALETTE["ink"])
    plots.note(axes,
               "Ensemble venceu" if best == len(values) - 1
               else f"Ensemble não superou {labels[best]}",
               ESTIMATOR_COLORS["ensemble"])
    plots.legend(axes, loc="upper center", fontsize=8)

    return plots.save(figure, "comparativo_membros_mse", *FIGURE_FOLDER)


def chart_feature_importance(forest, extra_trees, feature_names, top_n: int = 15):
    """
    Importância Gini média entre os dois membros que fazem média de árvores.

    O Gradient Boosting fica de fora de propósito: suas importâncias vêm de um
    processo sequencial de ajuste de resíduos e não estão em pé de igualdade com
    as dos membros por bagging, então a média dos três misturaria dois
    significados diferentes.
    """
    figure, axes = plots.new_axes(figsize=(11, 6))

    mean_importance = (forest.feature_importances_ + extra_trees.feature_importances_) / 2
    count = min(top_n, len(feature_names))
    order = np.argsort(mean_importance)[-count:][::-1]

    shades = plots.plt.cm.Blues(np.linspace(0.9, 0.4, count))
    axes.barh(range(count), mean_importance[order], color=shades, edgecolor="white",
              label="Importância Gini média (RF + ExtraTrees)")
    axes.set_yticks(range(count))
    axes.set_yticklabels([feature_names[index] for index in order], fontsize=9)
    axes.invert_yaxis()
    axes.set_xlabel("Importância (Gini)")

    plots.note(axes,
               f"Mais influente: {feature_names[order[0]]} "
               f"({mean_importance[order[0]]:.3f})", PALETTE["ink"])
    plots.legend(axes, loc="lower right", fontsize=8)

    return plots.save(figure, "importancia_features", *FIGURE_FOLDER)


def chart_metrics_table(validation_scores, test_scores):
    figure, axes = plots.new_axes(figsize=(7, 3.4))
    axes.axis("off")
    axes.grid(False)

    header = ["Métrica", "Validação", "Teste"]
    body = [
        ["MSE", f"{validation_scores.mse:.4f}", f"{test_scores.mse:.4f}"],
        ["RMSE", f"{validation_scores.rmse:.4f}", f"{test_scores.rmse:.4f}"],
        ["MAE", f"{validation_scores.mae:.4f}", f"{test_scores.mae:.4f}"],
        ["MAPE", f"{validation_scores.mape:.2f}%", f"{test_scores.mape:.2f}%"],
        ["R²", f"{validation_scores.r2:.4f}", f"{test_scores.r2:.4f}"],
    ]

    table = axes.table(cellText=body, colLabels=header, cellLoc="center",
                       loc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor(PALETTE["grid"])
        if row == 0:
            cell.set_facecolor(PALETTE["blue"])
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#EFF6FF" if column != 0 else "#F1F5F9")

    return plots.save(figure, "tabela_metricas", *FIGURE_FOLDER)


# ── Pipeline ─────────────────────────────────────────────────────────────────

@console.guard("Falha ao treinar o ensemble de Árvore Aleatória.")
def main() -> None:
    console.banner(
        "ENSEMBLE DE ÁRVORE ALEATÓRIA",
        f"VotingRegressor (RF + Gradient Boosting + Extra Trees) — meta MSE < {MSE_GOAL}",
    )
    runtime.announce()

    console.step(1, 6, "Carregando dados e derivando features")
    frame = data.load()
    engineered = engineer_features(frame)
    feature_names = [column for column in engineered.columns if column != data.TARGET]
    console.detail(f"{len(frame)} amostras | {len(data.FEATURES)} variáveis medidas "
                   f"+ {len(feature_names) - len(data.FEATURES)} derivadas")

    features = engineered[feature_names].to_numpy(dtype=float)
    target = engineered[data.TARGET].to_numpy(dtype=float).ravel()

    if LOG_TARGET:
        console.detail(f"log1p aplicado ao alvo: "
                       f"[{target.min():.2f}, {target.max():.2f}] → "
                       f"[{np.log1p(target).min():.4f}, {np.log1p(target).max():.4f}]")
        target = np.log1p(target)

    console.step(2, 6, "Dividindo os dados")
    x_fit, x_test, y_fit, y_test = train_test_split(
        features, target, test_size=TEST_SIZE, random_state=SEED
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_fit, y_fit, test_size=VALIDATION_SIZE, random_state=SEED
    )

    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_validation_scaled = scaler.transform(x_validation)
    x_test_scaled = scaler.transform(x_test)
    x_fit_scaled = scaler.transform(x_fit)
    console.detail(f"treino {len(x_train)} | validação {len(x_validation)} | "
                   f"teste {len(x_test)}")

    console.step(3, 6, f"Buscando hiperparâmetros — {SEARCH_ITERATIONS} iterações, "
                       f"{CV_FOLDS}-fold, por estimador")
    folds = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    best_parameters: dict[str, dict] = {}

    # Barra geral com um passo por estimador; dentro dela, uma barra por busca,
    # contando os `n_iter x folds` ajustes que o joblib despacha.
    total_fits = SEARCH_ITERATIONS * CV_FOLDS
    with progress.Tracker("Busca por estimador", total=len(SEARCH_SPACES)) as tracker:
        for key, space in SEARCH_SPACES.items():
            search = RandomizedSearchCV(
                build_estimator(key, {}), space,
                n_iter=SEARCH_ITERATIONS,
                scoring="neg_mean_squared_error",
                cv=folds, random_state=SEED, n_jobs=-1, verbose=0,
            )
            with tracker.stage(f"{ESTIMATOR_LABELS[key]} — busca",
                               total=total_fits) as stage:
                with progress.joblib_stage(stage):
                    search.fit(x_train_scaled, y_train,
                               sample_weight=sample_weights(y_train))

            best_parameters[key] = search.best_params_
            tracker.log(f"{ESTIMATOR_LABELS[key]} — melhor MSE CV "
                        f"{-search.best_score_:.4f}")
            tracker.advance_overall()

    console.step(4, 6, "Treinando o ensemble em treino + validação")
    ensemble = VotingRegressor(estimators=[
        (key, build_estimator(key, parameters))
        for key, parameters in best_parameters.items()
    ])
    with progress.Tracker("Ensemble", total=1) as tracker:
        with tracker.stage("VotingRegressor — ajuste final") as stage:
            with progress.joblib_stage(stage):
                ensemble.fit(x_fit_scaled, y_fit, sample_weight=sample_weights(y_fit))
        tracker.advance_overall()

    console.step(5, 6, "Avaliando ensemble e membros")

    def to_original_scale(values: np.ndarray) -> np.ndarray:
        return np.expm1(values) if LOG_TARGET else values

    raw_prediction_test = ensemble.predict(x_test_scaled)

    prediction_validation = to_original_scale(ensemble.predict(x_validation_scaled))
    prediction_test = to_original_scale(raw_prediction_test)
    y_validation_scale = to_original_scale(y_validation)
    y_test_scale = to_original_scale(y_test)

    # MSE na escala em que o modelo foi de fato treinado. Guardado à parte
    # porque é essa a escala em que a meta do projeto faz sentido numérico.
    mse_training_scale = float(np.mean((y_test - raw_prediction_test) ** 2))

    validation_scores = metrics.evaluate(y_validation_scale, prediction_validation,
                                         "Ensemble — validação")
    test_scores = metrics.evaluate(y_test_scale, prediction_test, "Ensemble — teste")

    # Os clones já ajustados que o ensemble guarda, não cópias reajustadas do
    # zero. Um `.fit()` separado aqui treinaria os membros sem os pesos amostrais
    # que o ensemble usou, e o gráfico de comparação estaria medindo dois
    # procedimentos de treino diferentes em vez de dois algoritmos diferentes.
    member_scores = {
        key: metrics.evaluate(
            y_test_scale,
            to_original_scale(member.predict(x_test_scaled)),
            ESTIMATOR_LABELS[key],
        )
        for key, member in ensemble.named_estimators_.items()
    }

    metrics.show([*member_scores.values(), test_scores],
                 headers=["Modelo", "MSE", "RMSE", "MAE", "MAPE", "R²"])

    # Guarda o resultado do teste para o menu mostrar depois, sem
    # treinar de novo.
    scoreboard.record("random_forest_ensemble",
                      "Árvore Aleatória — ensemble", test_scores,
                      mse_training_scale=mse_training_scale)

    console.step(6, 6, "Gerando gráficos e salvando artefatos")
    written = []
    written += chart_predicted_vs_actual(y_validation_scale, prediction_validation,
                                         validation_scores, "validacao", PALETTE["blue"])
    written += chart_predicted_vs_actual(y_test_scale, prediction_test,
                                         test_scores, "teste", PALETTE["blue_light"])
    written += chart_residuals(y_validation_scale, prediction_validation,
                               "validacao", PALETTE["blue"])
    written += chart_residuals(y_test_scale, prediction_test, "teste", PALETTE["blue_light"])
    written += chart_metrics_table(validation_scores, test_scores)
    written += chart_member_comparison(member_scores, test_scores)
    written += chart_feature_importance(ensemble.named_estimators_["rf"],
                                        ensemble.named_estimators_["et"],
                                        feature_names)

    paths.ensure(paths.RF_ENSEMBLE_DIR)
    dump(ensemble, paths.RF_ENSEMBLE_DIR / "ensemble_model.joblib")
    dump(scaler, paths.RF_ENSEMBLE_DIR / "scaler.joblib")
    dump({"features": feature_names, "log_target": LOG_TARGET},
         paths.RF_ENSEMBLE_DIR / "metadata.joblib")

    measured = mse_training_scale if MSE_GOAL_SCALE == "log1p" else test_scores.mse
    scale_label = "log1p(CBR)" if MSE_GOAL_SCALE == "log1p" else "CBR (%)"
    goal_reached = measured < MSE_GOAL

    console.result_panel(
        "ENSEMBLE TREINADO" if goal_reached else "ENSEMBLE TREINADO — META NÃO ATINGIDA",
        [
            f"MSE em {scale_label}: {measured:.4f} | meta < {MSE_GOAL}",
            f"MSE na escala real do CBR: {test_scores.mse:.4f}",
            f"RMSE {test_scores.rmse:.4f} | MAE {test_scores.mae:.4f} | "
            f"R² {test_scores.r2:.4f}",
            "",
            f"{len(written)} gráfico(s) em "
            f"{plots.relative(paths.FIGURES_DIR.joinpath(*FIGURE_FOLDER))}",
            f"Modelo em {plots.relative(paths.RF_ENSEMBLE_DIR)}",
        ],
        success=goal_reached,
    )

    # Veredito final: o que os números acima significam para quem vai usar
    # o modelo. Fica por último de propósito — é a linha que alguém lê ao
    # voltar ao terminal depois de um treino longo.
    metrics.report(test_scores)


if __name__ == "__main__":
    main()
