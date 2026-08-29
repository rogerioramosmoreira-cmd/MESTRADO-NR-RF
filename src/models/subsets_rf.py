"""
Árvore Aleatória sobre os cinco conjuntos de variáveis (C1–C5).

Treina uma floresta independente por conjunto e as compara, para responder qual
grupo de ensaios de laboratório realmente carrega o sinal do CBR:

    C1  granulometria                      (6 peneiras)
    C2  plasticidade                       (LL, IP)
    C3  compactação                        (umidade ótima, densidade máxima)
    C4  granulometria + plasticidade       (8 variáveis)
    C5  todas as variáveis                 (10 variáveis)

Cada conjunto tem sua própria busca de hiperparâmetros, seu próprio scaler e seu
próprio modelo salvo, então nenhuma informação do conjunto completo vaza para os
conjuntos reduzidos.

Execução:  python src/models/subsets_rf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import console, dependencies  # noqa: E402

dependencies.require("core")

import numpy as np  # noqa: E402
from joblib import dump  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.model_selection import (KFold, RandomizedSearchCV,  # noqa: E402
                                     cross_val_predict, train_test_split)
from sklearn.preprocessing import MinMaxScaler  # noqa: E402

from core import (  # noqa: E402
    data, metrics, paths, plots, progress, runtime, scoreboard,
)
from core.data import SUBSET_ORDER, SUBSETS, Subset  # noqa: E402
from core.subset_report import SubsetResult, render_model_charts, save_results  # noqa: E402

MODEL_NAME = "random_forest"

SEED = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.15

# Menos iterações que o script de modelo único (150): aqui a busca roda cinco
# vezes, e o que importa é a comparação entre conjuntos, não espremer o último
# milésimo de MSE de qualquer um deles.
# O segundo valor vale apenas com MLL_FAST=1 — ver core/runtime.py.
SEARCH_ITERATIONS = runtime.budget(60, 10)
CV_FOLDS = runtime.budget(5, 3)

# Comprime a distribuição do CBR, assimétrica à direita, para que os erros
# fiquem equilibrados em toda a faixa em vez de dominados pelos poucos valores
# muito altos.
LOG_TARGET = True

# Amostras acima deste CBR são raras; dar mais peso a elas evita que a floresta
# trate a ponta alta como ruído.
WEIGHT_THRESHOLD = 25.0
WEIGHT_RARE = 3.0
WEIGHT_COMMON = 1.0

SEARCH_SPACE = {
    "n_estimators": list(range(100, 801, 50)),
    "max_depth": list(range(3, 31, 2)) + [None],
    "min_samples_split": list(range(2, 15)),
    "min_samples_leaf": list(range(1, 10)),
    "max_features": ["sqrt", "log2"] + np.linspace(0.1, 1.0, 10).round(2).tolist(),
    "max_samples": np.linspace(0.5, 1.0, 6).round(2).tolist(),
}


def sample_weights(target: np.ndarray) -> np.ndarray:
    """Dá às amostras raras de CBR alto mais peso que às comuns."""
    threshold = np.log1p(WEIGHT_THRESHOLD) if LOG_TARGET else WEIGHT_THRESHOLD
    return np.where(target > threshold, WEIGHT_RARE, WEIGHT_COMMON).astype(np.float32)


def train_subset(frame, subset: Subset, tracker: progress.Tracker) -> SubsetResult:
    """Roda o ciclo completo de treino/ajuste/avaliação de um único conjunto."""
    features, target = data.matrices(frame, subset)

    if LOG_TARGET:
        target = np.log1p(target)

    # A mesma SEED em todos os conjuntos garante que cada um veja exatamente as
    # mesmas linhas em treino/validação/teste. Sem isso, um conjunto poderia
    # parecer melhor apenas por ter tirado uma divisão de teste mais fácil.
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

    tracker.log(
        f"treino {x_train.shape[0]} | validação {x_validation.shape[0]} | "
        f"teste {x_test.shape[0]} | variáveis {x_train.shape[1]}"
    )

    folds = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=SEED, n_jobs=-1),
        SEARCH_SPACE,
        n_iter=SEARCH_ITERATIONS,
        scoring="neg_mean_squared_error",
        cv=folds,
        random_state=SEED,
        n_jobs=-1,
        verbose=0,
    )

    # A busca faz um ajuste por combinação e por dobra; esse produto é o total
    # exato que a barra precisa.
    total_fits = SEARCH_ITERATIONS * CV_FOLDS
    with tracker.stage(f"{subset.key.upper()} — busca de hiperparâmetros",
                       total=total_fits) as stage:
        with progress.joblib_stage(stage):
            search.fit(x_train_scaled, y_train, sample_weight=sample_weights(y_train))

    tracker.log(f"melhor MSE CV: {-search.best_score_:.4f}")

    forest = RandomForestRegressor(**search.best_params_, random_state=SEED, n_jobs=-1)
    # O ajuste final é uma chamada única do sklearn, sem etapas observáveis:
    # barra pulsante em vez de determinada.
    with tracker.stage(f"{subset.key.upper()} — treino final") as stage:
        with progress.joblib_stage(stage):
            forest.fit(x_fit_scaled, y_fit, sample_weight=sample_weights(y_fit))

    # `forest` foi ajustado em x_fit (treino+validação), então prever o conjunto
    # de validação devolveria erro de treino, não de generalização — otimista por
    # construção (data leakage). A estimativa honesta vem do CV out-of-fold com
    # a melhor configuração, ajustada só no treino: cada previsão vem de um
    # modelo que nunca viu aquela linha.
    forest_cv = RandomForestRegressor(**search.best_params_, random_state=SEED,
                                      n_jobs=-1)
    with tracker.stage(f"{subset.key.upper()} — validação cruzada",
                       total=CV_FOLDS) as stage:
        with progress.joblib_stage(stage):
            prediction_validation = cross_val_predict(
                forest_cv, x_train_scaled, y_train, cv=folds, n_jobs=-1,
                params={"sample_weight": sample_weights(y_train)},
            )
    y_validation = y_train

    prediction_test = forest.predict(x_test_scaled)

    # As métricas são sempre reportadas na escala original do CBR, nunca em
    # log1p, para que um MSE daqui seja diretamente comparável com o dos outros
    # scripts.
    if LOG_TARGET:
        prediction_validation = np.expm1(prediction_validation)
        prediction_test = np.expm1(prediction_test)
        y_validation = np.expm1(y_validation)
        y_test = np.expm1(y_test)

    destination = paths.SUBSETS_DIR / MODEL_NAME / subset.key
    destination.mkdir(parents=True, exist_ok=True)
    dump(forest, destination / "rf_model.joblib")
    dump(scaler, destination / "scaler.joblib")

    return SubsetResult(
        model=MODEL_NAME,
        subset_key=subset.key,
        validation=metrics.evaluate(y_validation, prediction_validation, subset.name),
        test=metrics.evaluate(y_test, prediction_test, subset.name),
        y_validation=y_validation,
        prediction_validation=prediction_validation,
        y_test=y_test,
        prediction_test=prediction_test,
        parameters={**search.best_params_, "cv_mse": -search.best_score_},
        importances=forest.feature_importances_,
    )


@console.guard("Falha ao treinar a Árvore Aleatória por conjuntos.")
def main() -> None:
    console.banner(
        "ÁRVORE ALEATÓRIA — COMPARAÇÃO DE CONJUNTOS C1–C5",
        "Um modelo independente por conjunto de variáveis",
    )
    runtime.announce()

    console.step(1, 3, "Carregando dataset")
    frame = data.load()
    console.ok(f"{len(frame)} amostras completas, {len(data.FEATURES)} variáveis disponíveis")

    console.step(2, 3, f"Treinando {len(SUBSET_ORDER)} conjuntos")
    results: list[SubsetResult] = []
    with progress.Tracker("Conjuntos C1–C5", total=len(SUBSET_ORDER)) as tracker:
        for key in SUBSET_ORDER:
            subset = SUBSETS[key]
            tracker.log(f"{subset.name} — {subset.description}")
            result = train_subset(frame, subset, tracker)
            tracker.log(f"{subset.name} — MSE teste {result.test.mse:.4f} "
                        f"| R² {result.test.r2:.4f}")
            tracker.advance_overall()
            results.append(result)

    console.step(3, 3, "Gerando gráficos e salvando resultados")
    metrics.show([result.test for result in results])

    with progress.Tracker("Gráficos", total=1) as tracker:
        with tracker.stage("gerando figuras"):
            written = render_model_charts(results)
        tracker.advance_overall()
    results_path = save_results(MODEL_NAME, results)

    best = min(results, key=lambda result: result.test.mse)

    # O placar do menu guarda um resultado por item, e o item aqui são
    # os cinco conjuntos juntos: registra o melhor deles, nomeando qual é.
    scoreboard.record("subsets_rf", "Conjuntos C1–C5 — Árvore Aleatória", best.test,
                      conjunto=best.subset.name)
    console.result_panel(
        "ÁRVORE ALEATÓRIA — CONJUNTOS C1–C5",
        [
            f"Melhor conjunto: {best.subset.name} ({best.subset.description})",
            f"MSE {best.test.mse:.4f} | RMSE {best.test.rmse:.4f} | R² {best.test.r2:.4f}",
            "",
            f"{len(written)} arquivo(s) de gráfico em "
            f"{plots.relative(paths.FIGURES_DIR / 'subsets' / MODEL_NAME)}",
            f"Modelos em {plots.relative(paths.SUBSETS_DIR / MODEL_NAME)}",
            f"Métricas em {plots.relative(results_path)}",
        ],
    )

    # Veredito final: o que os números acima significam para quem vai usar
    # o modelo. Fica por último de propósito — é a linha que alguém lê ao
    # voltar ao terminal depois de um treino longo.
    metrics.report(best.test,
                   notes=[f"Avaliação do melhor conjunto ({best.subset.name}); "
                          "os cinco estão na tabela acima."])


if __name__ == "__main__":
    main()
