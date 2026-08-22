"""
Rede Neural (MLP) sobre os cinco conjuntos de variáveis (C1–C5).

A contrapartida de `subsets_rf.py`: os mesmos cinco conjuntos, a mesma semente
de divisão e as mesmas métricas, treinados com um perceptron multicamadas em vez
de uma floresta. Rodar os dois é o que torna honesta a comparação de
`subsets_comparison.py` — mesmas linhas, mesma escala, mesmos números.

    C1  granulometria                      (6 peneiras)
    C2  plasticidade                       (LL, IP)
    C3  compactação                        (umidade ótima, densidade máxima)
    C4  granulometria + plasticidade       (8 variáveis)
    C5  todas as variáveis                 (10 variáveis)

Execução:  python src/models/subsets_mlp.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Silencia os logs C++ de inicialização do TensorFlow antes de importá-lo; eles
# rolariam a saída estruturada do console para fora da tela.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from core import console, dependencies  # noqa: E402

dependencies.require("neural")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from joblib import dump  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import MinMaxScaler  # noqa: E402
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # noqa: E402
from tensorflow.keras.layers import (  # noqa: E402
    BatchNormalization,
    Dense,
    Dropout,
    Input,
    LeakyReLU,
)
from tensorflow.keras.models import Sequential  # noqa: E402
from tensorflow.keras.optimizers import Adam  # noqa: E402
from tensorflow.keras.regularizers import l2  # noqa: E402

from core import (  # noqa: E402
    data, metrics, paths, plots, progress, runtime, scoreboard,
)
from core.data import SUBSET_ORDER, SUBSETS, Subset  # noqa: E402
from core.subset_report import SubsetResult, render_model_charts, save_results  # noqa: E402

tf.get_logger().setLevel("ERROR")

MODEL_NAME = "mlp"

SEED = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.15

LOG_TARGET = True
WEIGHT_THRESHOLD = 20.0
WEIGHT_RARE = 3.0
WEIGHT_COMMON = 1.0

# Orçamento de busca por conjunto. Mantido pequeno porque roda cinco vezes; o
# script de conjunto único `mlp.py` continua sendo o ajustado exaustivamente.
# O segundo valor vale apenas com MLL_FAST=1 — ver core/runtime.py.
SEARCH_ITERATIONS = runtime.budget(12, 3)
SEARCH_EPOCHS = runtime.budget(100, 20)
FINAL_EPOCHS = runtime.budget(400, 60)
FINAL_PATIENCE = runtime.budget(25, 8)

LEARNING_RATES = [1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4]
BATCH_SIZES = [8, 16, 24, 32, 48]
DROPOUTS = [0.0, 0.05, 0.1, 0.15, 0.2]

L2_PENALTY = 0.001
LEAKY_SLOPE = 0.1


def build_network(n_features: int, dropout: float, learning_rate: float) -> Sequential:
    """
    Monta o regressor 128 → 64 → 32 → 16 → 1.

    O dropout diminui em direção à saída: as camadas largas do início suportam
    regularização pesada, as estreitas não suportam sem perder sinal.
    """
    network = Sequential([
        Input(shape=(n_features,)),

        Dense(128, kernel_regularizer=l2(L2_PENALTY)),
        BatchNormalization(),
        LeakyReLU(negative_slope=LEAKY_SLOPE),
        Dropout(dropout),

        Dense(64, kernel_regularizer=l2(L2_PENALTY)),
        BatchNormalization(),
        LeakyReLU(negative_slope=LEAKY_SLOPE),
        Dropout(dropout * 0.7),

        Dense(32, kernel_regularizer=l2(L2_PENALTY)),
        BatchNormalization(),
        LeakyReLU(negative_slope=LEAKY_SLOPE),
        Dropout(dropout * 0.4),

        Dense(16),
        LeakyReLU(negative_slope=LEAKY_SLOPE),

        Dense(1),  # saída linear — regressão sem limite superior
    ])
    network.compile(optimizer=Adam(learning_rate=learning_rate),
                    loss="mae", metrics=["mse"])
    return network


def sample_weights(target: np.ndarray) -> np.ndarray:
    """Dá às amostras raras de CBR alto mais peso que às comuns."""
    threshold = np.log1p(WEIGHT_THRESHOLD) if LOG_TARGET else WEIGHT_THRESHOLD
    return np.where(target > threshold, WEIGHT_RARE, WEIGHT_COMMON).astype(np.float32)


def train_subset(frame, subset: Subset, tracker: progress.Tracker) -> SubsetResult:
    """Roda o ciclo completo de busca/treino/avaliação de um único conjunto."""
    features, target = data.matrices(frame, subset)

    if LOG_TARGET:
        target = np.log1p(target)

    # Semente idêntica à de subsets_rf.py — os dois modelos precisam ver as
    # mesmas linhas nas mesmas divisões para que suas métricas sejam
    # comparáveis.
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

    tf.keras.utils.set_random_seed(SEED)

    generator = np.random.RandomState(SEED)
    candidates = [
        {
            "learning_rate": float(generator.choice(LEARNING_RATES)),
            "batch_size": int(generator.choice(BATCH_SIZES)),
            "dropout": float(generator.choice(DROPOUTS)),
        }
        for _ in range(SEARCH_ITERATIONS)
    ]

    best_configuration: dict = {}
    best_loss = np.inf

    # Uma etapa da barra por candidato testado.
    with tracker.stage(f"{subset.key.upper()} — busca de hiperparâmetros",
                       total=SEARCH_ITERATIONS) as stage:
        for position, candidate in enumerate(candidates, start=1):
            trial = build_network(x_train_scaled.shape[1],
                                  candidate["dropout"], candidate["learning_rate"])
            history = trial.fit(
                x_train_scaled, y_train,
                epochs=SEARCH_EPOCHS,
                batch_size=candidate["batch_size"],
                validation_data=(x_validation_scaled, y_validation),
                sample_weight=sample_weights(y_train),
                callbacks=[EarlyStopping(monitor="val_loss", patience=5,
                                         restore_best_weights=True, verbose=0)],
                verbose=0,
            )
            loss = min(history.history["val_loss"])
            if loss < best_loss:
                best_loss, best_configuration = loss, candidate
            stage.describe(f"{subset.key.upper()} — candidato {position}"
                           f"/{SEARCH_ITERATIONS} (melhor MAE {best_loss:.4f})")
            stage.advance()
            # O Keras mantém no grafo todos os modelos já construídos; limpar a
            # sessão entre tentativas impede a memória de subir ao longo de
            # 5 conjuntos x 12 candidatos.
            tf.keras.backend.clear_session()

    tracker.log(
        f"melhor config: lr={best_configuration['learning_rate']:.4g} "
        f"batch={best_configuration['batch_size']} "
        f"dropout={best_configuration['dropout']:.2f} → MAE val {best_loss:.4f}"
    )

    network = build_network(x_fit_scaled.shape[1],
                            best_configuration["dropout"],
                            best_configuration["learning_rate"])

    # Uma etapa da barra por época. O total é o teto de épocas: o early stopping
    # normalmente interrompe bem antes, então esta barra costuma terminar
    # incompleta — e é isso que se espera dela.
    with tracker.stage(f"{subset.key.upper()} — treino final",
                       total=FINAL_EPOCHS) as stage:
        history = network.fit(
            x_fit_scaled, y_fit,
            epochs=FINAL_EPOCHS,
            batch_size=best_configuration["batch_size"],
            validation_data=(x_validation_scaled, y_validation),
            sample_weight=sample_weights(y_fit),
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=FINAL_PATIENCE,
                              restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8,
                                  min_lr=1e-6, verbose=0),
                progress.keras_callback(stage),
            ],
            verbose=0,
        )

    epochs_run = len(history.history["loss"])
    tracker.log(f"parou na época {epochs_run} de {FINAL_EPOCHS}")

    prediction_validation = network.predict(x_validation_scaled, verbose=0).ravel()
    prediction_test = network.predict(x_test_scaled, verbose=0).ravel()

    if LOG_TARGET:
        prediction_validation = np.expm1(prediction_validation)
        prediction_test = np.expm1(prediction_test)
        y_validation = np.expm1(y_validation)
        y_test = np.expm1(y_test)

    destination = paths.SUBSETS_DIR / MODEL_NAME / subset.key
    destination.mkdir(parents=True, exist_ok=True)
    network.save(destination / "mlp_model.keras")
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
        parameters={**best_configuration, "epochs": epochs_run,
                    "val_mae": float(best_loss)},
        history={key: [float(value) for value in values]
                 for key, values in history.history.items()},
    )


@console.guard("Falha ao treinar a Rede Neural por conjuntos.")
def main() -> None:
    console.banner(
        "REDE NEURAL — COMPARAÇÃO DE CONJUNTOS C1–C5",
        "Um MLP independente por conjunto de variáveis",
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
    scoreboard.record("subsets_mlp", "Conjuntos C1–C5 — Rede Neural", best.test,
                      conjunto=best.subset.name)
    console.result_panel(
        "REDE NEURAL — CONJUNTOS C1–C5",
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
