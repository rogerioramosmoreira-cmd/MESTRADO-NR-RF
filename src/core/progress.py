"""
Barras de progresso do pipeline.

Um indicador giratório informa que algo está acontecendo, mas não informa
*quanto falta*. Um treino que leva quinze minutos precisa dizer em que ponto
está, e este módulo fornece barras determinadas sempre que o total é conhecido
de antemão:

    [4/6] Em busca dos Hiperparâmetros
    [|||||||||||||||||||||       ] 70%

Onde o total é conhecido:

  - busca de hiperparâmetros da Árvore Aleatória — `n_iter x folds` ajustes,
    contados um a um pelo joblib
  - busca da Rede Neural — uma etapa por candidato
  - treino final da rede — uma etapa por época
  - laço dos conjuntos C1–C5 — uma etapa por conjunto

A barra é escrita direto na saída padrão com retorno de carro (`\\r`), sem
depender de biblioteca de terminal. Isso funciona em qualquer console — o
console legado do Windows, o Terminal novo, um terminal de IDE — e não entra em
conflito com o `rich`, que só admite uma exibição ao vivo por vez.

Quando a saída não é um terminal (redirecionada para arquivo, lida por outro
processo), reescrever a mesma linha não faria sentido: nesse caso a barra é
impressa em marcos de 10%, uma linha por marco, para que o log continue legível.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from typing import Iterator

from core import console

# Aparência da barra.
BAR_WIDTH = 28
BAR_FILLED = "|"
BAR_EMPTY = " "

# Intervalo mínimo entre redesenhos. Uma busca de 1500 ajustes escreveria 1500
# vezes na tela para mostrar as mesmas 100 posições; por outro lado, redesenhar
# só quando o percentual muda deixa a tela congelada em etapas lentas. Um piso
# de tempo resolve os dois: o relógio de decorrido continua andando mesmo
# quando o percentual não muda, mostrando que o processo está vivo.
_MIN_REDRAW_SECONDS = 0.5

# Em saída não interativa, imprime a cada 10%.
_LOG_STEP_PERCENT = 10


def _is_terminal() -> bool:
    """
    Diz se a saída padrão é um terminal capaz de reescrever a linha.

    O painel de saída de algumas IDEs não se declara terminal, e nesse caso a
    barra viraria marcos de 10% mesmo estando visível para quem assiste. A
    variável `MLL_BAR` força a decisão quando a detecção automática erra:

        MLL_BAR=1   sempre desenha a barra animada
        MLL_BAR=0   sempre imprime marcos, uma linha por 10%
        (ausente)   detecta sozinho
    """
    forced = os.environ.get("MLL_BAR", "").strip().lower()
    if forced in {"1", "true", "sim"}:
        return True
    if forced in {"0", "false", "nao", "não"}:
        return False

    try:
        return bool(sys.stdout.isatty())
    except (AttributeError, ValueError):
        return False


class Bar:
    """
    Uma barra `[|||||||     ] 70%` redesenhada no lugar.

    `total=None` significa duração desconhecida: nesse caso não há percentual a
    mostrar, e a barra vira um marcador simples de atividade.
    """

    def __init__(self, total: int | None, indent: str = ""):
        self.total = total if (total is None or total > 0) else None
        self.indent = indent
        self.current = 0
        self._last_percent = -1
        self._last_draw = 0.0
        self._started = time.monotonic()
        self._interactive = _is_terminal()
        self._closed = False
        self._drawn = False
        self._width = 0
        self._ticker = None
        self._stop_ticker = None
        # A thread do relógio e a thread principal escrevem na mesma linha;
        # sem trava, um avanço no meio de um redesenho embaralharia a saída.
        import threading

        self._lock = threading.Lock()

    def start(self) -> None:
        """
        Desenha a barra vazia imediatamente e liga o relógio.

        Sem isto, a barra só apareceria no primeiro avanço — e o primeiro lote
        de uma busca pesada pode levar dezenas de segundos, tempo em que a tela
        ficaria parada e o treino pareceria travado.

        Uma thread de fundo redesenha a barra a cada meio segundo. Ela existe
        porque o progresso chega em saltos: entre dois lotes do joblib, ou
        durante um ajuste único que não reporta nada, nenhum avanço acontece e
        a tela ficaria estática. Com o relógio andando, dá para distinguir um
        processo lento de um processo travado.
        """
        self._render(force=True)

        if not self._interactive:
            return

        import threading

        self._stop_ticker = threading.Event()

        def loop() -> None:
            while not self._stop_ticker.wait(_MIN_REDRAW_SECONDS):
                if self._closed:
                    return
                self._render()

        # Daemon: se o processo principal morrer, esta thread não pode segurar
        # o encerramento do programa.
        self._ticker = threading.Thread(target=loop, daemon=True)
        self._ticker.start()

    # ── Avanço ───────────────────────────────────────────────────────────────

    def advance(self, step: int = 1) -> None:
        if self._closed:
            return
        self.current += step
        if self.total is not None:
            # Um total estimado por baixo (por exemplo, lotes do joblib maiores
            # que o previsto) não pode fazer a barra passar de 100%.
            self.current = min(self.current, self.total)
        self._render()

    def set_total(self, total: int) -> None:
        """Define o total quando ele só é conhecido depois de começar."""
        self.total = total if total > 0 else None

    def tick(self) -> None:
        """Redesenha para atualizar o relógio, sem alterar o progresso."""
        self._render()

    # ── Desenho ──────────────────────────────────────────────────────────────

    @property
    def percent(self) -> int:
        if not self.total:
            return 0
        return int(self.current * 100 / self.total)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._started

    def _remaining(self) -> float | None:
        """Estimativa do tempo restante, pela média do que já foi feito."""
        if not self.total or self.current <= 0:
            return None
        rate = self.elapsed / self.current
        return rate * (self.total - self.current)

    @staticmethod
    def _clock(seconds: float | None) -> str:
        if seconds is None:
            return "--:--"
        seconds = int(max(seconds, 0))
        if seconds >= 3600:
            return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}"
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    def _bar_text(self) -> str:
        if not self.total:
            # Duração desconhecida: um bloco que percorre a barra de um lado ao
            # outro. Não mede nada — só mostra que o processo continua vivo.
            block = 4
            span = BAR_WIDTH - block
            position = int(self.elapsed * 6) % (span * 2) if span > 0 else 0
            offset = position if position <= span else span * 2 - position
            bar = (BAR_EMPTY * offset + BAR_FILLED * block
                   + BAR_EMPTY * (span - offset))
            return (f"{self.indent}[{bar}]  ---  "
                    f"{self._clock(self.elapsed)} decorrido")

        filled = int(BAR_WIDTH * self.current / self.total)
        bar = BAR_FILLED * filled + BAR_EMPTY * (BAR_WIDTH - filled)
        return (f"{self.indent}[{bar}] {self.percent:3d}%  "
                f"{self._clock(self.elapsed)} decorrido"
                f" | restam ~{self._clock(self._remaining())}")

    def _render(self, force: bool = False) -> None:
        with self._lock:
            percent = self.percent
            now = time.monotonic()

            if self._interactive:
                if (not force
                        and percent == self._last_percent
                        and now - self._last_draw < _MIN_REDRAW_SECONDS):
                    return
                self._last_percent = percent
                self._last_draw = now

                text = self._bar_text()
                # Preenche com espaços até a maior largura já usada: um texto
                # que encurta (de "restam ~10:00" para "restam ~09:59")
                # deixaria restos da versão anterior na linha reescrita.
                padding = max(0, self._width - len(text))
                self._width = max(self._width, len(text))
                sys.stdout.write("\r" + text + " " * padding)
                sys.stdout.flush()
                self._drawn = True
                return

            # Saída não interativa: um marco a cada 10%, em linhas separadas.
            if self.total is None:
                return
            marker = percent - (percent % _LOG_STEP_PERCENT)
            if force or marker > self._last_percent:
                self._last_percent = marker
                print(self._bar_text(), flush=True)
                self._drawn = True

    def close(self) -> None:
        """Completa a barra, para o relógio e encerra a linha."""
        if self._closed:
            return
        self._closed = True

        # Parado antes do desenho final, para que a thread não escreva por cima
        # da linha já encerrada.
        if self._stop_ticker is not None:
            self._stop_ticker.set()
        if self._ticker is not None:
            self._ticker.join(timeout=_MIN_REDRAW_SECONDS * 2)

        if self.total:
            self.current = self.total
        self._render(force=True)

        # Quebra de linha só se algo chegou a ser desenhado; caso contrário
        # abriria uma linha em branco à toa.
        if self._drawn and self._interactive:
            sys.stdout.write("\n")
            sys.stdout.flush()


class Tracker:
    """
    Sequência numerada de etapas, cada uma com sua barra.

    Produz exatamente o formato pedido: uma linha `[k/n] descrição` marcando em
    que ponto do script estamos, e logo abaixo a barra daquela etapa.

        [1/5] C1 — busca de hiperparâmetros
        [|||||||||||||               ] 47%
    """

    def __init__(self, description: str, total: int):
        self.description = description
        self.total = total
        self.position = 0

    def __enter__(self) -> "Tracker":
        return self

    def __exit__(self, *exception) -> None:
        return None

    def log(self, message: str) -> None:
        """Linha de apoio, no mesmo recuo das demais mensagens do projeto."""
        console.detail(message)

    def advance_overall(self, step: int = 1) -> None:
        """Registra que uma etapa terminou."""
        self.position = min(self.position + step, self.total)

    @contextmanager
    def stage(self, description: str, total: int | None = None) -> Iterator["Stage"]:
        """
        Abre uma etapa: imprime `[k/n] descrição` e desenha a barra abaixo.

        O contador `k` é a posição já concluída mais um — a etapa que está
        rodando agora, e não a que acabou de terminar.
        """
        index = min(self.position + 1, self.total)
        console.step(index, self.total, description)

        bar = Bar(total)
        bar.start()
        try:
            yield Stage(bar)
        finally:
            # Fechada no `finally` para que uma exceção no meio do treino não
            # deixe a barra pela metade, sem quebra de linha, embaralhando a
            # mensagem de erro que vem logo em seguida.
            bar.close()


@contextmanager
def bar(total: int | None = None) -> Iterator["Stage"]:
    """
    Abre uma barra avulsa, sem imprimir linha de etapa.

    Para os scripts que já anunciam a fase por conta própria — `[4/6] Buscando
    hiperparâmetros...` — e só precisam da barra logo abaixo:

        [4/6] Buscando hiperparâmetros
        [|||||||||||||||||||         ]  70%  02:31 decorrido | restam ~01:04

    Usar o `Tracker` nesses casos imprimiria um segundo contador, concorrendo
    com o que o script já mostra.
    """
    instance = Bar(total)
    instance.start()
    try:
        yield Stage(instance)
    finally:
        instance.close()


class Stage:
    """Referência à barra da etapa corrente, entregue por `Tracker.stage`."""

    def __init__(self, bar: Bar):
        self._bar = bar

    def advance(self, step: int = 1) -> None:
        self._bar.advance(step)

    def set_total(self, total: int) -> None:
        self._bar.set_total(total)

    def describe(self, description: str) -> None:
        """
        Aceito por compatibilidade com quem chama, mas sem efeito.

        A barra ocupa uma linha só e é reescrita no lugar; trocar o texto dela
        no meio deixaria restos da descrição anterior na tela.
        """
        return None


# ── Integração com o joblib (scikit-learn) ───────────────────────────────────

@contextmanager
def joblib_stage(stage: Stage) -> Iterator[None]:
    """
    Faz cada ajuste concluído pelo joblib avançar a barra da etapa.

    O scikit-learn não expõe callback de progresso, mas toda busca com
    `n_jobs != 1` passa pelo `BatchCompletionCallBack` do joblib, que é chamado
    uma vez por lote concluído e conhece o tamanho do lote. Substituí-lo
    temporariamente dá a contagem exata de ajustes.

    É um detalhe interno do joblib, então qualquer falha ao instalar o gancho
    apenas desiste da contagem — a busca roda igual, só sem barra.
    """
    try:
        import joblib.parallel as joblib_parallel

        original = joblib_parallel.BatchCompletionCallBack
    except (ImportError, AttributeError):
        yield
        return

    class CountingCallBack(original):
        def __call__(self, *args, **kwargs):
            stage.advance(self.batch_size)
            return super().__call__(*args, **kwargs)

    joblib_parallel.BatchCompletionCallBack = CountingCallBack
    try:
        yield
    finally:
        # Restaurado sempre: deixar a classe trocada afetaria toda chamada de
        # joblib feita depois, inclusive as que não têm barra nenhuma.
        joblib_parallel.BatchCompletionCallBack = original


# ── Integração com o Keras ───────────────────────────────────────────────────

def keras_callback(stage: Stage):
    """
    Devolve um callback do Keras que avança a barra a cada época.

    Importa o TensorFlow só na chamada, para que este módulo continue
    utilizável nos scripts que não dependem dele.

    A barra costuma terminar antes de 100%: o early stopping interrompe o
    treino assim que a validação estaciona, e o total é o teto de épocas, não
    a previsão de quantas serão usadas.
    """
    from tensorflow.keras.callbacks import Callback

    class StageProgress(Callback):
        def on_epoch_end(self, epoch, logs=None):
            stage.advance(1)

    return StageProgress()
