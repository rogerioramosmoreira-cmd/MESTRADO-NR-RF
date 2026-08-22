"""
Verificação de dependências.

Responde a uma pergunta antes de qualquer modelo rodar: *as bibliotecas de que
este script precisa estão realmente instaladas?* Um pacote ausente é reportado
como um problema nomeado e solucionável, com a linha exata de `pip install` — e
não como um `ImportError` 400 linhas depois do início do treino.

As verificações usam `importlib.util.find_spec`, que resolve um módulo sem
executá-lo. Importar o TensorFlow só para ver se ele existe custaria vários
segundos e imprimiria o ruído de inicialização dele.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import sys
from dataclasses import dataclass
from typing import Sequence

from core import console

# Nome de importação -> nome da distribuição no PyPI. Eles diferem com
# frequência suficiente (`sklearn` / `scikit-learn`, `PIL` / `pillow`) para que
# adivinhar não seja opção.
DISTRIBUTION_NAMES = {
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "PIL": "pillow",
    "skimage": "scikit-image",
    "yaml": "PyYAML",
    "tensorflow": "tensorflow",
    "keras": "keras",
}

# Grupos de dependências. Um script declara o grupo de que precisa, em vez de
# listar pacotes soltos, então acrescentar uma dependência é uma mudança de uma
# linha só, aqui.
GROUPS: dict[str, tuple[str, ...]] = {
    "core": ("numpy", "pandas", "matplotlib", "joblib", "sklearn"),
    "plots": ("numpy", "pandas", "matplotlib", "seaborn"),
    "neural": ("numpy", "pandas", "matplotlib", "joblib", "sklearn", "tensorflow"),
    "dashboard": ("numpy", "pandas", "matplotlib", "joblib", "sklearn", "streamlit"),
    "console": ("rich",),
    # Leitura de planilha. O pandas delega o .xlsx ao openpyxl; sem ele, a falha
    # apareceria como um ImportError no meio da conversão.
    "excel": ("pandas", "openpyxl"),
}


@dataclass(frozen=True)
class Dependency:
    """Um módulo exigido e o que foi encontrado dele neste interpretador."""

    module: str
    installed: bool
    version: str | None

    @property
    def distribution(self) -> str:
        return DISTRIBUTION_NAMES.get(self.module, self.module)


def _installed_version(module: str) -> str | None:
    """Lê a versão da distribuição sem importar o módulo."""
    try:
        return importlib.metadata.version(DISTRIBUTION_NAMES.get(module, module))
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:  # noqa: BLE001 - metadado é auxiliar, nunca fatal
        return None


def inspect(modules: Sequence[str]) -> list[Dependency]:
    """Resolve cada módulo e informa se ele é importável."""
    found: list[Dependency] = []
    for module in modules:
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, ValueError):
            # Um pacote quebrado ou parcialmente desinstalado pode levantar erro
            # aqui; isso continua sendo "inutilizável", que é o que interessa a
            # quem chamou.
            spec = None
        found.append(Dependency(module, spec is not None, _installed_version(module)))
    return found


def install_command(missing: Sequence[Dependency]) -> str:
    """Monta a linha exata de pip que resolve as ausências reportadas."""
    packages = " ".join(dependency.distribution for dependency in missing)
    return f"{sys.executable} -m pip install {packages}"


def report(modules: Sequence[str], title: str = "Dependências") -> list[Dependency]:
    """Imprime a situação de cada módulo e devolve a lista completa."""
    results = inspect(modules)
    rows = []
    for dependency in results:
        if dependency.installed:
            status = "instalado"
            version = dependency.version or "-"
        else:
            status = "AUSENTE"
            version = "-"
        rows.append((dependency.distribution, status, version))

    console.section(title)
    console.metrics_table(rows, headers=["Biblioteca", "Status", "Versão"])
    return results


def require(group: str, quiet: bool = False) -> None:
    """
    Verifica um grupo de dependências e encerra com uma mensagem acionável se
    algum módulo estiver ausente.

    Chamada no início de todo script executável. Sai com código 2 — distinto de
    uma falha de treino (1) — para que o menu consiga separar "não instalado"
    de "rodou e quebrou".
    """
    modules = GROUPS.get(group)
    if modules is None:
        raise KeyError(f"Grupo de dependências desconhecido: {group!r}. "
                       f"Disponíveis: {sorted(GROUPS)}")

    results = inspect(modules)
    missing = [dependency for dependency in results if not dependency.installed]

    if not missing:
        if not quiet:
            versions = ", ".join(
                f"{d.distribution} {d.version}" for d in results if d.version
            )
            console.ok(f"Dependências verificadas — {versions}" if versions
                       else "Dependências verificadas.")
        return

    report(modules, title=f"Dependências ausentes — grupo '{group}'")
    console.result_panel(
        "DEPENDÊNCIAS AUSENTES",
        [
            f"Faltam {len(missing)} biblioteca(s): "
            + ", ".join(d.distribution for d in missing),
            "",
            "Instale com:",
            f"  {install_command(missing)}",
            "",
            "Ou instale tudo de uma vez:",
            f"  {sys.executable} -m pip install -r requirements.txt",
        ],
        success=False,
    )
    sys.exit(2)


def check_all() -> bool:
    """
    Verifica todos os grupos declarados de uma vez, com a tabela completa.

    É o diagnóstico detalhado, para quando o usuário quer ver versão por versão.
    Devolve True quando nada está faltando.
    """
    every_module = sorted({module for group in GROUPS.values() for module in group})
    results = report(every_module, title="Diagnóstico de Bibliotecas")
    missing = [dependency for dependency in results if not dependency.installed]

    if missing:
        console.error(
            f"{len(missing)} biblioteca(s) ausente(s): "
            + ", ".join(d.distribution for d in missing)
        )
        console.detail(install_command(missing))
        return False

    console.ok("Todas as bibliotecas necessárias estão instaladas.")
    return True


def missing_modules() -> list[Dependency]:
    """Todos os módulos declarados que não estão instalados."""
    every_module = sorted({module for group in GROUPS.values() for module in group})
    return [dependency for dependency in inspect(every_module)
            if not dependency.installed]


def broken_groups(missing: Sequence[Dependency] | None = None) -> dict[str, list[str]]:
    """
    Mapeia cada grupo incompleto para as distribuições que faltam nele.

    Serve para o menu dizer *o que deixa de funcionar*, e não apenas o que está
    ausente: sem TensorFlow o projeto inteiro continua utilizável, só os modelos
    de rede neural é que não rodam.
    """
    absent = {d.module for d in (missing if missing is not None else missing_modules())}
    broken: dict[str, list[str]] = {}
    for group, modules in GROUPS.items():
        gaps = [DISTRIBUTION_NAMES.get(m, m) for m in modules if m in absent]
        if gaps:
            broken[group] = gaps
    return broken


def startup_check() -> dict[str, list[str]]:
    """
    Verificação automática executada assim que o projeto abre.

    Resume em uma linha quando está tudo certo — a checagem não deve gastar meia
    tela toda vez que o menu aparece. Quando falta alguma coisa, mostra o que
    parou de funcionar e o comando exato de instalação.

    Não interrompe a execução: uma biblioteca ausente costuma inutilizar apenas
    parte do catálogo, e bloquear o menu inteiro por causa disso impediria o
    usuário de rodar os modelos que continuam perfeitamente funcionais.

    Devolve o mapa de grupos incompletos (vazio quando está tudo instalado).
    """
    console.section("Verificação automática de bibliotecas")

    missing = missing_modules()
    if not missing:
        every_module = sorted({m for group in GROUPS.values() for m in group})
        installed = inspect(every_module)
        console.ok(
            f"{len(installed)} bibliotecas verificadas, nenhuma ausente."
        )
        console.detail(", ".join(
            f"{d.distribution} {d.version}" for d in installed if d.version
        ))
        return {}

    broken = broken_groups(missing)

    console.error(
        f"{len(missing)} biblioteca(s) ausente(s): "
        + ", ".join(d.distribution for d in missing)
    )
    for group, gaps in broken.items():
        console.detail(f"grupo '{group}' incompleto — falta {', '.join(gaps)}")

    # O grupo 'core' é a base de que todo modelo depende. Se ele está
    # incompleto, nada roda — dizer que "os demais continuam funcionando"
    # nesse caso seria falso.
    if "core" in broken:
        impact = ["Nenhum modelo vai rodar: o grupo 'core' está incompleto."]
    else:
        impact = ["Os modelos que dependem dessas bibliotecas vão falhar.",
                  "Os demais continuam funcionando normalmente."]

    console.result_panel(
        "INSTALE ANTES DE CONTINUAR",
        [
            *impact,
            "",
            "Instale o que falta:",
            f"  {install_command(missing)}",
            "",
            "Ou tudo de uma vez:",
            f"  {sys.executable} -m pip install -r requirements.txt",
        ],
        success=False,
    )
    return broken
