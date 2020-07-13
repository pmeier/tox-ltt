from typing import Optional

import tox
from tox import reporter
from tox.action import Action
from tox.config import Parser
from tox.venv import VirtualEnv

import light_the_torch as ltt
from light_the_torch.computation_backend import CPUBackend


@tox.hookimpl
def tox_addoption(parser: Parser) -> None:
    parser.add_testenv_attribute(
        name="disable_light_the_torch",
        type="bool",
        help="disable installing PyTorch distributions with light-the-torch",
        default=False,
    )
    parser.add_testenv_attribute(
        name="force_cpu",
        type="bool",
        help="force CPU as computation backend",
        default=False,
    )


@tox.hookimpl
def tox_testenv_install_deps(venv: VirtualEnv, action: Action) -> None:
    envconfig = venv.envconfig
    config = envconfig.config

    if config.isolated_build and envconfig.envname == config.isolated_build_env:
        return None

    if envconfig.disable_light_the_torch:
        action.setactivity("light-the-torch", "skip")
        reporter.verbosity1(
            (
                "Skipping installation with light-the-torch since "
                "'disable_light_the_torch = True' is configured."
            ),
        )
        return None

    requirements = [dep_config.name for dep_config in venv.get_resolved_dependencies()]

    if not envconfig.skip_install:
        requirements.append(venv.package.strpath)

    if not requirements:
        return None

    action.setactivity("finddeps-light-the-torch", "")

    dists = ltt.resolve_dists(requirements)

    if not dists:
        reporter.verbosity1(
            (
                "Skipping installation with light-the-torch since no PyTorch "
                "distributions were found in the dependencies and requirements."
            ),
        )
        return None

    computation_backend: Optional[CPUBackend]
    if envconfig.force_cpu:
        reporter.verbosity1(
            (
                "Using CPU as computation backend instead of auto-detecting since "
                "'force_cpu = True' is configured."
            ),
        )
        computation_backend = CPUBackend()
    else:
        computation_backend = None

    links = ltt.find_links(dists, computation_backend=computation_backend)

    action.setactivity("installdeps-light-the-torch", ", ".join(links))
    venv.run_install_command(links, action)
