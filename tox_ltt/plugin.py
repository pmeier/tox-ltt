import subprocess
from typing import Any, List, Optional, Sequence, cast

import tox
from tox import reporter
from tox.action import Action
from tox.config import Parser, TestenvConfig
from tox.venv import VirtualEnv

import light_the_torch as ltt
from light_the_torch.cli import make_ltt_parser
from light_the_torch.computation_backend import CPUBackend


def extract_ltt_option_help(subcommand: str, option: str) -> str:
    def extract(seq: Sequence, attr: str, eq_cond: Any) -> Any:
        reduced_seq = [item for item in seq if getattr(item, attr) == eq_cond]
        assert len(reduced_seq) == 1
        return reduced_seq[0]

    ltt_parser = make_ltt_parser()

    argument_group = extract(ltt_parser._action_groups, "title", "subcommands")
    sub_parsers = extract(argument_group._actions, "dest", "subcommand")
    subcommand_parser = sub_parsers.choices[subcommand]
    return cast(str, extract(subcommand_parser._actions, "dest", option).help)


@tox.hookimpl
def tox_addoption(parser: Parser) -> None:
    parser.add_testenv_attribute(
        name="disable_light_the_torch",
        type="bool",
        help="disable installing PyTorch distributions with light-the-torch",
        default=False,
    )
    parser.add_testenv_attribute(
        name="pytorch_channel",
        type="string",
        help=extract_ltt_option_help("install", "channel"),
        default="stable",
    )
    parser.add_testenv_attribute(
        name="pytorch_force_cpu",
        type="bool",
        help=extract_ltt_option_help("install", "force_cpu"),
        default=False,
    )
    parser.add_testenv_attribute(
        name="force_cpu",
        type="bool",
        help="Deprecated alias of 'pytorch_force_cpu'.",
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
        if envconfig.usedevelop:
            path = config.setupdir.strpath
        else:
            path = venv.package.strpath
        if envconfig.extras:
            path += f"[{','.join(envconfig.extras)}]"
        requirements.append(path)

    if not requirements:
        return None

    action.setactivity("finddeps-light-the-torch", "")

    dists = ltt.extract_dists(requirements)
    dists = remove_extras(dists)

    if not dists:
        reporter.verbosity1(
            (
                "Skipping installation with light-the-torch since no PyTorch "
                "distributions were found in the dependencies and requirements."
            ),
        )
        return None

    links = ltt.find_links(
        dists,
        computation_backend=get_computation_backend(envconfig),
        channel=envconfig.pytorch_channel,
        python_version=get_python_version(envconfig),
    )

    action.setactivity("installdeps-light-the-torch", ", ".join(links))
    venv.run_install_command(links, action)


# TODO: this should probably implemented in light-the-torch
def remove_extras(dists: List[str]) -> List[str]:
    return [dist.split(";")[0] for dist in dists]


def _resolve_force_cpu(new: bool, legacy: bool) -> bool:
    if legacy:
        reporter.warning("The option 'force_cpu' was renamed to 'pytorch_force_cpu'.")
        return True

    return new


def get_computation_backend(envconfig: TestenvConfig) -> Optional[CPUBackend]:
    force_cpu = _resolve_force_cpu(envconfig.pytorch_force_cpu, envconfig.force_cpu)
    if not force_cpu:
        return None

    reporter.verbosity1(
        (
            "Using CPU as computation backend instead of auto-detecting since "
            "'pytorch_force_cpu = True' is configured."
        ),
    )
    return CPUBackend()


def get_python_version(envconfig: TestenvConfig) -> str:
    output = subprocess.check_output((envconfig.basepython, "--version"))
    return output.decode("utf-8").strip()[7:]
