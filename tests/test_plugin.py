import shutil

import pytest

from light_the_torch.computation_backend import CPUBackend


@pytest.fixture
def patch_extract_dists(mocker):
    def patch_extract_dists_(return_value=None):
        if return_value is None:
            return_value = []
        return mocker.patch(
            "tox_ltt.plugin.ltt.extract_dists", return_value=return_value
        )

    return patch_extract_dists_


@pytest.fixture
def patch_find_links(mocker):
    def patch_find_links_(return_value=None):
        if return_value is None:
            return_value = []
        return mocker.patch("tox_ltt.plugin.ltt.find_links", return_value=return_value)

    return patch_find_links_


@pytest.fixture
def install_mock(mocker):
    return mocker.patch("tox.venv.VirtualEnv.run_install_command")


def get_pyproject_toml():
    return """
    [build-system]
    requires = ["setuptools", "wheel"]
    build-backend = "setuptools.build_meta"
    """


def get_setup_py():
    return """
    from setuptools import setup

    setup()
    """


def get_setup_cfg(name, version, install_requires=None, extra_requires=None):
    lines = ["[metadata]", f"name = {name}", f"version = {version}"]

    if install_requires is not None:
        lines.extend(("[options]", "install_requires = "))
        lines.extend([f"\t{req}" for req in install_requires])

    if extra_requires is not None:
        lines.extend(("[options.extras_require]", "extra = "))
        lines.extend([f"\t{req}" for req in extra_requires])

    return "\n".join(lines)


def get_tox_ini(
    basepython=None,
    disable_light_the_torch=None,
    pytorch_channel=None,
    pytorch_force_cpu=None,
    force_cpu=None,
    deps=None,
    skip_install=False,
    usedevelop=False,
    extra=False,
    pep517=True,
):
    lines = ["[tox]", "envlist = py"]

    if pep517:
        lines.append("isolated_build = True")

    lines.extend(("[testenv]", "requires = ", "\ttox-ltt",))

    if basepython is not None:
        lines.append(f"basepython = {basepython}")
    if skip_install:
        lines.append("skip_install = True")
    if usedevelop:
        lines.append("usedevelop = True")
    if extra:
        lines.append("extras = extra")
    if disable_light_the_torch is not None:
        lines.append(f"disable_light_the_torch = {disable_light_the_torch}")
    if pytorch_channel is not None:
        lines.append(f"pytorch_channel = {pytorch_channel}")
    if pytorch_force_cpu is not None:
        lines.append(f"pytorch_force_cpu = {pytorch_force_cpu}")
    if force_cpu is not None:
        lines.append(f"force_cpu = {force_cpu}")
    if deps is not None:
        lines.append("deps = ")
        lines.extend([f"\t{dep}" for dep in deps])

    return "\n".join(lines)


@pytest.fixture
def tox_ltt_initproj(initproj):
    def tox_ltt_initproj_(
        name="foo",
        version="1.2.3",
        basepython=None,
        install_requires=None,
        extra_requires=None,
        disable_light_the_torch=None,
        pytorch_channel=None,
        pytorch_force_cpu=None,
        force_cpu=None,
        deps=None,
        skip_install=False,
        usedevelop=False,
        pep517=True,
    ):
        filedefs = {
            "setup.cfg": get_setup_cfg(
                name,
                version,
                install_requires=install_requires,
                extra_requires=extra_requires,
            ),
            "tox.ini": get_tox_ini(
                basepython=basepython,
                skip_install=skip_install,
                usedevelop=usedevelop,
                extra=extra_requires is not None,
                disable_light_the_torch=disable_light_the_torch,
                pytorch_channel=pytorch_channel,
                pytorch_force_cpu=pytorch_force_cpu,
                force_cpu=force_cpu,
                deps=deps,
                pep517=pep517,
            ),
        }
        if pep517:
            filedefs["pyproject.toml"] = get_pyproject_toml()
        else:
            filedefs["setup.py"] = get_setup_py()
        return initproj(
            f"{name}-{version}", filedefs=filedefs, add_missing_setup_py=False
        )

    return tox_ltt_initproj_


def test_help_ini(cmd):
    result = cmd("--help-ini")
    result.assert_success(is_run_test_env=False)
    assert "disable_light_the_torch" in result.out
    assert "pytorch_channel" in result.out
    assert "pytorch_force_cpu" in result.out


@pytest.mark.slow
def test_tox_ltt_disabled(patch_extract_dists, tox_ltt_initproj, cmd):
    mock = patch_extract_dists()
    tox_ltt_initproj(disable_light_the_torch=True)

    result = cmd()

    result.assert_success(is_run_test_env=False)
    mock.assert_not_called()


@pytest.mark.slow
def test_tox_ltt_pytorch_channel(patch_find_links, tox_ltt_initproj, cmd, install_mock):
    channel = "channel"

    mock = patch_find_links()
    tox_ltt_initproj(deps=("torch",), pytorch_channel=channel)

    result = cmd()

    result.assert_success(is_run_test_env=False)

    _, kwargs = mock.call_args
    assert kwargs["channel"] == channel


@pytest.mark.slow
def test_tox_ltt_pytorch_force_cpu(
    patch_find_links, tox_ltt_initproj, cmd, install_mock
):
    mock = patch_find_links()
    tox_ltt_initproj(deps=("torch",), pytorch_force_cpu=True)

    result = cmd()

    result.assert_success(is_run_test_env=False)

    _, kwargs = mock.call_args
    assert kwargs["computation_backends"] == CPUBackend()


@pytest.mark.slow
def test_tox_ltt_force_cpu_legacy(
    patch_find_links, tox_ltt_initproj, cmd, install_mock
):
    mock = patch_find_links()
    tox_ltt_initproj(deps=("torch",), force_cpu=True)

    result = cmd()

    result.assert_success(is_run_test_env=False)

    _, kwargs = mock.call_args
    assert kwargs["computation_backends"] == CPUBackend()


def test_tox_ltt_no_requirements(
    patch_extract_dists, tox_ltt_initproj, cmd, install_mock
):
    mock = patch_extract_dists()
    tox_ltt_initproj(skip_install=True)

    result = cmd()

    result.assert_success(is_run_test_env=False)
    mock.assert_not_called()


@pytest.mark.slow
def test_tox_ltt_no_pytorch_dists(
    patch_find_links, tox_ltt_initproj, cmd, install_mock
):
    mock = patch_find_links()

    deps = ("light-the-torch",)
    tox_ltt_initproj(deps=deps)

    result = cmd()

    result.assert_success(is_run_test_env=False)
    mock.assert_not_called()


@pytest.mark.slow
def test_tox_ltt_direct_pytorch_dists(
    patch_find_links, tox_ltt_initproj, cmd, install_mock
):
    mock = patch_find_links()

    deps = ("torch", "torchaudio", "torchtext", "torchvision")
    dists = set(deps)
    tox_ltt_initproj(deps=deps)

    result = cmd()

    result.assert_success(is_run_test_env=False)

    args, _ = mock.call_args
    assert set(args[0]) == dists


@pytest.mark.slow
def test_tox_ltt_indirect_pytorch_dists(
    patch_find_links, tox_ltt_initproj, cmd, install_mock
):
    mock = patch_find_links()

    deps = ("git+https://github.com/pmeier/pystiche@v0.5.0",)
    dists = {"torch>=1.5.0", "torchvision>=0.6.0"}
    tox_ltt_initproj(deps=deps)

    result = cmd()

    result.assert_success(is_run_test_env=False)

    args, _ = mock.call_args
    assert set(args[0]) == dists


def test_tox_ltt_project_pytorch_dists(
    subtests, patch_find_links, tox_ltt_initproj, cmd, install_mock
):
    mock = patch_find_links()

    install_requires = ("torch>=1.5.0", "torchvision>=0.6.0")
    dists = set(install_requires)

    for pep517 in (True, False):
        mock.reset()
        with subtests.test(pep517=pep517):
            tox_ltt_initproj(install_requires=install_requires, pep517=pep517)

            result = cmd()

            result.assert_success(is_run_test_env=False)

            args, _ = mock.call_args
            assert set(args[0]) == dists


def test_tox_ltt_project_extra_pytorch_dists(
    subtests, patch_find_links, tox_ltt_initproj, cmd, install_mock
):
    mock = patch_find_links()

    extra_requires = ("torch>=1.5.0", "torchvision>=0.6.0")
    dists = set(extra_requires)

    for pep517 in (True, False):
        mock.reset()
        with subtests.test(pep517=pep517):
            tox_ltt_initproj(extra_requires=extra_requires, pep517=pep517)

            result = cmd()

            result.assert_success(is_run_test_env=False)

            args, _ = mock.call_args
            assert set(args[0]) == dists


def test_tox_ltt_project_usedevelop(
    patch_find_links, tox_ltt_initproj, cmd, install_mock
):
    mock = patch_find_links()
    install_requires = ("torch>=1.5.0", "torchvision>=0.6.0")
    dists = set(install_requires)
    tox_ltt_initproj(install_requires=install_requires, usedevelop=True, pep517=False)

    result = cmd()

    result.assert_success(is_run_test_env=False)

    args, _ = mock.call_args
    assert set(args[0]) == dists


@pytest.fixture
def other_basepythons(current_tox_py):
    current_minor = int(current_tox_py[-1])
    basepythons = (f"python3.{minor}" for minor in {6, 7, 8} - {current_minor})
    return [
        basepython for basepython in basepythons if shutil.which(basepython) is not None
    ]


@pytest.mark.slow
def test_tox_ltt_other_basepython(
    subtests,
    mock_venv,
    patch_extract_dists,
    patch_find_links,
    install_mock,
    tox_ltt_initproj,
    cmd,
    other_basepythons,
):
    def canonical_to_tox(version):
        major, minor, _ = version.split(".")
        return f"python{major}.{minor}"

    deps = ["torch"]
    patch_extract_dists(return_value=deps)
    mock = patch_find_links()

    for basepython in other_basepythons:
        mock.reset()

        with subtests.test(basepython=basepython):
            tox_ltt_initproj(basepython=basepython, deps=deps)

            result = cmd()
            result.assert_success()

            _, kwargs = mock.call_args
            python_version = kwargs["python_version"]
            assert canonical_to_tox(python_version) == basepython
