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
        return mocker.patch()

    return patch_extract_dists_


@pytest.fixture
def patch_find_links(mocker):
    def patch_find_links_(return_value=None):
        if return_value is None:
            return_value = []
            return mocker.patch(
                "tox_ltt.plugin.ltt.find_links", return_value=return_value
            )
        return mocker.patch()

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


def get_setup_cfg(name, version, install_requires=None):
    lines = ["[metadata]", f"name = {name}", f"version = {version}"]

    if install_requires is not None:
        lines.extend(("[options]", "install_requires = "))
        lines.extend([f"\t{req}" for req in install_requires])

    return "\n".join(lines)


def get_tox_ini(
    disable_light_the_torch=None,
    force_cpu=None,
    deps=None,
    skip_install=False,
    pep517=True,
):

    lines = ["[tox]", "envlist = py"]

    if pep517:
        lines.append("isolated_build = True")

    lines.extend(("[testenv]", "requires = ", "\ttox-ltt",))

    if skip_install:
        lines.append("skip_install = True")
    if disable_light_the_torch is not None:
        lines.append(f"disable_light_the_torch = {disable_light_the_torch}")
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
        install_requires=None,
        disable_light_the_torch=None,
        force_cpu=None,
        deps=None,
        skip_install=False,
        pep517=True,
    ):
        filedefs = {
            "setup.cfg": get_setup_cfg(
                name, version, install_requires=install_requires
            ),
            "tox.ini": get_tox_ini(
                skip_install=skip_install,
                disable_light_the_torch=disable_light_the_torch,
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
    assert "force_cpu" in result.out


@pytest.mark.slow
def test_tox_ltt_disabled(patch_extract_dists, tox_ltt_initproj, cmd):
    mock = patch_extract_dists()
    tox_ltt_initproj(disable_light_the_torch=True)

    result = cmd()

    result.assert_success(is_run_test_env=False)
    mock.assert_not_called()


@pytest.mark.slow
def test_tox_ltt_force_cpu(patch_find_links, tox_ltt_initproj, cmd, install_mock):
    mock = patch_find_links()
    tox_ltt_initproj(deps=("torch",), force_cpu=True)

    result = cmd()

    result.assert_success(is_run_test_env=False)

    _, kwargs = mock.call_args
    assert kwargs["computation_backend"] == CPUBackend()


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
