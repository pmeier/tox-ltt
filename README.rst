tox-ltt
=======

.. start-badges

.. list-table::
    :stub-columns: 1

    * - package
      - |license| |status|
    * - code
      - |black| |mypy| |lint|
    * - tests
      - |tests| |coverage|

.. end-badges

`tox <https://tox.readthedocs.io/en/latest/>`_ plugin for
`light-the-torch <https://github.com/pmeier/light-the-torch>`_ .

.. code-block:: sh

  $ pip install tox tox-ltt
  $ tox --help-ini
  disable_light_the_torch <bool>   default: False
  disable installing PyTorch distributions with light-the-torch

  force_cpu       <bool>   default: False
  force CPU as computation backend
  [...]

.. note::

  If you have access to ``tox>=3.2`` you can use ``tox-ltt`` with the ``requires``
  keyword:

  .. code-block::

    [tox]

    [textenv]
    requires =
      tox-ltt
    disable_light_the_torch = False
    force_cpu = False


.. |license|
  image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
    :target: https://opensource.org/licenses/BSD-3-Clause
    :alt: License

.. |status|
  image:: https://www.repostatus.org/badges/latest/wip.svg
    :alt: Project Status: WIP
    :target: https://www.repostatus.org/#wip

.. |black|
  image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: black
   
.. |mypy|
  image:: http://www.mypy-lang.org/static/mypy_badge.svg
    :target: http://mypy-lang.org/
    :alt: mypy

.. |lint|
  image:: https://github.com/pmeier/tox-ltt/workflows/lint/badge.svg
    :target: https://github.com/pmeier/tox-ltt/actions?query=workflow%3Alint+branch%3Amaster
    :alt: Lint status via GitHub Actions

.. |tests|
  image:: https://github.com/pmeier/tox-ltt/workflows/tests/badge.svg
    :target: https://github.com/pmeier/tox-ltt/actions?query=workflow%3Atests+branch%3Amaster
    :alt: Test status via GitHub Actions

.. |coverage|
  image:: https://codecov.io/gh/pmeier/tox-ltt/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/pmeier/tox-ltt
    :alt: Test coverage via codecov.io
