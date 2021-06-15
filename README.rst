``hera_sim``: Simple visibility and instrumental systematics simulator for the HERA array
=========================================================================================

|Build Status| |Coverage Status|

Basic simulation package for HERA-like redundant interferometric arrays.

For a tutorial and overview of available features, check out the Jupyter
notebook: ``docs/tutorials/hera_sim tour.ipynb``.

Installation
------------

Conda users
~~~~~~~~~~~

If you are using conda, the following command will install all
dependencies which it can handle natively:

``$ conda install -c conda-forge numpy scipy pyuvdata attrs h5py healpy pyyaml``

If you are creating a new development environment, consider using the
included environment file:

``$ conda env create -f ci/tests.yaml``

This will create a fresh environment with all required dependencies, as
well as those required for testing. Then follow the pip-only
instructions below to install ``hera_sim`` itself.

Pip-only install
~~~~~~~~~~~~~~~~

Simply use ``pip install -e .`` or run
``pip install git+git://github.com/HERA-Team/hera_sim``.

For a development install (tests and documentation), run
``pip install -e .[dev]``.

Other optional extras can be installed as well. To use
baseline-dependent averaging functionality, install the extra ``[bda]``.
For the ability to simulate redundant gains, install ``[cal]``. To
enable GPU functionality on some of the methods (especially visibility
simulators), install ``[gpu]``. ## Documentation

https://hera-sim.readthedocs.io/en/latest/

Versioning
----------

We use semantic versioning (``major``.\ ``minor``.\ ``patch``) for the
``hera_sim`` package (see https://semver.org). To briefly summarize, new
``major`` versions include API-breaking changes, new ``minor`` versions
add new features in a backwards-compatible way, and new ``patch``
versions implement backwards-compatible bug fixes.

.. |Build Status| image:: https://github.com/HERA-Team/hera_sim/workflows/Tests/badge.svg
   :target: https://github.com/HERA-Team/hera_sim
.. |Coverage Status| image:: https://coveralls.io/repos/github/HERA-Team/hera_sim/badge.svg?branch=master
   :target: https://coveralls.io/github/HERA-Team/hera_sim?branch=master
