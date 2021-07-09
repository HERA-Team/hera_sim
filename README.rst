hera_sim
========

|Build Status| |Coverage Status| |RTD|

**Basic simulation package for HERA-like redundant interferometric arrays.**

Features
--------

* **Systematic Models:** Many models of instrumental systematics in various forms,
  eg. thermal noise, RFI, bandpass gains, cross-talk, cable reflections and foregrounds.
* **HERA-tuned:** All models have defaults tuned to HERA, with various default "sets"
  available (eg.H1C, H2C)
* **Interoperability:** Interoperability with ``pyuvdata`` datasets and ``pyuvsim``
  configurations.
* **Ease-of-use:** High-level interface for adding multiple systematics to existing
  visibilities in a self-consistent way.
* **Visibility Simulation:** A high-level interface for visbility simulation that is
  compatible with the configuration definition from ``pyuvsim`` but is able to call
  multiple simulator implementations.
* **Convenience:** Methods for adjusting simulated data to match the times/baselines of
  a reference dataset.

Documentation
-------------

At `ReadTheDocs <https://hera-sim.readthedocs.io/en/latest/>`_.
In particular, for a tutorial and overview of available features, check out the
`tour <https://hera-sim.readthedocs.io/en/latest/tutorials/hera_sim_tour.html>`_.

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

Developer install
~~~~~~~~~~~~~~~~~
For a development install (tests and documentation), run
``pip install -e .[dev]``.

Other optional extras can be installed as well. To use
baseline-dependent averaging functionality, install the extra ``[bda]``.
For the ability to simulate redundant gains, install ``[cal]``. To
enable GPU functionality on some of the methods (especially visibility
simulators), install ``[gpu]``.

As the repository is becoming quite large, you may also wish to perform
a shallow clone to retrieve only the recent commits and history. This makes
the clone faster and avoid bottleneck in CI pipelines.

Provide an argument ``--depth 1`` to the ``git clone`` command to copy only
the latest revision of the repository.

``git clone -–depth [depth] git@github.com:HERA-Team/hera_sim.git``

Versioning
----------

We use semantic versioning (``major``.\ ``minor``.\ ``patch``) for the
``hera_sim`` package (see `SemVer documentation <https://semver.org>`_).
To briefly summarize, new
``major`` versions include API-breaking changes, new ``minor`` versions
add new features in a backwards-compatible way, and new ``patch``
versions implement backwards-compatible bug fixes.

.. |Build Status| image:: https://github.com/HERA-Team/hera_sim/workflows/Tests/badge.svg
   :target: https://github.com/HERA-Team/hera_sim
.. |Coverage Status| image:: https://coveralls.io/repos/github/HERA-Team/hera_sim/badge.svg?branch=master
   :target: https://coveralls.io/github/HERA-Team/hera_sim?branch=master
.. |RTD| image:: https://readthedocs.org/projects/hera-sim/badge/?version=latest
   :target: https://hera-sim.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
