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

The best and easiest way to install ``hera_sim`` is with ``uv``::

  uv add hera_sim

Or if you are using the pip-style interface::

  uv pip install hera_sim

You can install optional extras as well -- for exapmle to get all the dependencies
required for simulating visibilities::

  uv add hera_sim --extra vis

Available extras are ``vis``, ``bda``, and ``cal`` (for also installing hera-calibration).

Conda users
~~~~~~~~~~~

If you are using conda, the following command will install all
dependencies which it can handle natively::

  conda install -c conda-forge numpy scipy pyuvdata attrs h5py healpy pyyaml

Developer install
~~~~~~~~~~~~~~~~~
If you are planning on developing ``hera_sim``, you should use ``uv``::

  git clone git@github.com/hera-team/hera_sim.git
  cd hera_sim
  uv sync --all-extras

As the repository is becoming quite large, you may also wish to perform
a shallow clone to retrieve only the recent commits and history. This makes
the clone faster and avoid bottleneck in CI pipelines.

Provide an argument ``--depth 1`` to the ``git clone`` command to copy only
the latest revision of the repository.

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
