# `hera_sim`: Simple simulation package for the HERA array

[![Build Status](https://github.com/HERA-Team/hera_sim/workflows/Tests/badge.svg)](https://github.com/HERA-Team/hera_sim)
[![Coverage Status](https://coveralls.io/repos/github/HERA-Team/hera_sim/badge.svg?branch=master)](https://coveralls.io/github/HERA-Team/hera_sim?branch=master)




Basic simulation package for HERA-like redundant interferometric
arrays.

For a tutorial and overview of available features, check out the
Jupyter notebook: `docs/tutorials/hera_sim tour.ipynb`.

## Installation

We have tried to make `hera_sim` as easy to install as possible, however it depends
on a few packages which can give some problems. Because of this, a simple
``pip`` install will **not** work on a clean environment. This may be remedied in the
future.

### Conda users
If you are using conda, the following command will install all dependencies which it
can handle natively:

``$ conda install -c conda-forge numpy scipy pyuvdata aipy>=3.0 attrs h5py healpy pyyaml``

If you are creating a new development environment, consider using the included environment
file:

``$ conda env create -f ci/tests.yaml``

This will create a fresh environment with all required dependencies, as well as those
required for testing. If you also would like to compile the documentation, use (after
the above):

``$ pip install -r docs/requirements.txt``

Finally, install `hera_sim` with

``$ pip install [-e] .``

For a **development install**, use the optional extra `[dev]`.


### Pip-only install
If you are not going to use conda, then you must follow the following process (the reason
that this is not a one-step process is because some of the packages inadvertantly require
other packages to be installed before they can even be installed, let alone used, and ``pip``
reads the ``setup.py`` file for each package before installing anything).

First install `numpy`, as one of the dependencies requires `numpy` to be installed before
it can be installed itself:

``pip install numpy``

Then to install this repo, either download and run ``pip install -e .`` or
run

``pip install git+git://github.com/HERA-Team/hera_sim``.

For a development install (tests and documentation), run `pip install -e .[dev]`.

## Documentation
https://hera-sim.readthedocs.io/en/latest/

## Versioning
We use semantic versioning (`major`.`minor`.`patch`) for the `hera_sim` package (see
https://semver.org). To briefly summarize, new `major` versions include API-breaking changes, new `minor` versions add new features in a backwards-compatible way, and new `patch` versions implement backwards-compatible bug fixes.
