[![Build Status](https://travis-ci.org/HERA-Team/hera_sim.svg?branch=master)](https://travis-ci.org/HERA-Team/hera_sim)
[![Coverage Status](https://coveralls.io/repos/github/HERA-Team/hera_sim/badge.svg?branch=master)](https://coveralls.io/github/HERA-Team/hera_sim?branch=master)

`hera_sim`: Simple simulation package
-------------------------------------

Basic simulation package for HERA-like redundant interferometric 
arrays. 

For a tutorial and overview of available features, check out the 
Jupyter notebook: `docs/tutorials/hera_sim tour.ipynb`.

Installation
------------
Before installing anything, you *must* manually install ``aipy`` using conda:

```
$ conda install -c conda-forge aipy
```

However, ``hera_sim`` depends on several packages which are conda-installable,
and if you are using ``conda``, you may wish to install them using conda 
instead of letting pip install them upon installation:

``$ conda install -c conda-forge numpy scipy pyuvdata aipy mpi4py``

Then to install this repo, either download and run ``pip install -e .`` or
run 

``pip install git+git://github.com/HERA-Team/hera_sim``.

Documentation
-------------
https://hera-sim.readthedocs.io/en/latest/