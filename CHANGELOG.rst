=========
Changelog
=========


v0.4.0 [2021.05.01]
===================

Added
-----

- New features added to ``vis_cpu``
    - Analytic beam interpolation
        - Instead of gridding the beam and interpolating the grid using splines,
          the beam can be interpolated directly by calling its ``interp`` method.
        - The user specifies this by passing ``use_pixel_beams=False`` to ``vis_cpu``.
    - A simple MPI parallelization scheme
        - Simulation scripts may be run using ``mpirun/mpiexec``
        - The user imports ``mpi4py`` into their script and passes
          ``mpi_comm=MPI.COMM_WORLD`` to vis_cpu
    - New ``PolyBeam`` and ``PerturbedPolyBeam`` analytic beams (classes)
        - Derived from ``pyuvsim.Analytic beam``
        - Based on axisymmetric Chebyshev polynomial fits to the Fagnoni beam.
        - PerturbedPolyBeam is capable of expressing a range of non-redundancy effects,
          including per-beam stretch factors, perturbed sidelobes, and
          ellipticity/rotation.

v0.3.0 [2019.12.10]
===================

Added
-----
- New sub-package ``simulators``
    - ``VisibilitySimulators`` class
        - Provides a common interface to interferometric visibility simulators.
          Users instantiate one of its subclasses and provide input antenna and
          sky scenarios.
        - ``HealVis`` subclass
        - Provides an interface to the ``healvis`` visibility simulator.
    - ``VisCPU`` subclass
        - Provides an interface to the ``viscpu`` visibility simulator.
    - ``conversions`` module
        - Not intended to be interfaced with by the end user; it provides useful
          coordinate transformations for ``VisibilitySimulators``.

v0.2.0 [2019.11.20]
===================

Added
-----
- Command-line Interface
    - Use anywhere with `hera_sim run [options] INPUT`
    - Tutorial available on readthedocs

- Enhancement of `run_sim` method of `Simulator` class
   - Allows for each simulation component to be returned
      - Components returned as a list of 2-tuples `(model_name, visibility)`
      - Components returned by specifying `ret_vis=True` in their kwargs

- Option to seed random number generators for various methods
   - Available via the `Simulator.add_` methods by specifying the kwarg \
     `seed_redundantly=True`
   - Seeds are stored in `Simulator` object, and may be saved as a `npy` \
     file when using the `Simulator.write_data` method

- New YAML tag `!antpos`
   - Allows for antenna layouts to be constructed using `hera_sim.antpos` \
     functions by specifying parameters in config file

Fixed
-----

- Changelog formatting for v0.1.0 entry

Changed
-------

- Implementation of `defaults` module
   - Allows for semantic organization of config files
   - Parameters that have the same name take on the same value
      - e.g. `std` in various `rfi` functions only has one value, even if \
        it's specified multiple times

v0.1.0 [2019.08.28]
===================

Added
-----

- New module `interpolators`
   - Classes intended to be interfaced with by end-users:
      - `Tsky`
         - Provides an interface for generating a sky temperature \
           interpolation object when provided with a `.npz` file \
           and interpolation kwargs.
      - `Beam`, `Bandpass`
         - Provides an interface for generating either a `poly1d` or \
           `interp1d` interpolation object when provided with an \
           appropriate datafile.

- New module `defaults`
   - Provides an interface which allows the user to dynamically adjust \
     default parameter settings for various `hera_sim` functions.

- New module `__yaml_constructors`
   - Not intended to be interfaced with by the end user; this module just \
     provides a location for defining new YAML tags to be used in conjunction \
     with the `defaults` module features and the `Simulator.run_sim` method.

- New directory `config`
   - Provides a location to store configuration files.

Fixed
-----

Changed
-------

- HERA-specific variables had their definitions removed from the codebase.
  Objects storing these variables still exist in the codebase, but their
  definitions now come from loading in data stored in various new files
  added to the `data` directory.

v0.0.1
======

- Initial released version
