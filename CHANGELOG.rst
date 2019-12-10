=========
Changelog
=========


v0.2.0 [2019.12.10]
==========

Added
-----

- New sub-package `simulators`
  - `VisibilitySimulators` class
     - Provides a common interface to interferometric visibility simulators.
       Users instantiate one of its subclasses and provide input antenna and
       sky scenarios.
    - `HealVis` subclass
      - Provides an interface to the `healvis` visibility simulator.
    - `VisCPU` subclass
      - Provides an interface to the `viscpu` visibility simulator.
  - `conversions` module
    - Not intended to be interfaced with by the end user; it provides useful
      coordinate transformations for `VisibilitySimulators`.

Fixed
-----

Changed
-------

v0.1.0 [2019.08.28]
===================

Added
-----

- New module `interpolators`
  - Classes intended to be interfaced with by end-users:
    - `Tsky`
      - Provides an interface for generating a sky temperature interpolation
        object when provided with a `.npz` file and interpolation kwargs.
    - `Beam`, `Bandpass`
      - Provides an interface for generating either a `poly1d` or `interp1d` 
        interpolation object when provided with an appropriate datafile.

- New module `defaults`
  - Provides an interface which allows the user to dynamically adjust default
    parameter settings for various `hera_sim` functions.

- New module `__yaml_constructors`
  - Not intended to be interfaced with by the end user; this module just
    provides a location for defining new YAML tags to be used in conjunction
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
