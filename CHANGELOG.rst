=========
Changelog
=========


Unreleased
==========

Added
-----
- `@component` and `@model` decorators for specifying whether a given object is
  a simulation component or a model for a given simulation component
  - Add some notes detailing this
- Functionality to specify antenna configurations in `io.empty_uvdata` using a 
  function call, rather than being required to generate the array beforehand

Fixed
-----
- Silent bug in `Simulator.add_foregrounds`
  - Incorrect handling of datasets with multiple polarizations
    - Updated to handle multiple polarizations correctly
  - Inconsistent sky realizations
    - Sky realizations made consistent within redundant groups (or do we
      want this to be consistent across *all* baselines?)

Changed
-------
- HERA-specific variables removed from code and placed in season configuration objects
  - **The following variables have been affected by this change:
    - `noise.HERA_Tsky_mdl`
    - `noise.HERA_BEAM_POLY`
    - `sigchain.HERA_NRAO_BANDPASS`
    - `rfi.HERA_STATIONS`**
- Default handling updated to be more semantic; this involves reformatting the
  various configuration YAMLs in `config`
- Various helper functions made hidden and generalized where appropriate
  - **The following helper functions have been modified:
    - ...**

- **API updated so that it is consistent within a single module, and tends to 
  follow a regular format throughout the `hera_sim` package
  - The following modules have been affected by this API change:
    - ...**

v0.1.0 [DATE]
=============

Added
-----

- :module: interpolators
  - :class: Tsky
    - A class for generating a sky temperature interpolation object from
      an appropriately formatted `.npz` file. See docstring for details
      on how these `.npz` files must be formatted.
    - Behaves just like an interpolation object; is basically a wrapper
      for `scipy.interpolate.RectBivariateSpline`.
  - :class: freq_interp1d
    - A class for generating a one-dimensional interpolation object which
      is assumed to be a function of frequency. Intended to be used to model
      frequency-dependent beam sizes and noiseless bandpass gain responses.
    - Currently supports use of either polynomial interpolators (`numpy.poly1d`)
      or spline interpolators (`scipy.interpolate.interp1d`). See class docstring
      for details on how to generate these objects.

- :module: defaults
  - :class: _Defaults
    - A class for handling the dynamic switching of function kwarg default values
      in an interactive environment; kwarg defaults are specified in a configuration
      YAML file. Interpolation objects may be specified by providing a path to
      the `.npy` or `.npz` file, along with any interpolation kwargs, needed to
      instantiate one of the interpolation objects in :module: interpolators. See
      one of the configuration files in the `config` directory for an example of how
      to format a configuration YAML.
    - This class is intended to exist as a singleton; as such, it is not accessible 
      the end-user. The end-user must interface with an instance of this class, which
      is accessible via `hera_sim.defaults`.

- :module: __yaml_constructors
  - A helper module that creates new YAML tags which may be used to specify 
    interpolation objects in a configuration YAML. This module is not intended to 
    be interfaced with by the end-user.

- Added various `.npy` and `.npz` files which contain parameters for HERA-specific
  variables to the `data` directory.

- Added a `config` directory which stores configuration YAMLs

v0.0.1 [???]
============

- Initial released version
