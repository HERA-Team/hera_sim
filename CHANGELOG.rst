=========
Changelog
=========

------
v0.1.0
------

^^^^^^^^^
Additions
^^^^^^^^^
* :module: interpolators
  * :class: Tsky
    * A class for generating a sky temperature interpolation object from
      an appropriately formatted `.npz` file. See docstring for details
      on how these `.npz` files must be formatted.
    * Behaves just like an interpolation object; is basically a wrapper
      for `scipy.interpolate.RectBivariateSpline`.
  * :class: freq_interp1d
    * A class for generating a one-dimensional interpolation object which
      is assumed to be a function of frequency. Intended to be used to model
      frequency-dependent beam sizes and noiseless bandpass gain responses.
    * Currently supports use of either polynomial interpolators (`numpy.poly1d`)
      or spline interpolators (`scipy.interpolate.interp1d`). See class docstring
      for details on how to generate these objects.

* :module: defaults
  * :class: _Defaults
    * A class for handling the dynamic switching of function kwarg default values
      in an interactive environment; kwarg defaults are specified in a configuration
      YAML file. Interpolation objects may be specified by providing a path to
      the `.npy` or `.npz` file, along with any interpolation kwargs, needed to
      instantiate one of the interpolation objects in :module: interpolators. See
      one of the configuration files in the `config` directory for an example of how
      to format a configuration YAML.
    * This class is intended to exist as a singleton; as such, it is not accessible 
      the end-user. The end-user must interface with an instance of this class, which
      is accessible via `hera_sim.defaults`.

* :module: __yaml_constructors
  * A helper module that creates new YAML tags which may be used to specify 
    interpolation objects in a configuration YAML. This module is not intended to 
    be interfaced with by the end-user.
