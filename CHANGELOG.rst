=========
Changelog
=========

v1.0.0 [???]
============

Added
-----

Fixed
-----

Changed
-------

- All functions that take frequencies and LSTs as arguments have had their signatures
  changed to ``func(lsts, freqs, \*args, \*\*kwargs)``.
- Changes to handling of functions which employ a fringe or delay filtering step with
  variable keywords for the filters used. Filter keywords are now specified with
  individual dictionaries called ``delay_filter_kwargs`` or ``fringe_filter_kwargs``,
  depending on the filter used. Functions affected by this change are
  ``diffuse_foreground`` and ``noiselike_eor``.
- Changes to parameters that shared the same name but represented conceptually different
  objects in various functions. Functions affected by this are:
  ``utils.rough_delay_filter``, ``utils.rough_fringe_filter`` and most RFI functions.
- The :func:`~hera_sim.io.empty_uvdata` function had its default keyword values
  set to ``None``. The keywords accepted by this function have also been changed to
  match their names in ``pyuvsim.simsetup.initialize_uvdata_from_keywords``.
- Changes to parameters in most RFI models. Optional parameters that are common to many
  models (but should not share the same name), such as ``std`` or ``chance``, have been
  prefixed with the model name and an underscore (e.g. ``dtv_chance``). Various other
  parameters have been renamed for consistency/clarity. Note that the ``freq_min`` and
  ``freq_max`` parameters for ``rfi_dtv`` have been replaced by the single parameter
  ``dtv_band``, which is a tuple denoting the edges of the DTV band in GHz.
- Functions in ``utils`` now use ``freqs`` instead of ``fqs``.

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
    - Use anywhere with ``hera_sim run [options] INPUT``
    - Tutorial available on readthedocs

- Enhancement of ``run_sim`` method of ``Simulator`` class
   - Allows for each simulation component to be returned
      - Components returned as a list of 2-tuples ``(model_name, visibility)``
      - Components returned by specifying ``ret_vis=True`` in their kwargs

- Option to seed random number generators for various methods
   - Available via the ``Simulator.add_`` methods by specifying the kwarg \
     ``seed_redundantly=True``
   - Seeds are stored in ``Simulator`` object, and may be saved as a ``npy`` \
     file when using the ``Simulator.write_data`` method

- New YAML tag ``!antpos``
   - Allows for antenna layouts to be constructed using ``hera_sim.antpos`` \
     functions by specifying parameters in config file

Fixed
-----

- Changelog formatting for v0.1.0 entry

Changed
-------

- Implementation of ``defaults`` module
   - Allows for semantic organization of config files
   - Parameters that have the same name take on the same value
      - e.g. ``std`` in various ``rfi`` functions only has one value, even if \
        it's specified multiple times

v0.1.0 [2019.08.28]
===================

Added
-----

- New module ``interpolators``
   - Classes intended to be interfaced with by end-users:
      - ``Tsky``
         - Provides an interface for generating a sky temperature \
           interpolation object when provided with a ``.npz`` file \
           and interpolation kwargs.
      - ``Beam``, ``Bandpass``
         - Provides an interface for generating either a ``poly1d`` or \
           ``interp1d`` interpolation object when provided with an \
           appropriate datafile.

- New module ``defaults``
   - Provides an interface which allows the user to dynamically adjust \
     default parameter settings for various ``hera_sim`` functions.

- New module ``__yaml_constructors``
   - Not intended to be interfaced with by the end user; this module just \
     provides a location for defining new YAML tags to be used in conjunction \
     with the ``defaults`` module features and the ``Simulator.run_sim`` method.

- New directory ``config``
   - Provides a location to store configuration files.

Fixed
-----

Changed
-------

- HERA-specific variables had their definitions removed from the codebase.
  Objects storing these variables still exist in the codebase, but their
  definitions now come from loading in data stored in various new files
  added to the ``data`` directory.

v0.0.1
======

- Initial released version
