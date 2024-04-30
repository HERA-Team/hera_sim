=========
Changelog
=========

dev
===

Added
-----
- Classes subclassed from ``SimulationComponent`` now have a ``is_randomized``
  class attribute that informs the ``Simulator`` of whether it should provide
  a ``BitGenerator`` to the class when simulating the component.
  - Classes which use a random component should now have a ``rng`` attribute,
    which should be treated in the same manner as other model parameters. In
    other words, random states are now effectively treated as model parameters.
- New simulator class ``FFTVis`` that uses the ``fftvis`` package to simulate
  visibilities. This is a CPU-based visibility simulator that is faster than
  ``MatVis`` for large, compact arrays.

Changed
-------
- All random number generation now uses the new ``numpy`` API.
  - Rather than seed the global random state, a new ``BitGenerator`` is made
    with whatever random seed is desired.
  - The Simulator API has remained virtually unchanged, but the internal logic
    that handles random state management has received a significant update.

Deprecated
----------

- Support for Python 3.9 has been dropped.

Fixed
-----
- API calls for pyuvdata v2.4.0.

v4.1.0 [2023.06.26]
===================
This release heavily focuses on performance of the visibility simulators, and in
particular the ``VisCPU`` simulator.

Fixed
-----
- Passing ``spline_interp_opts`` now correctly pipes these options through to the
  visibility simulators.

Added
-----
- New ``_blt_order_kws`` class-attribute for ``VisibilitySimulator`` subclasses, that
  can be used to create the mock metadata in an order corresponding to that required
  by the simulator (instead of re-ordering after data creation, which can take some
  time).
- New optional ``compress_data_model()`` method on ``VisibilitySimulator`` subclasses
  that allows unnecessary metadata in the ``UVData`` object to be dropped before
  simulation (can be restored afterwards with the associated ``restore_data_model()``).
  This can reduce peak memory usage.
- New ``check_antenna_conjugation`` parameter for the ``VisCPU`` simulator, to allow
  turning off checks for antenna conjugation, which takes time and is unnecessary for
  mock datasets.
- Dependency on ``hera-cli-utils`` which adds options like ``--log-level`` and ``--profile``
  to ``hera-sim-vis.py``.
- Option to use a taper in generating a bandpass.
- ``utils.tanh_window`` function for generating a two-sided tanh window.
- ``interpolators.Reflection`` class for building a complex reflection
  coefficient interpolator from a ``npz`` archive.
- Reflection coefficient and beam integral ``npz`` archives for the phase 1
  and phase 4 systems (i.e., dipole feed and Vivaldi feed).

Changed
-------
- ``run_check_acceptability`` is now ``False`` by default when constructing simulations
  from obsparams configuration files, to improve performance.
- For ``VisCPU`` simulator, we no longer copy the whole data array when simulating, but
  instead just fill the existing one, to save on peak RAM.
- Made ``VisCPU._reorder_vis()`` much faster (like 99% time reduction).
- The ``--compress`` option to ``hera-sim-vis.py`` is no longer a boolean flag but
  takes a file argument. This file will be written as a cache of the baseline-time indices
  required to keep when compressing by redundancy.

v4.0.0 [2023.05.22]
===================

Breaking Changes
----------------
- Removed the ``HealVis`` wrapper. Use ``pyuvsim`` instead.

Changed
-------
- Updated package to always use future array shapes for ``pyuvdata`` objects (this
  includes updates to ``PolyBeam`` and ``Simulator`` objects amongst others).

v3.1.1 [2023.02.23]
===================

Changed
-------
- Coupling matrix calculation in :class:`~.sigchain.MutualCoupling` has been updated
  to correctly calculate the coupling coefficients from the provided E-field beam.

v3.1.0 [2023.01.17]
===================

Added
-----
- :class:`~.sigchain.MutualCoupling` class that simulates the systematic described in Josaitis
  et al. 2021.
- New class attributes for the :class:`~.SimulationComponent` class:
    - ``return_type`` specifies what type of return value to expect;
    - ``attrs_to_pull`` specifies which ``Simulator`` attributes to use.
- Some helper functions for :class:`~.sigchain.MutualCoupling` matrix multiplications.
- More attributes from the underlying ``UVData`` object exposed to the :class:`~.Simulator`.

Changed
-------
- ``Simulator._update_args`` logic has been improved.
- :class:`~.Simulator` attributes ``lsts``, ``times``, and ``freqs`` are no longer cached.

v3.0.0
======

Removed
-------

- Finally removed ability to set ``use_pixel_beams`` and ``bm_pix`` on the VisCPU
  simulator. This was removed in v1.0.0 of ``vis_cpu``.
- Official support for py37.

Internals
---------

- Added isort and pyupgrade pre-commit hooks for cleaner code.

v2.3.4 [2022.06.08]
===================

Added
-----
- ``NotImplementedError`` raised when trying to simulate noise using an interpolated
  sky temperature and phase-wrapped LSTs.
- More comparison tests of pyuvsim wrapper.

Fixed
-----
- Inferred integration time in ``ThermalNoise`` when phase-wrapped LSTs are used.
- Added ``**kwargs`` to ``PolyBeam.interp`` method to match UVBeam.
- healvis wrapper properly sets cross-pol visibilities to zero.

Changed
-------
- Temporarily forced all UVData objects in the code to use current array shapes.

v2.3.3 [2022.02.21]
===================

Added
-----
- ``adjustment.interpolate_to_reference`` now supports interpolating in time when
  there is a phase wrap in LST.

Changed
-------
- Some logical statements in ``adjustment.interpolate_to_reference`` were changed
  to use binary operators on logical arrays instead of e.g. ``np.logical_or``.

v2.3.2 [2022.02.18]
===================

Added
-----
- ``_extract_kwargs`` attribute added to the ``SimulationComponent`` class. This
  attribute is used by the ``Simulator`` to determine which optional parameters
  should actually be extracted from the data.
- ``antpair`` optional parameter added to the ``ThermalNoise`` class. This is
  used to determine whether to simulate noise via the radiometer equation (as is
  appropriate for a cross-correlation) or to just add a bias from the receiver
  temperature (which is our proxy for what should happen to an auto-correlation).

Fixed
-----
- The ``Simulator`` class now correctly uses the auto-correlations to simulate
  noise for the cross-correlations.

v2.3.1 [2022.01.19]
===================

Fixed
-----
- Using the ``normalize_beams`` option is now possible with the ``from_config``
  class method.

v2.3.0 [2022.01.19]
===================

Added
-----
- ``normalize_beams`` option in ``ModelData`` class. Setting this parameter to
  ``True`` enforces peak-normalization on all of the beams used in the simulation.
  The default behavior is to not peak-normalize the beams.

v2.2.1 [2022.01.14]
===================

Added
-----
- ``OverAirCrossCoupling`` now has a parameter ``amp_norm``. This lets the user
  decide at what distance from the receiverator the gain of the emitted signal
  is equal to the base amplitude.

Fixed
-----
- ``OverAirCrossCoupling`` now only simulates the systematic for cross-correlations.
- ``ReflectionSpectrum`` class had its ``is_multiplicative`` attribute set to True.

v2.2.0 [2022.01.13]
===================

Added
-----
- New ``ReflectionSpectrum`` class to generate multiple reflections over a
  specified range of delays/amplitudes.

Fixed
-----
- Corrected some parameter initializations in ``sigchain`` module.

v2.1.0 [2022.01.12]
===================

Added
-----
- New ``OverAirCrossCoupling`` class to better model crosstalk in H1C data.

Changed
-------
- Slightly modified ``Simulator`` logic for automatically choosing parameter values.
  This extends the number of cases the class can handle, but will be changed in a
  future update.

v2.0.0 [2021.11.16]
===================

Added
-----
- New VisibilitySimulator interface. See the `<https://hera-sim.readthedocs.io/en/latest/tutorials/visibility_simulator.html> Visibility Simulator Tutorial`_
  for details. This is a breaking change for usage of the visibility simulators, and
  includes more robust handling of polarization, fixed ordering of data when put back
  into the ``UVData`` objects, more native support for using ``pyradiosky`` to define
  the sky model, and improved support for ``vis_cpu``.
- Interface directly to the ``pyuvsim`` simulation engine.
- Ability to load tutorial data from the installed package.
- New and refactored tests for visibility simulations.

Fixed
-----
- default ``feed_array`` for ``PolyBeam`` fixed.

Changed
-------
- Updated tutorial for the visibility simulator interface (see above link).
- ``vis_cpu``  made an optional extra
- removed the ``conversions`` module, which is now in the ``vis_cpu`` package.
- Can now properly use ``pyuvdata>=2.2.0``.


v1.1.1 [2021.08.21]
===================

Added
-----
- Add a Zernike polynomial beam model.

v1.1.0 [2021.08.04]
===================

Added
-----
- Enable polarization support for ``vis_cpu`` (handles polarized primary beams, but
  only Stokes I sky model so far)
- Add a polarized version of the analytic PolyBeam model.

v1.0.2 [2021.07.01]
===================

Fixed
-----
- Bug in retrieval of unique LSTs by :class:`~.Simulator` when a blt-order other than
  time-baseline is used has been fixed. LSTs should now be correctly retrieved.
- :func:`~.io.empty_uvdata` now sets the ``phase_type`` attribute to "drift".

v1.0.1 [2021.06.30]
===================

Added
-----

Fixed
-----
- Discrepancy in :class:`~.foregrounds.PointSourceForeground` documentation and actual
  implementation has been resolved. Simulated foregrounds now look reasonable.

Changed
-------
- The time parameters used for generating an example ``Simulator`` instance in the tutorial
  have been updated to match their description.
- :class:`~.Simulator` tutorial has been changed slightly to account for the foreground fix.

v1.0.0 [2021.06.16]
===================

Added
-----
- :mod:`~.adjustment` module from HERA Phase 1 Validation work
   - :func:`~.adjustment.adjust_to_reference`
      - High-level interface for making one set of data comply with another set of data.
        This may involve rephasing or interpolating in time and/or interpolating in
        frequency. In the case of a mismatch between the two array layouts, this algorithm
        will select a subset of antennas to provide the greatest number of unique baselines
        that remain in the downselected array.
  - All other functions in this module exist only to modularize the above function.
- :mod:`~.cli_utils` module providing utility functions for the CLI simulation script.
- :mod:`~.components` module providing an abstract base class for simulation components.
   - Any new simulation components should be subclassed from the
     :class:`~.components.SimulationComponent` ABC. New simulation components subclassed
     appropriately are automatically discoverable by the :class:`~.Simulator` class. A MWE
     for subclassing new components is as follows::

        @component
        class Component:
            pass

        class Model(Component):
            ...

     The ``Component`` base class tracks any models subclassed from it and makes it
     discoverable to the :class:`~.Simulator`.
- New "season" configuration (called ``"debug"``), intended to be used for debugging
  the :class:`~.Simulator` when making changes that might not be easily tested.
- :func:`~.io.chunk_sim_and_save` function from HERA Phase 1 Validation work
   - This function allows the user to write a :class:`pyuvdata.UVData` object to disk
     in chunks of some set number of integrations per file (either specified directly,
     or specified implicitly by providing a list of reference files). This is very
     useful for taking a large simulation and writing it to disk in a way that mimics
     how the correlator writes files to disk.
- Ability to generate noise visibilities based on autocorrelations from the data.
  This is achieved by providing a value for the ``autovis`` parameter in
  the ``thermal_noise`` function (see :class:`~.noise.ThermalNoise`).
- The :func:`~.sigchain.vary_gains_in_time` provides an interface for taking a gain
  spectrum and applying time variation (linear, sinusoidal, or noiselike) to any of
  the reflection coefficient parameters (amplitude, phase, or delay).
- The :class:`~.sigchain.CrossCouplingSpectrum` provides an interface for generating
  multiple realizations of the cross-coupling systematic spaced logarithmically in
  amplitude and linearly in delay. This is ported over from the Validation work.

Fixed
-----
- The reionization signal produced by ``eor.noiselike_eor`` is now guaranteed to
  be real-valued for autocorrelations (although the statistics of the EoR signal for
  the autocorrelations still need to be investigated for correctness).

Changed
-------

- **API BREAKING CHANGES**
   - All functions that take frequencies and LSTs as arguments have had their signatures
     changed to ``func(lsts, freqs, *args, **kwargs)``.
   - Functions that employ :func:`~.utils.rough_fringe_filter` or
     :func:`~.utils.rough_delay_filter` as part of the visibility calculation now have
     parameters ``delay_filter_kwargs`` and/or ``fringe_filter_kwargs``, which are
     dictionaries that are ultimately passed to the filtering functions.
     ``foregrounds.diffuse_foreground`` and ``eor.noiselike_eor`` are both affected by this.
   - Some parameters have been renamed to enable simpler handling of package-wide defaults.
     Parameters that have been changed are:
      - ``filter_type`` -> ``delay_filter_type`` in :func:`~.utils.gen_delay_filter`
      - ``filter_type`` -> ``fringe_filter_type`` in :func:`~.utils.gen_fringe_filter`
      - ``chance`` -> ``impulse_chance`` in ``rfi_impulse`` (see :class:`~.rfi.Impulse`)
      - ``strength`` -> ``impulse_strength`` in ``rfi_impulse`` (see :class:`~.rfi.Impulse`)
      - Similar changes were made in ``rfi_dtv`` (:class:`~.rfi.DTV`) and ``rfi_scatter``
        (:class:`~.rfi.Scatter`).
   - Any occurrence of the parameter ``fqs`` has been replaced with ``freqs``.
   - The ``noise.jy2T`` function was moved to :mod:`~.utils` and renamed. See
     :func:`~.utils.jansky_to_kelvin`.
   - The parameter ``fq0`` has been renamed to ``f0`` in :class:`~.rfi.RfiStation`.
   - The ``_listify`` function has been moved from :mod:`~.rfi` to :mod:`~.utils`.
   - ``sigchain.HERA_NRAO_BANDPASS`` no longer exists in the code, but may be loaded from
     the file ``HERA_H1C_BANDPASS.npy`` in the ``data`` directory.
- Other Changes
   - The :class:`~.Simulator` has undergone many changes that make the class much easier
     to use, while also providing a handful of extra features. The new :class:`~.Simulator`
     provides the following features:
      - A universal :meth:`~.Simulator.add` method for applying any of the effects
        implemented in ``hera_sim``, as well as any custom effects defined by the user.
      - A :meth:`~.Simulator.get` method that retrieves any previously simulated effect.
      - The option to apply a simulated effect to only a subset of antennas, baselines,
        and/or polarizations, accessed through using the ``vis_filter`` parameter.
      - Multiple modes of seeding the random state to achieve a higher degree of realism
        than previously available.
      - The :meth:`~.Simulator.calculate_filters` method pre-calculates the fringe-rate
        and delay filters for the entire array and caches the result. This provides a
        marginal-to-modest speedup for small arrays, but can provide a significant
        speedup for very large arrays. Benchmarking results TBD.
      - An instance of the :class:`~.Simulator` may be generated with an empty call to
        the class if any of the season defaults are active (or if the user has provided
        some other sufficiently complete set of default settings).
      - Some of the methods for interacting with the underlying :class:`pyuvdata.UVData`
        object have been exposed to the :class:`~.Simulator` (e.g. ``get_data``).
      - An easy reference to the :func:`~.io.chunk_sim_and_save` function.
   - :mod:`~.foregrounds`, :mod:`~.eor`, :mod:`~.noise`, :mod:`~.rfi`,
     :mod:`~.antpos`, and :mod:`~.sigchain` have been modified to implement the
     features using callable classes. The old functions still exist for
     backwards-compatibility, but moving forward any additions to visibility or
     systematics simulators should be implemented using callable classes and be
     appropriately subclassed from :class:`~.components.SimulationComponent`.
   - :func:`~.io.empty_uvdata` has had almost all of its parameter values set to default as
     ``None``. Additionally, the ``n_freq``, ``n_times``, ``antennas`` parameters are being
     deprecated and will be removed in a future release.
   - :func:`~.noise.white_noise` is being deprecated. This function has been moved to the
     utility module and can be found at :func:`~.utils.gen_white_noise`.

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
