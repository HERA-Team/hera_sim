"""Re-imagining of the simulation module."""

import functools
import inspect
import os
import sys
import warnings
import yaml
import time
from pathlib import Path

import numpy as np
from cached_property import cached_property
from pyuvdata import UVData
from astropy import constants as const
from typing import Type, Union, Tuple, Sequence, Optional, Dict

from . import io
from . import utils
from .defaults import defaults
from . import __version__
from .components import SimulationComponent


# Define some commonly used types for typing purposes.
AntPairPol = Tuple[int, int, str]
AntPair = Tuple[int, int]
AntPol = Tuple[int, str]
Component = Union[str, Type[SimulationComponent], SimulationComponent]


# wrapper for the run_sim method, necessary for part of the CLI
def _generator_to_list(func, *args, **kwargs):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        result = list(func(*args, **kwargs))
        return None if result == [] else result

    return new_func


class Simulator:
    """Class for managing a simulation.

    """

    # TODO: figure out how typing works for this
    def __init__(self, data=None, defaults_config=None, **kwargs):
        """Simulate visibilities and instrumental effects for an entire array.

        Parameters
        ----------
        data
            ``pyuvdata.UVData`` object to use for the simulation or path to a
            UVData-supported file.
        defaults_config
            Path to defaults configuraiton, seasonal keyword, or configuration
            dictionary for setting default simulation parameters. See tutorial
            on setting defaults for further information.
        kwargs
            Parameters to use for initializing UVData object if none is provided.
            If ``data`` is a file path, then these parameters are used when reading
            the file. Otherwise, the parameters are used in creating a ``UVData``
            object using ``io.empty_uvdata``.

        Attributes
        ----------
        data: ``pyuvdata.UVData``
            Object containing simulated visibilities and metadata.
        extras: dict
            Dictionary to use for storing extra parameters.
        antpos: dict
            Dictionary pairing antenna numbers to ENU positions in meters.
        lsts: np.ndarray of float
            Observed LSTs in radians.
        freqs: np.ndarray of float
            Observed frequencies in GHz.
        times: np.ndarray of float
            Observed times in JD.
        pols: list of str
            Polarization strings.
        """
        # TODO: add ability for user to specify parameter names to look for on
        # parsing call signature
        # create some utility dictionaries
        self._components = {}
        self.extras = {}  # FIXME: we can just use self.data.extras
        self._seeds = {}
        self._antpairpol_cache = {}

        # apply and activate defaults if specified
        if defaults_config:
            self.apply_defaults(defaults_config)

        # actually initialize the UVData object stored in self.data
        self._initialize_data(data, **kwargs)
        for param in ("Ntimes", "Nfreqs", "Nblts", "Npols", "Nbls"):
            setattr(self, param, getattr(self.data, param))
        self.Nants = len(self.antpos)

    @cached_property
    def antpos(self):
        # TODO: docstring
        """
        """
        antpos, ants = self.data.get_ENU_antpos(pick_data_ants=True)
        return dict(zip(ants, antpos))

    @cached_property
    def lsts(self):
        """Observed Local Sidereal Times in radians."""
        # This process retrieves the unique LSTs while respecting phase wraps.
        unique_lsts, inverse_inds, counts = np.unique(
            self.data.lst_array, return_inverse=True, return_counts=True
        )
        return unique_lsts[inverse_inds[:: counts[0]]]

    @cached_property
    def freqs(self):
        """Frequencies in GHz."""
        return np.unique(self.data.freq_array) / 1e9

    @cached_property
    def times(self):
        """Simulation times in JD."""
        return np.unique(self.data.time_array)

    @cached_property
    def pols(self):
        """Array of polarization strings."""
        return self.data.get_pols()

    @cached_property
    def reds(self):
        """
        List of redundant groups; each entry is a list of baseline numbers.
        """
        return self.data.get_redundancies()[0]

    def apply_defaults(self, config, refresh=True):
        """
        Apply the provided default configuration.

        Equivalent to calling ``hera_sim.defaults`` with the same parameters.
        See ``hera_sim.defaults.set`` documentation for further details.
        """
        defaults.set(config, refresh=refresh)

    def add(
        self,
        component: Component,
        *,
        add_vis: bool = True,
        ret_vis: bool = False,
        seed: Optional[str] = None,
        vis_filter: Optional[Sequence] = None,
        **kwargs,
    ):
        """
        Simulate an effect then apply and/or return the result.

        Parameters
        ----------
        component: str or SimulationComponent subclass
            Effect to be simulated. This can either be an alias of the effect,
            or the class (or instance thereof) that simulates the effect.
        add_vis: bool, optional
            Whether to apply the effect to the simulated data. Default is True.
        ret_vis: bool, optional
            Whether to return the simulated effect. If the effect is a per-antenna
            effect, then a dictionary mapping antenna numbers to ``np.ndarray``s is
            returned. Otherwise, a ``pyuvdata.UVData.data_array``-style array is
            returned. Default is False,
        seed: str or int, optional
            How to seed the random number generator. Can either directly provide
            a seed as an integer, or use one of the supported keywords. See
            .. meth:: print_seed_types for information on supported keywords.
            Default is to not seed the random number generator.
        vis_filter: iterable, optional
            Iterable specifying which antennas/polarizations for which the effect
            should be simulated. See documentation of .. meth:: _apply_filter for
            details of supported formats and functionality.
        **kwargs
            Optional keyword arguments for the provided ``component``.
        """
        # Obtain a callable reference to the simulation component model.
        model = self._get_component(component)
        model_key = self._get_model_name(component)
        if not isinstance(model, SimulationComponent):
            model = model(**kwargs)
        self._sanity_check(model)  # Check for component ordering issues.
        self._antpairpol_cache[model_key] = []  # Initialize this model's cache.

        # Simulate the effect by iterating over baselines and polarizations.
        data = self._iteratively_apply(
            model,
            add_vis=add_vis,
            ret_vis=ret_vis,
            vis_filter=vis_filter,
            antpairpol_cache=self._antpairpol_cache[model_key],
            seed=seed,
            **kwargs,
        )  # This is None if ret_vis is False

        if add_vis:
            # Record the component simulated and the parameters used.
            if defaults._override_defaults:
                for param in model.kwargs:
                    if param not in kwargs and param in defaults():
                        kwargs[param] = defaults(param)
            self._update_history(model, **kwargs)
            if seed is not None:
                kwargs["seed"] = seed
                self._update_seeds(model_key)
            if vis_filter is not None:
                kwargs["vis_filter"] = vis_filter
            self._components[model_key] = kwargs
        else:
            del self._antpairpol_cache[model_key]

        return data

    def get(
        self,
        component: Component,
        key: Optional[Union[int, str, AntPair, AntPairPol]] = None,
    ) -> Union[np.ndarray, Dict[int, np.ndarray]]:
        """
        Retrieve an effect that was previously simulated.

        Parameters
        ----------
        component
            Effect that is to be retrieved. See .. meth:: add for more details.
        key
            Key for retrieving simulated effect. Possible choices are as follows:
                An integer may specify either a single antenna (for per-antenna
                effects) or be a ``pyuvdata``-style baseline integer.
                A string specifying a polarization can be used to retrieve the
                effect for every baseline for the specified polarization.
                A length-2 tuple of integers can be used to retrieve the effect
                for that baseline for all polarizations.
                A length-3 tuple specifies a particular baseline and polarization
                for which to retrieve the effect.
            Not specifying a key results in the effect being returned for all
            baselines (or antennas, if the effect is per-antenna) and polarizations.

        Returns
        -------
        effect
            The simulated effect appropriate for the provided key. Return type
            depends on the effect being simulated and the provided key. See the
            tutorial Jupyter notebook for the ``Simulator`` for example usage.
        """
        # Retrieve the model and verify it has been simulated.
        model = self._get_component(component)
        model_key = self._get_model_name(component)
        if model_key not in self._components:
            raise ValueError("The provided component has not yet been simulated.")

        # Parse the key and verify that it's properly formatted.
        ant1, ant2, pol = self._parse_key(key)
        self._validate_get_request(model, ant1, ant2, pol)

        # Prepare to re-simulate the effect.
        kwargs = self._components[model_key].copy()
        seed = kwargs.pop("seed", None)
        vis_filter = kwargs.pop("vis_filter", None)
        antpairpol_cache = self._antpairpol_cache[model_key]
        if not isinstance(model, SimulationComponent):
            model = model(**kwargs)

        if model.is_multiplicative:
            # We'll get a dictionary back, so the handling is different.
            gains = self._iteratively_apply(
                model,
                add_vis=False,
                ret_vis=True,
                seed=seed,
                vis_filter=vis_filter,
                **kwargs,
            )
            if ant1 is not None:
                if pol:
                    return gains[pol][ant1]
                return {_pol: gains[_pol][ant1] for _pol in self.pols}
            else:
                if pol:
                    return gains[pol]
                return gains

        # Specifying neither antenna implies the full array's data is desired.
        if ant1 is None and ant2 is None:
            # Simulate the effect
            data = self._iteratively_apply(
                model,
                add_vis=False,
                ret_vis=True,
                seed=seed,
                vis_filter=vis_filter,
                antpairpol_cache=None,
                **kwargs,
            )

            # Trim the data if a specific polarization is requested.
            if pol is None:
                return data
            pol_ind = self.pols.index(pol)
            return data[:, 0, :, pol_ind]

        # We're only simulating for a particular baseline.
        # First, find out if it needs to be conjugated.
        try:
            blt_inds = self.data.antpair2ind(ant1, ant2)
            # I don't think this will ever execute, but just in case...
            if blt_inds.size == 0:
                raise ValueError  # pragma: no cover
            conj_data = False
        except ValueError:  # pragma: no cover
            blt_inds = self.data.antpair2ind(ant2, ant1)
            conj_data = True

        # We've got three different seeding cases to work out.
        if seed == "initial":
            # Initial seeding means we need to do the whole array.
            data = self._iteratively_apply(
                model,
                add_vis=False,
                ret_vis=True,
                seed=seed,
                vis_filter=vis_filter,
                antpairpol_cache=None,
                **kwargs,
            )[blt_inds, 0, :, :]
            if conj_data:  # pragma: no cover
                data = np.conj(data)
            if pol is None:
                return data
            pol_ind = self.data.get_pols().index(pol)
            return data[..., pol_ind]
        elif seed == "redundant":
            if conj_data:
                self._seed_rng(seed, model, ant2, ant1, pol)
            else:
                self._seed_rng(seed, model, ant1, ant2, pol)
        elif seed is not None:
            self._seed_rng(seed, model, ant1, ant2, pol)

        # Prepare the model parameters, then simulate and return the effect.
        if pol is None:
            data_shape = (self.lsts.size, self.freqs.size, len(self.pols))
            pols = self.pols
            return_slice = (slice(None),) * 3
        else:
            data_shape = (self.lsts.size, self.freqs.size, 1)
            pols = (pol,)
            return_slice = (slice(None), slice(None), 0)
        data = np.zeros(data_shape, dtype=np.complex)
        for i, _pol in enumerate(pols):
            args = self._initialize_args_from_model(model)
            args = self._update_args(args, ant1, ant2, pol)
            args.update(kwargs)
            if conj_data:
                self._seed_rng(seed, model, ant2, ant1, _pol)
            else:
                self._seed_rng(seed, model, ant1, ant2, _pol)
            data[..., i] = model(**args)
        if conj_data:
            data = np.conj(data)
        return data[return_slice]

    def plot_array(self):
        """Generate a plot of the array layout in ENU coordinates.

        """
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel("East Position [m]", fontsize=12)
        ax.set_ylabel("North Position [m]", fontsize=12)
        ax.set_title("Array Layout", fontsize=12)
        dx = 0.25
        for ant, pos in self.antpos.items():
            ax.plot(pos[0], pos[1], color="k", marker="o")
            ax.text(pos[0] + dx, pos[1] + dx, ant)
        return fig

    def refresh(self):
        """Refresh the Simulator object.

        This zeros the data array, resets the history, and clears the
        instance's _components dictionary.
        """
        self.data.data_array = np.zeros(self.data.data_array.shape, dtype=np.complex128)
        self.data.history = ""
        self._components.clear()
        self._antpairpol_cache.clear()
        self._seeds.clear()
        self.extras.clear()

    def write(self, filename, save_format="uvh5", **kwargs):
        # TODO: docstring
        """
        """
        try:
            getattr(self.data, f"write_{save_format}")(filename, **kwargs)
        except AttributeError:
            raise ValueError(
                "The save_format must correspond to a write method in UVData."
            )

    # TODO: Determine if we want to provide the user the option to retrieve
    # simulation components as a return value from run_sim. Remove the
    # _generator_to_list wrapper if we do not make that a feature.
    @_generator_to_list
    def run_sim(self, sim_file=None, **sim_params):
        # TODO: docstring
        """
        """
        # make sure that only sim_file or sim_params are specified
        if not (bool(sim_file) ^ bool(sim_params)):
            raise ValueError(
                "Either an absolute path to a simulation configuration "
                "file or a dictionary of simulation parameters may be "
                "passed, but not both. Please only pass one of the two."
            )

        # read the simulation file if provided
        if sim_file is not None:
            with open(sim_file, "r") as config:
                try:
                    sim_params = yaml.load(config.read(), Loader=yaml.FullLoader)
                except Exception:
                    raise IOError("The configuration file was not able to be loaded.")

        # loop over the entries in the configuration dictionary
        for component, params in sim_params.items():
            # make sure that the parameters are a dictionary
            if not isinstance(params, dict):
                raise TypeError(
                    "The parameters for {component} are not formatted "
                    "properly. Please ensure that the parameters for "
                    "each component are specified using a "
                    "dictionary.".format(component=component)
                )

            # add the component to the data
            value = self.add(component, **params)

            # if the user wanted to return the data, then
            if value is not None:
                yield component, value

    def chunk_sim_and_save(
        self,
        save_dir,
        ref_files=None,
        Nint_per_file=None,
        prefix=None,
        sky_cmp=None,
        state=None,
        filetype="uvh5",
        clobber=True,
    ):
        """
        Chunk a simulation in time and write to disk.

        This function is a thin wrapper around :func:`io.chunk_sim_and_save`;
        please see that function's documentation for more information.
        """
        io.chunk_sim_and_save(
            self.data,
            save_dir,
            ref_files=ref_files,
            Nint_per_file=Nint_per_file,
            prefix=prefix,
            sky_cmp=sky_cmp,
            state=state,
            filetype=filetype,
            clobber=clobber,
        )
        return

    # -------------- Legacy Functions -------------- #
    # TODO: write a deprecated wrapper function
    def add_eor(self, model, **kwargs):
        """
        Add an EoR-like model to the visibilities. See :meth:`add` for
        more details.
        """
        return self.add(model, **kwargs)

    def add_foregrounds(self, model, **kwargs):
        """
        Add foregrounds to the visibilities. See :meth:`add` for
        more details.
        """

        return self.add(model, **kwargs)

    def add_noise(self, model, **kwargs):
        """
        Add thermal noise to the visibilities. See :meth:`add` for
        more details.
        """
        return self.add(model, **kwargs)

    def add_rfi(self, model, **kwargs):
        """Add RFI to the visibilities. See :meth:`add` for more details."""
        return self.add(model, **kwargs)

    def add_gains(self, **kwargs):
        """
        Apply bandpass gains to the visibilities. See :meth:`add` for
        more details.
        """
        return self.add("gains", **kwargs)

    def add_sigchain_reflections(self, ants=None, **kwargs):
        """
        Apply reflection gains to the visibilities. See :meth:`add` for
        more details.
        """
        kwargs.update(ants=ants)
        return self.add("reflections", **kwargs)

    def add_xtalk(self, model="gen_whitenoise_xtalk", bls=None, **kwargs):
        """Add crosstalk to the visibilities. See :meth:`add` for more details."""
        kwargs.update(vis_filter=bls)
        return self.add(model, **kwargs)

    @staticmethod
    def _apply_filter(vis_filter, ant1, ant2, pol):
        """Determine whether to filter the visibility for (ant1, ant2, pol).

        Functionally, ``vis_filter`` specifies which (ant1, ant2, pol) tuples
        will have a simulated effect propagated through the ``_iteratively_apply``
        method. ``vis_filter`` acts as a logical equivalent of a passband filter.

        Parameters
        ----------
        vis_filter
            Either a polarization string, antenna number, baseline, antpairpol
            (baseline + polarization), collection of antenna numbers and/or
            polarization strings, or collection of such keys.
        ant1, ant2, pol
            Baseline + polarization to compare against the provided filter.

        Returns
        -------
        apply_filter
            False if the provided antpairpol satisfies any of the keys provided
            in ``vis_filter``; True otherwise. See examples for details.

        Examples
        --------
        ``vis_filter`` = (0,)
        returns: False for any baseline including antenna 0
            -> only baselines including antenna 0 have a simulated effect applied.

        ``vis_filter`` = ('xx',)
        returns: False if ``pol == "xx"`` else True
            -> only polarization "xx" has a simulated effect applied.

        ``vis_filter`` = (0, 1, 'yy')
        returns: False if ``(ant1, ant2, pol) in [(0, 1, 'yy'), (1, 0, 'yy)]``
            -> only baseline (0,1), or its conjugate, with polarization 'yy' will
            have a simulated effect applied.
        """
        # find out whether or not multiple keys are passed
        multikey = any(isinstance(key, (list, tuple)) for key in vis_filter)
        # iterate over the keys, find if any are okay
        if multikey:
            apply_filter = [
                Simulator._apply_filter(key, ant1, ant2, pol) for key in vis_filter
            ]
            # if a single filter says to let it pass, then do so
            return all(apply_filter)
        elif all(item is None for item in vis_filter):
            # support passing tuple of None
            return False
        elif len(vis_filter) == 1:
            # check if the polarization matches, since the only
            # string identifiers should be polarization strings
            # TODO: add support for antenna strings (e.g. 'auto')
            if isinstance(vis_filter, str):
                return not pol == vis_filter[0]
            # otherwise assume that this is specifying an antenna
            else:
                return not vis_filter[0] in (ant1, ant2)
        elif len(vis_filter) == 2:
            # there are three cases: two polarizations are specified;
            # an antpol is specified; a baseline is specified
            # first, handle the case of two polarizations
            if all(isinstance(key, str) for key in vis_filter):
                return pol not in vis_filter
            # otherwise it's simple
            else:
                return not all(key in (ant1, ant2, pol) for key in vis_filter)
        elif len(vis_filter) == 3:
            # assume it's a proper antpairpol
            return not (
                vis_filter == [ant1, ant2, pol] or vis_filter == [ant2, ant1, pol]
            )
        else:
            # assume it's some list of antennas/polarizations
            return not any(key in (ant1, ant2, pol) for key in vis_filter)

    def _initialize_data(self, data, **kwargs):
        # TODO: docstring
        """
        """
        if data is None:
            self.data = io.empty_uvdata(**kwargs)
        elif isinstance(data, (str, Path)):
            self.data = self._read_datafile(data, **kwargs)
            self.extras["data_file"] = data
        elif isinstance(data, UVData):
            self.data = data
        else:
            raise TypeError("Unsupported type.")  # make msg better

    def _initialize_args_from_model(self, model):
        # TODO: docstring
        """
        """
        model_params = self._get_model_parameters(model)
        model_params = {k: v for k, v in model_params.items() if v is inspect._empty}

        # pull the lst and frequency arrays as required
        args = {
            param: getattr(self, param)
            for param in model_params
            if param in ("lsts", "freqs")
        }

        model_params.update(args)

        return model_params

    def _iterate_antpair_pols(self):
        # TODO: docstring
        """
        """
        for ant1, ant2, pol in self.data.get_antpairpols():
            blt_inds = self.data.antpair2ind((ant1, ant2))
            pol_ind = self.data.get_pols().index(pol)
            yield ant1, ant2, pol, blt_inds, pol_ind

    # TODO: think about how to streamline this algorithm and make it more readable
    # In particular, make the logic for adding/returning the effect easier to follow.
    def _iteratively_apply(
        self,
        model,
        add_vis=True,
        ret_vis=False,
        seed=None,
        vis_filter=None,
        antpairpol_cache=None,
        **kwargs,
    ):
        # TODO: docstring
        """
        """
        # do nothing if neither adding nor returning the effect
        if not add_vis and not ret_vis:
            warnings.warn(
                "You have chosen to neither add nor return the effect "
                "you are trying to simulate, so nothing will be "
                "computed. This warning was raised for the model: "
                "{model}".format(model=self._get_model_name(model))
            )
            return

        # make an empty list for antpairpol cache if it's none
        if antpairpol_cache is None:
            antpairpol_cache = []

        # pull lsts/freqs if required and find out which extra
        # parameters are required
        base_args = self._initialize_args_from_model(model)

        # get a copy of the data array
        data_copy = self.data.data_array.copy()

        # find out if the model is multiplicative
        is_multiplicative = getattr(model, "is_multiplicative", None)

        # handle user-defined functions as the passed model
        if is_multiplicative is None:
            warnings.warn(
                "You are attempting to compute a component but have "
                "not specified an ``is_multiplicative`` attribute for "
                "the component. The component will be added under "
                "the assumption that it is *not* multiplicative."
            )
            is_multiplicative = False

        # Pre-simulate gains.
        if is_multiplicative:
            gains = {}
            args = self._update_args(base_args)
            args.update(kwargs)
            for pol in self.pols:
                if seed:
                    seed = self._seed_rng(seed, model, pol=pol)
                gains[pol] = model(**args)

        for ant1, ant2, pol, blt_inds, pol_ind in self._iterate_antpair_pols():
            # Determine whether or not to filter the result.
            apply_filter = self._apply_filter(
                utils._listify(vis_filter), ant1, ant2, pol
            )

            # Check if this antpairpol or its conjugate have been simulated.
            bl_in_cache = (ant1, ant2, pol) in antpairpol_cache
            conj_in_cache = (ant2, ant1, pol) in antpairpol_cache

            if seed == "redundant" and conj_in_cache:
                # Ensure that V_ij = conj(V_ji).
                seed = self._seed_rng(seed, model, ant2, ant1, pol)
            elif seed is not None:
                seed = self._seed_rng(seed, model, ant1, ant2, pol)

            # Prepare the actual arguments to be used.
            use_args = self._update_args(base_args, ant1, ant2, pol)
            use_args.update(kwargs)

            # Cache simulated antpairpols if not filtered out.
            if not (bl_in_cache or conj_in_cache or apply_filter):
                antpairpol_cache.append((ant1, ant2, pol))

            # Check whether we're simulating a gain or a visibility.
            if is_multiplicative:
                # Calculate the complex gain, but only apply it if requested.
                gain = gains[pol][ant1] * np.conj(gains[pol][ant2])
                if not apply_filter:
                    data_copy[blt_inds, 0, :, pol_ind] *= gain
            else:
                # if the conjugate baseline has been simulated and
                # the RNG was only seeded initially, then we should
                # not re-simulate to ensure invariance under complex
                # conjugation and swapping antennas
                if conj_in_cache and seed is None:
                    conj_blts = self.data.antpair2ind((ant2, ant1))
                    vis = (data_copy - self.data.data_array)[
                        conj_blts, 0, :, pol_ind
                    ].conj()
                else:
                    # TODO: see if it's not too complicated to use cached
                    # delay/fringe filters here.
                    vis = model(**use_args)

                # filter what's actually having data simulated
                if apply_filter:
                    vis = np.zeros_like(vis)

                # and add it in
                data_copy[blt_inds, 0, :, pol_ind] += vis

        # return the component if desired
        # this is a little complicated, but it's done this way so that
        # there aren't *three* copies of the data array floating around
        # this is to minimize the potential of triggering a MemoryError
        if ret_vis:
            data_copy -= self.data.data_array
            # the only time we're allowed to have add_vis be False is
            # if ret_vis is True, and nothing happens if both are False
            # so this is the *only* case where we'll have to reset the
            # data array
            if add_vis:
                self.data.data_array += data_copy
            # return the gain dictionary if gains are simulated
            if is_multiplicative:
                return gains
            # otherwise return the actual visibility simulated
            else:
                return data_copy
        else:
            self.data.data_array = data_copy

    @staticmethod
    def _read_datafile(datafile, **kwargs):
        # TODO: docstring
        """
        """
        uvd = UVData()
        uvd.read(datafile, read_data=True, **kwargs)
        return uvd

    def _seed_rng(self, seed, model, ant1=None, ant2=None, pol=None):
        # TODO: docstring
        """
        """
        if not isinstance(seed, str):
            raise TypeError("The seeding mode must be specified as a string.")
        if seed == "redundant":
            if ant1 is None or ant2 is None:
                raise TypeError(
                    "A baseline must be specified in order to "
                    "seed by redundant group."
                )
            # Determine the key for the redundant group this baseline is in.
            bl_int = self.data.antnums_to_baseline(ant1, ant2)
            key = (next(reds for reds in self.reds if bl_int in reds)[0],)
            if pol:
                key += (pol,)
            # seed the RNG accordingly
            np.random.seed(self._get_seed(model, key))
            return "redundant"
        elif seed == "once":
            # this option seeds the RNG once per iteration of
            # _iteratively_apply, using the same seed every time
            # this is appropriate for antenna-based gains (where the
            # entire gain dictionary is simulated each time), or for
            # something like PointSourceForeground, where objects on
            # the sky are being placed randomly
            key = (pol,) if pol else 0
            np.random.seed(self._get_seed(model, key))
            return "once"
        elif seed == "initial":
            # this seeds the RNG once at the very beginning of
            # _iteratively_apply. this would be useful for something
            # like ThermalNoise
            key = (pol,) if pol else -1
            np.random.seed(self._get_seed(model, key))
            return None
        else:
            raise ValueError("Seeding mode not supported.")

    def _update_args(self, args, ant1=None, ant2=None, pol=None):
        # TODO: docstring
        """
        """
        # TODO: add helpful warning if unexpected arguments in signature
        # helper for getting the correct parameter name
        def key(requires):
            return list(args)[requires.index(True)]

        # find out what needs to be added to args
        # for antenna-based gains
        _requires_ants = [param.startswith("ant") for param in args]
        requires_ants = any(_requires_ants)
        # for sky components
        _requires_bl_vec = [param.startswith("bl") for param in args]
        requires_bl_vec = any(_requires_bl_vec)
        # for cross-coupling xtalk
        _requires_vis = [param.find("vis") != -1 for param in args]
        requires_vis = any(_requires_vis)

        # check if this is an antenna-dependent quantity; should
        # only ever be true for gains (barring future changes)
        if requires_ants:
            new_param = {key(_requires_ants): self.antpos}
        # check if this is something requiring a baseline vector
        # current assumption is that these methods require the
        # baseline vector to be provided in nanoseconds
        elif requires_bl_vec:
            bl_vec = self.antpos[ant2] - self.antpos[ant1]
            bl_vec_ns = bl_vec * 1e9 / const.c.value
            new_param = {key(_requires_bl_vec): bl_vec_ns}
        # check if this is something that depends on another
        # visibility. as of now, this should only be cross coupling
        # crosstalk
        elif requires_vis:
            autovis = self.data.get_data(ant1, ant1, pol)
            new_param = {key(_requires_vis): autovis}
        else:
            new_param = {}
        # update appropriately and return
        use_args = args.copy()
        use_args.update(new_param)

        # there should no longer be any unspecified, required parameters
        # so this *shouldn't* error out
        use_args = {
            key: value
            for key, value in use_args.items()
            if not type(value) is inspect.Parameter
        }

        if any([val is inspect._empty for val in use_args.values()]):
            warnings.warn(
                "One of the required parameters was not extracted. "
                "Please check that the parameters for the model you "
                "are trying to add are detectable by the Simulator. "
                "The Simulator will automatically find the following "
                "required parameters: \nlsts \nfreqs \nAnything that "
                "starts with 'ant' or 'bl'\n Anything containing 'vis'."
            )

        return use_args

    @staticmethod
    def _get_model_parameters(model):
        """Retrieve the full model signature (init + call) parameters."""
        init_params = inspect.signature(model.__class__).parameters
        call_params = inspect.signature(model).parameters
        # this doesn't work correctly if done on one line
        model_params = {}
        for params in (call_params, init_params):
            for parameter, value in params.items():
                model_params[parameter] = value.default
        model_params.pop("kwargs", None)
        return model_params

    @staticmethod
    def _get_component(component):
        """Cast input ``component`` to a SimulationComponent or instance thereof."""
        try:
            if issubclass(component, SimulationComponent):
                # support passing user-defined classes that inherit from
                # the SimulationComponent base class to add method
                return component
            else:
                # issubclass will not raise a TypeError in python <= 3.6
                raise TypeError
        except TypeError:
            # this is raised if ``component`` is not a class
            if component.__class__.__name__ == "function":
                raise TypeError(
                    "You are attempting to add a component that is "
                    "modeled using a function. Please convert the "
                    "function to a callable class and try again."
                )
            if callable(component):
                return component
            if not isinstance(component, str):
                # TODO: update this error message to reflect the
                # change in allowed component types
                raise TypeError(
                    "``component`` must be either a class which "
                    "derives from ``SimulationComponent`` or an "
                    "instance of a callable class, or a function, "
                    "whose signature is:\n"
                    "func(lsts, freqs, *args, **kwargs)\n"
                    "If it is none of the above, then it must be "
                    "a string which corresponds to the name of a "
                    "``hera_sim`` class or an alias thereof."
                )

            # TODO: make this a private method _check_registry
            # keep track of all known aliases in case desired
            # component isn't found in the search
            all_aliases = []
            for registry in SimulationComponent.__subclasses__():
                for model in registry.__subclasses__():
                    aliases = (model.__name__,)
                    aliases += getattr(model, "_alias", ())
                    aliases = [alias.lower() for alias in aliases]
                    for alias in aliases:
                        all_aliases.append(alias)
                    if component.lower() in aliases:
                        return model

            # if this part is executed, then the model wasn't found, so
            string_of_aliases = ", ".join(set(all_aliases))
            raise UnboundLocalError(
                f"The component '{component}' wasn't found. The "
                f"following aliases are known: \n{string_of_aliases}\n"
                "Please ensure that the component you are trying "
                "to add is a subclass of a registry."
            )

    def _generate_seed(self, model, key):
        # TODO: docstring
        """
        """
        model = self._get_model_name(model)
        # for the sake of randomness
        np.random.seed(int(time.time() * 1e6) % 2 ** 32)
        if model not in self._seeds:
            self._seeds[model] = {}
        self._seeds[model][key] = np.random.randint(2 ** 32)

    def _get_seed(self, model, key):
        # TODO: docstring
        """
        """
        model = self._get_model_name(model)
        if model not in self._seeds:
            self._generate_seed(model, key)
        # TODO: handle conjugate baselines here instead of other places
        if key not in self._seeds[model]:
            self._generate_seed(model, key)
        return self._seeds[model][key]

    @staticmethod
    def _get_model_name(model):
        # TODO: docstring
        """
        """
        if isinstance(model, str):
            return model.lower()
        try:
            return model.__name__.lower()
        except AttributeError:
            # check if it's a user defined function
            if model.__class__.__name__ == "function":
                # don't allow users to pass functions, only classes
                # TODO: find out if this check always happens before
                # _get_component is called
                raise TypeError(
                    "You are trying to simulate an effect using a custom function. "
                    "Please refer to the tutorial for instructions regarding how "
                    "to define new simulation components compatible with the Simulator."
                )
            else:
                return model.__class__.__name__.lower()

    def _parse_key(self, key: Union[int, str, AntPair, AntPairPol]) -> AntPairPol:
        """Convert a key of at-most length-3 to an (ant1, ant2, pol) tuple."""
        if key is None:
            ant1, ant2, pol = None, None, None
        elif isinstance(key, int):
            # Figure out if it's an antenna or baseline integer
            if key in self.antpos:
                ant1, ant2, pol = key, None, None
            else:
                ant1, ant2 = self.data.baseline_to_antnums(key)
                pol = None
        elif isinstance(key, str):
            if key.lower() in ("auto", "cross"):
                raise NotImplementedError("Functionality not yet supported.")
            ant1, ant2, pol = None, None, key
        else:
            try:
                iter(key)
                if len(key) not in (2, 3):
                    raise TypeError
            except TypeError:
                raise ValueError(
                    "Key must be an integer, string, antenna pair, or antenna "
                    "pair with a polarization string."
                )
            if len(key) == 2:
                if all(type(val) is int for val in key):
                    ant1, ant2 = key
                    pol = None
                else:
                    ant1, pol = key
                    ant2 = None
            else:
                ant1, ant2, pol = key
        return ant1, ant2, pol

    def _sanity_check(self, model):
        # TODO: docstring
        """
        """
        has_data = not np.all(self.data.data_array == 0)
        is_multiplicative = getattr(model, "is_multiplicative", False)
        contains_multiplicative_effect = any(
            self._get_component(component).is_multiplicative
            for component in self._components
        )

        if is_multiplicative and not has_data:
            warnings.warn(
                "You are trying to compute a multiplicative "
                "effect, but no visibilities have been "
                "simulated yet."
            )
        elif not is_multiplicative and contains_multiplicative_effect:
            warnings.warn(
                "You are adding visibilities to a data array "
                "*after* multiplicative effects have been "
                "introduced."
            )

    def _update_history(self, model, **kwargs):
        # TODO: docstring
        """
        """
        component = self._get_model_name(model)
        vis_filter = kwargs.pop("vis_filter", None)
        msg = f"hera_sim v{__version__}: Added {component} using parameters:\n"
        for param, value in defaults._unpack_dict(kwargs).items():
            msg += f"{param} = {value}\n"
        if vis_filter is not None:
            msg += "Effect simulated for the following antennas/baselines/pols:\n"
            msg += ", ".join(vis_filter)
        self.data.history += msg

    def _update_seeds(self, model_name=None):
        """Update the seeds in the extra_keywords property."""
        seed_dict = {}
        for component, seeds in self._seeds.items():
            if model_name is not None and component != model_name:
                continue

            if len(seeds) == 1:
                seed = list(seeds.values())[0]
                key = "_".join([component, "seed"])
                seed_dict[key] = seed
            else:
                # This should only be raised for seeding by redundancy.
                # Each redundant group is denoted by the *first* baseline
                # integer for the particular redundant group. See the
                # _generate_redundant_seeds method for reference.
                for bl_int, seed in seeds.items():
                    key = "_".join([component, "seed", str(bl_int)])
                    seed_dict[key] = seed

        # Now actually update the extra_keywords dictionary.
        self.data.extra_keywords.update(seed_dict)

    def _validate_get_request(
        self, model: Component, ant1: int, ant2: int, pol: str
    ) -> None:
        """Verify that the provided antpairpol is appropriate given the model."""
        if ant1 is None and ant2 is None:
            if pol is None or pol in self.pols:
                return
            else:
                raise ValueError(f"Polarization {pol} not found.")

        if pol is not None and pol not in self.pols:
            raise ValueError(f"Polarization {pol} not found.")

        if model.is_multiplicative:
            if ant1 is not None and ant2 is not None:
                raise ValueError(
                    "At most one antenna may be specified when retrieving "
                    "a multiplicative effect."
                )
        else:
            if (ant1 is None) ^ (ant2 is None):
                raise ValueError(
                    "Either no antennas or a pair of antennas must be provided "
                    "when retrieving a non-multiplicative effect."
                )
            if ant1 not in self.antpos or ant2 not in self.antpos:
                raise ValueError("At least one antenna is not in the array layout.")
