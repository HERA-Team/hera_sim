"""Module containing a high-level interface for :mod:`hera_sim`.

This module defines the :class:`Simulator` class, which provides the user
with a high-level interface to all of the features provided by :mod:`hera_sim`.
For detailed instructions on how to manage a simulation using the
:class:`Simulator`, please refer to the tutorials.
"""

import contextlib
import functools
import inspect
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union

import numpy as np
import yaml
from astropy import constants as const
from cached_property import cached_property
from deprecation import deprecated
from pyuvdata import UVData
from pyuvdata import utils as uvutils

from . import __version__, io, utils
from .components import SimulationComponent, get_model, list_all_components
from .defaults import defaults

_add_depr = deprecated(
    deprecated_in="1.0", removed_in="2.0", details="Use the :meth:`add` method instead."
)

# Define some commonly used types for typing purposes.
AntPairPol = tuple[int, int, str]
AntPair = tuple[int, int]
AntPol = tuple[int, str]
Component = Union[str, type[SimulationComponent], SimulationComponent]


# wrapper for the run_sim method, necessary for part of the CLI
def _generator_to_list(func, *args, **kwargs):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        result = list(func(*args, **kwargs))
        return None if result == [] else result

    return new_func


class Simulator:
    """Simulate visibilities and/or instrumental effects for an entire array.

    Parameters
    ----------
    data
        :class:`pyuvdata.UVData` object to use for the simulation or path to a
        UVData-supported file.
    defaults_config
        Path to defaults configuraiton, seasonal keyword, or configuration
        dictionary for setting default simulation parameters. See tutorial
        on setting defaults for further information.
    redundancy_tol
        Position tolerance for finding redundant groups, in meters. Default is
        1 meter.
    kwargs
        Parameters to use for initializing UVData object if none is provided.
        If ``data`` is a file path, then these parameters are used when reading
        the file. Otherwise, the parameters are used in creating a ``UVData``
        object using :func:`~.io.empty_uvdata`.

    Attributes
    ----------
    data : :class:`pyuvdata.UVData` instance
        Object containing simulated visibilities and metadata.
    extras : dict
        Dictionary to use for storing extra parameters.
    antpos : dict
        Dictionary pairing antenna numbers to ENU positions in meters.
    lsts : np.ndarray of float
        Observed LSTs in radians.
    freqs : np.ndarray of float
        Observed frequencies in GHz.
    times : np.ndarray of float
        Observed times in JD.
    pols : list of str
        Polarization strings.
    red_grps : list of list of int
        Redundant baseline groups. Each entry is a list containing the baseline
        integer for each member of that redundant group.
    red_vecs : list of :class:`numpy.ndarray` of float
        Average of all the baselines for each redundant group.
    red_lengths : list of float
        Length of each redundant baseline.
    """

    def __init__(
        self,
        *,
        data: Optional[Union[str, UVData]] = None,
        defaults_config: Optional[Union[str, dict]] = None,
        redundancy_tol: float = 1.0,
        **kwargs,
    ):
        # TODO: add ability for user to specify parameter names to look for on
        # parsing call signature
        # Create some utility dictionaries.
        self._components = {}
        self._seeds = {}
        self._antpairpol_cache = {}
        self._filter_cache = {"delay": {}, "fringe": {}}

        # apply and activate defaults if specified
        if defaults_config:
            self.apply_defaults(defaults_config)

        # actually initialize the UVData object stored in self.data
        self._initialize_data(data, **kwargs)
        self._calculate_reds(tol=redundancy_tol)
        self.extras = self.data.extra_keywords
        for param in ("Ntimes", "Nfreqs", "Nblts", "Npols", "Nbls"):
            setattr(self, param, getattr(self.data, param))
        self.Nants = len(self.antpos)

        # Let's make some helpful methods from the UVData object available
        for attr in ("data", "flags", "antpairs", "antpairpols", "pols"):
            setattr(self, f"get_{attr}", getattr(self.data, f"get_{attr}"))

    @property
    def antenna_numbers(self):
        return self.data.antenna_numbers

    @property
    def ant_1_array(self):
        return self.data.ant_1_array

    @property
    def ant_2_array(self):
        return self.data.ant_2_array

    @property
    def polarization_array(self):
        return self.data.polarization_array

    @property
    def data_array(self):
        """Array storing the visibilities."""
        return self.data.data_array

    @property
    def antpos(self):
        """Mapping between antenna numbers and ENU positions in meters."""
        antpos, ants = self.data.get_ENU_antpos(pick_data_ants=True)
        return dict(zip(ants, antpos))

    @property
    def lsts(self):
        """Observed Local Sidereal Times in radians."""
        # This process retrieves the unique LSTs while respecting phase wraps.
        _, unique_inds = np.unique(self.data.time_array, return_index=True)
        return self.data.lst_array[unique_inds]

    @property
    def freqs(self):
        """Frequencies in GHz."""
        return np.unique(self.data.freq_array) / 1e9

    @property
    def times(self):
        """Simulation times in JD."""
        return np.unique(self.data.time_array)

    @property
    def pols(self):
        """Array of polarization strings."""
        return self.data.get_pols()

    @cached_property
    def integration_time(self):
        """Integration time, assuming it's identical across baselines."""
        return np.mean(self.data.integration_time)

    @cached_property
    def channel_width(self):
        """Channel width, assuming each channel is the same width."""
        return np.mean(self.data.channel_width)

    def apply_defaults(self, config: Optional[Union[str, dict]], refresh: bool = True):
        """
        Apply the provided default configuration.

        Equivalent to calling :meth:`~hera_sim.defaults.set` with the same parameters.

        Parameters
        ----------
        config
            If given, either a path pointing to a defaults configuration
            file, a string identifier of a particular config (e.g. 'h1c')
            or a dictionary of configuration parameters
            (see :class:`~.defaults.Defaults`).
        refresh
            Whether to refresh the defaults.
        """
        defaults.set(config, refresh=refresh)

    def calculate_filters(
        self,
        *,
        delay_filter_kwargs: Optional[dict[str, Union[float, str]]] = None,
        fringe_filter_kwargs: Optional[dict[str, Union[float, str, np.ndarray]]] = None,
    ):
        """
        Pre-compute fringe-rate and delay filters for the entire array.

        Parameters
        ----------
        delay_filter_kwargs
            Extra parameters necessary for generating a delay filter. See
            :func:`utils.gen_delay_filter` for details.
        fringe_filter_kwargs
            Extra parameters necessary for generating a fringe filter. See
            :func:`utils.gen_fringe_filter` for details.
        """
        delay_filter_kwargs = delay_filter_kwargs or {}
        fringe_filter_kwargs = fringe_filter_kwargs or {}
        self._calculate_delay_filters(**delay_filter_kwargs)
        self._calculate_fringe_filters(**fringe_filter_kwargs)

    def add(
        self,
        component: Component,
        *,
        add_vis: bool = True,
        ret_vis: bool = False,
        seed: Optional[Union[str, int]] = None,
        vis_filter: Optional[Sequence] = None,
        component_name: Optional[str] = None,
        **kwargs,
    ) -> Optional[Union[np.ndarray, dict[int, np.ndarray]]]:
        """
        Simulate an effect then apply and/or return the result.

        Parameters
        ----------
        component
            Effect to be simulated. This can either be an alias of the effect,
            or the class (or instance thereof) that simulates the effect.
        add_vis
            Whether to apply the effect to the simulated data. Default is True.
        ret_vis
            Whether to return the simulated effect. Nothing is returned by default.
        seed
            How to seed the random number generator. Can either directly provide
            a seed as an integer, or use one of the supported keywords. See
            tutorial for using the :class:`Simulator` for supported seeding modes.
            Default is to use a seed based on the current random state.
        vis_filter
            Iterable specifying which antennas/polarizations for which the effect
            should be simulated. See tutorial for using the :class:`Simulator` for
            details of supported formats and functionality.
        component_name
            Name to use when recording the parameters used for simulating the effect.
            Default is to use the name of the class used to simulate the effect.
        **kwargs
            Optional keyword arguments for the provided ``component``.

        Returns
        -------
        effect
            The simulated effect; only returned if ``ret_vis`` is set to ``True``.
            If the simulated effect is multiplicative, then a dictionary mapping
            antenna numbers to the per-antenna effect (as a ``np.ndarray``) is
            returned. Otherwise, the effect for the entire array is returned with
            the same structure as the ``pyuvdata.UVData.data_array`` that the
            data is stored in.
        """
        # Obtain a callable reference to the simulation component model.
        model = self._get_component(component)
        model_key = (
            component_name if component_name else self._get_model_name(component)
        )
        if not isinstance(model, SimulationComponent):
            model = model(**kwargs)
        self._sanity_check(model)  # Check for component ordering issues.
        self._antpairpol_cache[model_key] = []  # Initialize this model's cache.
        if seed is None and add_vis:
            warnings.warn(
                "You have not specified how to seed the random state. "
                "This effect might not be exactly recoverable.",
                stacklevel=2,
            )

        # Record the component simulated and the parameters used.
        if defaults._override_defaults:
            for param in getattr(model, "kwargs", {}):
                if param not in kwargs and param in defaults():
                    kwargs[param] = defaults(param)
        self._components[model_key] = kwargs.copy()
        self._components[model_key]["alias"] = component

        # Simulate the effect by iterating over baselines and polarizations.
        data = self._iteratively_apply(
            model,
            add_vis=add_vis,
            ret_vis=ret_vis,
            vis_filter=vis_filter,
            antpairpol_cache=self._antpairpol_cache[model_key],
            seed=seed,
            model_key=model_key,
            **kwargs,
        )  # This is None if ret_vis is False

        if add_vis:
            self._update_history(model, **kwargs)
            if seed:
                self._components[model_key]["seed"] = seed
                self._update_seeds(model_key)
            if vis_filter is not None:
                self._components[model_key]["vis_filter"] = vis_filter
        else:
            del self._antpairpol_cache[model_key]
            del self._components[model_key]
            if self._seeds.get(model_key, None):
                del self._seeds[model_key]

        return data

    def get(
        self,
        component: Component,
        key: Optional[Union[int, str, AntPair, AntPairPol]] = None,
    ) -> Union[np.ndarray, dict[int, np.ndarray]]:
        """
        Retrieve an effect that was previously simulated.

        Parameters
        ----------
        component
            Effect that is to be retrieved. See :meth:`add` for more details.
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
            tutorial Jupyter notebook for the :class:`Simulator` for example usage.

        Notes
        -----
        This will only produce the correct output if the simulated effect is
        independent of the data itself. If the simulated effect contains a
        randomly-generated component, then the random seed must have been set
        when the effect was initially simulated.
        """
        # Retrieve the model and verify it has been simulated.
        if component in self._components:
            model = self._get_component(self._components[component]["alias"])
            model_key = component
        else:
            model = self._get_component(component)
            model_key = self._get_model_name(component)
            if model_key not in self._components:
                raise ValueError("The provided component has not yet been simulated.")

        # Parse the key and verify that it's properly formatted.
        ant1, ant2, pol = self._parse_key(key)
        self._validate_get_request(model, ant1, ant2, pol)

        # Prepare to re-simulate the effect.
        kwargs = self._components[model_key].copy()
        kwargs.pop("alias")  # To handle multiple instances of simulating an effect.
        seed = kwargs.pop("seed", None)
        vis_filter = kwargs.pop("vis_filter", None)
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
                model_key=model_key,
                **kwargs,
            )
            if ant1 is not None:
                if pol:
                    return gains[(ant1, pol)]
                return {key: gain for key, gain in gains.items() if ant1 in key}
            else:
                if pol:
                    return {key: gain for key, gain in gains.items() if pol in key}
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
                model_key=model_key,
                **kwargs,
            )

            # Trim the data if a specific polarization is requested.
            if pol is None:
                return data
            pol_ind = self.pols.index(pol)
            return data[:, :, pol_ind]

        # We're only simulating for a particular baseline.
        # (The validation check ensures this is the case.)
        # First, find out if it needs to be conjugated.
        try:
            blt_inds = self.data.antpair2ind(ant1, ant2)
            if blt_inds is None:
                raise ValueError
            conj_data = False
        except ValueError:
            blt_inds = self.data.antpair2ind(ant2, ant1)
            conj_data = True

        # We have three different seeding cases to work out.
        if seed == "initial":
            # Initial seeding means we need to do the whole array.
            data = self._iteratively_apply(
                model,
                add_vis=False,
                ret_vis=True,
                seed=seed,
                vis_filter=vis_filter,
                antpairpol_cache=None,
                model_key=model_key,
                **kwargs,
            )[blt_inds, :, :]
            if conj_data:  # pragma: no cover
                data = np.conj(data)
            if pol is None:
                return data
            pol_ind = self.data.get_pols().index(pol)
            return data[..., pol_ind]

        # Figure out whether we need to do a polarization selection.
        if pol is None:
            data_shape = (self.lsts.size, self.freqs.size, len(self.pols))
            pols = self.pols
            return_slice = (slice(None),) * 3
        else:
            data_shape = (self.lsts.size, self.freqs.size, 1)
            pols = (pol,)
            return_slice = (slice(None), slice(None), 0)

        # Prepare the model parameters, then simulate and return the effect.
        data = np.zeros(data_shape, dtype=complex)
        for i, _pol in enumerate(pols):
            args = self._initialize_args_from_model(model)
            args = self._update_args(args, model, ant1, ant2, pol)
            args.update(kwargs)
            if conj_data:
                _, rng = self._seed_rng(
                    seed, model, ant2, ant1, _pol, model_key=model_key
                )
            else:
                _, rng = self._seed_rng(
                    seed, model, ant1, ant2, _pol, model_key=model_key
                )
            args["rng"] = rng
            data[..., i] = model(**args)
        if conj_data:
            data = np.conj(data)
        return data[return_slice]

    def plot_array(self):
        """Generate a plot of the array layout in ENU coordinates."""
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
        """Refresh the object.

        This zeros the data array, resets the history, and clears the
        instance's ``_components`` dictionary.
        """
        self.data.data_array = np.zeros(self.data.data_array.shape, dtype=complex)
        self.data.history = ""
        self._components.clear()
        self._antpairpol_cache.clear()
        self._seeds.clear()
        self._filter_cache = {"delay": {}, "fringe": {}}
        self.extras.clear()

    def write(self, filename, save_format="uvh5", **kwargs):
        """Write the ``data`` to disk using a ``pyuvdata``-supported filetype."""
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
        """
        Run an entire simulation.

        Parameters
        ----------
        sim_file
            Path to a configuration file specifying simulation parameters.
            Required if ``sim_params`` is not provided.
        **sim_params
            Once-nested dictionary mapping simulation components to models,
            with each model mapping to a dictionary of parameter-value pairs.
            Required if ``sim_file`` is not provided.

        Returns
        -------
        components
            List of simulation components that were generated with the
            parameter ``ret_vis`` set to ``True``, returned in the order
            that they were simulated. This is only returned if there is
            at least one simulation component with ``ret_vis`` set to
            ``True`` in its configuration file/dictionary.

        Examples
        --------
        Suppose we have the following configuration dictionary::

            sim_params = {
                "pntsrc_foreground": {"seed": "once", "nsrcs": 500},
                "gains": {"seed": "once", "dly_rng": [-20, 20], "ret_vis": True},
                "reflections": {"seed": "once", "dly_jitter": 10},
            }

        Invoking this method with ``**sim_params`` as its argument will simulate
        visibilities appropriate for a sky with 500 point sources, generate
        bandpass gains for each antenna and apply the effect to the foreground
        data, then generate cable reflections with a Gaussian jitter in the
        reflection delays with a standard deviation of 10 ns and apply the
        effect to the data. The return value will be a list with one entry:
        a dictionary mapping antenna numbers to their associated bandpass gains.

        The same effect can be achieved by writing a YAML file that is loaded
        into a dictionary formatted as above. See the :class:`Simulator` tutorial
        for a more in-depth explanation of how to use this method.
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
            with open(sim_file) as config:
                try:
                    sim_params = yaml.load(config.read(), Loader=yaml.FullLoader)
                except Exception:
                    raise OSError("The configuration file was not able to be loaded.")

        # loop over the entries in the configuration dictionary
        for component, params in sim_params.items():
            # make sure that the parameters are a dictionary
            if not isinstance(params, dict):
                raise TypeError(
                    f"The parameters for {component} are not formatted "
                    "properly. Please ensure that the parameters for "
                    "each component are specified using a dictionary."
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

        This function is a thin wrapper around :func:`~.io.chunk_sim_and_save`;
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

    # -------------- Legacy Functions -------------- #
    @_add_depr
    def add_eor(self, model, **kwargs):
        """Add an EoR-like model to the visibilities."""
        return self.add(model, **kwargs)

    @_add_depr
    def add_foregrounds(self, model, **kwargs):
        """Add foregrounds to the visibilities."""
        return self.add(model, **kwargs)

    @_add_depr
    def add_noise(self, model, **kwargs):
        """Add thermal noise to the visibilities."""
        return self.add(model, **kwargs)

    @_add_depr
    def add_rfi(self, model, **kwargs):
        """Add RFI to the visibilities."""
        return self.add(model, **kwargs)

    @_add_depr
    def add_gains(self, **kwargs):
        """Apply bandpass gains to the visibilities."""
        return self.add("gains", **kwargs)

    @_add_depr
    def add_sigchain_reflections(self, ants=None, **kwargs):
        """Apply reflections to the visibilities. See :meth:`add` for details."""
        if ants is not None:
            kwargs.update(vis_filter=ants)
        return self.add("reflections", **kwargs)

    @_add_depr
    def add_xtalk(self, model="gen_whitenoise_xtalk", bls=None, **kwargs):
        """Add crosstalk to the visibilities. See :meth:`add` for more details."""
        if bls is not None:
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
        result: only baselines including antenna 0 have a simulated effect applied.

        ``vis_filter`` = ('xx',)
        returns: False if ``pol == "xx"`` else True
        result: only polarization "xx" has a simulated effect applied.

        ``vis_filter`` = (0, 1, 'yy')
        returns: False if ``(ant1, ant2, pol) in [(0, 1, 'yy'), (1, 0, 'yy)]``
        result: only baseline (0,1), or its conjugate, with polarization "yy" will
        have a simulated effect applied.
        """
        # If multiple keys are passed, do this recursively...
        multikey = any(isinstance(key, (list, tuple)) for key in vis_filter)
        if multikey:
            apply_filter = [
                Simulator._apply_filter(key, ant1, ant2, pol) for key in vis_filter
            ]
            return all(apply_filter)  # and approve if just one key fits.
        elif all(item is None for item in vis_filter):
            # Support passing a list of None.
            return False
        elif len(vis_filter) == 1:
            # For now, assume a string specifies a polarization.
            if isinstance(vis_filter[0], str):
                return not pol == vis_filter[0]
            # Otherwise, assume that this specifies an antenna.
            else:
                return vis_filter[0] not in (ant1, ant2)
        elif len(vis_filter) == 2:
            # TODO: This will need to be updated when we support ant strings.
            # Three cases: two pols; an ant+pol; a baseline.
            # If it's two polarizations, then make sure this pol is one of them.
            if all(isinstance(key, str) for key in vis_filter):
                return pol not in vis_filter
            # If it's an ant+pol, make sure both the antenna and pol are present.
            elif any(isinstance(key, str) for key in vis_filter):
                return not all(key in (ant1, ant2, pol) for key in vis_filter)
            # Otherwise, make sure the baseline is correct.
            else:
                return not (
                    utils._listify(vis_filter) == [ant1, ant2]
                    or utils._listify(vis_filter) == [ant2, ant1]
                )
        elif len(vis_filter) == 3:
            # Assume it's a proper antpairpol.
            return not (
                utils._listify(vis_filter) == [ant1, ant2, pol]
                or utils._listify(vis_filter) == [ant2, ant1, pol]
            )
        else:
            # Assume it's some list of antennas/polarizations.
            pols = []
            ants = []
            for key in vis_filter:
                if isinstance(key, str):
                    pols.append(key)
                elif isinstance(key, int):
                    ants.append(key)
            # We want polarization and ant1 or ant2 in the filter.
            # This would be used in simulating e.g. a few feeds that have an
            # abnormally high system temperature.
            return not (pol in pols and (ant1 in ants or ant2 in ants))

    def _calculate_reds(self, tol=1.0):
        """Calculate redundant groups and populate class attributes."""
        groups, centers, lengths = self.data.get_redundancies(tol=tol)
        self.red_grps = groups
        self.red_vecs = centers
        self.red_lengths = lengths

    def _calculate_delay_filters(
        self,
        *,
        standoff: float = 0.0,
        delay_filter_type: Optional[str] = "gauss",
        min_delay: Optional[float] = None,
        max_delay: Optional[float] = None,
        normalize: Optional[float] = None,
    ):
        """
        Calculate delay filters for each redundant group.

        Parameters
        ----------
        standoff
            Extra extent in delay that the filter extends out to in order to
            allow for suprahorizon emission. Should be specified in nanoseconds.
            Default buffer is zero.
        delay_filter_type
            String specifying the filter profile. See :func:`utils.gen_delay_filter`
            for details.
        min_delay
            Minimum absolute delay of the filter, in nanoseconds.
        max_delay
            Maximum absolute delay of the filter, in nanoseconds.
        normalize
            Normalization of the filter such that the output power is the product
            of the input power and the normalization factor.

        See Also
        --------
        :func:`utils.gen_delay_filter`
        """
        # Note that this is not the most efficient way of caching the filters;
        # however, this is algorithmically very simple--just use one filter per
        # redundant group. This could potentially be improved in the future,
        # but it should work fine for our purposes.
        for red_grp, bl_len in zip(self.red_grps, self.red_lengths):
            bl_len_ns = bl_len / const.c.to("m/ns").value
            bl_int = sorted(red_grp)[0]
            delay_filter = utils.gen_delay_filter(
                self.freqs,
                bl_len_ns,
                standoff=standoff,
                delay_filter_type=delay_filter_type,
                min_delay=min_delay,
                max_delay=max_delay,
                normalize=normalize,
            )
            self._filter_cache["delay"][bl_int] = delay_filter

    def _calculate_fringe_filters(
        self, *, fringe_filter_type: Optional[str] = "tophat", **filter_kwargs
    ):
        """
        Calculate fringe-rate filters for all baselines.

        Parameters
        ----------
        fringe_filter_type
            The fringe-rate filter profile.
        filter_kwargs
            Other parameters necessary for specifying the filter. These
            differ based on the filter profile.

        See Also
        --------
        :func:`utils.gen_fringe_filter`
        """
        # This uses the same simplistic approach as the delay filter
        # calculation does--just do one filter per redundant group.
        for red_grp, (blx, _bly, _blz) in zip(self.red_grps, self.red_vecs):
            ew_bl_len_ns = blx / const.c.to("m/ns").value
            bl_int = sorted(red_grp)[0]
            fringe_filter = utils.gen_fringe_filter(
                self.lsts,
                self.freqs,
                ew_bl_len_ns,
                fringe_filter_type=fringe_filter_type,
                **filter_kwargs,
            )
            self._filter_cache["fringe"][bl_int] = fringe_filter

    def _initialize_data(self, data: Optional[Union[str, Path, UVData]], **kwargs):
        """
        Initialize the ``data`` attribute with a ``UVData`` object.

        Parameters
        ----------
        data
            Either a ``UVData`` object or a path-like object to a file
            that can be loaded into a ``UVData`` object. If not provided,
            then sufficient keywords for initializing a ``UVData`` object
            must be provided. See :func:`io.empty_uvdata` for more
            information on which keywords are needed.

        Raises
        ------
        TypeError
            If the provided value for ``data`` is not an object that can
            be cast to a ``UVData`` object.
        """
        if data is None:
            self.data = io.empty_uvdata(**kwargs)
        elif isinstance(data, (str, Path)):
            self.data = self._read_datafile(data, **kwargs)
            self.data.extra_keywords["data_file"] = data
        elif isinstance(data, UVData):
            self.data = data
        else:
            raise TypeError(
                "data type not understood. Only a UVData object or a path to "
                "a UVData-compatible file may be passed as the data parameter. "
                "Otherwise, keywords must be provided to build a UVData object."
            )

        if not self.data.future_array_shapes:  # pragma: nocover
            self.data.use_future_array_shapes()

    def _initialize_args_from_model(self, model):
        """
        Retrieve the LSTs and/or frequencies required for a model.

        Parameters
        ----------
        model: callable
            Model whose argspec is to be inspected and recovered.

        Returns
        -------
        model_params: dict
            Dictionary mapping positional argument names to either an
            ``inspect._empty`` object or the relevant parameters pulled
            from the ``Simulator`` object. The only parameters that are
            not ``inspect._empty`` are "lsts" and "freqs", should they
            appear in the model's argspec.

        Examples
        --------
        Suppose we have the following function::

            def func(freqs, ants, other=None):
                pass

        The returned object would be a dictionary with keys ``freqs`` and
        ``ants``, with the value for ``freqs`` being ``self.freqs`` and
        the value for ``ants`` being ``inspect._empty``. Since ``other``
        has a default value, it will not be in the returned dictionary.
        """
        model_params = self._get_model_parameters(model)
        model_params = {
            k: v
            for k, v in model_params.items()
            if v is inspect._empty or k in model.attrs_to_pull
        }

        # Pull any attributes from the Simulator that are required.
        args = {}
        for param, value in model_params.items():
            if hasattr(self, param) and value in (None, inspect._empty):
                args[param] = getattr(self, param)

        model_params.update(args)

        return model_params

    def _iterate_antpair_pols(self):
        """Loop through all baselines and polarizations."""
        for ant1, ant2, pol in self.data.get_antpairpols():
            blt_inds = self.data.antpair2ind((ant1, ant2))
            pol_ind = self.data.get_pols().index(pol)
            if blt_inds is not None:
                yield ant1, ant2, pol, blt_inds, pol_ind

    def _iteratively_apply(
        self,
        model: SimulationComponent,
        *,
        add_vis: bool = True,
        ret_vis: bool = False,
        seed: str | int | None = None,
        vis_filter: Sequence | None = None,
        antpairpol_cache: Sequence[AntPairPol] | None = None,
        model_key: str | None = None,
        **kwargs,
    ) -> Union[np.ndarray, dict[int, np.ndarray]] | None:
        """
        Simulate an effect for an entire array.

        This method loops over every baseline and polarization in order
        to simulate the effect ``model`` for the full array. The result
        is optionally applied to the simulation's data and/or returned.

        Parameters
        ----------
        model
            Callable model used to simulate an effect.
        add_vis
            Whether to apply the effect to the simulation data. Default
            is to apply the effect.
        ret_vis
            Whether to return the simulated effect. Default is to not
            return the effect. Type of returned object depends on whether
            the effect is multiplicative or not.
        seed
            Either an integer specifying the seed to be used in setting
            the random state, or one of a select few keywords. Default
            is to use the current random state. See :meth:`_seed_rng`
            for descriptions of the supported seeding modes.
        vis_filter
            List of antennas, baselines, polarizations, antenna-polarization
            pairs, or antpairpols for which to simulate the effect. This
            specifies which of the above the effect is to be simulated for,
            and anything that does not meet the keys specified in this list
            does not have the effect applied to it. See :meth:`_apply_filter`
            for more details.
        antpairpol_cache
            List of (ant1, ant2, pol) tuples specifying which antpairpols have
            already had the effect simulated. Not intended for use by the
            typical end-user.
        model_key
            String identifying the model component being computed. This is
            handed around to ensure that random number generation schemes using
            the "initial" seeding routine can be recovered via ``self.get``.
        kwargs
            Extra parameters passed to ``model``.

        Returns
        -------
        effect: np.ndarray or dict
            The simulated effect. Only returned if ``ret_vis`` is set to True.
            If the effect is *not* multiplicative, then the returned object
            is an ndarray; otherwise, a dictionary mapping antenna numbers
            to ndarrays is returned.
        """
        # There's nothing to do if we're neither adding nor returning.
        if not add_vis and not ret_vis:
            warnings.warn(
                "You have chosen to neither add nor return the effect "
                "you are trying to simulate, so nothing will be "
                f"computed. This warning was raised for the model: {model_key}",
                stacklevel=2,
            )
            return

        # Initialize the antpairpol cache if we need to.
        if antpairpol_cache is None:
            antpairpol_cache = []

        # Pull relevant parameters from Simulator.
        # Also make placeholders for antenna/baseline dependent parameters.
        base_args = self._initialize_args_from_model(model)

        # Get a copy of the data array.
        data_copy = self.data.data_array.copy()

        # Pull useful auxilliary parameters.
        is_multiplicative = getattr(model, "is_multiplicative", None)
        is_smooth_in_freq = getattr(model, "is_smooth_in_freq", True)
        if is_multiplicative is None:
            warnings.warn(
                "You are attempting to compute a component but have "
                "not specified an ``is_multiplicative`` attribute for "
                "the component. The component will be added under "
                "the assumption that it is *not* multiplicative.",
                stacklevel=2,
            )
            is_multiplicative = False

        # Pre-simulate gains.
        if is_multiplicative:
            gains = {}
            args = self._update_args(base_args, model)
            args.update(kwargs)
            for pol in self.data.get_feedpols():
                if seed:
                    seed, rng = self._seed_rng(
                        seed, model, pol=pol, model_key=model_key
                    )
                    args["rng"] = rng
                polarized_gains = model(**args)
                for ant, gain in polarized_gains.items():
                    gains[(ant, pol)] = gain

        # Determine whether to use cached filters, and which ones to use if so.
        model_kwargs = getattr(model, "kwargs", {})
        use_cached_filters = any("filter" in key for key in model_kwargs)
        get_delay_filter = (
            is_smooth_in_freq
            and "delay_filter_kwargs" not in kwargs
            and "delay_filter_kwargs" in model_kwargs
            and bool(self._filter_cache["delay"])
        )
        get_fringe_filter = (
            "fringe_filter_kwargs" not in kwargs
            and "fringe_filter_kwargs" in model_kwargs
            and bool(self._filter_cache["fringe"])
        )
        use_cached_filters &= get_delay_filter or get_fringe_filter

        if model.return_type == "full_array":
            args = self._update_args(base_args, model)
            args.update(kwargs)
            if seed:
                if seed == "redundant":
                    warnings.warn(
                        "You are trying to set the random state once per "
                        "redundant group while simulating an effect that "
                        "computes the entire visibility matrix in one go. "
                        "Any randomness in the simulation component may not "
                        "come out as expected--please check your settings."
                        f"This warning was raised for model: {model_key}",
                        stacklevel=2,
                    )
                seed, rng = self._seed_rng(model, model_key=model_key)
                args["rng"] = rng
            data_copy += model(**args)
        else:
            # Iterate over the array and simulate the effect as-needed.
            for ant1, ant2, pol, blt_inds, pol_ind in self._iterate_antpair_pols():
                # Determine whether or not to filter the result.
                apply_filter = self._apply_filter(
                    utils._listify(vis_filter), ant1, ant2, pol
                )
                if apply_filter:
                    continue

                # Check if this antpairpol or its conjugate have been simulated.
                bl_in_cache = (ant1, ant2, pol) in antpairpol_cache
                conj_in_cache = (ant2, ant1, pol) in antpairpol_cache

                # Seed the random number generator.
                key = (ant2, ant1, pol) if conj_in_cache else (ant1, ant2, pol)
                seed, rng = self._seed_rng(seed, model, *key, model_key=model_key)

                # Prepare the actual arguments to be used.
                use_args = self._update_args(base_args, model, ant1, ant2, pol)
                use_args.update(kwargs)
                if model.is_randomized:
                    use_args["rng"] = rng
                if use_cached_filters:
                    filter_kwargs = self._get_filters(
                        ant1,
                        ant2,
                        get_delay_filter=get_delay_filter,
                        get_fringe_filter=get_fringe_filter,
                    )
                    use_args.update(filter_kwargs)

                # Cache simulated antpairpols if not filtered out.
                if not (bl_in_cache or conj_in_cache or apply_filter):
                    antpairpol_cache.append((ant1, ant2, pol))

                # Check whether we're simulating a gain or a visibility.
                if is_multiplicative:
                    # Calculate the complex gain, but only apply it if requested.
                    gain = gains[(ant1, pol[0])] * np.conj(gains[(ant2, pol[1])])
                    data_copy[blt_inds, :, pol_ind] *= gain
                else:
                    # I don't think this will ever be executed, but just in case...
                    if conj_in_cache and seed is None:  # pragma: no cover
                        conj_blts = self.data.antpair2ind((ant2, ant1))
                        vis = (data_copy - self.data.data_array)[
                            conj_blts, :, pol_ind
                        ].conj()
                    else:
                        vis = model(**use_args)

                    # and add it in
                    data_copy[blt_inds, :, pol_ind] += vis

        # return the component if desired
        # this is a little complicated, but it's done this way so that
        # there aren't *three* copies of the data array floating around
        # this is to minimize the potential of triggering a MemoryError
        if ret_vis:
            # return the gain dictionary if gains are simulated
            if is_multiplicative:
                return gains
            data_copy -= self.data.data_array
            # the only time we're allowed to have add_vis be False is
            # if ret_vis is True, and nothing happens if both are False
            # so this is the *only* case where we'll have to reset the
            # data array
            if add_vis:
                self.data.data_array += data_copy
            # otherwise return the actual visibility simulated
            return data_copy
        else:
            self.data.data_array = data_copy

    @staticmethod
    def _read_datafile(datafile: Union[str, Path], **kwargs) -> UVData:
        """Read a file as a ``UVData`` object.

        Parameters
        ----------
        datafile
            Path to a file containing visibility data readable by ``pyuvdata``.
        **kwargs
            Arguments passed to the ``UVData.read`` method.

        Returns
        -------
        UVData
            The read-in data object.
        """
        uvd = UVData()
        uvd.read(datafile, read_data=True, **kwargs)
        return uvd

    def _seed_rng(self, seed, model, ant1=None, ant2=None, pol=None, model_key=None):
        """
        Set the random state according to the provided parameters.

        This is a helper function intended to be used solely in the
        :meth:`_iteratively_apply` method. It exists in order to ensure that
        the simulated data is as realistic as possible, assuming the user
        understands the proper choice of seeding method to use for the
        various effects that can be simulated.

        Parameters
        ----------
        seed
            Either the random seed to use (when provided as an integer),
            or one of the following keywords:

                ``"once"``:
                    The random state is set to the same value for
                    every baseline and polarization; one unique seed is
                    created for each model that uses this seeding mode.
                    This is recommended for simulating point-source foregrounds
                    and per-antenna effects.
                ``"redundant"``:
                    The random state is only uniquely set once per redundant
                    group for a given model. This is recommended for simulating
                    diffuse foregrounds and the reionization signal.
                ``"initial"``:
                    The random state is set at the very beginning of the
                    iteration over the array. This is essentially the same as
                    using a seeding mode of ``None``, though not identical.
                    This is recommended for simulating thermal noise, or for
                    simulating an effect that has a random component that
                    changes between baselines.

        model
            Name of the model for which to either recover or cache the seed.
            This is used to lookup random state seeds in the :attr:`_seeds`
            dictionary.
        ant1
            First antenna in the baseline.
        ant2
            Second antenna in the baseline (for baseline-dependent effects).
        pol
            Polarization string.
        model_key
            Identifier for retrieving the model parameters from the
            ``self._components`` attribute. This is only needed for ensuring
            that random effects using the "initial" seed can be recovered
            with the ``self.get`` method.

        Returns
        -------
        updated_seed
            Either the input seed or ``None``, depending on the provided seed.
            This is just used to ensure that the logic for setting the random
            state in the :meth:`_iteratively_apply` routine works out.
        rng
            The random number generator to be used for producing the random effect.

        Raises
        ------
        TypeError
            The provided seed is not ``None``, an integer, or a string.
        ValueError
            Two cases: one, the ``"redundant"`` seeding mode is being used
            and a baseline isn't provided; two, the seed is a string, but
            is not one of the supported seeding modes.
        """
        model_key = model_key or self._get_model_name(model)
        if seed is None:
            rng = self._components[model_key].get("rng", np.random.default_rng())
            return (None, rng)
        if isinstance(seed, int):
            return (seed, np.random.default_rng(seed))
        if not isinstance(seed, str):
            raise TypeError(
                "The seeding mode must be specified as a string or integer. "
                "If an integer is provided, then it will be used as the seed."
            )
        if seed == "redundant":
            if ant1 is None or ant2 is None:
                raise ValueError(
                    "A baseline must be specified in order to "
                    "seed by redundant group."
                )
            # Determine the key for the redundant group this baseline is in.
            bl_int = self.data.antnums_to_baseline(ant1, ant2)
            key = (next(reds for reds in self.red_grps if bl_int in reds)[0],)
            if pol:
                key += (pol,)
            # seed the RNG accordingly
            seed = self._get_seed(model_key, key)
            return ("redundant", np.random.default_rng(seed))
        elif seed == "once":
            # this option seeds the RNG once per iteration of
            # _iteratively_apply, using the same seed every time
            # this is appropriate for antenna-based gains (where the
            # entire gain dictionary is simulated each time), or for
            # something like PointSourceForeground, where objects on
            # the sky are being placed randomly
            key = (pol,) if pol else 0
            seed = self._get_seed(model_key, key)
            return ("once", np.random.default_rng(seed))
        elif seed == "initial":
            # this seeds the RNG once at the very beginning of
            # _iteratively_apply. this would be useful for something
            # like ThermalNoise
            key = (pol,) if pol else -1
            rng = np.random.default_rng(self._get_seed(model_key, key))
            self._components[model_key]["rng"] = rng
            return (None, rng)
        else:
            raise ValueError("Seeding mode not supported.")

    def _update_args(self, args, model, ant1=None, ant2=None, pol=None):
        """
        Scan the provided arguments and pull data as necessary.

        This method searches the provided dictionary for various positional
        arguments that can be determined by data stored in the ``Simulator``
        instance. Please refer to the source code to see what argument
        names are searched for and how their values are obtained.

        Parameters
        ----------
        args: dict
            Dictionary mapping names of positional arguments to either
            a value pulled from the ``Simulator`` instance or an
            ``inspect._empty`` object. See .. meth: _initialize_args_from_model
            for details on what to expect (these two methods are always
            called in conjunction with one another).
        model: SimulationComponent
            The model being simulated. The model will define which attributes
            should be pulled from the ``Simulator``.
        ant1: int, optional
            Required parameter if an autocorrelation visibility or a baseline
            vector is in the keys of ``args``.
        ant2: int, optional
            Required parameter if a baseline vector is in the keys of ``args``.
        pol: str, optional
            Polarization string. Currently not used.
        """
        # TODO: review this and see if there's a smarter way to do it.
        new_params = {}
        for param, attr in model.attrs_to_pull.items():
            if param in ("autovis", "autovis_i"):
                new_params[param] = self.data.get_data(ant1, ant1, pol)
            elif param == "autovis_j":
                new_params[param] = self.data.get_data(ant2, ant2, pol)
            elif param == "bl_vec":
                bl_vec = self.antpos[ant2] - self.antpos[ant1]
                new_params[param] = bl_vec / const.c.to("m/ns").value
            elif param == "antpair":
                new_params[param] = (ant1, ant2)
            else:
                # The parameter can be retrieved directly from the Simulator
                new_params[param] = getattr(self, attr)

        use_args = args.copy()
        use_args.update(new_params)
        return use_args

    def _get_filters(
        self,
        ant1: int,
        ant2: int,
        *,
        get_delay_filter: bool = True,
        get_fringe_filter: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        Retrieve delay and fringe filters from the cache.

        Parameters
        ----------
        ant1
            First antenna in the baseline.
        ant2
            Second antenna in the baseline.
        get_delay_filter
            Whether to retrieve the delay filter.
        get_fringe_filter
            Whether to retrieve the fringe filter.

        Returns
        -------
        filters
            Dictionary containing the fringe and delay filters that
            have been pre-calculated for the provided baseline.
        """
        filters = {}
        if not get_delay_filter and not get_fringe_filter:
            # Save some CPU cycles.
            return filters
        bl_int = self.data.antnums_to_baseline(ant1, ant2)
        conj_bl_int = self.data.antnums_to_baseline(ant2, ant1)
        is_conj = False
        for red_grp in self.red_grps:
            if bl_int in red_grp:
                key = sorted(red_grp)[0]
                break
            if conj_bl_int in red_grp:
                key = sorted(red_grp)[0]
                is_conj = True
                break
        if get_delay_filter:
            delay_filter = self._filter_cache["delay"][key]
            filters["delay_filter_kwargs"] = {"delay_filter": delay_filter}
        if get_fringe_filter:
            fringe_filter = self._filter_cache["fringe"][key]
            if is_conj:
                # Fringes are seen to move in the opposite direction.
                fringe_filter = fringe_filter[::-1, :]
            filters["fringe_filter_kwargs"] = {"fringe_filter": fringe_filter}
        return filters

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
    def _get_component(
        component: Union[str, type[SimulationComponent], SimulationComponent],
    ) -> Union[SimulationComponent, type[SimulationComponent]]:
        """Normalize a component to be either a class or instance."""
        if isinstance(component, str):
            try:
                return get_model(component)
            except KeyError:
                raise ValueError(
                    f"The model {component!r} does not exist. The following models are "
                    f"available: \n{list_all_components()}."
                )
        elif isinstance(component, SimulationComponent):
            return component
        else:
            with contextlib.suppress(TypeError):
                if issubclass(component, SimulationComponent):
                    return component
            raise TypeError(
                "The input type for the component was not understood. "
                "Must be a string, or a class/instance of type 'SimulationComponent'. "
                f"Available component models are:\n{list_all_components()}"
            )

    def _generate_seed(self, model, key):
        """Generate a random seed and cache it in the ``self._seeds`` attribute.

        Parameters
        ----------
        model
            The name of the model to retrieve the random seed for, as it would
            appear in the ``self._components`` attribute. (This should always
            correspond to the ``model_key`` determined in the ``self.add`` method.)
        key
            The key to use for tracking the random seed. This is only really
            used for keeping track of random seeds that are set per polarization
            or per redundant group.
        """
        # Just to make it extra random.
        rng = np.random.default_rng()
        if model not in self._seeds:
            self._seeds[model] = {}
        self._seeds[model][key] = rng.integers(2**32)

    def _get_seed(self, model, key):
        """Retrieve or generate a random seed given a model and key.

        Parameters
        ----------
        model
            The name of the model to retrieve the random seed for, as it would
            appear in the ``self._components`` attribute. (This should always
            correspond to the ``model_key`` determined in the ``self.add`` method.)
        key
            The key to use for tracking the random seed. This is only really
            used for keeping track of random seeds that are set per polarization
            or per redundant group.

        Returns
        -------
        seed
            The random seed to use for setting the random state.
        """
        if model not in self._seeds:
            self._generate_seed(model, key)
        if key not in self._seeds[model]:
            self._generate_seed(model, key)
        return self._seeds[model][key]

    @staticmethod
    def _get_model_name(model):
        """Find out the (lowercase) name of a provided model."""
        if isinstance(model, str):
            return model.lower()
        elif isinstance(model, SimulationComponent):
            return model.__class__.__name__.lower()
        else:
            with contextlib.suppress(TypeError):
                if issubclass(model, SimulationComponent):
                    return model.__name__.lower()

            raise TypeError(
                "You are trying to simulate an effect using a custom function. "
                "Please refer to the tutorial for instructions regarding how "
                "to define new simulation components compatible with the Simulator."
            )

    def _parse_key(self, key: Union[int, str, AntPair, AntPairPol]) -> AntPairPol:
        """Convert a key of at-most length-3 to an (ant1, ant2, pol) tuple."""
        valid_pols = {
            k.lower()
            for k in {
                **uvutils.POL_STR2NUM_DICT,
                **uvutils.JONES_STR2NUM_DICT,
                **uvutils.CONJ_POL_DICT,
            }
        }
        valid_pols.update({"jee", "jen", "jne", "jnn"})

        def checkpol(pol):
            if pol is None:
                return None

            if not isinstance(pol, str):
                raise TypeError(f"Invalid polarization type: {type(pol)}.")

            if pol.lower() not in valid_pols:
                raise ValueError(f"Invalid polarization string: {pol}.")

            return pol

        if key is None:
            ant1, ant2, pol = None, None, None
        elif np.issubdtype(type(key), np.integer):
            # Figure out if it's an antenna or baseline integer
            if key in self.antpos:
                ant1, ant2, pol = key, None, None
            else:
                ant1, ant2 = self.data.baseline_to_antnums(key)
                pol = None
        elif isinstance(key, str):
            if key.lower() in ("auto", "cross"):
                raise NotImplementedError("Functionality not yet supported.")
            key = checkpol(key)
            ant1, ant2, pol = None, None, key
        else:

            def intify(x):
                return x if x is None else int(x)

            try:
                iter(key)  # ensure it's iterable
                if len(key) not in (2, 3):
                    raise TypeError

                if len(key) == 2:
                    if all(isinstance(val, int) for val in key):
                        ant1, ant2 = key
                        pol = None
                    else:
                        ant1, pol = intify(key[0]), checkpol(key[1])
                        ant2 = None
                else:
                    ant1, ant2, pol = intify(key[0]), intify(key[1]), checkpol(key[2])

            except TypeError:
                raise ValueError(
                    "Key must be an integer, string, antenna pair, or antenna "
                    f"pair with a polarization string. Got {key}."
                )
        return ant1, ant2, pol

    def _sanity_check(self, model):
        """Check that simulation components are applied sensibly."""
        has_data = not np.all(self.data.data_array == 0)
        is_multiplicative = getattr(model, "is_multiplicative", False)
        contains_multiplicative_effect = any(
            self._get_component(component["alias"]).is_multiplicative
            for component in self._components.values()
        )

        if is_multiplicative and not has_data:
            warnings.warn(
                "You are trying to compute a multiplicative "
                "effect, but no visibilities have been simulated yet.",
                stacklevel=1,
            )
        elif not is_multiplicative and contains_multiplicative_effect:
            warnings.warn(
                "You are adding visibilities to a data array "
                "*after* multiplicative effects have been introduced.",
                stacklevel=1,
            )

    def _update_history(self, model, **kwargs):
        """Record the component simulated and its parameters in the history."""
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
        if getattr(model, "is_multiplicative", False):
            pols = self.data.get_feedpols()
            pol_type = "Feed"
        else:
            pols = self.pols
            pol_type = "Visibility"
        if ant1 is None and ant2 is None:
            if pol is None or pol in pols:
                return
            else:
                raise ValueError(f"{pol_type} polarization {pol} not found.")

        if pol is not None and pol not in pols:
            raise ValueError(f"{pol_type} polarization {pol} not found.")

        if getattr(model, "is_multiplicative", False):
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
