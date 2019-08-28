"""
Primary interface module for hera_sim, defining a :class:`Simulator` class which provides a common API for all
effects produced by this package.
"""

import functools
import inspect
import sys
import warnings
import yaml
import time

import numpy as np
from cached_property import cached_property
from pyuvdata import UVData
from astropy import constants as const
from collections import OrderedDict

from . import io
from . import sigchain
from .interpolators import Tsky
from .version import version


class CompatibilityException(ValueError):
    pass

def _get_model(mod, name):
    return getattr(sys.modules["hera_sim." + mod], name)


class _model(object):
    """
    A decorator that defines a "model" addition for the Simulator class.

    The basic functionality of the model is to:

    1. Provide keywords "add_vis" and "ret_vis" to enable adding the resulting
       visibilities to the underlying dataset or returning the added visibilities.
    2. Automatically locate a callable model provided either that callable
       or a string function name (the module from which the callable is imported
       can be passed to the decorator, but is by default intepreted as the last
       part of the model name).
    3. Add a comment to the `history` of the UVData object concerning what
       exactly has ben added.
    """

    def __init__(self, base_module=None, multiplicative=False):
        self.base_module = base_module
        self.multiplicative = multiplicative

    def __call__(self, func, *args, **kwargs):
        name = func.__name__

        @functools.wraps(func)
        def new_func(obj, *args, **kwargs):

            # If "ret_vis" is set, then we want to return the visibilities
            # that are being added to the base. If add_vis is set to False,
            # we need to
            add_vis = kwargs.pop("add_vis", True)

            ret_vis = kwargs.pop("ret_vis", False)
            if not add_vis:
                ret_vis = True

            if ret_vis:
                initial_vis = obj.data.data_array.copy()

            # If this is a multiplicative model, and *no* additive models
            # have been called, raise a warning.
            if self.multiplicative and np.all(obj.data.data_array == 0):
                warnings.warn("You are trying to determine visibilities that depend on preceding visibilities, but " +
                              "no previous vis have been created.")
            elif not self.multiplicative and (hasattr(obj, "_added_models") and any([x[1] for x in obj._added_models])):
                # some of the previous models were multiplicative, and now we're trying to add.
                warnings.warn("You are adding absolute visibilities _after_ determining visibilities that should " +
                              "depend on these. Please re-consider.")

            if "model" in inspect.getargspec(func)[0]: # TODO: needs to be updated for python 3
                # Cases where there is a choice of model
                model = args[0] if args else kwargs.pop("model")

                # If the model is a str, get its actual callable.
                if isinstance(model, str):
                    if self.base_module is None:
                        self.base_module = name[4:]  # get the bit after the "add"

                    model = _get_model(self.base_module, model)

                func(obj, model, **kwargs)

                if not isinstance(model, str):
                    method = model.__name__
                else:
                    method = model

                method = "using {} ".format(method)
            else:
                # For cases where there is no choice of model.
                method = ""
                func(obj, *args, **kwargs)

            if add_vis:
                msg = "\nhera_sim v{version}: Added {component} {method_name}with kwargs: {kwargs}"
                obj.data.history += msg.format(
                    version=version,
                    component="".join(name.split("_")[1:]),
                    method_name=method,
                    kwargs=kwargs,
                )

                # Add this particular model to a cache of "added models" for this sim.
                # This can be gotten from history, but easier just to keep it here.
                if not hasattr(obj, "_added_models"):
                    obj._added_models = [(name, self.multiplicative)]
                else:
                    obj._added_models += [(name, self.multiplicative)]

            # Here actually return something.
            if ret_vis:
                res = obj.data.data_array - initial_vis

                # If we don't want to add the visibilities, set them back
                # to the original before returning.
                if not add_vis:
                    obj.data.data_array[:] = initial_vis[:]

                return res

        return new_func


class Simulator:
    """
    Primary interface object for hera_sim.

    Produces visibility simulations with various independent sky- and instrumental-effects, and offers the resulting
    visibilities in :class:`pyuvdata.UVData` format.
    """
 
    # make a dictionary whose values point to the various methods
    # used to add different simulation components
    SIMULATION_COMPONENTS = OrderedDict(
                            { 'noiselike_eor':'add_eor',
                              'diffuse_foreground':'add_foregrounds',
                              'pntsrc_foreground':'add_foregrounds',
                              'thermal_noise':'add_noise',
                              'rfi_stations':'add_rfi',
                              'rfi_impulse':'add_rfi',
                              'rfi_scatter':'add_rfi',
                              'rfi_dtv':'add_rfi',
                              'gains':'add_gains',
                              'sigchain_reflections':'add_sigchain_reflections',
                              'gen_whitenoise_xtalk':'add_xtalk',
                              'gen_cross_coupling_xtalk':'add_xtalk'
                              } )

    def __init__(
            self,
            data_filename=None,
            data = None,
            refresh_data=False,
            n_freq=None,
            n_times=None,
            antennas=None,
            **kwargs
    ):
        """
        Initialise the object either from file or by creating an empty object.

        Args:
            data_filename (str, optional): filename of data to be read, in ``pyuvdata``-compatible format. If not
                given, an empty :class:`pyuvdata.UVdata` object will be created from scratch. *Deprecated since
                v0.0.1, will be removed in v0.1.0. Use `data` instead*.
            data (str or :class:`UVData`): either a string pointing to data to be read (i.e. the same as
                `data_filename`), or a UVData object.
            refresh_data (bool, optional): if reading data from file, this can be used to manually set the data to zero,
                and remove flags. This is useful for using an existing file as a template, but not using its data.
            n_freq (int, optional): if `data_filename` not given, this is required and sets the number of frequency
                channels.
            n_times (int, optional): if `data_filename` is not given, this is required and sets the number of obs
                times.
            antennas (dict, optional): if `data_filename` not given, this is required. See docs of
                :func:`~io.empty_uvdata` for more details.

        Other Args:
            All other arguments are sent either to :func:`~UVData.read` (if `data_filename` is given) or
            :func:`~io.empty_uvdata` if not. These all have default values as defined in the documentation for those
            objects, and are therefore optional.

        Raises:
            :class:`CompatibilityException`: if the created/imported data has attributes which are in conflict
                with the assumptions made in the models of this Simulator.

        """

        if data_filename is not None:
            warnings.warn("`data_filename` is deprecated, please use `data` instead", DeprecationWarning)
            
        self.data_filename = data_filename

        if self.data_filename is None and data is None:
            # Create an empty UVData object.

            # Ensure required parameters have been set.
            if n_freq is None:
                raise ValueError("if data_filename and data not given, n_freq must be given")
            if n_times is None:
                raise ValueError("if data_filename and data not given, n_times must be given")
            if antennas is None:
                raise ValueError("if data_filename and data not given, antennas must be given")

            # Actually create it
            self.data = io.empty_uvdata(
                nfreq=n_freq,
                ntimes=n_times,
                ants=antennas,
                **kwargs
            )

        else:
            if type(data) is str:
                self.data_filename = data

            if self.data_filename is not None:
                # Read data from file.
                self.data = self._read_data(self.data_filename, **kwargs)

                # Reset data to zero if user desires.
                if refresh_data:
                    self.data.data_array[:] = 0.0
                    self.data.flag_array[:] = False
                    self.data.nsample_array[:] = 1.0
            else:
                self.data = data

        # Assume the phase type is drift unless otherwise specified.
        if self.data.phase_type == "unknown":
            self.data.set_drift()

        # what does this line do?
        self.data.baseline_array

        # add redundant bl groups to UVData object's extra keywords
        self.data.extra_keywords['reds'] = self.data.get_baseline_redundancies()[0]

        # Check if the created/read data is compatible with the assumptions of
        # this class.
        self._check_compatibility()

    @cached_property
    def antpos(self):
        """
        Dictionary of {antenna: antenna_position} for all antennas in the data.
        """
        antpos, ants = self.data.get_ENU_antpos(pick_data_ants=True)
        return dict(zip(ants, antpos))

    @staticmethod
    def _read_data(filename, **kwargs):
        uv = UVData()
        uv.read(filename, read_data=True, **kwargs)
        return uv

    def write_data(self, filename, file_type="uvh5", **kwargs):
        """
        Write current UVData object to file.

        Args:
            filename (str): filename to write to.
            file_type: (str): one of "miriad", "uvfits" or "uvh5" (i.e. any of the supported write methods of
                :class:`pyuvdata.UVData`) which determines which write method to call.
            **kwargs: keyword arguments sent directly to the write method chosen.
        """
        try:
            getattr(self.data, "write_%s" % file_type)(filename, **kwargs)
        except AttributeError:
            raise ValueError("The file_type must correspond to a write method in UVData.")

    def _check_compatibility(self):
        """
        Merely checks the compatibility of the data with the assumptions of the simulator class and its modules.
        """
        if self.data.phase_type != "drift":
            raise CompatibilityException("The phase_type of the data must be 'drift'.")

    def _iterate_antpair_pols(self):
        """
        Iterate through antenna pairs and polarizations in the data object
        """

        for ant1, ant2, pol in self.data.get_antpairpols():
            blt_inds = self.data.antpair2ind((ant1, ant2))
            pol_ind = self.data.get_pols().index(pol)
            yield ant1, ant2, pol, blt_inds, pol_ind

    def _apply_vis(self, model, ant1, ant2, blt_ind, pol_ind, **kwargs):
        # get freqs from zeroth spectral window
        fqs = self.data.freq_array[0] * 1e-9
        lsts = self.data.lst_array[blt_ind]
        bl_vec = (self.antpos[ant1] - self.antpos[ant2]) * 1e9 / const.c.value
        vis = model(lsts=lsts, fqs=fqs, bl_vec=bl_vec, **kwargs)
        self.data.data_array[blt_ind, 0, :, pol_ind] += vis

    def _generate_seeds(self, model):
        np.random.seed(int(time.time()))
        seeds = np.random.randint(2**32, size=len(self.data.extra_keywords['reds']))
        self.data.extra_keywords["{}_seeds".format(str(model))] = seeds
    
    def _get_seed(self, ant1, ant2, model):
        seeds = self.data.extra_keywords["{}_seeds".format(str(model))]
        bl = self.data.antnums_to_baseline(ant1, ant2)
        key = []
        for reds in self.data.extra_keywords['reds']:
            key.append(bl in reds)
        return seeds[key.index(True)]

    @_model()
    def add_eor(self, model, **kwargs):
        """
        Add an EoR-like model to the visibilities.

        Args:
            model (str or callable): either a string name of a model function existing in :mod:`~hera_sim.eor`, or
                a callable which has the signature ``fnc(lsts, fqs, bl_vec, **kwargs)``.
            ret_vis (bool, optional): whether to return the visibilities that are being added to to the base
                data as a new array. Default False.
            add_vis (bool, optional): whether to add the calculated visibilities to the underlying data array.
                Default True.
            **kwargs: keyword arguments sent to the EoR model function, other than `lsts`, `fqs` and `bl_vec`.
        """
        seed_redundantly = kwargs.pop("seed_redundantly", False)
        if seed_redundantly:
            self._generate_seeds(model)

        for ant1, ant2, pol, blt_ind, pol_ind in self._iterate_antpair_pols():
            if seed_redundantly:
                seed = self._get_seed(ant1, ant2, model)
                np.random.seed(seed)

            self._apply_vis(model, ant1, ant2, blt_ind, pol_ind, **kwargs)

    @_model()
    def add_foregrounds(self, model, **kwargs):
        """
        Add a foreground model to the visibilities.

        Args:
            model (str or callable): either a string name of a model function existing in :mod:`~hera_sim.foregrounds`,
                or a callable which has the signature ``fnc(lsts, fqs, bl_vec, **kwargs)``.
            ret_vis (bool, optional): whether to return the visibilities that are being added to to the base
                data as a new array. Default False.
            add_vis (bool, optional): whether to add the calculated visibilities to the underlying data array.
                Default True.
            **kwargs: keyword arguments sent to the foregournd model function, other than `lsts`, `fqs` and `bl_vec`.
        """
        seed_redundantly = kwargs.pop("seed_redundantly", False)
        if seed_redundantly:
            self._generate_seeds(model)

        # account for multiple polarizations if effect is polarized
        check_pol = True if "Tsky_mdl" in inspect.signature(model).parameters \
                         and len(self.data.get_pols()) > 1 \
                         else False

        if check_pol:
            assert "pol" in kwargs.keys(), \
                    "Please specify which polarization the sky temperature " \
                    "model corresponds to by passing in a value for the " \
                    "kwarg 'pol'."
            vis_pol = kwargs.pop("pol")
            assert vis_pol in self.data.get_pols(), \
                    "You are attempting to use a polarization not included " \
                    "in the Simulator object you are working with. You tried " \
                    "to use the polarization {}, but the Simulator object you " \
                    "are working with only has the following polarizations: " \
                    "{}".format(vis_pol, self.data.get_pols())

        for ant1, ant2, pol, blt_ind, pol_ind in self._iterate_antpair_pols():
            if seed_redundantly:
                seed = self._get_seed(ant1, ant2, model)
                np.random.seed(seed)

            if check_pol:
                if pol == vis_pol:
                    self._apply_vis(model, ant1, ant2, blt_ind, pol_ind, **kwargs)
            else:
                self._apply_vis(model, ant1, ant2, blt_ind, pol_ind, **kwargs)

    @_model()
    def add_noise(self, model, **kwargs):
        """
        Add thermal noise to the visibilities.

        Args:
            model (str or callable): either a string name of a model function existing in :mod:`~hera_sim.noise`,
                or a callable which has the signature ``fnc(lsts, fqs, bl_len_ns, omega_p, **kwargs)``.
            ret_vis (bool, optional): whether to return the visibilities that are being added to to the base
                data as a new array. Default False.
            add_vis (bool, optional): whether to add the calculated visibilities to the underlying data array.
                Default True.
            **kwargs: keyword arguments sent to the noise model function, other than `lsts`, `fqs` and `bl_len_ns`.
        """
        for ant1, ant2, pol, blt_ind, pol_ind in self._iterate_antpair_pols():
            # this doesn't need to be seeded, does it?
            lsts = self.data.lst_array[blt_ind]

            self.data.data_array[blt_ind, 0, :, pol_ind] += model(
                lsts=lsts, fqs=self.data.freq_array[0] * 1e-9, **kwargs
            )

    @_model()
    def add_rfi(self, model, **kwargs):
        """
        Add RFI to the visibilities.

        Args:
            model (str or callable): either a string name of a model function existing in :mod:`~hera_sim.rfi`,
                or a callable which has the signature ``fnc(lsts, fqs, **kwargs)``.
            ret_vis (bool, optional): whether to return the visibilities that are being added to to the base
                data as a new array. Default False.
            add_vis (bool, optional): whether to add the calculated visibilities to the underlying data array.
                Default True.
            **kwargs: keyword arguments sent to the RFI model function, other than `lsts` or `fqs`.
        """
        for ant1, ant2, pol, blt_ind, pol_ind in self._iterate_antpair_pols():
            lsts = self.data.lst_array[blt_ind]

            # XXX this should be seeded according to the time corresponding to blt_ind

            # RFI added in-place (giving rfi= does not seem to work here)
            self.data.data_array[blt_ind, 0, :, 0] += model(
                lsts=lsts,
                # Axis 0 is spectral windows, of which at this point there are always 1.
                fqs=self.data.freq_array[0] * 1e-9,
                **kwargs
            )

    @_model(multiplicative=True)
    def add_gains(self, **kwargs):
        """
        Apply mock gains to visibilities.

        Currently this consists of a bandpass, and cable delays & phases.

        Args:
            ret_vis (bool, optional): whether to return the visibilities that are being added to to the base
                data as a new array. Default False.
            add_vis (bool, optional): whether to add the calculated visibilities to the underlying data array.
                Default True.
            **kwargs: keyword arguments sent to the gen_gains method in :mod:~`hera_sim.sigchain`.
        """

        gains = sigchain.gen_gains(
            fqs=self.data.freq_array[0] * 1e-9, ants=self.data.get_ants(), **kwargs
        )

        for ant1, ant2, pol, blt_ind, pol_ind in self._iterate_antpair_pols():
            self.data.data_array[blt_ind, 0, :, pol_ind] = sigchain.apply_gains(
                vis=self.data.data_array[blt_ind, 0, :, pol_ind],
                gains=gains,
                bl=(ant1, ant2)
            )

    @_model(multiplicative=True)
    def add_sigchain_reflections(self, ants=None, **kwargs):
        """
        Apply signal chain reflections to visibilities.

        Args:
            ants: list of antenna numbers to add reflections to
            **kwargs: keyword arguments sent to the gen_reflection_gains method in :mod:~`hera_sim.sigchain`.
        """
        if ants is None:
            ants = self.data.get_ants()
            
        # generate gains
        gains = sigchain.gen_reflection_gains(self.data.freq_array[0], ants, **kwargs)

        for ant1, ant2, pol, blt_ind, pol_ind in self._iterate_antpair_pols():
            self.data.data_array[blt_ind, 0, :, pol_ind] = sigchain.apply_gains(
                vis=self.data.data_array[blt_ind, 0, :, pol_ind],
                gains=gains,
                bl=(ant1, ant2)
            )

    @_model('sigchain', multiplicative=True)
    def add_xtalk(self, model='gen_whitenoise_xtalk', bls=None, **kwargs):
        """
        Add crosstalk to visibilities.

        Args:
            bls (list of 3-tuples, optional): ant-pair-pols to add xtalk to.
            **kwargs: keyword arguments sent to the model :meth:~`hera_sim.sigchain.{model}`.
        """
        freqs = self.data.freq_array[0]
        for ant1, ant2, pol, blt_ind, pol_ind in self._iterate_antpair_pols():
            if bls is not None and (ant1, ant2, pol) not in bls:
                continue
            if model.__name__ == 'gen_whitenoise_xtalk':
                xtalk = model(freqs, **kwargs)
            elif model.__name__ == 'gen_cross_coupling_xtalk':
                # for now uses ant1 ant1 for auto correlation vis
                autovis = self.data.get_data(ant1, ant1, pol)
                xtalk = model(freqs, autovis, **kwargs)

            self.data.data_array[blt_ind, 0, :, pol_ind] = sigchain.apply_xtalk(
                vis=self.data.data_array[blt_ind, 0, :, pol_ind],
                xtalk=xtalk
            )
    
    
    def run_sim(self, sim_file=None, **sim_params):
        """
        Accept a dictionary or YAML file of simulation parameters and add in
        all of the desired simulation components to the Simulator object.

        Args:
            sim_file (str, optional): string providing a path to a YAML file
                The YAML file must be configured so that the dictionary
                generated by yaml.load() will follow the format required of
                `sim_params`. Note that any simulation components which
                require a `Tsky_mdl` parameter must have the value
                corresponding to the `Tsky_mdl` key be formatted as a
                dictionary such that the :class:~interpolators.Tsky class
                can construct a `Tsky_mdl` interpolation object from the
                dictionary items. See the :class:~interpolators.Tsky docstring
                for details on how the `Tsky_mdl` dictionary should be
                formatted.
            
            **sim_params (dict, optional): dictionary of simulation parameters.
                Each parameter in this unpacked dictionary must take the form
                model = {param_name: param_value, ...}, where `model` denotes
                which simulation component is to be added, and the dictionary
                provides all the model kwargs that the user wishes to set. Any
                model kwargs not provided will assume their default values.

        Raises:
            AssertionError:
                One (and *only* one) of the above arguments must be provided. If 
                *both* sim_file and sim_params are provided, then this function
                will raise an AssertionError.

            KeyError:
                Raised if the `sim_file` YAML is not configured such that all
                `Tsky_mdl` entries have a `file` key. The value corresponding
                to this key should be a `.npz` file from which an interpolation
                object may be created. See the :class:~interpolators.Tsky
                docstring for information on how the `.npz` file should be
                formatted.

            TypeError:
                Raised if the `sim_file` YAML is not configured such that all
                `Tsky_mdl` entries have dictionaries as their values.
        """

        # keep track of which components don't use models
        uses_no_model = []
        for key, val in self.SIMULATION_COMPONENTS.items():
            func = getattr(self, val)
            # raise a NotImplementedError if using Py2
            if sys.version_info.major < 3 or \
               sys.version_info.major > 3 and sys.version_info.minor < 4:
                raise NotImplementedError("Please use a version of Python >= 3.4.")
            if 'model' not in inspect.signature(func).parameters:
                uses_no_model.append(key)

        assert sim_file is not None or sim_params, \
                'Either a path to a simulation file or a dictionary of ' \
                'simulation parameters must be provided.'

        assert sim_file is None or not sim_params, \
                'Either a simulation configuration file or a dictionary ' \
                'of simulation parameters may be passed, but not both. ' \
                'Please choose only one of the two to pass as an argument.'

        # if a path to a simulation file is provided, then read it in
        if sim_file is not None:
            with open(sim_file, 'r') as doc:
                try:
                    sim_params = yaml.load(doc.read(), Loader=yaml.FullLoader)
                except:
                    print('Check your configuration file. Something broke.')
                    sys.exit()

        for model, params in sim_params.items():
            assert model in self.SIMULATION_COMPONENTS.keys(), \
                    'Models must be supported by hera_sim. ' \
                    "'{}' is currently not supported.".format(model)

            assert isinstance(params, dict), \
                    'Values of sim_params must be dictionaries. ' \
                    "The values for '{}' do not comply.".format(model)

            # since this currently only supports python 3.4 or newer, we can
            # assume that all dicts are ordered
            add_component = getattr(self, self.SIMULATION_COMPONENTS[model])
            params = sim_params[model]
            if model in uses_no_model:
                add_component(**params)
            else:
                add_component(model, **params)

