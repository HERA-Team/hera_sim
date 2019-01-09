"""
Primary interface module for hera_sim, defining a :class:`Simulator` class which provides a common API for all
effects produced by this package.
"""

import functools
import inspect
import sys

from pyuvdata import UVData

from . import io
from . import sigchain
from .version import version


def _get_model(mod, name):
    return getattr(sys.modules["hera_sim." + mod], name)


class _model(object):

    def __init__(self, base_module=None):
        self.base_module = base_module

    def __call__(self, func, *args, **kwargs):
        name = func.__name__

        @functools.wraps(func)
        def new_func(obj, *args, **kwargs):
            if "model" in inspect.getargspec(func)[0]:
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
                func(obj, **kwargs)

            msg = "\nhera_sim v{version}: Added {component} {method_name}with kwargs: {kwargs}"

            obj.data.history += msg.format(
                version=version,
                component="".join(name.split("_")[1:]),
                method_name=method,
                kwargs=kwargs,
            )

        return new_func


class Simulator:
    """
    Primary interface object for hera_sim.

    Produces visibility simulations with various independent sky- and instrumental-effects, and offers the resulting
    visibilities in :class:`UVData` format
    """

    def __init__(
        self,
        data_filename=None,
        n_freq=None,
        n_times=None,
        antennas=None,
        ant_pairs=None,
        pols=None,
        time_per_integ=10.7,
        min_freq=0.1,
        channel_bw=0.1 / 1024.0,
        instrument="hera_sim",
        telescope_location=None,
        telescope_lat_lon_alt=None,
        object_name="sim_data",
        start_jd=2458119.5,
        vis_units="uncalib",
        **kwargs
    ):

        self.data_filename = data_filename

        if self.data_filename is None:
            # Create an empty UVData object.

            # Ensure required parameters have been set.
            assert (
                n_freq is not None
            ), "if data_filename not given, n_freq must be given"
            assert (
                n_times is not None
            ), "if data_filename not given, n_times must be given"
            assert (
                antennas is not None
            ), "if data_filename not given, antennas must be given"
            assert (
                ant_pairs is not None
            ), "if data_filename not given, ant_pairs must be given"

            # Default values for mutable parameters
            if pols is None:
                pols = ["xx"]
            if telescope_location is None:
                telescope_location = io.HERA_LOCATION
            if telescope_lat_lon_alt is None:
                telescope_lat_lon_alt = io.HERA_LAT_LON_ALT

            # Actually create it
            self.data = io.empty_uvdata(
                nfreq=n_freq,
                ntimes=n_times,
                ants=antennas,
                antpairs=ant_pairs,
                pols=pols,
                time_per_integ=time_per_integ,
                min_freq=min_freq,
                channel_bw=channel_bw,
                instrument=instrument,
                telescope_location=telescope_location,
                telescope_lat_lon_alt=telescope_lat_lon_alt,
                object_name=object_name,
                start_jd=start_jd,
                vis_units=vis_units,
            )

        else:
            self.data = self._read_data(self.data_filename, pols, **kwargs)

    def _read_data(self, filename, polarizations, **kwargs):
        uv = UVData()
        uv.read(filename, polarizations=polarizations, read_data=True, **kwargs)
        return uv

    def write_data(self, filename, file_type=None, **kwargs):
        """
        Write current UVData object to file.

        Args:
            filename (str): filename to write to. Suffix of filename will determine the type of file to be written,
                unless `file_type` is set.
            file_type: (str): one of "miriad", "uvfits" or "uvh5" (i.e. any of the supported write methods of
                :class:`pyuvdata.UVData`) which determines which write method to call.
            **kwargs: keyword arguments sent directly to the write method chosen.
        """
        if file_type is None:
            try:
                file_type = filename.split(".")[-1]
            except IndexError:
                raise ValueError(
                    "if no file_type is given, the filename must be suffixed with either .miriad, .uvfits or .uvh5"
                )

        if file_type not in ["miriad", "uvfits", "uvh5"]:
            raise ValueError("file_type must be one of 'miriad', 'uvfits' or 'uvh5'")

        getattr(self.data, "write_%s" % filename.split(".")[-1])(filename, **kwargs)

    def _iterate_baselines(self, with_conj=True):
        """
        Iterate through baselines in the data object
        """
        for i, ant1 in enumerate(self.data.antenna_numbers):
            for j, ant2 in enumerate(
                self.data.antenna_numbers[(i + 1 if with_conj else 0) :]
            ):
                # Get the Nblts indices of this baseline (and its conjugate)
                yield ant1, ant2, self.data.antpair2ind(
                    ant1, ant2, ordered=not with_conj
                )

    @_model()
    def add_eor(self, model, **kwargs):
        """
        Add an EoR-like model to the visibilities.

        Args:
            model (str or callable): either a string name of a model function existing in :mod:`~hera_sim.eor`, or
                a callable which has the signature ``fnc(lsts, fqs, bl_len_ns, **kwargs)``.
            **kwargs: keyword arguments sent to the EoR model function, other than `lsts`, `fqs` and `bl_len_ns`.
        """
        for ant1, ant2, ind in self._iterate_baselines():
            lsts = self.data.lst_array[ind]
            bl_len_m = self.data.uvw_array[ind][
                0, 0
            ]  # just the E-W baseline length at this point.

            vis = model(
                lsts=lsts,
                fqs=self.data.freq_array[0]
                * 1e-9,  # Axis 0 is spectral windows, of which at this point there are always 1.
                bl_len_ns=bl_len_m * 1e9 / 3e8,
                **kwargs
            )

            self.data.data_array[
                ind, 0, :, 0
            ] += vis  # TODO: not sure about only using first pol.

    @_model()
    def add_foregrounds(self, model, **kwargs):
        """
        Add a foreground model to the visibilities.

        Args:
            model (str or callable): either a string name of a model function existing in :mod:`~hera_sim.foregrounds`,
                or a callable which has the signature ``fnc(lsts, fqs, bl_len_ns, **kwargs)``.
            **kwargs: keyword arguments sent to the foregournd model function, other than `lsts`, `fqs` and `bl_len_ns`.
        """
        for ant1, ant2, ind in self._iterate_baselines():
            lsts = self.data.lst_array[ind]
            bl_len_m = self.data.uvw_array[ind][
                0, 0
            ]  # just the E-W baseline length at this point.

            vis = model(
                lsts=lsts,
                fqs=self.data.freq_array[0]
                * 1e-9,  # Axis 0 is spectral windows, of which at this point there are always 1.
                bl_len_ns=bl_len_m * 1e9 / 3e8,
                **kwargs
            )

            self.data.data_array[
                ind, 0, :, 0
            ] += vis  # TODO: not sure about only using first pol.

    @_model()
    def add_noise(self, model, **kwargs):
        """
        Add thermal noise to the visibilities.

        Args:
            model (str or callable): either a string name of a model function existing in :mod:`~hera_sim.noise`,
                or a callable which has the signature ``fnc(lsts, fqs, bl_len_ns, **kwargs)``.
            **kwargs: keyword arguments sent to the noise model function, other than `lsts`, `fqs` and `bl_len_ns`.
        """
        for ant1, ant2, ind in self._iterate_baselines():
            lsts = self.data.lst_array[ind]

            self.data.data_array[ind, 0, :, 0] += model(
                lsts=lsts, fqs=self.data.freq_array[0] * 1e-9, **kwargs
            )

    @_model()
    def add_reflections(self, model, **kwargs):
        """
        Add auto- or cross-reflections to data visibilities.

        Args:
            model (str or callable): either a string name of a model function existing in :mod:`~hera_sim.reflections`,
                or a callable which has the signature ``fnc(vis, fqs, **kwargs)``.
            **kwargs: keyword arguments sent to the reflections model function, other than `vis` and `fqs`. Common
                parameters are `dly`, `phs` and `amp`.
        """
        for ant1, ant2, ind in self._iterate_baselines():
            # the following performs the modification in-place
            self.data.data_array[ind, 0, :, 0] = model(
                vis=self.data.data_array[
                    ind, 0, :, 0
                ],  # pass the ntimes x nfreqs part of the visibilities.
                freqs=self.data.freq_array[0]
                * 1e-9,  # Axis 0 is spectral windows, of which at this point there are always 1.
                **kwargs
            )

    @_model()
    def add_rfi(self, model, **kwargs):
        """
        Add RFI to the visibilities.

        Args:
            model (str or callable): either a string name of a model function existing in :mod:`~hera_sim.rfi`,
                or a callable which has the signature ``fnc(lsts, fqs, **kwargs)``.
            **kwargs: keyword arguments sent to the RFI model function, other than `lsts` or `fqs`.
        """
        for ant1, ant2, ind in self._iterate_baselines():
            lsts = self.data.lst_array[ind]

            # RFI added in-place
            model(
                lsts=lsts,
                fqs=self.data.freq_array[0]
                * 1e-9,  # Axis 0 is spectral windows, of which at this point there are always 1.
                rfi=self.data.data_array[ind, 0, :, 0],
                **kwargs
            )

    @_model()
    def add_gains(self, **kwargs):
        """
        Add gains to visibilities.

        Args:
            **kwargs: keyword arguments sent to the gen_gains method in :mod:~`hera_sim.sigchain`.
        """

        gains = sigchain.gen_gains(
            freqs=self.data.freq_array[0] * 1e-9, ants=self.data.get_ants(), **kwargs
        )

        for ant1, ant2, ind in self._iterate_baselines(with_conj=False):
            self.data.data_array[ind, 0, :, 0] = sigchain.apply_gains(
                vis=self.data.data_array[ind, 0, :, 0], gains=gains, bl=(ant1, ant2)
            )

    @_model()
    def add_xtalk(self, **kwargs):
        """
        Add crosstalk to visibilities.

        Args:
            **kwargs: keyword arguments sent to the gen_xtalk method in :mod:~`hera_sim.sigchain`.
        """

        xtalk = sigchain.gen_xtalk(freqs=self.data.freq_array[0] * 1e-9, **kwargs)

        # At the moment, the cross-talk function applies the same cross talk to every baseline/time.
        # Not sure if this is good or not.
        self.data.data_array[:, 0, :, 0] = sigchain.apply_xtalk(
            vis=self.data.data_array[:, 0, :, 0], xtalk=xtalk
        )
