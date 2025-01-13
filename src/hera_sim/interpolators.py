"""This module provides interfaces to different interpolation classes."""

import warnings
from os import path

import numpy as np
from cached_property import cached_property
from scipy.interpolate import RectBivariateSpline, interp1d

from hera_sim import DATA_PATH

INTERP_OBJECTS = {"1d": ("beam", "bandpass", "reflection"), "2d": ("Tsky_mdl",)}


def _check_path(datafile):
    # if the datafile is not an absolute path, assume it's in the data folder
    if not path.isabs(datafile):
        datafile = path.join(DATA_PATH, datafile)
    # make sure that the path exists
    assert path.exists(datafile), (
        "If datafile is not an absolute path, then it is assumed to "
        "exist in the hera_sim.data folder. The datafile passed could "
        "not be found; please ensure that the path to the file exists"
    )
    return datafile


def _read_npy(npy):
    """Load in contents of a .npy file."""
    return np.array(np.load(npy))


def _read_npz(npz):
    """Load in contents of a .npz file."""
    # We have to convert to dict to read the data in, instead of lazy-loading.
    # Otherwise, Interpolator is not pickleable.
    return dict(np.load(npz, allow_pickle=True))


def _read(datafile):
    ext = path.splitext(datafile)[1]
    if ext == ".npy":
        return _read_npy(datafile)
    elif ext == ".npz":
        return _read_npz(datafile)
    else:
        raise ValueError(f"File type {ext!r} not supported.")


class Interpolator:
    """Base interpolator class.

    Parameters
    ----------
    datafile : str
        Path to the file to be used to generate the interpolation object.
        Must be either a .npy or .npz file, depending on which type of
        interpolation object is desired. If path is not absolute, then the
        file is assumed to exist in the `data` directory of ``hera_sim`` and
        is modified to reflect this assumption.
    interp_kwargs : unpacked dict, optional
        Passed to the interpolation method used to make the interpolator.
    """

    def __init__(self, datafile, **interp_kwargs):
        self._datafile = _check_path(datafile)
        self._data = _read(self._datafile)
        self._interp_kwargs = interp_kwargs


# TODO: add support for multiple polarizations
class Tsky(Interpolator):
    """Sky temperature interpolator.

    Parameters
    ----------
    datafile : str
        Passed to superclass constructor. Must be a ``.npz`` file with the
        following archives:

        * ``tsky``:
          Array of sky temperature values in units of Kelvin; must
          have shape=(NPOLS, NLSTS, NFREQS).
        * ``freqs``:
          Array of frequencies at which the tsky model is evaluated,
          in units of GHz; must have shape=(NFREQS,).
        * ``lsts``:
          Array of LSTs at which the tsky model is evaulated, in
          units of radians; must have shape=(NLSTS,).
        * ``meta``:
          Dictionary of metadata describing the data stored in the npz
          file. Currently it only needs to contain an entry 'pols',
          which lists the polarizations such that their order agrees
          with the ordering of arrays along the tsky axis-0. The user
          may choose to also save the units of the frequency, lst, and
          tsky arrays as strings in this dictionary.
    interp_kwargs : unpacked dict, optional
        Extend interp_kwargs parameter for superclass to allow for the
        specification of which polarization to use via the key 'pol'. If
        'pol' is specified, then it must be one of the polarizations listed
        in the 'meta' dictionary.

    Attributes
    ----------
    freqs : np.ndarray
        Frequency array used to construct the interpolator object. Has
        units of GHz and shape=(NFREQS,).
    lsts : np.ndarray
        LST array used to construct the interpolator object. Has units of
        radians and shape=(NLSTS,).
    tsky : np.ndarray
        Sky temperature array used to construct the interpolator object.
        Has units of Kelvin and shape=(NPOLS, NLSTS, NFREQS).
    meta : dict
        Dictionary containing some metadata relevant to the interpolator.
    pol : str, default 'xx'
        Polarization appropriate for the sky temperature model. Must be
        one of the polarizations stored in the 'meta' dictionary.

    Raises
    ------
    AssertionError:
        Raised if any of the required npz keys are not found or if the
        tsky array does not have shape=(NPOLS, NLSTS, NFREQS).
    """

    def __init__(self, datafile, **interp_kwargs):
        super().__init__(datafile, **interp_kwargs)
        self._check_npz_format()
        self.pol = self._interp_kwargs.pop("pol", "xx")
        self._check_pol(self.pol)

    def __call__(self, lsts, freqs):
        """Evaluate the Tsky model at the specified lsts and freqs."""
        return self._interpolator(lsts, freqs)

    @property
    def freqs(self) -> np.ndarray:
        """Frequencies of the interpolation data."""
        return self._data["freqs"]

    @property
    def tsky(self) -> np.ndarray:
        """Measured values of the sky temperature to be interpolated."""
        return self._data["tsky"]

    @property
    def lsts(self) -> np.ndarray:
        """Times at which the sky temperature is measured."""
        return self._data["lsts"]

    @property
    def meta(self):
        """Metadata about the measured/model sky temperature."""
        return self._data["meta"][None][0]

    @cached_property
    def _interpolator(self):
        """Construct an interpolation object.

        Uses class attributes to construct an interpolator using the
        ``scipy.interpolate.RectBivariateSpline`` interpolation class.
        """
        # get index of tsky's 0-axis corresponding to pol
        pol_index = self.meta["pols"].index(self.pol)

        # get the tsky data
        tsky_data = self.tsky[pol_index]

        # TODO: make LST wrapping a little smarter
        # (that is, are 10 extra LSTs on each side really needed?)
        dlst = np.mean(np.diff(self.lsts))
        wrap_length = 10
        if (self.lsts[0] - dlst * wrap_length > 0) or (
            self.lsts[-1] + dlst * wrap_length < 2 * np.pi
        ):
            warnings.warn(
                "The provided LSTs do not sufficiently cover [0, 2*pi). "
                "The interpolated sky temperature may have unexpected behavior "
                "near 0 and 2*pi.",
                stacklevel=1,
            )

        lsts = np.concatenate(
            [
                self.lsts[-wrap_length:] - 2 * np.pi,
                self.lsts,
                self.lsts[:wrap_length] + 2 * np.pi,
            ]
        )
        tsky_data = np.concatenate(
            [
                tsky_data[-wrap_length:],
                tsky_data,
                tsky_data[:wrap_length],
            ]
        )

        # now make the interpolation object
        return RectBivariateSpline(lsts, self.freqs, tsky_data, **self._interp_kwargs)

    def _check_npz_format(self):
        """Check that the npz archive is formatted properly."""
        # TODO: change all assert statements to appropriate error raises
        assert "freqs" in self._data.keys(), (
            "The frequencies corresponding to the sky temperature array "
            "must be provided. They must be saved to the npz file using "
            "the key 'freqs'."
        )
        assert "lsts" in self._data.keys(), (
            "The LSTs corresponding to the sky temperature array must "
            "be provided. They must be saved to the npz file using the "
            "key 'lsts'."
        )
        assert "tsky" in self._data.keys(), (
            "The sky temperature array must be saved to the npz file "
            "using the key 'tsky'."
        )
        assert "meta" in self._data.keys(), (
            "The npz file must contain a metadata dictionary that can "
            "be accessed with the key 'meta'. This dictionary should "
            "provide information about the units of the various arrays "
            "and the polarizations of the sky temperature array."
        )

        if not np.all(self.freqs[1:] > self.freqs[:-1]):
            raise ValueError("Frequencies must be strictly increasing.")

        if not np.all(self.lsts[1:] > self.lsts[:-1]):
            raise ValueError("LSTs must be strictly increasing.")

        # check that tsky has the correct shape
        assert self.tsky.shape == (
            len(self.meta["pols"]),
            self.lsts.size,
            self.freqs.size,
        ), (
            "The tsky array is incorrectly shaped. Please ensure that "
            "the tsky array has shape (NPOLS, NLSTS, NFREQS)."
        )

    def _check_pol(self, pol):
        """Check that the desired polarization is in the meta dict."""
        assert pol in self.meta["pols"], (
            "Polarization must be in the metadata's polarization tuple. "
            "The metadata contains the following polarizations: "
            "{}".format(self.meta["pols"])
        )


class FreqInterpolator(Interpolator):
    """Frequency interpolator.

    Parameters
    ----------
    datafile : str
        Passed to the superclass constructor.

    interp_kwargs : unpacked dict, optional
        Extends superclass interp_kwargs parameter by checking for the key
        'interpolator' in the dictionary. The 'interpolator' key should
        have the value 'poly1d' or 'interp1d'; these correspond to the
        `np.poly1d` and `scipy.interpolate.interp1d` objects, respectively.
        If the 'interpolator' key is not found, then it is assumed that
        a `np.poly1d` object is to be used for the interpolator object.

    Raises
    ------
    AssertionError
        This is raised if the choice of interpolator and the required type
        of the ref_file do not agree (i.e. trying to make a 'poly1d' object
        using a .npz file as a reference). An AssertionError is also raised
        if the .npz for generating an 'interp1d' object does not have the
        correct arrays in its archive.
    """

    def __init__(self, datafile, obj_type=None, **interp_kwargs):
        super().__init__(datafile, **interp_kwargs)
        self._interp_type = self._interp_kwargs.pop("interpolator", "poly1d")
        self._obj = obj_type
        self._check_format()

    def __call__(self, freqs):
        """Evaluate the interpolation object at the given frequencies."""
        return self._interpolator(freqs)

    @cached_property
    def _interpolator(self):
        """Construct the interpolator object."""
        if self._interp_type == "poly1d":
            return np.poly1d(self._data)

        # if not using poly1d, then need to get some parameters for
        # making the interp1d object
        obj = self._data[self._obj]
        freqs = self._data["freqs"]

        # use a cubic spline by default, but override this if the user
        # specifies a different kind of interpolator
        kind = self._interp_kwargs.pop("kind", "cubic")
        return interp1d(freqs, obj, kind=kind, **self._interp_kwargs)

    def _check_format(self):
        """Check that class attributes are appropriately formatted."""
        assert self._interp_type in (
            "poly1d",
            "interp1d",
        ), "Interpolator choice must either be 'poly1d' or 'interp1d'."

        if self._interp_type == "interp1d":
            assert path.splitext(self._datafile)[1] == ".npz", (
                "In order to use an 'interp1d' object, the reference file "
                "must be a '.npz' file."
            )
            assert self._obj in self._data.keys() and "freqs" in self._data.keys(), (
                "You've chosen to use an interp1d object for modeling the "
                f"{self._obj}. Please ensure that the `.npz` archive has the following "
                f"keys: 'freqs', '{self._obj}'"
            )
        else:
            # we can relax this a bit and allow for users to also pass a npz
            # with the same keys as in the above case, but it seems silly to
            # use a polynomial interpolator instead of a spline interpolator in
            # this case
            assert path.splitext(self._datafile)[1] == ".npy", (
                "In order to use a 'poly1d' object, the reference file "
                "must be a .npy file that contains the coefficients for "
                "the polynomial fit in decreasing order."
            )


class Beam(FreqInterpolator):
    """Beam interpolation object.

    Parameters
    ----------
    datafile : str
        Passed to the superclass constructor.
    interp_kwargs : unpacked dict, optional
        Passed to the superclass constructor.
    """

    def __init__(self, datafile, **interp_kwargs):
        super().__init__(datafile, obj_type="beam", **interp_kwargs)


class Bandpass(FreqInterpolator):
    """Bandpass interpolation object.

    Parameters
    ----------
    datafile : str
        Passed to the superclass constructor.
    interp_kwargs : unpacked dict, optional
        Passed to the superclass constructor.
    """

    def __init__(self, datafile, **interp_kwargs):
        super().__init__(datafile, obj_type="bandpass", **interp_kwargs)


class Reflection(FreqInterpolator):
    """Complex reflection coefficient interpolator."""

    def __init__(self, datafile, **interp_kwargs):
        if "interpolator" not in interp_kwargs:
            interp_kwargs["interpolator"] = "interp1d"
        super().__init__(datafile, obj_type="reflection", **interp_kwargs)

    @cached_property
    def _re_interp(self):
        interp_kwargs = {"kind": "cubic"}
        interp_kwargs.update(self._interp_kwargs)
        return interp1d(
            self._data["freqs"], self._data[self._obj].real, **interp_kwargs
        )

    @cached_property
    def _im_interp(self):
        interp_kwargs = {"kind": "cubic"}
        interp_kwargs.update(self._interp_kwargs)
        return interp1d(
            self._data["freqs"], self._data[self._obj].imag, **interp_kwargs
        )

    def __call__(self, freqs):
        return self._re_interp(freqs) + 1j * self._im_interp(freqs)
