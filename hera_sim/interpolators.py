"""
This module provides interfaces to different interpolation classes.
"""

import numpy as np
from cached_property import cached_property
from scipy.interpolate import RectBivariateSpline, interp1d
from hera_sim.data import DATA_PATH
from os import path

INTERP_OBJECTS = {"1d": ("beam", "bandpass",),
                  "2d": ("Tsky_mdl", ) }

def _check_path(datafile):
    # if the datafile is not an absolute path, assume it's in the data folder
    if not path.isabs(datafile):
        datafile = path.join(DATA_PATH, datafile)
    # make sure that the path exists
    assert path.exists(datafile), \
            "If datafile is not an absolute path, then it is assumed to " \
            "exist in the hera_sim.data folder. The datafile passed could " \
            "not be found; please ensure that the path to the file exists"
    return datafile

def _read_npy(npy):
    return np.load(_check_path(npy))

# TODO: Update module to have a base `interpolator` class, from which `Tsky`
# and `freq_interp1d` both inherit. Additionally, either create subclasses of 
# `freq_interp1d` called `Beam` and `Bandpass`, or change the assumed naming
# convention for arrays in `.npz` archives used to make `interp1d` objects.
# The latter option is likely better. _check_path may instead be a member
# function of the `interpolator` class. Figure out what to do about the
# `_read_npy` function.

class Interpolator:
    """
    This class serves as the base interpolator class from which all other 
    hera_sim interpolators inherit.
    """
    def __init__(self, datafile, **interp_kwargs):
        self._datafile = _check_path(datafile)
        self._data = np.load(self._datafile, allow_pickle=True)
        self._interp_kwargs = interp_kwargs
    """
    TODO: fill out the docstring
    """

class Tsky(Interpolator):
    """
    This class provides an interface for converting paths to npz files into
    interpolation objects. In particular, this class takes a path to a npz
    file and creates an interpolation object for a sky temperature model given
    the frequencies and LSTs (specified in the npz file) at which the sky
    temperature array is evaluated.
    """
    def __init__(self, datafile, **interp_kwargs):
        Interpolator.__init__(self, datafile, **interp_kwargs)
        self._check_npz_format()
        self.pol = self._interp_kwargs.pop("pol", "xx")
        self._check_pol(self.pol)

    """
    Initialize the Tsky object from a path to a npz file with the required
    information.

    Args:
        datafile (str): path to a npz file
            the npz file must contain the following information:
                array of sky temperature values in units of Kelvins
                this must be accessible via the key 'tsky' and must
                have shape=(NPOLS, NLSTS, NFREQS).

                array of frequencies at which the tsky model is evaluated
                in units of GHz. this should have shape=(NFREQS,) and
                should be accessible via the key 'freqs'.

                array of LSTs at which the tsky model is evaulated in
                units of radians. this should have shape=(NLSTS,) and
                should be accessible via the key 'lsts'.

                dictionary of metadata describing the data stored in the npz
                file. currently it only needs to contain an entry 'pols', 
                which lists the polarizations such that their order agrees
                with the ordering of arrays along the tsky axis-0. the user
                may choose to also save the units of the frequency, lst, and
                tsky arrays as strings in this dictionary. this dictionary
                must be accessible via the key 'meta'.

        **interp_kwargs (dict, optional): dictionary of kwargs for creating
            the interpolation object. These are used to create an instance of
            the scipy.interpolate.RectBivariateSpline class. Additionally, 
            the polarization used should be stored in this dictionary as the 
            value corresponding to the key `pol`.

    Raises:
        AssertionError: if any of the required npz keys are not found. these
            are 'freqs', 'lsts', 'tsky', and 'meta'. See the contents of the
            docstring relating to the datafile argument for a more detailed
            discussion. An AssertionError is also raised if the shape of the
            tsky array is not (NPOLS, NLSTS, NFREQS).
    """

    def __call__(self, lsts, freqs):
        return self._interpolator(lsts, freqs)

    @property
    def freqs(self):
        return self._data['freqs']

    @property
    def tsky(self):
        return self._data['tsky']

    @property
    def lsts(self):
        return self._data['lsts']

    @property
    def meta(self):
        return self._data['meta'][None][0]

    @cached_property
    def _interpolator(self):
        # get index of tsky's 0-axis corresponding to pol
        j = self.meta['pols'].index(self.pol)
        
        # get the tsky data
        tsky_data = self.tsky[j]

        # do some wrapping in LST
        lsts = np.concatenate([self.lsts[-10:]-2*np.pi,
                               self.lsts,
                               self.lsts[:10]+2*np.pi])
        tsky_data = np.concatenate([tsky_data[-10:], tsky_data, tsky_data[:10]])

        # now make the interpolation object
        return RectBivariateSpline(lsts, self.freqs, tsky_data, **self._interp_kwargs)

    def _check_npz_format(self):
        # check that the npz has all the required objects archived
        assert 'freqs' in self._data.keys(), \
                "The frequencies corresponding to the sky temperature array " \
                "must be provided. They must be saved to the npz file using " \
                "the key 'freqs'."
        assert 'lsts' in self._data.keys(), \
                "The LSTs corresponding to the sky temperature array must " \
                "be provided. They must be saved to the npz file using the " \
                "key 'lsts'."
        assert 'tsky' in self._data.keys(), \
                "The sky temperature array must be saved to the npz file " \
                "using the key 'tsky'."
        assert 'meta' in self._data.keys(), \
                "The npz file must contain a metadata dictionary that can " \
                "be accessed with the key 'meta'. This dictionary should " \
                "provide information about the units of the various arrays " \
                "and the polarizations of the sky temperature array."
        # check that tsky has the correct shape
        assert self.tsky.shape==(len(self.meta['pols']),
                                 self.lsts.size, self.freqs.size), \
                "The tsky array is incorrectly shaped. Please ensure that " \
                "the tsky array has shape (NPOLS, NLSTS, NFREQS)."

    def _check_pol(self, pol):
        assert pol in self.meta['pols'], \
                "Polarization must be in the metadata's polarization tuple. " \
                "The metadata contains the following polarizations: " \
                "{}".format(self.meta['pols'])

class FreqInterpolator(Interpolator):
    """
    This class provides an interface for creating either a numpy.poly1d or a 
    scipy.interpolate.interp1d interpolation object from a reference file. This 
    class is intended to be used primarily as a helper function for various 
    functions in the hera_sim repository.
    """
    def __init__(self, datafile, **interp_kwargs):
        Interpolator.__init__(self, datafile, **interp_kwargs)
        self._interp_type = self._interp_kwargs.pop("interpolator", "poly1d")
        self._obj = self._interp_kwargs.pop("obj", None)
        self._check_format()
    
    """
    Initialize an interpolation object from a given reference file and choice of 
    interpolator.

    Args:
        datafile (str):
            Path to the reference file to be used for making the 1d-interpolator
            This must be either a .npy or .npz file, and the path may either be
            relative or absolute. If the path is relative, then the file is 
            assumed to exist in the data directory. The choice of .npy or .npz
            format depends on which interpolation method is chosen.

        interpolator (str):
            Choice of interpolation object to be used. Must be either 'poly1d' 
            or 'interp1d'. If 'poly1d' is chosen, then ref_file must be a 
            .npy file. If 'interp1d' is chosen, then ref_file must be a .npz 
            file with two arrays stored in its archive: 'freqs' and 'beam'.

        **interp_kwargs (dict, optional):
            Keyword arguments to be passed to the interpolator in the case that
            'interp1d' is chosen. `obj` must be a key in `interp_kwargs`; it 
            should specify whether the object represented is a beam size or a
            bandpass response (denoted by the values 'beam' and 'bandpass', 
            respectively).

    Raises:
        AssertionError:
            This is raised if the choice of interpolator and the required type
            of the ref_file do not agree (i.e. trying to make a 'poly1d' object
            using a .npz file as a reference). An AssertionError is also raised
            if the .npz for generating an 'interp1d' object does not have the
            correct arrays in its archive.
        
        ValueError:
            This is raised if 'obj' is not a key in `interp_kwargs`. This behavior
            was introduced based on the assumption that a `.npz` file would have
            a `freqs` array and an `obj` (either 'beam' or 'bandpass', depending
            on the value of `obj`) array in its archive. This will likely be 
            dropped in a future release.
    """

    def __call__(self, freqs):
        return self._interpolator(freqs)

    @cached_property
    def _interpolator(self):
        if self._interp_type=='poly1d':
            return np.poly1d(self._data)
        else:
            # if not using poly1d, then need to get some parameters for
            # making the interp1d object
            obj = self._data[self._obj]
            freqs = self._data['freqs']

            # use a cubic spline by default, but override this if the user
            # specifies a different kind of interpolator
            kind = self._interp_kwargs.pop('kind', 'cubic')
            return interp1d(freqs, obj, kind=kind, **self._interp_kwargs)

    def _check_format(self):
        if self._obj is None:
            raise ValueError("Please specify what type of object the " \
                             "interpolator represents by using the `object` " \
                             "kwarg. The currently supported object types are " \
                             "{}.".format(INTERP_OBJECTS['1d'])
                             )
        else:
            assert self._obj in INTERP_OBJECTS['1d'], \
                    "The specified object type is not supported. The currently " \
                    "supported object types are: {}".format(INTERP_OBJECTS['1d'])

        assert self._interp_type in ('poly1d', 'interp1d'), \
                "Interpolator choice must either be 'poly1d' or 'interp1d'."

        if self._interp_type=='interp1d':
            assert path.splitext(self._datafile)[1] == '.npz', \
                    "In order to use an 'interp1d' object, the reference file " \
                    "must be a '.npz' file."
            assert self._obj in self._data.keys() and 'freqs' in self._data.keys(), \
                    "You've chosen to use an interp1d object for modeling the " \
                    "{}. Please ensure that the `.npz` archive has the following " \
                    "keys: 'freqs', '{}'".format(self._obj, self._obj)
        else:
            # we can relax this a bit and allow for users to also pass a npz
            # with the same keys as in the above case, but it seems silly to
            # use a polynomial interpolator instead of a spline interpolator in
            # this case
            assert path.splitext(self._datafile)[1] == '.npy', \
                    "In order to use a 'poly1d' object, the reference file " \
                    "must be a .npy file that contains the coefficients for " \
                    "the polynomial fit in decreasing order."

