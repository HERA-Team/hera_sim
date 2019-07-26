"""
This module provides interfaces to different interpolation classes.
"""

import numpy as np
from cached_property import cached_property
from scipy.interpolate import RectBivariateSpline

class Tsky:
    """
    This class provides an interface for converting hooks to npz files into
    interpolation objects. In particular, this class takes a hook to a npz
    file and creates an interpolation object for a sky temperature model given
    the frequencies and LSTs (specified in the npz file) at which the sky
    temperature array is evaluated.
    """
    def __init__(self, datafile, pol='xx', **interp_kwargs):
        self.datafile = datafile
        self._data = np.load(self.datafile, allow_pickle=True)
        self._check_npz_format()
        self._check_pol(pol)
        self.pol = pol
        self._interp_kwargs = interp_kwargs

    """
    Initialize the Tsky object from a hook to a npz file with the required
    information.

    Args:
        datafile (str): hook to a npz file
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

        pol (str, optional; default='xx'): string specifying which polarization
            to create the interpolation object for. pol must be in the list of
            polarizations saved in the metadata dictionary.

        **interp_kwargs (dict, optional): dictionary of kwargs for creating
            the interpolation object. these are used to create an instance of
            the scipy.interpolate.RectBivariateSpline class.

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

