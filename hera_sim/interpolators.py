"""
This module provides interfaces to different interpolation classes.
"""

import numpy as np
from cached_property import cached_property
from scipy.interpolate import RectBivariateSpline

class Tsky:
    """
    # TODO: fill in docstring
    """
    def __init__(self, datafile, pol='xx', **interp_kwargs):
        self.datafile = datafile
        self._data = np.load(self.datafile, allow_pickle=True)
        self._check_npz_format()
        self._check_pol(pol)
        self.pol = pol
        self._interp_kwargs = interp_kwargs

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

