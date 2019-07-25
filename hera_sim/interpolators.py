"""
This module provides interfaces to different interpolation classes.
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline

class Tsky:
    """
    # TODO: fill in docstring
    """
    def __init__(self, datafile, **interp_kwargs):
        self.datafile = datafile
        self._data = np.load(self.datafile, allow_pickle=True)
        # make sure the correct keys are used
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
        # now make sure that the sky temperature array has the correct shape
        # this should be (NPOLS, NLSTS, NFREQS) or (NPOLS, NFREQS, NLSTS)
        # XXX this is somewhat strict, but useful in the case NFREQS==NLSTS
        # XXX however, in the case NFREQS==NLSTS, it is possible for the
        # XXX sky temperature to be shaped incorrectly and pass this assertion
        assert self.tsky.shape==(len(self.meta['pols']),
                                 self.lsts.size, self.freqs.size), \
                "The tsky array is incorrectly shaped. Please ensure that " \
                "the tsky array has shape (NPOLS, NLSTS, NFREQS)."

        self._interpolators = {}

        # fill in interpolators
        # XXX do we want to make a hidden member function for generating
        # XXX the interpolator objects?
        for j, pol in enumerate(self.meta['pols']):
            tsky_data = self.tsky[j]

            # fill in information outside of measured LST values for
            # interpolator stability
            lsts = np.concatenate([self.lsts[-10:]-2*np.pi,
                                   self.lsts,
                                   self.lsts[:10]+2*np.pi])
            tsky_data = np.concatenate([tsky_data[-10:],
                                        tsky_data,
                                        tsky_data[:10]])
            
            # now make the interpolation object
            self._interpolators[pol] = RectBivariateSpline(lsts, 
                                                           self.freqs,
                                                           tsky_data,
                                                           **interp_kwargs)

    @property
    def freqs(self):
        # XXX units handling?
        return self._data['freqs']

    @property
    def tsky(self):
        # XXX units handling?
        return self._data['tsky']

    @property
    def lsts(self):
        # XXX units handling?
        return self._data['lsts']
    # XXX should we assert that units must be of a certain type, or do we allow
    # XXX for more flexibility and provide conversions between units?

    @property
    def meta(self):
        return self._data['meta'][None][0]

    def get_interpolator(self, pol='xx'):
        self._check_pol(pol)
        return self._interpolators[pol]

    def resample_Tsky(self, lsts, freqs, pol='xx'):
        self._check_pol(pol)
        return self._interpolators[pol](lsts, freqs)

    def _check_pol(self, pol):
        assert pol in self.meta['pols'], \
                "Polarization must be in the metadata's polarization tuple. " \
                "The metadata contains the following polarizations: " \
                "{}".format(self.meta['pols'])

