"""A module for modeling HERA signal chains."""

import numpy as np
from .data import DATA_PATH

def get_bandpass(fqs):
    """
    A function for retrieving base bandpass gains for a given array of
    frequencies. This model uses a polynomial fit to the HERA H1C bandpass.

    Args:
        fqs (array-like): shape=(NFREQS,); GHz
            array of frequencies at which to evaluate the bandpass gains

    Returns:
        bandpass_gains (array-like): shape=(NFREQS,); dimensionless
            array of voltage gains at the given frequencies
    """
    HERA_BANDPASS = np.load('{}/HERA_H1C_BANDPASS.npy'.format(DATA_PATH))
    return np.polyval(HERA_BANDPASS, fqs)
