"""A module for modeling HERA signal chains."""

import numpy as np

from .data import DATA_PATH

def get_bandpass(fqs):
    """
    A function for calculating noiseless bandpass gains at the specified
    frequencies. The current implementation uses a polynomial fit.

    Args:
        fqs (array-like): shape=(NFREQS,), GHz
            array of frequencies at which to calculate gains

    Returns:
        bandpass_gains (array-like): shape=(NFREQS,), dimensionless
            array of voltage gains calculated at the specified frequencies
    """
    HERA_BANDPASS = np.load('{}/HERA_H2C_BANDPASS.npy'.format(DATA_PATH))
    return np.polyval(HERA_BANDPASS, fqs)
