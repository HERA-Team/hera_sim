"""A module for modeling HERA signal chains."""

import numpy as np
from .data import DATA_PATH

def get_bandpass(fqs):
    HERA_BANDPASS = np.load('{}/HERA_H1C_BANDPASS.npy'.format(DATA_PATH))
    return np.polyval(HERA_BANDPASS, fqs)
