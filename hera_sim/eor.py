'''A module for generating a rough eor-like signal.'''

import numpy as np
from scipy import interpolate
import aipy
from scipy.signal import windows
from . import noise
from . import utils


def noiselike_eor(lsts, fqs, bl_len_ns, eor_amp=1e-5, 
                  fringe_filter_type='tophat', **fringe_filter_kwargs):
    """
    Generate a noise-like, fringe-filtered EoR visibility.

    Args:
        lsts : ndarray with LSTs [radians]
        fqs : ndarray with frequencies [GHz]
        bl_len_ns : float, East-West baseline length [nanosec]
        eor_amp : float, amplitude of EoR signal [arbitrary]
        fringe_filter_type : str, type of fringe-rate filter, see utils.gen_fringe_filter()
        fringe_filter_kwargs : kwargs given fringe_filter_type, see utils.gen_fringe_filter()

    Returns: 
        vis : 2D ndarray holding simulated complex visibility
        fringe_filter : fringe-rate filter applied to data
    """
    # generate white noise in frate and freq space
    data = noise.white_noise((len(lsts), len(fqs))) * eor_amp

    # generate fringe filter in frate & freq space
    fringe_filter = utils.gen_fringe_filter(lsts, fqs, bl_len_ns, filter_type=fringe_filter_type, **fringe_filter_kwargs)
 
    # apply to data
    data = np.fft.fft(np.fft.ifft(data, axis=0) * fringe_filter, axis=0)

    return data, fringe_filter

