"""
"""
import os
import warnings
import numpy as np
from pyuvsim.simsetup import initialize_uvdata_from_keywords
from .data import DATA_PATH
from .defaults import _defaults

HERA_LAT_LON_ALT = np.load(os.path.join(DATA_PATH, "HERA_LAT_LON_ALT.npy"))

# this decorator allows the parameters specified in the function
# signature to be overridden by the defaults module
@_defaults
def empty_uvdata(Ntimes=None, start_time=2456658.5, # Jan 1 2014
                 integration_time=None, array_layout=None,
                 Nfreqs=None, start_freq=None, channel_width=None,
                 n_freq=None, n_times=None, antennas=None, # back-compat
                 **kwargs):
    # TODO: docstring
    """
    """
    # issue a deprecation warning if any old parameters are used
    if any([param is not None for param in (n_freq, n_times, antennas)]):
        warnings.warn(
            "The n_freq, n_times, and antennas parameters are being " \
            "deprecated and will be removed in version ???. Please " \
            "update your code to use the Nfreqs, Ntimes, and " \
            "array_layout parameters instead.", DeprecationWarning
        )
        
    # for backwards compatability
    if n_freq is not None:
        Nfreqs = n_freq
    if n_times is not None:
        Ntimes = n_times
    if antennas is not None:
        array_layout = antennas

    # only specify defaults this way for 
    # things that are *not* season-specific
    polarization_array = kwargs.pop("polarization_array", ['xx'])
    telescope_location = list(
        kwargs.pop("telescope_location", HERA_LAT_LON_ALT)
    )
    telescope_name = kwargs.pop("telescope_name", "hera_sim")
    write_files = kwargs.pop("write_files", False)

    uvd = initialize_uvdata_from_keywords(
        Ntimes=Ntimes,
        start_time=start_time,
        integration_time=integration_time,
        Nfreqs=Nfreqs,
        start_freq=start_freq,
        channel_width=channel_width,
        array_layout=array_layout,
        polarization_array=polarization_array,
        telescope_location=telescope_location,
        telescope_name=telescope_name,
        write_files=write_files,
        complete=True,
        **kwargs
    )

    # remove this once abscal is OK to use different conventions
    uvd.conjugate_bls(convention="ant1<ant2")

    return uvd
