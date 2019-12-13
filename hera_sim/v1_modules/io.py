"""
"""
import os
import numpy as np
from pyuvsim.simsetup import initialize_uvdata_from_keywords
from .data import DATA_PATH
from .defaults import _defaults

HERA_LAT_LON_ALT = np.load(os.path.join(DATA_PATH, "HERA_LAT_LON_ALT.npy"))

# this decorator allows the parameters specified in the function
# signature to be overridden by the defaults module
@_defaults
def empty_uvdata(Ntimes=None, start_time=None, integration_time=None,
                 Nfreqs=None, start_freq=None, channel_width=None,
                 array_layout=None, **kwargs):
    # TODO: docstring
    """
    """
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
