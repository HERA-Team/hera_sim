"""
"""
import os
import numpy as np
from pyuvsim.simsetup import initialize_uvdata_from_keywords
from .data import DATA_PATH

HERA_LAT_LON_ALT = np.load(os.path.join(DATA_PATH, "HERA_LAT_LON_ALT.npy"))

def empty_uvdata(**kwargs):
    # TODO: docstring
    """
    """
    # pull from defaults?
    # only use defaults for things that are *not* season-specific
    telescope_location = kwargs.pop("telescope_location", HERA_LAT_LON_ALT)
    telescope_name = kwargs.pop("telescope_name", "hera_sim")
    write_files = kwargs.pop("write_files", False)

    uvd = initialize_uvdata_from_keywords(
        telescope_location=telescope_location,
        telescope_name=telescope_name,
        write_files=write_files,
        complete=True,
        **kwargs
    )

    # remove this once abscal is OK to use different conventions
    uvd.conjuage_bls(convention="ant1<ant2")

    return uvd
