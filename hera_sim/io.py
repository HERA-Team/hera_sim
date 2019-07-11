"""
A module containing routines for interfacing data produced by `hera_sim` with other
codes, especially UVData.
"""
import numpy as np
from pyuvsim.simsetup import initialize_uvdata_from_keywords

HERA_LAT_LON_ALT = (
    np.degrees(-0.53619179912885),
    np.degrees(0.3739944696510935),
    1073.0000000074506
)


def empty_uvdata(nfreq, ntimes, ants, **kwargs):
    """Construct and return a full :class:`pyuvdata.UVData` object, with empty
    `data_array`.

    This function is merely a thin wrapper around
    :func:`pyuvsim.simsetup.initialize_uvdata_from_keywords`, providing some defaults
    aligned with a nominal HERA telescope.

    Args:
        nfreq (int):
            The number of frequency channels to include.
        ntimes (int):
            The number of times to observe.
        ants (dict):
            A dictionary mapping an integer to a three-tuple of ENU co-ordinates for
            each antenna. These antennas can be down-selected via keywords.
        **kwargs:
            All kwargs are passed directly to
            :func:`pyuvsim.simsetup.initialize_uvdata_from_keywords`. However,
            some extra defaults are set:
                | telescope_location: the site of HERA
                | telescope_name: "hera_sim"
                | start_freq: 100 MHz
                | channel_width: 100 MHz / 1024
                | integration_time: 10.7 sec
                | start_time: 2458119.5 JD
                | end_time: start_time + ntimes*integration_time
                | polarizations : ['xx']
                | write_files: False

    Returns:
        A :class:`pyuvdata.UVData` object, unfilled.

    """
    start_time = kwargs.get("start_time", 2458119.5)
    integration_time = kwargs.get("integration_time", 10.7)

    uv = initialize_uvdata_from_keywords(
        antenna_layout_filename=None,  # To keep consistency with old hera_sim empty_uvdata
        array_layout=ants,
        telescope_location=list(kwargs.get("telescope_location", HERA_LAT_LON_ALT)),
        telescope_name=kwargs.get("telescope_name", "hera_sim"),
        Nfreqs=nfreq,
        start_freq=kwargs.get("start_freq", 1e8),
        freq_array=None,  # To keep consistency with old hera_sim empty_uvdata
        channel_width=kwargs.get("channel_width", 1e8 / 1024.),
        Ntimes=ntimes,
        integration_time=integration_time,
        start_time=start_time,
        end_time=kwargs.get("end_time", start_time + ntimes * integration_time),
        time_array=None,  # To keep consistency with old hera_sim empty_uvdata
        polarization_array=kwargs.get("polarization_array", ['xx']),
        write_files=kwargs.get("write_files", False),
        complete=True,  # Fills out the UVData object
        **kwargs
    )

    return uv
