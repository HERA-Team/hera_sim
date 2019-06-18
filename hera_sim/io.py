"""
A module containing routines for interfacing data produced by `hera_sim` with other
codes, especially UVData.
"""
import numpy as np
from pyuvsim.simsetup import initialize_uvdata_from_keywords
from pyuvsim.uvsim import init_uvdata_out
import copy

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
                | polarization_array: np.array([-5])
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
        polarizations=kwargs.get("polarizations", ['xx']),
        polarization_array=kwargs.get("polarization_array", np.array([-5])),
        write_files=kwargs.get("write_files", False),
        **kwargs
    )[0]  # TODO: The 0 index needs to be removed when pyuvsim is fixed.

    init_uvdata_out(uv)
    return uv


def init_uvdata_out(uv_in, inplace=True):
    """
    Initialize an empty uvdata object to fill with simulated data.

    NOTE: this is a copy of the function from pyuvsim, but streamlined.
    Args:
        uv_in: The input uvdata object.
               This is usually an incomplete object, containing only metadata.
        source_list_name: Name of source list file or mock catalog.
        obs_param_file: Name of observation parameter config file
        telescope_config_file: Name of telescope config file
        antenna_location_file: Name of antenna location file
    """
    if not inplace:
        uv_obj = copy.deepcopy(uv_in)
    else:
        uv_obj = uv_in

    uv_obj.set_drift()
    uv_obj.vis_units = 'Jy'

    uv_obj.instrument = uv_obj.telescope_name
    uv_obj.set_lsts_from_time_array()
    uv_obj.spw_array = np.array([0])
    if uv_obj.Nfreqs == 1:
        uv_obj.channel_width = 1.  # Hz
    else:
        uv_obj.channel_width = np.diff(uv_obj.freq_array[0])[0]
    uv_obj.set_uvws_from_antenna_positions()
    if uv_obj.Ntimes == 1:
        uv_obj.integration_time = np.ones_like(uv_obj.time_array, dtype=np.float64)  # Second
    else:
        # Note: currently only support a constant spacing of times
        uv_obj.integration_time = (np.ones_like(uv_obj.time_array, dtype=np.float64)
                                   * np.diff(np.unique(uv_obj.time_array))[0] * (24. * 60**2))  # Seconds

    # Clear existing data, if any
    uv_obj.data_array = np.zeros((uv_obj.Nblts, uv_obj.Nspws, uv_obj.Nfreqs, uv_obj.Npols), dtype=np.complex)
    uv_obj.flag_array = np.zeros((uv_obj.Nblts, uv_obj.Nspws, uv_obj.Nfreqs, uv_obj.Npols), dtype=bool)
    uv_obj.nsample_array = np.ones_like(uv_obj.data_array, dtype=float)

    uv_obj.extra_keywords = {}

    # TODO: this needs to be fixed in the original metadata setter
    uv_obj.telescope_location = list(uv_obj.telescope_location)
    uv_obj.antenna_names = uv_obj.antenna_numbers.astype('str')
    uv_obj.check()

    return uv_obj