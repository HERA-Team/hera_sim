"""Methods for input/output of data."""

import os
import re
import warnings
from collections.abc import Sequence

import numpy as np
import pyuvdata
from pyuvdata import UVData
from pyuvsim.simsetup import initialize_uvdata_from_keywords

from . import DATA_PATH
from .defaults import _defaults

HERA_LAT_LON_ALT = np.load(DATA_PATH / "HERA_LAT_LON_ALT.npy")


# this decorator allows the parameters specified in the function
# signature to be overridden by the defaults module
@_defaults
def empty_uvdata(
    Ntimes=None,
    start_time=2456658.5,  # Jan 1 2014
    integration_time=None,
    array_layout: dict[int, Sequence[float]] = None,
    Nfreqs=None,
    start_freq=None,
    channel_width=None,
    n_freq=None,
    n_times=None,
    antennas=None,  # back-compat
    conjugation=None,
    **kwargs,
):
    """Create an empty UVData object with given specifications.

    Parameters
    ----------
    Ntimes : int, optional
        NUmber of unique times in the data object.
    start_time : float, optional
        Starting time (Julian date) by default 2456658.5
    array_layout : dict, optional
        Specify an array layout. Keys should be integers specifying antenna numbers,
        and values should be length-3 sequences of floats specifying ENU positions.
    Nfreqs : int, optional
        Number of frequency channels in the data object
    start_freq : float, optional
        Lowest frequency channel, by default None
    channel_width : float, optional
        Channel width, by default None
    n_freq : int, optional
        Alias for ``Nfreqs``
    n_times : int, optional
        Alias for ``Ntimes``.
    antennas : dict, optional
        Alias for array_layout for backwards compatibility.
    **kwargs
        Passed to :func:`pyuvsim.simsetup.initialize_uvdata_from_keywords`

    Returns
    -------
    UVData
        An empty UVData object with given specifications.
    """
    # issue a deprecation warning if any old parameters are used
    if any(param is not None for param in (n_freq, n_times, antennas)):
        warnings.warn(
            "The n_freq, n_times, and antennas parameters are being "
            "deprecated and will be removed in the future. Please "
            "update your code to use the Nfreqs, Ntimes, and "
            "array_layout parameters instead.",
            category=DeprecationWarning,
            stacklevel=2,
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
    polarization_array = kwargs.pop("polarization_array", ["xx"])
    telescope_location = [
        float(x) for x in kwargs.pop("telescope_location", HERA_LAT_LON_ALT)
    ]

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
        **kwargs,
    )
    # This is a bit of a hack, but this seems like the only way?
    if pyuvdata.__version__ < "2.2.0":
        uvd.set_drift()
    elif next(iter(uvd.phase_center_catalog.values()))["cat_type"] != "unprojected":
        uvd.fix_phase()

    # TODO: the following is a hack patch for pyuvsim which should be fixed there.
    if "x_orientation" in kwargs and uvd.x_orientation is None:
        uvd.x_orientation = kwargs["x_orientation"]

    if conjugation is not None:
        uvd.conjugate_bls(convention=conjugation)

    # Ensure we're using future array shapes
    if not uvd.future_array_shapes:  # pragma: no cover
        uvd.use_future_array_shapes()

    return uvd


def chunk_sim_and_save(
    sim_uvd,
    save_dir,
    ref_files=None,
    Nint_per_file=None,
    prefix=None,
    sky_cmp=None,
    state=None,
    filetype="uvh5",
    clobber=True,
):
    """
    Chunk the simulation data to match the reference file and write to disk.

    Chunked files have the following naming convention:
    ``save_dir/[{prefix}.]{jd_major}.{jd_minor}[.{sky_cmp}][.{state}].{filetype}``.
    The entires in brackets are optional and may be omitted.

    Parameters
    ----------
    sim_uvd : :class:`pyuvdata.UVData`
        :class:`pyuvdata.UVData` object containing the simulation data
        to chunk and write to disk.
    save_dir : str or path-like object
        Path to the directory where the chunked files will be saved.
    ref_files : iterable of str
        Iterable of filepaths to use for reference when chunking. This must
        be specified if ``Nint_per_file`` is not specified. This determines
        (and overrides, if also provided) ``Nint_per_file`` if provided.
    Nint_per_file : int, optional
        Number of integrations per chunked file. This must be specified
        if ``ref_files`` is not specified.
    prefix : str, optional
        Prefix of file basename. Default is to add no prefix.
    sky_cmp : str, optional
        String denoting which sky component has been simulated. Should
        be one of the following: ('foregrounds', 'eor', 'sum').
    state : str, optional
        String denoting whether the file is the true sky or corrupted.
    filetype : str, optional
        Format to use when writing files to disk. Must be a filetype
        supported by :class:`pyuvdata.UVData`. Default is uvh5.
    clobber : bool, optional
        Whether to overwrite any existing files that share the new
        filenames. Default is to overwrite files.
    """
    if not isinstance(sim_uvd, UVData):
        raise ValueError("sim_uvd must be a UVData object.")

    write_method = getattr(sim_uvd, f"write_{filetype}", None)
    if write_method is None:
        raise ValueError("Write method not supported.")

    if ref_files is None and Nint_per_file is None:
        raise ValueError(
            "Either a glob of reference files or the number of integrations "
            "per file must be provided."
        )

    # Pull the number of integrations per file if needed.
    if ref_files is not None:
        uvd = UVData()
        uvd.read(ref_files[0], read_data=False)
        Nint_per_file = uvd.Ntimes
        jd_pattern = re.compile(r"\.(?P<major>[0-9]{7})\.(?P<minor>[0-9]{5}).")

    # Pull the simulation times, then start the chunking process.
    sim_times = np.unique(sim_uvd.time_array)
    Nfiles = int(np.ceil(sim_uvd.Ntimes / Nint_per_file))
    for Nfile in range(Nfiles):
        # Figure out filing and slicing information.
        if ref_files is not None:
            jd = re.search(jd_pattern, str(ref_files[Nfile])).groupdict()
            jd = float(f"{jd['major']}.{jd['minor']}")
            uvd = UVData()
            uvd.read(ref_files[Nfile], read_data=False)
            times = np.unique(uvd.time_array)
        else:
            start_ind = Nfile * Nint_per_file
            jd = np.round(sim_times[start_ind], 5)
            this_slice = slice(start_ind, start_ind + Nint_per_file)
            times = sim_times[this_slice]
        filename = f"{jd:.5f}.{filetype}"
        if prefix is not None:
            filename = f"{prefix}." + filename
        if sky_cmp is not None:
            filename = filename.replace(f".{filetype}", f".{sky_cmp}.{filetype}")
        if state is not None:
            filename = filename.replace(f".{filetype}", f".{state}.{filetype}")
        save_path = os.path.join(save_dir, filename)

        # Chunk it and write to disk.
        this_uvd = sim_uvd.select(times=times, inplace=False)
        getattr(this_uvd, f"write_{filetype}")(save_path, clobber=clobber)

        # Delete the temporary UVData object to speed things up a bit.
        del this_uvd
