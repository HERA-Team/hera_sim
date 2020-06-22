"""Module providing tools for adjusting simulation data/metadata to a reference."""

import copy
import os
import pathlib

import numpy as np
from astropy import units
from scipy.interpolate import interp1d, RectBivariateSpline
from warnings import warn

from pyuvdata import UVData
from pyuvdata.utils import polstr2num
from .simulate import Simulator
from .utils import _listify

try:
    import hera_cal
except ImportError:
    warn("hera_cal is not installed. Certain features will be unavailable.")


def adjust_to_reference(
    target,
    reference,
    interpolate=True,
    use_reference_positions=False,
    position_tolerance=1,
    relabel_antennas=True,
    conjugation_convention=None,
    overwrite_telescope_metadata=False,
    verbose=False,
):
    """
    Modify target data/metadata to be consistent with some reference.

    Parameters
    ----------
    target: str, path-like object, or :class:`pyuvdata.UVData`-like object
        Target object to modify. If it is a path-like object, then it is
        loaded into a :class:`pyuvdata.UVData` object.
    reference: str, path-like object, or :class:`pyuvdata.UVData` instance
        Reference to use when modifying target object. May be an ordered
        collection of path-like objects. Only the metadata is required.
    interpolate: bool, optional
        Whether to interpolate the target data in time, or whether to rephase.
        Default is to interpolate in time using a cubic spline.
    use_reference_positions: bool, optional
        Whether to use the reference antenna positions or a shifted version of
        the original target object's antenna positions. When adjusting simulated
        data to observational data, this parameter decides whether to use the
        simulated antenna positions or the observational data's antenna positions.
        Default is to use the original target object's antenna positions, but
        shift them in a way that produces the maximal overlap with the reference
        antenna positions.
    position_tolerance: float or array-like of float, optional
        Tolerance for adjusting antenna positions, in meters. Default is one
        meter in x, y, and z. Specifying a single number will use the same value
        for each component; specifying a length-3 array of floats will use those
        values for the individual components, in ENU-coordinates.
    relabel_antennas: bool, optional
        Whether to rewrite antenna numbers/names when modifying the target
        antenna array to optimally match the reference antenna array. Default
        is to relabel the target object's antennas with the reference antenna
        labels.
    conjugation_convention: str, optional
        Which conjugation convention to enforce after performing the antenna
        array adjustment step. Default is to not enforce a convention.
    overwrite_telescope_metadata: bool, optional
        Whether to overwrite the target object's telescope metadata with that
        of the reference data. Default is to leave the target object's telescope
        metadata unchanged.
    verbose: bool, optional
        Whether to print brief statements noting progress. Default is to not
        print updates.

    Returns
    -------
    modified_data : :class:`pyuvdata.UVData` or :class:`Simulator` object
        :class:`pyuvdata.UVData` object containing the modified target data.
        If the initial target data was passed as a :class:`Simulator` object,
        then the returned object is also a :class:`Simulator` object.

    Notes
    -----
    The modified data will have its time and LST metadata modified to exactly
    match the time and LST metadata of the reference data for all reference LSTs
    that lie within the bounds of the target object's LSTs. Additionally, only a
    subset of the original antenna array will remain, and the remaining antennas
    will have their positions modified (either shifted or completely overwritten)
    such that the modified positions give the largest intersection with the
    reference antenna array, up to the specified tolerance. The flag and nsamples
    arrays do not have their values modified, but may be reshaped.

    Unless the integration times for the target and reference match each other
    exactly, it is strongly recommended that you use cubic spline interpolation
    rather than the rephasing provided here (which is just a thin wrapper around
    a rephasing tool from :package:`hera_cal`), since small discrepancies in
    integration times can result in some integrations effectively being skipped
    by the rephasing tool, producing discontinuities in the rephased data.

    Finally, a word of caution: if the target object has its data compressed by
    redundancy, then it is possible that the antenna array adjustment step may
    trigger a memory overflow. The current algorithm requires that every antenna
    in the target object has data associated with it, and ensures that this is
    true by inflating the data by redundancy. (It is not clear if it is even
    possible to perform the antenna array adjustment step for an object with
    arbitrary compression by redundancy.)
    """
    if verbose:
        print("Validating positional arguments...")

    # Quickly check that the tolerance for the antenna adjustment is valid.
    # (Processing prior to antenna adjustment can be long, so do this early.)
    if np.isrealobj(position_tolerance):
        if np.isscalar(position_tolerance):
            position_tolerance = np.ones(3) * position_tolerance
        else:
            position_tolerance = np.array(position_tolerance).flatten()
            if position_tolerance.size != 3:
                raise ValueError(
                    "position_tolerance should be a scalar or length-3 array."
                )
    else:
        raise TypeError(
            "position_tolerance must be a real-valued scalar or length-3 array."
        )

    # Check if the target object is a Simulator, but work with a UVData object.
    target_is_simulator = isinstance(target, Simulator)
    target = _to_uvdata(target)

    # Pull the reference metadata.
    if not isinstance(reference, UVData):
        if isinstance(reference, Simulator):
            reference_metadata = reference.data.copy(metadata_only=True)
        else:
            reference_files = _listify(reference)
            _validate_file_list(reference_files, "reference")
            reference_metadata = UVData()
            reference_metadata.read(reference_files, read_data=False)
    else:
        reference_metadata = reference.copy(metadata_only=True)

    if verbose:
        if interpolate:
            print("Interpolating target data to reference data LSTs...")
        else:
            print("Rephasing target data to reference data LSTs...")

    if interpolate:
        target = interpolate_to_reference(target, reference_metadata)
    else:
        if not np.isclose(
            target.integration_time.mean(), reference_metadata.integration_time.mean()
        ):
            msg = "Target and reference integration times do not match. "
            msg += "This may result in discontinuities in the result."
            warn(msg)

        target = rephase_to_reference(target, reference_metadata)

    if verbose:
        print("Inflating target data by baseline redundancy...")
    target.inflate_by_redundancy()

    if verbose:
        print("Adjusting target's antenna array to optimally match reference...")
    target = match_antennas(
        target,
        reference_metadata,
        tol=position_tolerance,
        relabel_antennas=relabel_antennas,
        use_reference_positions=use_reference_positions,
        overwrite_telescope_metadata=overwrite_telescope_metadata,
    )

    if conjugation_convention is not None:
        if verbose:
            print(f"Conjugating target to {conjugation_convention} convention...")
        target.conjugate_bls(conjugation_convention)

    if target_is_simulator:
        target = Simulator(data=target)

    return target


def match_antennas(
    target,
    reference,
    tol=1.0,
    relabel_antennas=True,
    use_reference_positions=False,
    overwrite_telescope_metadata=False,
):
    pass


def interpolate_to_reference(
    target,
    reference=None,
    ref_times=None,
    ref_lsts=None,
    ref_freqs=None,
    axis="time",
    kind="cubic",
):
    """
    Interpolate target visibilities to reference times/frequencies. Interpolation
    may be along one axis or both. Interpolating data with a phase wrap or
    interpolating data to a phase-wrapped set of LSTs is currently unsupported.

    Parameters
    ----------
    target: :class:`pyuvdata.UVData` instance or :class:`Simulator` instance
        Object containing the visibility data and metadata which is to be
        interpolated to the reference LSTs and/or frequencies.
    reference: :class:`pyuvdata.UVData` instance or :class:`Simulator` instance
        Object containing reference metadata. Does not need to be provided if
        metadata arrays are provided (``ref_times``, ``ref_lsts``, and/or
        ``ref_freqs``, depending on which axis is used).
    ref_times: array-like of float
        Reference times corresponding to the LSTs the target will be interpolated
        to, in units of JD. Must be provided if ``reference`` is not provided and
        interpolation is along the time axis.
    ref_lsts: array-like of float
        Reference LSTs to interpolate the target data to, in units of radians.
        Must be provided if ``reference`` is not provided and interpolation is
        along the time axis.
    ref_freqs: array-like of float
        Reference frequencies to interpolate the target data to, in units of Hz.
        Must be provided if ``reference`` is not provided and interpolation is
        along frequency axis.
    axis: str, optional
        Axis to perform interpolation along. Must be one of the following: 'time',
        'freq', or 'both'. Default is to interpolate along the time axis.
    kind: str, optional
        Order of the spline interpolator used. A cubic spline is used by default.

    Returns
    -------
    modified_data: :class:`pyuvdata.UVData` instance or :class:`Simulator` instance
        Data adjusted so that its times match the reference times which correspond
        to overlapping LSTs. The type of object returned is the same as the type
        of object passed.
    """
    # First, do a check of reference information
    if reference is not None:
        ref_time_to_lst_map = {
            ref_time: ref_lst
            for ref_time, ref_lst in zip(reference.time_array, reference.lst_array)
        }
        ref_times = np.array(list(ref_time_to_lst_map.keys()))
        ref_lsts = np.array(list(ref_time_to_lst_map.values()))
        ref_freqs = np.unique(reference.freq_array)
    else:
        if axis in ("time", "both") and (ref_lsts is None or ref_times is None):
            raise ValueError(
                "Time and LST reference information must be provided for "
                "interpolation along time-axis."
            )
        if axis in ("freq", "both") and ref_freqs is None:
            raise ValueError(
                "Frequency reference information must be provided for "
                "interpolation along frequency-axis."
            )

    target_is_simulator = isinstance(target, Simulator)
    target = _to_uvdata(target)
    target_time_to_lst_map = {
        target_time: target_lst
        for target_time, target_lst in zip(target.time_array, target.lst_array)
    }
    target_lsts = np.array(list(target_time_to_lst_map.values()))
    target_freqs = np.unique(target.freq_array)

    # TODO: figure out how to handle phase wraps
    def iswrapped(lsts):
        return np.any(lsts < lsts[0])

    if iswrapped(target_lsts) or iswrapped(ref_lsts):
        raise NotImplementedError(
            "Either the target LSTs or the reference LSTs have a phase wrap. "
            "This is currently not supported."
        )

    # Ensure reference parameters are a subset of target parameters.
    if axis in ("time", "both"):
        if np.any(np.logical_or(ref_lsts < target_lsts[0], ref_lsts > target_lsts[-1])):
            warn("Reference LSTs not a subset of target LSTs; clipping.")
            key = np.argwhere(
                np.logical_and(target_lsts[0] <= ref_lsts, ref_lsts <= target_lsts[-1])
            ).flatten()
            ref_times = ref_times[key]
            ref_lsts = ref_lsts[key]
    if axis in ("freq", "both"):
        if np.any(
            np.logical_or(ref_freqs < target_freqs[0], ref_freqs > target_freqs[-1])
        ):
            warn("Reference frequencies not a subset of target frequencies; clipping.")
            key = np.argwhere(
                np.logical_and(
                    target_freqs[0] <= ref_freqs, ref_freqs <= target_freqs[-1]
                )
            ).flatten()
            ref_freqs = ref_freqs[key]

    # Setup data/metadata objects that need to be non-trivially rewritten.
    if axis in ("time", "both"):
        new_Nblts = ref_times.size * target.Nbls
        new_time_array = np.empty(new_Nblts, dtype=float)
        new_lst_array = np.empty(new_Nblts, dtype=float)
        new_ant_1_array = np.empty(new_Nblts, dtype=int)
        new_ant_2_array = np.empty(new_Nblts, dtype=int)
        new_baseline_array = np.empty(new_Nblts, dtype=int)
        new_uvw_array = np.empty((new_Nblts, 3), dtype=float)
        if axis == "both":
            new_data_shape = (new_Nblts, 1, ref_freqs.size, target.Npols)
        else:
            new_data_shape = (new_Nblts, 1, target_freqs.size, target.Npols)
        new_data = np.zeros(new_data_shape, dtype=np.complex)
    else:
        new_data_shape = (target.Nblts, 1, ref_freqs.size, target.Npols)

    # Actually update metadata and interpolate the data.
    for i, bl in enumerate(target.get_antpairs()):
        if axis == "freq":
            for pol_ind, pol in enumerate(target.polarization_array):
                vis = target.get_data(bl + (pol,))
                this_blt_slice = target._key2inds(bl + (pol,))[0]
                re_spline = interp1d(target_freqs, vis.real, axis=1, kind=kind)
                im_spline = interp1d(target_freqs, vis.imag, axis=1, kind=kind)
                new_data[this_blt_slice, 0, :, pol_ind] = re_spline(
                    ref_freqs
                ) + 1j * im_spline(ref_freqs)
            continue

        # Preparation for updating metadata.
        ant1, ant2 = bl
        this_slice = slice(i, None, target.Nbls)
        old_blt = target._key2inds(bl)[0][0]  # As a reference
        this_uvw = target.uvw_array[old_blt]
        this_baseline = target.baseline_array[old_blt]

        # Now actually update the metadata.
        new_ant_1_array[this_slice] = ant1
        new_ant_2_array[this_slice] = ant2
        new_baseline_array[this_slice] = this_baseline
        new_uvw_array[this_slice] = this_uvw
        new_time_array[this_slice] = ref_times
        new_lst_array[this_slice] = ref_lsts

        # Update the data.
        for pol_ind, pol in enumerate(target.polarization_array):
            vis = target.get_data(bl + (pol,))
            if axis == "both":
                re_spline = RectBivariateSpline(
                    target_lsts, target_freqs, vis.real, kind=kind
                )
                im_spline = RectBivariateSpline(
                    target_lsts, target_freqs, vis.imag, kind=kind
                )
                new_data[this_slice, 0, :, pol_ind] = re_spline(
                    ref_lsts, ref_freqs
                ) + 1j * im_spline(ref_lsts, ref_freqs)
            else:
                re_spline = interp1d(target_lsts, vis.real, axis=0, kind=kind)
                im_spline = interp1d(target_lsts, vis.imag, axis=0, kind=kind)
                new_data[this_slice, 0, :, pol_ind] = re_spline(
                    ref_lsts
                ) + 1j * im_spline(ref_lsts)

    # Finally, update all of the metadata.
    if axis in ("freq", "both"):
        target.Nfreqs = ref_freqs.size
    if axis in ("time", "both"):
        target.Ntimes = ref_times.size
        target.time_array = new_time_array
        target.lst_array = new_lst_array
        target.ant_1_array = new_ant_1_array
        target.ant_2_array = new_ant_2_array
        target.baseline_array = new_baseline_array
        target.uvw_array = new_uvw_array
        target.blt_order = None

    # Now update the data-like attributes
    target.flag_array = np.zeros(new_data.shape, dtype=bool)
    target.nsample_array = np.ones(new_data.shape, dtype=float)
    target.data_array = new_data
    if target_is_simulator:
        target = Simulator(data=target)

    return target


def rephase_to_reference(
    target, reference=None, ref_times=None, ref_lsts=None,
):
    """
    Rephase target data to match overlapping reference LSTs. This function
    requires that ``hera_cal`` be installed.

    Parameters
    ----------
    target: :class:`pyuvdata.UVData` instance or :class:`Simulator` instance
        Object containing the visibility data and metadata which is to be
        rephased to reference LSTs.
    reference:
    """
    pass


def _validate_file_list(file_list, name="file list"):
    """Ensure all entries in the file list are path-like objects."""
    if not all(isinstance(item, (str, pathlib.Path)) for item in file_list):
        raise TypeError(
            f"{name} must be either a collection of path-like objects or a "
            "UVData object or Simulator object"
        )


def _to_uvdata(sim):
    """Convert input object to a :class:`pyuvdata.UVData` object."""
    if isinstance(sim, UVData):
        return sim
    elif isinstance(sim, Simulator):
        return sim.data
    elif isinstance(sim, (str, pathlib.Path)):
        uvd = UVData()
        uvd.read(sim)
        return uvd
    else:
        try:
            _ = iter(sim)
            if not all(os.path.exists(entry) for entry in sim):
                raise TypeError
            uvd = UVData()
            uvd.read(sim)
            return uvd
        except TypeError:
            raise TypeError("Input object could not be converted to UVData object.")
