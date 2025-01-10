"""Module providing tools for adjusting simulation data/metadata to a reference."""

import logging
import os
import pathlib
from warnings import warn

import numpy as np
from pyuvdata import UVData
from pyuvdata.utils import polnum2str
from scipy.interpolate import RectBivariateSpline, interp1d

from .simulate import Simulator
from .utils import _listify

try:
    # Import hera_cal functions.
    from hera_cal.abscal import get_d2m_time_map
    from hera_cal.io import to_HERAData
    from hera_cal.utils import lst_rephase

    HERA_CAL = True
except (ModuleNotFoundError, FileNotFoundError) as err:  # pragma: no cover
    if err is ModuleNotFoundError:
        missing = "hera-calibration"
    else:
        missing = "git"
    HERA_CAL = False


logger = logging.getLogger(__name__)


def adjust_to_reference(
    target,
    reference,
    interpolate=True,
    interpolation_axis="time",
    use_reference_positions=False,
    use_ENU_positions=False,
    position_tolerance=1,
    relabel_antennas=True,
    conjugation_convention=None,
    overwrite_telescope_metadata=False,
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
    interpolation_axis: str, optional
        Which axis to perform interpolation over; must be one of 'freq', 'time',
        or 'both'. Ignored if ``interpolate==False``. Default is 'time'.
    use_reference_positions: bool, optional
        Whether to use the reference antenna positions or a shifted version of
        the original target object's antenna positions. When adjusting simulated
        data to observational data, this parameter decides whether to use the
        simulated antenna positions or the observational data's antenna positions.
        Default is to use the original target object's antenna positions, but
        shift them in a way that produces the maximal overlap with the reference
        antenna positions.
    use_ENU_positions: bool, optional
        Whether to perform the antenna matching algorithm using ENU antenna
        positions. Default is to use whatever coordinates the antenna positions
        are used in the ``target`` and ``reference`` data.
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

    Returns
    -------
    modified_data : :class:`pyuvdata.UVData` or :class:`~.simulate.Simulator` object
        :class:`pyuvdata.UVData` object containing the modified target data.
        If the initial target data was passed as a :class:`~.simulate.Simulator` object,
        then the returned object is also a :class:`~.simulate.Simulator` object.

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
    a rephasing tool from ``hera_cal``), since small discrepancies in
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
    # Quickly check that the tolerance for the antenna adjustment is valid.
    # (Processing prior to antenna adjustment can be long, so do this early.)
    logger.info("Validating positional arguments...")
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
    if not target.future_array_shapes:  # pragma: no cover
        target.use_future_array_shapes()

    # Pull the reference metadata.
    if not isinstance(reference, UVData):
        if isinstance(reference, Simulator):
            reference_metadata = reference.data.copy(metadata_only=True)
        else:
            reference_files = _listify(reference)
            _validate_file_list(reference_files)
            reference_metadata = UVData()
            reference_metadata.read(reference_files, read_data=False)
    else:
        reference_metadata = reference.copy(metadata_only=True)

    if interpolate:
        logger.info("Interpolating target data to reference data LSTs...")
    else:
        logger.info("Rephasing target data to reference data LSTs...")

    if interpolate:
        target = interpolate_to_reference(
            target, reference_metadata, axis=interpolation_axis
        )
    else:
        target = rephase_to_reference(target, reference_metadata)

    # Check if target is compressed by redundancy.
    Nbls_with_autos = target.Nants_telescope * (target.Nants_telescope + 1) / 2
    Nbls_mod_autos = Nbls_with_autos - target.Nants_telescope
    if target.Nbls not in (Nbls_with_autos, Nbls_mod_autos):
        logger.info("Inflating target data by baseline redundancy...")
        target.inflate_by_redundancy()

    logger.info("Adjusting target's antenna array to optimally match reference...")
    # Make sure the reference metadata has had its times updated appropriately.
    if not interpolate or interpolation_axis in ("time", "both"):
        reference_metadata.select(times=target.time_array)
    target = match_antennas(
        target,
        reference_metadata,
        tol=position_tolerance,
        ENU=use_ENU_positions,
        relabel_antennas=relabel_antennas,
        use_reference_positions=use_reference_positions,
        overwrite_telescope_metadata=overwrite_telescope_metadata,
    )

    if conjugation_convention is not None:
        logger.info(f"Conjugating target to {conjugation_convention} convention...")
        target.conjugate_bls(conjugation_convention)

    if target_is_simulator:
        target = Simulator(data=target)

    return target


def match_antennas(
    target,
    reference,
    tol=1.0,
    ENU=False,
    relabel_antennas=True,
    use_reference_positions=False,
    overwrite_telescope_metadata=False,
):
    """
    Select the best-matched subset of antennas between target and reference.

    This function chooses a subset of antennas from ``target`` and ``reference``
    to keep so that the resulting subset has the greatest number of baselines
    and the set of baselines produced by one set of antennas is identical to
    the baselines produced by the other set of antennas. One can show that this
    optimal set of antennas can be obtained by finding the greatest intersection
    of the two sets of antennas, up to the symmetries allowed by how baselines
    are calculated from antenna positions (namely translation and reflection
    invariance).

    Parameters
    ----------
    target: :class:`pyuvdata.UVData` instance or :class:`~.simulate.Simulator` instance
        Object containing the data and metadata for which the antenna position
        adjustment is intended.
    reference: :class:`pyuvdata.UVData`
        Object containing the reference data, to which the target antenna array
        is to be matched. Only the metadata is needed.
    tol: float or array-like of float, optional
        Position tolerance for matching antenna positions, in meters. If a float
        is passed, then this is taken to be the tolerance in each dimension;
        otherwise, a length-3 array-like object of floats must be passed which
        specify the tolerance in each dimension. Default is 1 meter in each
        dimension (so a 1-meter cube neighborhood centered on each antenna).
    ENU: bool, optional
        Whether to perform the calculation in ENU coordinates. Default is to
        perform the calculation in whatever coordinate system the ``target`` and
        ``reference`` antenna positions are defined.
    relabel_antennas: bool, optional
        Whether to change the labels of the remaining ``target`` antennas to
        match those of the corresponding ``reference`` antennas. Default is to
        make the antenna labels match.
    use_reference_positions: bool, optional
        Whether to modify the remaining ``target`` antenna positions to exactly
        match those of the corresponding ``reference`` antennas. Default is to
        apply a translation to the remaining array, so that ideal antenna
        positions remain ideal in the case that ideal antenna positions are
        matched to real antenna positions.
    overwrite_telescope_metadata: bool, optional
        Whether to overwrite ``target`` telescope metadata (such as observatory
        position) to match ``reference`` telescope metadata. Default is to leave
        the ``target`` telescope metadata untouched.

    Returns
    -------
    modified_data: :class:`pyuvdata.UVData` or :class:`~.simulate.Simulator` instance
        Object containing the ``target`` data with its antenna array modified and
        a downselect performed to only keep data for remaining antennas.
    """
    # TODO: This function should be updated so that the antenna matching is
    # performed locally in ENU coordinates, though the process of updating to
    # that routine brings with it some questions that need to be thought about
    # carefully and should be discussed. In particular, it is important to
    # address whether it is acceptable to modify the target antenna positions
    # in a way that allows them to be reflected through the local origin. While
    # this is a mathematically necessary consideration for finding the optimal
    # match in non-exponential time, adjusting the antenna positions like-so
    # is pretty unrealistic.
    target_is_simulator = isinstance(target, Simulator)
    target = _to_uvdata(target)
    if not target.future_array_shapes:  # pragma: nocover
        target.use_future_array_shapes()
    target_copy = target.copy()
    reference = _to_uvdata(reference)
    if not reference.future_array_shapes:  # pragma: nocover
        reference.use_future_array_shapes()
    reference_metadata = reference.copy(metadata_only=True)

    # Find the best choice of mapping between antennas.
    target_antpos = _get_antpos(target, ENU=ENU)
    ref_antpos = _get_antpos(reference, ENU=ENU)
    # This contains the downselected and shifted target antenna positions.
    array_intersection, is_reflected = _get_array_intersection(
        target_antpos, ref_antpos, tol
    )
    target_to_reference_map = _get_antenna_map(array_intersection, ref_antpos, tol)
    reference_to_target_map = {
        ref_ant: target_ant for target_ant, ref_ant in target_to_reference_map.items()
    }

    # Select only antennas that remain in the overlap.
    target_copy.select(
        antenna_nums=list(target_to_reference_map.keys()), keep_all_metadata=False
    )
    reference_metadata.select(
        antenna_nums=list(target_to_reference_map.values()), keep_all_metadata=False
    )

    # Update antenna and possibly telescope metadata.
    if relabel_antennas:
        target_copy.ant_1_array = np.asarray(
            [
                target_to_reference_map[target_ant]
                for target_ant in target_copy.ant_1_array
            ]
        )
        target_copy.ant_2_array = np.asarray(
            [
                target_to_reference_map[target_ant]
                for target_ant in target_copy.ant_2_array
            ]
        )
        for i, bl in enumerate(target_copy.baseline_array):
            ant1, ant2 = target.baseline_to_antnums(bl)
            ant1 = target_to_reference_map[ant1]
            ant2 = target_to_reference_map[ant2]
            newbl = target.antnums_to_baseline(ant1, ant2)
            target_copy.baseline_array[i] = newbl

    attrs_to_update = tuple()
    if relabel_antennas:
        attrs_to_update += ("antenna_numbers", "antenna_names")
    if overwrite_telescope_metadata:
        attrs_to_update += (
            "telescope_location",
            "telescope_location_lat_lon_alt",
            "telescope_location_lat_lon_alt_degrees",
        )
    for attr in attrs_to_update:
        setattr(target_copy, attr, getattr(reference_metadata, attr))

    # Update the antenna positions; this is necessarily ugly.
    if use_reference_positions and relabel_antennas:
        # The antenna numbers and positions exactly match the reference.
        target_copy.antenna_positions = reference_metadata.antenna_positions
    elif use_reference_positions and not relabel_antennas:
        # We need to use the reference positions but keep the target ordering.
        target_copy.antenna_positions = np.array(
            [
                reference.antenna_positions[
                    reference.antenna_numbers.tolist().index(
                        target_to_reference_map[target_ant]
                    )
                ]
                for target_ant in target_copy.antenna_numbers
            ]
        )
    elif not use_reference_positions and relabel_antennas:
        # We need to shift the antennas and relabel them.
        target_copy.antenna_positions = np.array(
            [
                array_intersection[reference_to_target_map[ref_ant]]
                for ref_ant in reference_metadata.antenna_numbers
            ]
        )
    else:
        # We just need to shift the antenna positions.
        target_copy.antenna_positions = np.array(
            [
                array_intersection[target_ant]
                for target_ant in target_copy.antenna_numbers
            ]
        )

    target_copy._clear_key2ind_cache(target_copy)
    target_copy._clear_antpair2ind_cache(target_copy)

    # Now update the data... this will be a little messy.
    for antpairpol, vis in target.antpairpol_iter():
        ant1, ant2, pol = antpairpol
        # Skip this visibility if we dropped one of the antennas.
        if ant1 not in array_intersection or ant2 not in array_intersection:
            continue

        # Figure out the new antenna-pair.
        if relabel_antennas:
            new_antpairpol = (
                target_to_reference_map[ant1],
                target_to_reference_map[ant2],
                pol,
            )
        else:
            new_antpairpol = antpairpol

        # Figure out how to slice through the new data array.
        blts, conj_blts, pol_inds = target_copy._key2inds(new_antpairpol)

        if blts is not None:
            # The new baseline has the same conjugation as the old one.
            this_slice = (blts, slice(None), pol_inds[0].start)
        else:  # pragma: no cover
            # The new baseline is conjugated relative to the old one.
            # Given the handling of the antenna relabeling, this might not actually
            # ever be called.
            this_slice = (conj_blts, slice(None), pol_inds[1])
            vis = vis.conj()
            new_antpairpol = new_antpairpol[:2][::-1] + (pol,)
        # If we needed to reflect the entire array to find the best match, then
        # we need to make sure to conjugate the visibilities since the reflection
        # is effectively undone by baseline conjugation.
        if is_reflected:
            vis = vis.conj()

        # Update the data-like parameters.
        target_copy.data_array[this_slice] = vis
        target_copy.flag_array[this_slice] = target.get_flags(antpairpol)
        target_copy.nsample_array[this_slice] = target.get_nsamples(antpairpol)

    # Update the uvw array just to be safe.
    target_copy.set_uvws_from_antenna_positions()

    # Make sure to return a Simulator object if one was passed.
    if target_is_simulator:
        target_copy = Simulator(data=target_copy)

    return target_copy


def interpolate_to_reference(
    target,
    reference=None,
    ref_times=None,
    ref_lsts=None,
    ref_freqs=None,
    axis="time",
    kind="cubic",
    kt=3,
    kf=3,
):
    """
    Interpolate target visibilities to reference times/frequencies.

    Interpolation may be along one axis or both. Interpolating data with a phase wrap or
    interpolating data to a phase-wrapped set of LSTs is currently unsupported.

    Parameters
    ----------
    target: :class:`pyuvdata.UVData` or :class:`~.simulate.Simulator` instance
        Object containing the visibility data and metadata which is to be
        interpolated to the reference LSTs and/or frequencies.
    reference: :class:`pyuvdata.UVData` or :class:`~.simulate.Simulator` instance
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
    kt, kf: int, optional
        Degrees of the bivariate spline interpolator along the time and frequency
        axes, respectively. Default is to use a bicubic spline interpolator.

    Returns
    -------
    modified_data: :class:`pyuvdata.UVData` or :class:`~.simulate.Simulator` instance
        Data adjusted so that its times match the reference times which correspond
        to overlapping LSTs. The type of object returned is the same as the type
        of object passed.
    """
    # Check that valid axis argument is passed.
    if axis not in ("time", "freq", "both"):
        raise ValueError(
            "axis parameter must be one of the following: 'time', 'freq', or 'both'."
        )

    # Check reference information is sufficient.
    if reference is not None:
        if not isinstance(reference, UVData):
            try:
                reference = _to_uvdata(reference)
            except TypeError:
                raise TypeError("reference must be convertible to a UVData object.")

        ref_times, inds = np.unique(reference.time_array, return_index=True)
        ref_lsts = reference.lst_array[inds].copy()
        ref_freqs = np.unique(reference.freq_array)
    else:
        if axis in ("time", "both"):
            if ref_lsts is None or ref_times is None:
                raise ValueError(
                    "Time and LST reference information must be provided for "
                    "interpolation along time-axis."
                )
            if len(ref_times) != len(ref_lsts):
                raise ValueError("ref_times and ref_lsts must have the same length.")
            # Don't accidentally mess up the input LST array if there's a phase wrap.
            ref_lsts = ref_lsts.copy()
        if axis in ("freq", "both") and ref_freqs is None:
            raise ValueError(
                "Frequency reference information must be provided for "
                "interpolation along frequency-axis."
            )

    target_is_simulator = isinstance(target, Simulator)
    target = _to_uvdata(target)
    target_times, inds = np.unique(target.time_array, return_index=True)
    target_lsts = target.lst_array[inds].copy()
    target_freqs = np.unique(target.freq_array)

    def iswrapped(lsts):
        return np.any(lsts < lsts[0])

    # Ensure reference parameters are a subset of target parameters.
    if axis in ("time", "both"):
        # Raise an error if the phasing isn't trivial
        if not target._check_for_cat_type("unprojected").all():
            raise ValueError(
                "Time interpolation only supported for unprojected telescopes."
            )

        # Unwrap the LST axis if we have a phase wrap.
        if iswrapped(target_lsts):
            target_lsts[target_lsts < target_lsts[0]] += 2 * np.pi
        if iswrapped(ref_lsts):
            ref_lsts[ref_lsts < ref_lsts[0]] += 2 * np.pi

        if np.any((ref_lsts < target_lsts[0]) | (ref_lsts > target_lsts[-1])):
            warn("Reference LSTs not a subset of target LSTs; clipping.")
            is_in_range = (target_lsts[0] <= ref_lsts) & (ref_lsts <= target_lsts[-1])
            ref_times = ref_times[is_in_range]
            ref_lsts = ref_lsts[is_in_range]
    if axis in ("freq", "both"):
        if np.any((ref_freqs < target_freqs[0]) | (ref_freqs > target_freqs[-1])):
            warn("Reference frequencies not a subset of target frequencies; clipping.")
            is_in_range = (target_freqs[0] <= ref_freqs) & (
                ref_freqs <= target_freqs[-1]
            )
            ref_freqs = ref_freqs[is_in_range]

    # Setup data/metadata objects that need to be non-trivially rewritten.
    if axis in ("time", "both"):
        new_Nblts = ref_times.size * target.Nbls
        new_time_array = np.empty(new_Nblts, dtype=float)
        new_lst_array = np.empty(new_Nblts, dtype=float)
        new_ant_1_array = np.empty(new_Nblts, dtype=int)
        new_ant_2_array = np.empty(new_Nblts, dtype=int)
        new_baseline_array = np.empty(new_Nblts, dtype=int)
        new_uvw_array = np.empty((new_Nblts, 3), dtype=float)
        new_integration_times = np.empty(new_Nblts, dtype=float)
        new_phase_center_id_array = np.zeros(new_Nblts, dtype=int)
        new_phase_center_app_ra = np.empty(new_Nblts, dtype=float)
        new_phase_center_app_dec = (
            np.ones(new_Nblts, dtype=float) * target.phase_center_app_dec[0]
        )
        new_phase_center_frame_pa = np.zeros(new_Nblts, dtype=float)
        if axis == "both":
            new_data_shape = (new_Nblts, ref_freqs.size, target.Npols)
        else:
            new_data_shape = (new_Nblts, target_freqs.size, target.Npols)
        new_data = np.zeros(new_data_shape, dtype=complex)
    else:
        new_data_shape = (target.Nblts, ref_freqs.size, target.Npols)

    # Actually update metadata and interpolate the data.
    new_data = np.empty(new_data_shape, dtype=complex)
    history_update = "" if target.history.endswith("\n") else "\n"
    for i, antpair in enumerate(target.get_antpairs()):
        if axis == "freq":
            for pol_ind, pol in enumerate(target.polarization_array):
                vis = target.get_data(antpair + (pol,))
                this_blt_slice = target._key2inds(antpair + (pol,))[0]
                re_spline = interp1d(target_freqs, vis.real, axis=1, kind=kind)
                im_spline = interp1d(target_freqs, vis.imag, axis=1, kind=kind)
                new_data[this_blt_slice, :, pol_ind] = re_spline(
                    ref_freqs
                ) + 1j * im_spline(ref_freqs)
            continue

        # Preparation for updating metadata.
        ant1, ant2 = antpair
        this_slice = slice(i, None, target.Nbls)
        old_blts = target._key2inds(antpair)[0]  # As a reference
        this_uvw = target.uvw_array[old_blts][0]
        this_baseline = target.baseline_array[old_blts][0]
        this_integration_time = target.integration_time[old_blts][0]

        # Now actually update the metadata.
        new_ant_1_array[this_slice] = ant1
        new_ant_2_array[this_slice] = ant2
        new_baseline_array[this_slice] = this_baseline
        new_uvw_array[this_slice] = this_uvw
        new_time_array[this_slice] = ref_times
        new_lst_array[this_slice] = ref_lsts
        new_integration_times[this_slice] = this_integration_time
        phase_center_interp = interp1d(
            target_lsts, target.phase_center_app_ra[old_blts], kind="linear"
        )
        new_phase_center_app_dec[this_slice] = phase_center_interp(ref_lsts)

        # Update the data.
        for pol_ind, pol in enumerate(target.polarization_array):
            vis = target.get_data(antpair + (pol,))
            if axis == "both":
                re_spline = RectBivariateSpline(
                    target_lsts, target_freqs, vis.real, kx=kt, ky=kf
                )
                im_spline = RectBivariateSpline(
                    target_lsts, target_freqs, vis.imag, kx=kt, ky=kf
                )
                new_data[this_slice, :, pol_ind] = re_spline(
                    ref_lsts, ref_freqs
                ) + 1j * im_spline(ref_lsts, ref_freqs)
            else:
                re_spline = interp1d(target_lsts, vis.real, axis=0, kind=kind)
                im_spline = interp1d(target_lsts, vis.imag, axis=0, kind=kind)
                new_data[this_slice, :, pol_ind] = re_spline(ref_lsts) + 1j * im_spline(
                    ref_lsts
                )

    # Finally, update all of the metadata.
    if axis in ("freq", "both"):
        target.Nfreqs = ref_freqs.size
        target.freq_array = ref_freqs
        history_update += "Data interpolated in frequency with hera_sim.\n"
    if axis in ("time", "both"):
        target.Nblts = ref_times.size * target.Nbls
        target.Ntimes = ref_times.size
        target.time_array = new_time_array
        target.lst_array = new_lst_array % (2 * np.pi)
        target.ant_1_array = new_ant_1_array
        target.ant_2_array = new_ant_2_array
        target.baseline_array = new_baseline_array
        target.uvw_array = new_uvw_array
        target.integration_time = new_integration_times
        target.phase_center_app_dec = new_phase_center_app_dec
        target.phase_center_app_ra = new_phase_center_app_ra
        target.phase_center_id_array = new_phase_center_id_array
        target.phase_center_frame_pa = new_phase_center_frame_pa
        target.blt_order = None
        history_update += "Data interpolated in time with hera_sim.\n"

    # Now update the data-like attributes; assumes input is unflagged, unavged
    target.flag_array = np.zeros(new_data.shape, dtype=bool)
    target.nsample_array = np.ones(new_data.shape, dtype=float)
    target.data_array = new_data
    if target_is_simulator:
        target = Simulator(data=target)

    return target


def rephase_to_reference(target, reference=None, ref_times=None, ref_lsts=None):
    """
    Rephase target data to match overlapping reference LSTs.

    This function requires that ``hera_cal`` be installed.

    Parameters
    ----------
    target: :class:`pyuvdata.UVData` or :class:`~.simulate.Simulator` instance
        Object containing the visibility data and metadata which is to be
        rephased to reference LSTs.
    reference: :class:`pyuvdata.UVData`
        Object containing reference metadata. Must be provided if ``ref_times``
        and ``ref_lsts`` are not provided.
    ref_times: array-like of float
        Reference times in JD. Must be provided if ``reference`` is not provided.
    ref_lsts: array-like of float
        Reference LSTs in radians. Must be provided if ``reference`` is not
        provided.

    Returns
    -------
    rephased_data: :class:`pyuvdata.UVData` or :class:`~.simulate.Simulator` instance
        Object containing the rephased data with metadata updated appropriately.
        The times in this object are relabeled to be the reference times with
        LSTs sufficiently close to target LSTs for rephasing.
    """
    if not HERA_CAL:  # pragma: no cover
        raise NotImplementedError(
            "You must have ``hera-calibration`` installed to use this function."
        )

    # Convert target to a HERAData object.
    target_is_simulator = isinstance(target, Simulator)
    target = _to_uvdata(target)
    target = to_HERAData(target)

    # Validate the reference information.
    if reference is not None:
        if not isinstance(reference, UVData):
            try:
                reference = _to_uvdata(reference)
            except TypeError:
                raise TypeError("reference must be convertible to a UVData object.")

        ref_time_to_lst_map = dict(zip(reference.time_array, reference.lst_array))
        ref_times = np.array(list(ref_time_to_lst_map.keys()))
        ref_lsts = np.array(list(ref_time_to_lst_map.values()))
    else:
        if ref_times is None or ref_lsts is None:
            raise ValueError(
                "Both ref_times and ref_lsts must be provided if reference is not."
            )

        if len(ref_times) != len(ref_lsts):
            raise ValueError("ref_times and ref_lsts must have the same length.")

        ref_time_to_lst_map = dict(zip(ref_times, ref_lsts))

    # Construct the reference -> target time map.
    target_time_to_lst_map = dict(zip(target.time_array, target.lst_array))
    target_times = np.array(list(target_time_to_lst_map.keys()))
    target_lsts = np.array(list(target_time_to_lst_map.values()))
    ref_to_target_time_map = get_d2m_time_map(
        ref_times, ref_lsts, target_times, target_lsts
    )

    # Use only reference LSTs within the bounds of the target LSTs.
    if any(target_time is None for target_time in ref_to_target_time_map.values()):
        warn("Some reference LSTs not near target LSTs.")

    # Choose times/lsts this way to accommodate multiplicities.
    ref_times = np.array(
        [
            ref_time
            for ref_time, target_time in ref_to_target_time_map.items()
            if target_time is not None
        ]
    )
    ref_lsts = np.array([ref_time_to_lst_map[ref_time] for ref_time in ref_times])
    target_times = np.array(
        [
            target_time
            for target_time in ref_to_target_time_map.values()
            if target_time is not None
        ]
    )
    target_lsts = np.array(
        [target_time_to_lst_map[target_time] for target_time in target_times]
    )

    # Get rephasing amount for each integration.
    dlst = ref_lsts - target_lsts
    dlst = np.where(np.isclose(dlst, 0, atol=0.1), dlst, dlst - 2 * np.pi)

    # Notify the user if there's a discontinuity in rephasing amount.
    dlst_diff = np.diff(dlst)
    avg_diff = np.median(dlst_diff)
    if np.any(np.logical_not(np.isclose(dlst_diff, avg_diff, rtol=0.1))):
        warn(
            "Rephasing amount is discontinuous; there may be discontinuities "
            "in the rephased visibilities."
        )

    # Prepare the target data for rephasing.
    target.select(times=np.unique(target_times))
    data = target.build_datacontainers()[0]
    data.select_or_expand_times(target_times)
    antpos = data.antpos
    bls = {(ai, aj, pol): antpos[aj] - antpos[ai] for ai, aj, pol in data.bls()}
    lat = target.telescope_location_lat_lon_alt_degrees[0]
    new_Nblts = target.Nbls * target_times.size
    new_data = np.zeros((new_Nblts, target.Nfreqs, target.Npols), dtype=complex)
    new_time_array = np.empty(new_Nblts, dtype=float)
    new_lst_array = np.empty(new_Nblts, dtype=float)
    new_integration_times = np.empty(new_Nblts, dtype=float)
    new_ant_1_array = np.empty(new_Nblts, dtype=int)
    new_ant_2_array = np.empty(new_Nblts, dtype=int)
    new_baseline_array = np.empty(new_Nblts, dtype=int)
    new_uvw_array = np.empty((new_Nblts, 3), dtype=float)

    # Rephase and update the data/metadata; some repeated code here.
    for i, antpair in enumerate(target.get_antpairs()):
        ant1, ant2 = antpair
        this_slice = slice(i, None, target.Nbls)
        old_blts = target._key2inds(antpair)[0]  # As a reference
        this_uvw = target.uvw_array[old_blts][0]
        this_baseline = target.baseline_array[old_blts][0]
        this_integration_time = target.integration_time[old_blts][0]

        # Update the metadata.
        new_ant_1_array[this_slice] = ant1
        new_ant_2_array[this_slice] = ant2
        new_baseline_array[this_slice] = this_baseline
        new_uvw_array[this_slice] = this_uvw
        new_time_array[this_slice] = ref_times
        new_integration_times[this_slice] = this_integration_time

        # Update the data
        for pol_ind, pol in enumerate(target.polarization_array):
            pol = polnum2str(pol)
            antpairpol = antpair + (pol,)
            vis = data[antpairpol]
            bl = bls[antpairpol]
            new_data[this_slice, :, pol_ind] = lst_rephase(
                vis, bl, data.freqs, dlst, lat=lat, inplace=False, array=True
            )

    # Convert from HERAData object to UVData object
    _uvd = UVData()
    for attr in _uvd:
        setattr(_uvd, attr, getattr(target, attr))
    target = _uvd
    del _uvd

    # Times/LSTs need special treatment.
    for ref_time, target_time in ref_to_target_time_map.items():
        if target_time is None:
            continue
        this_slice = np.argwhere(new_time_array == ref_time).flatten()
        new_lst_array[this_slice] = ref_time_to_lst_map[ref_time]

    # Now update all of the data/metadata.
    target.Ntimes = np.unique(new_time_array).size
    target.Nblts = new_Nblts
    target.time_array = new_time_array
    target.lst_array = new_lst_array
    target.integration_time = new_integration_times
    target.ant_1_array = new_ant_1_array
    target.ant_2_array = new_ant_2_array
    target.baseline_array = new_baseline_array
    target.uvw_array = new_uvw_array
    target.data_array = new_data
    target.flag_array = np.zeros(new_data.shape, dtype=bool)
    target.nsample_array = np.ones(new_data.shape, dtype=float)
    target.blt_order = None

    # Convert to Simulator if needed, then return the result
    if target_is_simulator:
        target = Simulator(data=target)

    return target


def _validate_file_list(file_list):
    """Ensure all entries in the file list are path-like objects."""
    if not all(isinstance(item, (str, pathlib.Path)) for item in file_list):
        raise TypeError("Not all objects in the list are path-like.")
    if not all(os.path.exists(item) for item in file_list):
        raise ValueError("At least one path in the list does not exist.")


def _to_uvdata(sim):
    """Convert input object to a :class:`pyuvdata.UVData` object."""
    if isinstance(sim, UVData):
        return sim.copy()
    elif isinstance(sim, Simulator):
        return sim.data.copy()
    elif isinstance(sim, (str, pathlib.Path)):
        if not os.path.exists(sim):
            raise ValueError("Path to data file does not exist.")
        uvd = UVData()
        uvd.read(sim)
        return uvd
    else:
        try:
            _ = iter(sim)
            _validate_file_list(sim)
            uvd = UVData()
            uvd.read(sim)
            return uvd
        except (TypeError, ValueError) as err:
            if type(err) is TypeError:
                raise TypeError("Input object could not be converted to UVData object.")
            else:
                raise ValueError("At least one of the files does not exist.")


def _get_antpos(uvd, ENU=False):
    """Retrieve {ant: pos} dictionary from a UVData object."""
    if ENU:
        pos, ant = uvd.get_ENU_antpos()
    else:
        ant = uvd.antenna_numbers
        pos = uvd.antenna_positions

    return dict(zip(ant, pos))


def _get_array_intersection(antpos_1, antpos_2, tol=1.0):
    """
    Find the optimal intersection of two antenna arrays.

    For clarity, this function matches antennas in ``antpos_1`` to antennas in
    ``antpos_2`` and returns the modified ``antpos_1`` array. Note that the returned
    array will, in general, not have its antenna numbers match its original numbering
    scheme, since the array may be reflected through the origin in the matching
    process (since reflection through the origin does *not* have any effect on the
    set of baselines corresponding to the array).
    """
    # Make sure that antenna positions are numpy arrays
    antpos_1 = {ant: np.array(pos) for ant, pos in antpos_1.items()}
    antpos_2 = {ant: np.array(pos) for ant, pos in antpos_2.items()}
    optimal_translation = _get_optimal_translation(antpos_1, antpos_2, tol)
    new_antpos_1 = {ant: pos + optimal_translation for ant, pos in antpos_1.items()}
    ant_1_to_2_map = _get_antenna_map(new_antpos_1, antpos_2, tol)

    # Need to also check with one array reflected.
    reflected_antpos_1 = {ant: -pos for ant, pos in antpos_1.items()}
    optimal_translation_r = _get_optimal_translation(reflected_antpos_1, antpos_2, tol)
    reflected_shifted_antpos_1 = {
        ant: pos + optimal_translation_r for ant, pos in reflected_antpos_1.items()
    }
    alt_ant_map = _get_antenna_map(reflected_shifted_antpos_1, antpos_2, tol)

    # Choose the option with a greater number of antennas in the intersection.
    if len(ant_1_to_2_map) >= len(alt_ant_map):
        intersection = {ant_1: new_antpos_1[ant_1] for ant_1 in ant_1_to_2_map.keys()}
        is_reflected = False
    else:
        # Conjugation convention reverses (b_ij -> b_ji)
        intersection = {
            ant_1: reflected_shifted_antpos_1[ant_1] for ant_1 in alt_ant_map.keys()
        }
        is_reflected = True

    return intersection, is_reflected


def _get_antenna_map(antpos_1, antpos_2, tol=1.0):
    """Find a mapping between antenna numbers."""
    antenna_map = {}
    ant_2_array = list(antpos_2.keys())
    for ant_1, pos_1 in antpos_1.items():
        ant_2_index = np.argwhere(
            [np.allclose(pos_1, pos_2, atol=tol) for pos_2 in antpos_2.values()]
        ).flatten()
        if ant_2_index.size == 0:
            continue  # No match
        antenna_map[ant_1] = ant_2_array[ant_2_index[0]]

    return antenna_map


def _get_optimal_translation(antpos_1, antpos_2, tol=1.0):
    """Find the translation that maximizes overlap between antenna arrays."""
    translations = _build_translations(antpos_1, antpos_2, tol)

    # Calculate the number of overlapping antennas for each translation.
    intersection_sizes = {}
    for ant_map, translation in translations.items():
        shifted_antpos_1 = {ant: pos + translation for ant, pos in antpos_1.items()}
        Nintersections = len(_get_antenna_map(shifted_antpos_1, antpos_2, tol=tol))
        intersection_sizes[ant_map] = Nintersections

    # Choose the translation that produces the largest intersection.
    ant_map_keys = list(translations.keys())
    intersections_per_mapping = list(intersection_sizes.values())
    index = np.argmax(intersections_per_mapping)
    optimal_mapping = ant_map_keys[index]

    return translations[optimal_mapping]


def _build_translations(antpos_1, antpos_2, tol=1.0):
    """Build all possible translations that map at least one antenna to another."""
    # Make sure that antenna positions are numpy arrays.
    antpos_1 = {ant: np.array(pos) for ant, pos in antpos_1.items()}
    antpos_2 = {ant: np.array(pos) for ant, pos in antpos_2.items()}

    # Brute-force calculation of all translations.
    translations = {
        f"{ant_1}->{ant_2}": antpos_2[ant_2] - antpos_1[ant_1]
        for ant_1 in antpos_1.keys()
        for ant_2 in antpos_2.keys()
    }

    # Reduction to unique translations.
    unique_translations = {}
    for ant_map, translation in translations.items():
        if not any(
            np.allclose(translation, unique_translation, atol=tol)
            for unique_translation in unique_translations.values()
        ):
            unique_translations[ant_map] = translation

    return unique_translations
