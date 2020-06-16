"""Module providing tools for adjusting simulation data/metadata to a reference."""

import copy
import os

import numpy as np
from astropy import units
from scipy.interpolate import interp1d
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
    target: str, path-like object, or :class:`pyuvdata.UVData` instance
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
    pass
