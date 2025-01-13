"""Test the various simulation adjustment tools."""

import copy
import itertools
import logging
import os

import numpy as np
import pytest
from astropy import units
from pyuvdata import UVData
from pyuvdata.utils import antnums_to_baseline

from hera_sim import Simulator, adjustment, antpos, interpolators

# For use later when simulating foregrounds.
Tsky_mdl = interpolators.Tsky(datafile="HERA_Tsky_Reformatted.npz")
omega_p = interpolators.Beam(datafile="HERA_H1C_BEAM_POLY.npy", interpolator="poly1d")


# Useful helper functions for building arrays and finding antenna pairs.
def scale_array(antpos, sep=14.6):
    return {ant: np.array(pos) * sep for ant, pos in antpos.items()}


def get_antpairs(antpos):
    return list(itertools.combinations_with_replacement(antpos, 2))


def get_all_baselines(antpairs):
    """Get the set of all possible baseline integers, with conjugations.

    This is a bit of a hack, but it gives all unique antpairs with conjugations,
    represented as baseline integers. I'm assuming we'll never use these tools
    on arrays with more than 2048 antennas, so there's no point actually counting
    the number of antennas (which is the third argument to antnums_to_baseline).
    """
    ant_1_array = [antpair[0] for antpair in antpairs]
    ant_2_array = [antpair[1] for antpair in antpairs]
    return set(
        antnums_to_baseline(
            ant_1_array + ant_2_array, ant_2_array + ant_1_array, Nants_telescope=0
        )
    )


def baseline_integers_check_out(uvd, ants):
    """Check that the baseline integers in ``uvd`` are as expected.

    This is a bit of a roundabout check to make sure that all of the baseline
    integers in a UVData object whose antennas have been modified are what they
    are supposed to be. We don't necessarily know what the conjugation convention
    used is, so we check that the baseline integers are in the set of all possible
    baseline integers, given the set of antenna numbers we expect to be present,
    and that there are the correct number of baselines in the UVData object. This
    set of checks is sufficient for this end.
    """
    antpairs = get_antpairs(ants)  # Unique pairs of antennas, including autos.
    all_baseline_integers = get_all_baselines(antpairs)
    baseline_integers_found = set(uvd.baseline_array).issubset(all_baseline_integers)
    Nbls_match = len(antpairs) == uvd.Nbls
    return baseline_integers_found and Nbls_match


@pytest.fixture(scope="function")
def base_config():
    Nfreqs = 100
    start_freq = 100e6
    bandwidth = 50e6
    Ntimes = 100
    start_time = 2458799.0
    integration_time = 10.7
    # Use square array with HERA antenna separation.
    array_layout = {0: [0, 0, 0], 1: [1, 0, 0], 2: [0, 1, 0], 3: [1, 1, 0]}
    array_layout = scale_array(array_layout)
    config = dict(
        Nfreqs=Nfreqs,
        start_freq=start_freq,
        bandwidth=bandwidth,
        Ntimes=Ntimes,
        start_time=start_time,
        integration_time=integration_time,
        array_layout=array_layout,
    )
    return copy.deepcopy(config)


@pytest.fixture(scope="function")
def base_sim(base_config):
    return Simulator(**base_config)


def is_subarray(antpos_1, antpos_2, tol=0.1):
    # Checks if antpos_2 is a subarray of antpos_1
    return all(
        any(np.allclose(pos_1, pos_2, atol=tol) for pos_1 in antpos_1.values())
        for pos_2 in antpos_2.values()
    )


def arrays_are_equal(antpos_1, antpos_2, tol=0.1):
    # A = B iff A \subset B and B \subset A
    return is_subarray(antpos_1, antpos_2, tol) and is_subarray(antpos_2, antpos_1, tol)


def test_match_reflected_arrays():
    # Simple test: reflected right-triangles should match exactly
    # Right triangle, hypotenuse from (0,0) to (1,1), leg along x-axis.
    array_1 = {0: [0, 0, 0], 1: [1, 0, 0], 2: [1, 1, 0]}
    # Reflect triangle about hypotenuse; all baselines remain.
    array_2 = {0: [0, 0, 0], 1: [0, 1, 0], 2: [1, 1, 0]}

    array_intersection, is_reflected = adjustment._get_array_intersection(
        array_1, array_2, tol=0
    )
    assert arrays_are_equal(array_intersection, array_2, 0) and is_reflected


def test_match_subarray():
    # Same right triangle from above test for first array.
    array_1 = {0: [0, 0, 0], 1: [1, 0, 0], 2: [1, 1, 0]}
    # Pure EW/NS baselines agree, but diagnoal is rotated by 90 degrees.
    array_2 = {0: [0, 0, 0], 1: [1, 0, 0], 2: [0, 1, 0]}
    array_intersection = adjustment._get_array_intersection(array_1, array_2, tol=0)[0]
    # Only way to choose antennas with identical baselines is to choose one leg.
    assert len(array_intersection) == 2


def test_match_translated_array():
    # A simple translation should just be undone
    rng = np.random.default_rng(0)
    translation = rng.uniform(-1, 1, 3)
    array_1 = {0: [0, 0, 0], 1: [1, 0, 0], 2: [1, 1, 0]}
    array_2 = {ant: np.array(pos) - translation for ant, pos in array_1.items()}
    # Won't be an exact match to machine precision, so need some small tolerance.
    tol = 0.01
    # Get the translated version of array 1 that best matches the shifted array.
    array_intersection = adjustment._get_array_intersection(array_1, array_2, tol=tol)[
        0
    ]
    assert arrays_are_equal(array_intersection, array_2, tol=tol)


def test_exact_subarray_match():
    # A small hex array should be a subset of a larger hex array
    hex_array = antpos.HexArray(split_core=False, outriggers=0)
    hex_array_1 = hex_array(3)
    hex_array_2 = hex_array(2)
    array_intersection = adjustment._get_array_intersection(hex_array_1, hex_array_2)[0]
    assert arrays_are_equal(array_intersection, hex_array_2)


# This next set of antenna matching tests performs lots of metadata checks, which
# aren't really sensible to perform as separate tests--they're testing different
# modes of running the antenna matching algorithm on simulation-like objects. So
# these tests are broken up by the operating mode used for match_antennas.


def test_match_antennas_default_settings(base_config, base_sim):
    #
    # Make an array that's two parallel diagonal lines bounded by a length-2 square
    # with one vertex at the origin, confined to the upper-right quadrant of the plane.
    # The arrays look like so (new on left; original on right):
    #         4
    #      2  3         2  3
    #   0  1            0  1
    #
    new_array = {0: [0, 0, 0], 1: [1, 0, 0], 2: [1, 1, 0], 3: [2, 1, 0], 4: [2, 2, 0]}
    base_config["array_layout"] = scale_array(new_array)
    ref_sim = Simulator(**base_config)

    # The optimal intersection can be read off of the schematic above.
    overlap_array = {ant: base_config["array_layout"][ant] for ant in (0, 1, 2)}

    # Explicitly use the parameters that are chosen as the defaults, *except* for
    # the telescope metadata parameter; set that parameter to be True, just so the
    # lines of code are run (not much point checking, since the core functionality is
    # verified by checking other metadata; this just extends the number of attributes
    # that are adjusted in the target UVData object).
    modified_sim = adjustment.match_antennas(
        target=base_sim,
        reference=ref_sim,
        tol=1.0,
        ENU=False,
        relabel_antennas=True,
        use_reference_positions=False,
        overwrite_telescope_metadata=True,
    )

    # Check that the matched array is identical to the overlap array, up to labels.
    assert arrays_are_equal(overlap_array, modified_sim.antpos)

    # Check that the antenna labels are updated correctly.
    assert np.all(modified_sim.data.antenna_numbers == list(overlap_array.keys()))

    # Check that the baseline integers are correct.
    assert baseline_integers_check_out(uvd=modified_sim.data, ants=overlap_array)


# Second test: repeat the first, but do not relabel antennas.
def test_match_antennas_all_kwargs_false(base_sim, base_config):
    # Construct a reference object in the same way as the previous test.
    new_array = {0: [0, 0, 0], 1: [1, 0, 0], 2: [1, 1, 0], 3: [2, 1, 0], 4: [2, 2, 0]}
    base_config["array_layout"] = scale_array(new_array)
    ref_sim = Simulator(**base_config)
    overlap_array = {ant: base_config["array_layout"][ant] for ant in (0, 1, 2)}
    target_to_ref_ant_map = {0: 0, 1: 1, 3: 2}

    # Explicitly call match_antennas with all options False.
    modified_sim = adjustment.match_antennas(
        target=base_sim,
        reference=ref_sim,
        tol=1.0,
        ENU=False,
        relabel_antennas=False,
        use_reference_positions=False,
        overwrite_telescope_metadata=False,
    )

    # Check that the arrays are identical, with their original labels.
    assert arrays_are_equal(overlap_array, modified_sim.antpos)
    assert np.all(
        modified_sim.data.antenna_numbers == list(target_to_ref_ant_map.keys())
    )

    # Check that the baseline integers are correct.
    assert baseline_integers_check_out(
        uvd=modified_sim.data, ants=target_to_ref_ant_map
    )


@pytest.mark.xfail(reason="Bugfix needed as well as philosophical considerations.")
def test_match_antennas_using_ENU_positions(base_sim, base_config):
    # Construct reference object a la previous two tests.
    new_array = {0: [0, 0, 0], 1: [1, 0, 0], 2: [1, 1, 0], 3: [2, 1, 0], 4: [2, 2, 0]}
    base_config["array_layout"] = scale_array(new_array)
    ref_sim = Simulator(**base_config)
    modified_sim = adjustment.match_antennas(
        target=base_sim,
        reference=ref_sim,
        tol=1.0,
        ENU=True,
        relabel_antennas=True,
        use_reference_positions=False,
        overwrite_telescope_metadata=False,
    )

    # The Simulator.antpos attribute returns the antpos array in ENU coordinates.
    overlap_array_ENU = {ant: ref_sim.antpos[ant] for ant in (0, 1, 2)}

    # Check that the remaining antenna array is what's expected.
    assert np.all(modified_sim.data.antenna_numbers == list(overlap_array_ENU.keys()))
    assert arrays_are_equal(modified_sim.antpos, overlap_array_ENU)

    # Check that the baseline integers are correct.
    assert baseline_integers_check_out(uvd=modified_sim.data, ants=overlap_array_ENU)


# Increase complexity a bit: jitter antenna positions
def test_match_antennas_using_reference_positions_and_labels(base_sim, base_config):
    jitter_radius = 0.1  # 10 cm
    new_array = {0: [0, 0, 0], 1: [1, 0, 0], 2: [1, 1, 0], 3: [2, 1, 0], 4: [2, 2, 0]}
    base_config["array_layout"] = scale_array(new_array)
    rng = np.random.default_rng(0)
    new_array = {
        ant: pos + rng.uniform(-jitter_radius, jitter_radius, 3)
        for ant, pos in base_config["array_layout"].items()
    }  # Mess up the antenna positions by up to 10 cm in each direction, randomly.
    base_config["array_layout"] = new_array
    ref_sim = Simulator(**base_config)

    overlap_array = {ant: new_array[ant] for ant in (0, 1, 2)}
    # Use default parameters, but this time make the modified simulation use the
    # positions exactly as they are in the reference simulation.
    modified_sim = adjustment.match_antennas(
        target=base_sim,
        reference=ref_sim,
        tol=1.0,
        ENU=False,
        relabel_antennas=True,
        use_reference_positions=True,
        overwrite_telescope_metadata=False,
    )

    # Since we're using the reference positions, these should match exactly.
    assert arrays_are_equal(modified_sim.antpos, overlap_array, tol=1e-9)
    assert np.all(modified_sim.data.antenna_numbers == list(overlap_array.keys()))

    # Check the baseline integers.
    assert baseline_integers_check_out(uvd=modified_sim.data, ants=overlap_array)


# Slight variation of previous test.
def test_match_antennas_use_reference_positions_only(base_sim, base_config):
    jitter_radius = 0.1  # 10 cm
    new_array = {0: [0, 0, 0], 1: [1, 0, 0], 2: [1, 1, 0], 3: [2, 1, 0], 4: [2, 2, 0]}
    base_config["array_layout"] = scale_array(new_array)
    rng = np.random.default_rng(0)
    new_array = {
        ant: pos + rng.uniform(-jitter_radius, jitter_radius, 3)
        for ant, pos in base_config["array_layout"].items()
    }  # Mess up the antenna positions by up to 10 cm in each direction, randomly.
    base_config["array_layout"] = new_array
    ref_sim = Simulator(**base_config)

    # Overlap array has same positions as before, but different numbers.
    overlap_array = {ant: pos for ant, pos in new_array.items() if ant in (0, 1, 2)}
    overlap_array[3] = overlap_array.pop(2)  # Cheap trick to get the numbers right.

    modified_sim = adjustment.match_antennas(
        target=base_sim,
        reference=ref_sim,
        tol=1.0,
        ENU=False,
        relabel_antennas=False,
        use_reference_positions=True,
        overwrite_telescope_metadata=False,
    )

    # We're using the reference positions, so they should match exactly!
    assert arrays_are_equal(modified_sim.antpos, overlap_array, tol=1e-9)
    assert np.all(modified_sim.data.antenna_numbers == list(overlap_array.keys()))

    # Check that the baseline integers check out.
    assert baseline_integers_check_out(modified_sim.data, overlap_array)


def test_antenna_matching_scrambling_numbers(base_config, base_sim):
    # Jumble up the antenna numbers to something crazy, but make the array
    # matching trivial; make sure that the ordering of the antennas is right,
    # using both reference positions and labels.
    pass


def test_antenna_matching_conjugation(base_config, base_sim):
    # For reference, comparison of antenna arrays:
    #
    #  2 3    3 2
    #  0 1    1 0
    #
    new_config = copy.deepcopy(base_config)
    new_antenna_numbers = (1, 0, 3, 2)
    new_config["array_layout"] = {
        new_ant: base_config["array_layout"][old_ant]
        for new_ant, old_ant in enumerate(new_antenna_numbers)
    }  # Scramble the antenna numbers
    ref_sim = Simulator(**new_config)

    # Simulate some foregrounds so we can actually check conjugation details.
    base_sim.add(
        "diffuse_foreground", Tsky_mdl=Tsky_mdl, omega_p=omega_p, seed="redundant"
    )

    # Explicit is better than implicit.
    modified_sim = adjustment.match_antennas(
        target=base_sim,
        reference=ref_sim,
        tol=1.0,
        ENU=False,
        relabel_antennas=True,
        use_reference_positions=False,
        overwrite_telescope_metadata=False,
    )

    # (0, 1) maps to (1, 0)
    vis_1 = base_sim.data.get_data(0, 1, "xx")
    vis_2 = modified_sim.data.get_data(1, 0, "xx")
    assert np.allclose(vis_1, vis_2)

    # (0, 3) maps to (1, 2)
    vis_1 = base_sim.data.get_data(0, 3, "xx")
    vis_2 = modified_sim.data.get_data(1, 2, "xx")
    assert np.allclose(vis_1, vis_2)


# Same as above, but with broken array symmetry.
def test_antenna_matching_conjugation_asymmetric_arrays(base_config):
    # For reference, here are the antenna positions (right is reference):
    #
    #    3
    #  2
    #  0 1          0
    #               1
    #             2
    #
    antpos = {0: [0, 0, 0], 1: [1, 0, 0], 2: [0, 1, 0], 3: [1, 2, 0]}
    antpos = scale_array(antpos)
    ref_antpos = {0: [0, 0, 0], 1: [0, -1, 0], 2: [-1, -2, 0]}
    ref_antpos = scale_array(ref_antpos)
    base_config["array_layout"] = antpos
    sim = Simulator(**base_config)
    base_config["array_layout"] = ref_antpos
    ref_sim = Simulator(**base_config)

    # Add some foregrounds for comparisons; we're not looking at autocorrelations
    # and there are only unique cross-correlations here, so no need for seeding.
    sim.add("diffuse_foreground", Tsky_mdl=Tsky_mdl, omega_p=omega_p)

    # Same deal as previous test.
    modified_sim = adjustment.match_antennas(
        target=sim,
        reference=ref_sim,
        tol=1.0,
        ENU=False,
        relabel_antennas=True,
        use_reference_positions=False,
        overwrite_telescope_metadata=False,
    )

    # (0, 3) maps to (2, 0)
    vis_1 = sim.data.get_data(0, 3, "xx")
    vis_2 = modified_sim.data.get_data(2, 0, "xx")
    assert np.allclose(vis_1, vis_2)

    # (3, 2) maps to (1, 2)
    vis_1 = sim.data.get_data(3, 2, "xx")
    vis_2 = modified_sim.data.get_data(1, 2, "xx")
    assert np.allclose(vis_1, vis_2)


def test_interpolation_in_frequency_with_simulators(base_config, base_sim):
    base_config["Nfreqs"] = 200  # Increase frequency resolution.
    base_config["start_freq"] = 105e6  # Ensure reference sim freqs contained in target.
    base_config["bandwidth"] = 40e6
    ref_sim = Simulator(**base_config)

    # Simulate foregrounds.
    base_sim.add("diffuse_foreground", Tsky_mdl=Tsky_mdl, omega_p=omega_p)

    # Interpolate base_sim to ref_sim along the frequency axis.
    interpolated_sim = adjustment.interpolate_to_reference(
        target=base_sim, reference=ref_sim, axis="freq"
    )

    # Check that frequency metadata is updated appropriately.
    assert np.allclose(interpolated_sim.freqs, ref_sim.freqs)
    assert interpolated_sim.data.Nfreqs == base_config["Nfreqs"]


def test_interpolation_phased(base_config, base_sim):
    base_config["Nfreqs"] = 200  # Increase frequency resolution.
    base_config["start_freq"] = 105e6  # Ensure reference sim freqs contained in target.
    base_config["bandwidth"] = 40e6
    ref_sim = Simulator(**base_config)

    # Simulate foregrounds.
    base_sim.add("diffuse_foreground", Tsky_mdl=Tsky_mdl, omega_p=omega_p)
    base_sim.data.phase_center_catalog[0]["cat_type"] = "sidereal"

    # Interpolate base_sim to ref_sim along the frequency axis.
    with pytest.raises(
        ValueError, match="Time interpolation only supported for unprojected telescopes"
    ):
        adjustment.interpolate_to_reference(
            target=base_sim, reference=ref_sim, axis="time"
        )


def test_interpolation_in_frequency_with_array(base_config, base_sim):
    # Do the same as above, but this time pass a frequency array.
    ref_freqs = np.linspace(105e6, 145e6, 200)  # Hz
    base_sim.add("diffuse_foreground", Tsky_mdl=Tsky_mdl, omega_p=omega_p)
    interpolated_sim = adjustment.interpolate_to_reference(
        target=base_sim, ref_freqs=ref_freqs, axis="freq"
    )

    # Check frequency metadata; Simulator.freqs returns frequencies in GHz.
    assert np.allclose(interpolated_sim.freqs, ref_freqs / 1e9)
    assert interpolated_sim.data.Nfreqs == ref_freqs.size


def test_interpolation_in_frequency_is_consistent(base_config, base_sim):
    # Interpolate using both methods, confirm that results are identical.
    base_config["Nfreqs"] = 200  # Increase frequency resolution.
    base_config["start_freq"] = 105e6  # Ensure reference sim freqs contained in target.
    base_config["bandwidth"] = 40e6
    ref_sim = Simulator(**base_config)
    ref_freqs = ref_sim.freqs * 1e9  # Hz

    # Simulate foregrounds.
    base_sim.add("diffuse_foreground", Tsky_mdl=Tsky_mdl, omega_p=omega_p)

    # Interpolate base_sim to ref_sim along the frequency axis.
    interpolated_sim_A = adjustment.interpolate_to_reference(
        target=base_sim, reference=ref_sim, axis="freq"
    )

    # Do the same, but using a frequency array.
    interpolated_sim_B = adjustment.interpolate_to_reference(
        target=base_sim, ref_freqs=ref_freqs, axis="freq"
    )

    # The interpolated data arrays should be nearly identical.
    assert np.allclose(
        interpolated_sim_A.data.data_array, interpolated_sim_B.data.data_array
    )


def test_interpolation_in_time_with_simulators(base_config, base_sim):
    # Make a new set of observation times that are completely contained in the
    # base set of times.
    base_config["Ntimes"] = 150
    offset = base_config["integration_time"] * units.s.to("day")
    base_config["start_time"] = base_config["start_time"] + offset
    base_config["integration_time"] = 5.2
    ref_sim = Simulator(**base_config)

    # Simulate foregrounds.
    base_sim.add("diffuse_foreground", Tsky_mdl=Tsky_mdl, omega_p=omega_p)

    # Interpolate base_sim to ref_sim along the time axis.
    interpolated_sim = adjustment.interpolate_to_reference(
        target=base_sim, reference=ref_sim, axis="time"
    )

    # Check that the metadata checks out.
    assert interpolated_sim.data.Nblts == ref_sim.data.Nblts
    assert interpolated_sim.data.Ntimes == ref_sim.data.Ntimes
    assert np.allclose(interpolated_sim.lsts, ref_sim.lsts)
    assert np.allclose(interpolated_sim.times, ref_sim.times)


def test_interpolation_in_time_with_array(base_config, base_sim):
    # Same test as above, but now with passing reference times/lsts.
    base_config["Ntimes"] = 150
    offset = base_config["integration_time"] * units.s.to("day")
    base_config["start_time"] = base_config["start_time"] + offset
    base_config["integration_time"] = 5.2
    ref_sim = Simulator(**base_config)

    # Simulate foregrounds.
    base_sim.add("diffuse_foreground", Tsky_mdl=Tsky_mdl, omega_p=omega_p)

    interpolated_sim = adjustment.interpolate_to_reference(
        target=base_sim, ref_times=ref_sim.times, ref_lsts=ref_sim.lsts, axis="time"
    )

    # Same metadata checks as before.
    assert interpolated_sim.data.Nblts == ref_sim.data.Nblts
    assert interpolated_sim.data.Ntimes == ref_sim.data.Ntimes
    assert np.allclose(interpolated_sim.lsts, ref_sim.lsts)
    assert np.allclose(interpolated_sim.times, ref_sim.times)


def test_interpolation_in_time_is_consistent(base_config, base_sim):
    # Check that interpolating each way provides identical results.
    base_config["Ntimes"] = 150
    offset = base_config["integration_time"] * units.s.to("day")
    base_config["start_time"] = base_config["start_time"] + offset
    base_config["integration_time"] = 5.2
    ref_sim = Simulator(**base_config)

    # Simulate foregrounds.
    base_sim.add("diffuse_foreground", Tsky_mdl=Tsky_mdl, omega_p=omega_p)

    interpolated_sim_A = adjustment.interpolate_to_reference(
        target=base_sim, reference=ref_sim, axis="time"
    )
    interpolated_sim_B = adjustment.interpolate_to_reference(
        target=base_sim, ref_times=ref_sim.times, ref_lsts=ref_sim.lsts, axis="time"
    )

    assert np.allclose(
        interpolated_sim_A.data.data_array, interpolated_sim_B.data.data_array
    )


def test_interpolation_with_phase_wrap(base_config):
    # Make a simulation with times straddling the phase wrap at 2pi.
    base_config["start_time"] = 2458041.37335
    base_config["Ntimes"] = 100
    sim = Simulator(**base_config)

    # Extract a subset of the LSTs to interpolate to.
    ref_lsts = sim.lsts[10:-10]
    ref_times = sim.times[10:-10]

    # Just set all the data to unity for simplicity.
    sim.data.data_array[...] = 1

    # Actually do the interpolation.
    interp_sim = adjustment.interpolate_to_reference(
        target=sim, ref_times=ref_times, ref_lsts=ref_lsts, axis="time"
    )

    print(ref_lsts)
    print(interp_sim.lsts)
    assert np.allclose(interp_sim.data.data_array, 1)
    assert np.allclose(interp_sim.lsts, ref_lsts)
    assert np.allclose(interp_sim.times, ref_times)


def test_interpolation_both_axes_with_simulators(base_config, base_sim):
    # Make same modifications done in previous two tests.
    base_config["Ntimes"] = 150
    offset = base_config["integration_time"] * units.s.to("day")
    base_config["start_time"] = base_config["start_time"] + offset
    base_config["integration_time"] = 5.2
    base_config["Nfreqs"] = 200  # Increase frequency resolution.
    base_config["start_freq"] = 105e6  # Ensure reference sim freqs contained in target.
    base_config["bandwidth"] = 40e6
    ref_sim = Simulator(**base_config)

    # Simulate foregrounds.
    base_sim.add("diffuse_foreground", Tsky_mdl=Tsky_mdl, omega_p=omega_p)

    # Interpolate base_sim to ref_sim along both axes.
    interpolated_sim = adjustment.interpolate_to_reference(
        target=base_sim, reference=ref_sim, axis="both"
    )

    # Check that the metadata checks out.
    assert interpolated_sim.data.Nblts == ref_sim.data.Nblts
    assert interpolated_sim.data.Ntimes == ref_sim.data.Ntimes
    assert interpolated_sim.data.Nfreqs == ref_sim.data.Nfreqs
    assert np.allclose(interpolated_sim.lsts, ref_sim.lsts)
    assert np.allclose(interpolated_sim.times, ref_sim.times)
    assert np.allclose(interpolated_sim.freqs, ref_sim.freqs)


def test_interpolation_both_axes_with_arrays(base_config, base_sim):
    # Now do a check with passing reference arrays.
    base_config["Ntimes"] = 150
    offset = base_config["integration_time"] * units.s.to("day")
    base_config["start_time"] = base_config["start_time"] + offset
    base_config["integration_time"] = 5.2
    base_config["Nfreqs"] = 200  # Increase frequency resolution.
    base_config["start_freq"] = 105e6  # Ensure reference sim freqs contained in target.
    base_config["bandwidth"] = 40e6
    ref_sim = Simulator(**base_config)

    # Simulate foregrounds.
    base_sim.add("diffuse_foreground", Tsky_mdl=Tsky_mdl, omega_p=omega_p)

    # Do the interpolation.
    interpolated_sim = adjustment.interpolate_to_reference(
        target=base_sim,
        ref_times=ref_sim.times,
        ref_lsts=ref_sim.lsts,
        ref_freqs=ref_sim.freqs * 1e9,
        axis="both",
    )

    # Same checks as before.
    assert interpolated_sim.data.Nblts == ref_sim.data.Nblts
    assert interpolated_sim.data.Ntimes == ref_sim.data.Ntimes
    assert interpolated_sim.data.Nfreqs == ref_sim.data.Nfreqs
    assert np.allclose(interpolated_sim.lsts, ref_sim.lsts)
    assert np.allclose(interpolated_sim.times, ref_sim.times)
    assert np.allclose(interpolated_sim.freqs, ref_sim.freqs)


def test_interpolation_both_axes_is_consistent(base_config, base_sim):
    # Do the interpolation both ways and check that the interpolated data agree.
    base_config["Ntimes"] = 150
    offset = base_config["integration_time"] * units.s.to("day")
    base_config["start_time"] = base_config["start_time"] + offset
    base_config["integration_time"] = 5.2
    base_config["Nfreqs"] = 200  # Increase frequency resolution.
    base_config["start_freq"] = 105e6  # Ensure reference sim freqs contained in target.
    base_config["bandwidth"] = 40e6
    ref_sim = Simulator(**base_config)

    # Simulate foregrounds.
    base_sim.add("diffuse_foreground", Tsky_mdl=Tsky_mdl, omega_p=omega_p)

    # Do the interpolation both ways.
    interpolated_sim_A = adjustment.interpolate_to_reference(
        target=base_sim, reference=ref_sim, axis="both"
    )
    interpolated_sim_B = adjustment.interpolate_to_reference(
        target=base_sim,
        ref_times=ref_sim.times,
        ref_lsts=ref_sim.lsts,
        ref_freqs=ref_sim.freqs * 1e9,
        axis="both",
    )

    # The interpolated data should be identical.
    assert np.allclose(
        interpolated_sim_A.data.data_array, interpolated_sim_B.data.data_array
    )


def test_interpolate_exception_bad_axis(base_sim):
    with pytest.raises(ValueError) as err:
        adjustment.interpolate_to_reference(base_sim, base_sim, axis="nan")
    assert "axis parameter must be" in err.value.args[0]


def test_interpolate_exception_bad_reference_type(base_sim):
    with pytest.raises(TypeError) as err:
        adjustment.interpolate_to_reference(base_sim, 42)
    assert "reference must be convertible" in err.value.args[0]


def test_interpolate_exception_insufficient_time_data(base_sim):
    with pytest.raises(ValueError) as err:
        adjustment.interpolate_to_reference(base_sim, ref_times=base_sim.times)
    assert "Time and LST reference information" in err.value.args[0]


def test_interpolate_exception_insufficient_freq_data(base_sim):
    with pytest.raises(ValueError) as err:
        adjustment.interpolate_to_reference(base_sim, axis="freq")
    assert "Frequency reference information" in err.value.args[0]


def test_interpolate_exception_reference_time_and_lst_mismatch(base_sim):
    with pytest.raises(ValueError) as err:
        adjustment.interpolate_to_reference(
            base_sim, ref_times=[1, 2, 3], ref_lsts=[0, 1]
        )
    assert "ref_times and ref_lsts must have the same length." == err.value.args[0]


def test_interpolate_warning_partial_frequency_match(base_config, base_sim):
    base_config["start_freq"] = 120e6
    ref_sim = Simulator(**base_config)
    with pytest.warns(UserWarning) as record:
        adjustment.interpolate_to_reference(
            target=base_sim, reference=ref_sim, axis="both"
        )
    assert "Reference frequencies not a subset" in record[0].message.args[0]


def test_interpolate_partial_frequency_match(base_config, base_sim):
    base_config["start_freq"] = 120e6
    ref_sim = Simulator(**base_config)
    interpolated_sim = adjustment.interpolate_to_reference(
        target=base_sim, reference=ref_sim, axis="both"
    )
    overlapping_channels = np.unique(
        [np.argmin(np.abs(base_sim.freqs - freq)) for freq in ref_sim.freqs]
    )
    assert np.allclose(interpolated_sim.freqs, base_sim.freqs[overlapping_channels])


def test_interpolate_warning_partial_lst_match(base_config, base_sim):
    time_offset = 50 * base_config["integration_time"] * units.s.to("day")
    base_config["start_time"] = base_config["start_time"] + time_offset
    ref_sim = Simulator(**base_config)
    with pytest.warns(UserWarning) as record:
        adjustment.interpolate_to_reference(
            target=base_sim, reference=ref_sim, axis="both"
        )
    assert "Reference LSTs not a subset" in record[0].message.args[0]


def test_interpolate_partial_lst_match(base_config, base_sim):
    time_offset = 50 * base_config["integration_time"] * units.s.to("day")
    base_config["start_time"] = base_config["start_time"] + time_offset
    ref_sim = Simulator(**base_config)
    interpolated_sim = adjustment.interpolate_to_reference(
        target=base_sim, reference=ref_sim, axis="both"
    )
    overlapping_integrations = np.unique(
        [np.argmin(np.abs(base_sim.lsts - lst)) for lst in ref_sim.lsts]
    )
    assert np.allclose(interpolated_sim.lsts, base_sim.lsts[overlapping_integrations])


def test_rephase_to_reference_with_simulators(base_config, base_sim):
    # Check that things work okay for slightly offset times/LSTs.
    time_offset = 0.1 * base_config["integration_time"] * units.s.to("day")
    base_config["start_time"] = base_config["start_time"] + time_offset
    ref_sim = Simulator(**base_config)

    # Simulate foregrounds.
    base_sim.add("diffuse_foreground", Tsky_mdl=Tsky_mdl, omega_p=omega_p)

    # Rephase and check the metadata.
    rephased_sim = adjustment.rephase_to_reference(target=base_sim, reference=ref_sim)
    assert np.allclose(rephased_sim.lsts, ref_sim.lsts)
    assert np.allclose(rephased_sim.times, ref_sim.times)


def test_rephase_to_reference_with_array(base_config, base_sim):
    # Check that it works fine passing arrays instead.
    time_offset = 0.1 * base_config["integration_time"] * units.s.to("day")
    base_config["start_time"] = base_config["start_time"] + time_offset
    ref_sim = Simulator(**base_config)

    # Simulate foregrounds.
    base_sim.add("diffuse_foreground", Tsky_mdl=Tsky_mdl, omega_p=omega_p)

    # Rephase and check the metadata.
    rephased_sim = adjustment.rephase_to_reference(
        target=base_sim, ref_times=ref_sim.times, ref_lsts=ref_sim.lsts
    )
    assert np.allclose(rephased_sim.lsts, ref_sim.lsts)
    assert np.allclose(rephased_sim.times, ref_sim.times)


def test_rephase_to_reference_is_consistent(base_config, base_sim):
    time_offset = 0.1 * base_config["integration_time"] * units.s.to("day")
    base_config["start_time"] = base_config["start_time"] + time_offset
    ref_sim = Simulator(**base_config)

    # Simulate foregrounds.
    base_sim.add("diffuse_foreground", Tsky_mdl=Tsky_mdl, omega_p=omega_p)

    # Get the rephased data both ways.
    rephased_sim_A = adjustment.rephase_to_reference(target=base_sim, reference=ref_sim)
    rephased_sim_B = adjustment.rephase_to_reference(
        target=base_sim, ref_times=ref_sim.times, ref_lsts=ref_sim.lsts
    )

    # Check that the data arrays agree.
    assert np.allclose(rephased_sim_A.data.data_array, rephased_sim_B.data.data_array)


def test_rephasing_exception_bad_reference(base_sim):
    with pytest.raises(TypeError) as err:
        adjustment.rephase_to_reference(target=base_sim, reference=42)
    assert err.value.args[0] == "reference must be convertible to a UVData object."


def test_rephasing_exception_insufficient_time_information(base_sim):
    with pytest.raises(ValueError) as err:
        adjustment.rephase_to_reference(target=base_sim, ref_times=[1, 2, 3])
    assert "Both ref_times and ref_lsts" in err.value.args[0]


def test_rephasing_exception_mismatched_times_and_lsts(base_sim):
    with pytest.raises(ValueError) as err:
        adjustment.rephase_to_reference(
            target=base_sim, ref_times=[1, 2, 3], ref_lsts=[2, 2]
        )
    assert err.value.args[0] == "ref_times and ref_lsts must have the same length."


def test_rephasing_warning_out_of_bounds_lsts(base_sim):
    dt = np.mean(np.diff(base_sim.times))
    dlst = np.mean(np.diff(base_sim.lsts))
    with pytest.warns(UserWarning) as record:
        adjustment.rephase_to_reference(
            target=base_sim,
            ref_times=base_sim.times + 5 * dt,
            ref_lsts=base_sim.lsts + 5 * dlst,
        )
    assert record[0].message.args[0] == "Some reference LSTs not near target LSTs."


def test_rephasing_warning_discontinuous_dlst(base_config, base_sim):
    base_config["integration_time"] = 10.3  # Mismatch with integration time of 10.7 s
    ref_sim = Simulator(**base_config)
    with pytest.warns(UserWarning) as record:
        adjustment.rephase_to_reference(target=base_sim, reference=ref_sim)
    assert "Rephasing amount is discontinuous" in record[0].message.args[0]


# Make some objects to use for testing different types of input to adjust_to_reference
@pytest.fixture(scope="function")
def ref_sim(base_config):
    # Update the time parameters to be bounded by the base time parameters.
    time_offset = 4 * base_config["integration_time"] * units.s.to("day")
    base_config["Ntimes"] = 150
    base_config["integration_time"] = 5.2
    base_config["start_time"] = base_config["start_time"] + time_offset
    base_config["array_layout"].update({4: np.array([0.5, 2, 0]) * 14.6})
    sim = Simulator(**base_config)
    return sim


@pytest.fixture(scope="function", params=["sim", "uvd", "files"])
def reference(request, ref_sim, tmp_path):
    # Write files to the temporary directory if they don't exist.
    file_list = [tmp_path / f for f in os.listdir(tmp_path) if "ref" in f]
    if not file_list:
        ref_sim.chunk_sim_and_save(
            save_dir=str(tmp_path), Nint_per_file=30, prefix="ref"
        )
        file_list = [tmp_path / f for f in os.listdir(tmp_path) if "ref" in f]

    if request.param == "sim":
        return ref_sim
    elif request.param == "uvd":
        return ref_sim.data
    else:
        return file_list


@pytest.fixture(scope="function", params=["uvd", "files"])
def target(request, base_sim, tmp_path):
    # Simulate some visibilities.
    base_sim.add("diffuse_foreground", Tsky_mdl=Tsky_mdl, omega_p=omega_p)
    file_list = [tmp_path / f for f in os.listdir(tmp_path) if "base" in f]
    if not file_list:
        base_sim.chunk_sim_and_save(
            save_dir=str(tmp_path), Nint_per_file=20, prefix="base"
        )
        file_list = [tmp_path / f for f in os.listdir(tmp_path) if "base" in f]

    if request.param == "uvd":
        return base_sim.data
    else:
        return file_list


def test_adjust_to_reference_with_simulator_type_target(reference, base_sim):
    ref_uvd = adjustment._to_uvdata(reference)
    modified_sim = adjustment.adjust_to_reference(target=base_sim, reference=reference)
    # Just do a quick metadata check.
    assert np.allclose(
        np.unique(modified_sim.data.time_array), np.unique(ref_uvd.time_array)
    )
    assert np.allclose(
        np.unique(modified_sim.data.lst_array), np.unique(ref_uvd.lst_array)
    )
    assert arrays_are_equal(base_sim.antpos, modified_sim.antpos)
    assert baseline_integers_check_out(modified_sim.data, base_sim.antpos)


def test_adjust_to_reference_with_non_simulator_target(target, reference, base_sim):
    ref_uvd = adjustment._to_uvdata(reference)
    modified_uvd = adjustment.adjust_to_reference(target=target, reference=reference)
    modified_antpos = adjustment._get_antpos(modified_uvd, ENU=True)
    assert np.allclose(
        np.unique(modified_uvd.time_array), np.unique(ref_uvd.time_array)
    )
    assert np.allclose(np.unique(modified_uvd.lst_array), np.unique(ref_uvd.lst_array))
    assert arrays_are_equal(base_sim.antpos, modified_antpos)
    assert baseline_integers_check_out(modified_uvd, base_sim.antpos)


def test_adjust_to_reference_conjugation_works(base_sim, ref_sim):
    modified_sim = adjustment.adjust_to_reference(
        target=base_sim, reference=ref_sim, conjugation_convention="ant1<ant2"
    )
    assert np.all(modified_sim.data.ant_1_array <= modified_sim.data.ant_2_array)


@pytest.fixture(scope="function")
def redundant_sim(base_config):
    new_array = {
        0: [0, 0, 0],
        1: [1, 0, 0],
        2: [2, 0, 0],
        3: [0, 1, 0],
        4: [1, 1, 0],
        5: [2, 1, 0],
    }

    base_config["array_layout"] = scale_array(new_array)
    sim = Simulator(**base_config)
    sim.add("diffuse_foreground", Tsky_mdl=Tsky_mdl, omega_p=omega_p, seed="redundant")
    sim.data.compress_by_redundancy()
    return sim


def test_adjust_to_reference_verbosity_with_interpolating(
    ref_sim, redundant_sim, caplog
):
    with caplog.at_level(logging.INFO):
        adjustment.adjust_to_reference(
            target=redundant_sim, reference=ref_sim, conjugation_convention="ant1<ant2"
        )
    assert "Validating positional arguments..." in caplog.text
    assert "Interpolating target data to reference data LSTs..." in caplog.text
    assert "Inflating target data by baseline redundancy..." in caplog.text
    assert "Conjugating target to ant1<ant2 convention..." in caplog.text


def test_adjust_to_reference_verbosity_with_rephasing(ref_sim, redundant_sim, caplog):
    with caplog.at_level(logging.INFO):
        adjustment.adjust_to_reference(
            target=redundant_sim, reference=ref_sim, interpolate=False
        )
    assert "Validating positional arguments..." in caplog.text
    assert "Rephasing target data to reference data LSTs..." in caplog.text
    assert "Inflating target data by baseline redundancy..." in caplog.text


def test_position_tolerance_exception_bad_value():
    # This is the first thing that's checked, so the other input doesn't matter.
    with pytest.raises(ValueError) as err:
        adjustment.adjust_to_reference(0, 0, position_tolerance=[1, 1, 1, 1])
    assert "tolerance should be a scalar or length-3 array." in err.value.args[0]


def test_position_tolerance_exception_bad_type():
    with pytest.raises(TypeError) as err:
        adjustment.adjust_to_reference(0, 0, position_tolerance=1j)
    assert "must be a real-valued scalar or" in err.value.args[0]


def test_uvdata_converter(reference):
    uvd = adjustment._to_uvdata(reference)
    assert isinstance(uvd, UVData)
    if isinstance(reference, list):
        # Check that it works with a single file.
        uvd = adjustment._to_uvdata(reference[0])
        assert isinstance(uvd, UVData)


def test_to_uvdata_exception_nonexistent_files():
    bad_files = ["not_a_file.uvh5", "another_bad_file.uvh5"]
    with pytest.raises(ValueError) as err:
        adjustment._to_uvdata(bad_files)
    assert err.value.args[0] == "At least one of the files does not exist."


def test_to_uvdata_exception_bad_input_type():
    bad_input = [1]
    with pytest.raises(TypeError) as err:
        adjustment._to_uvdata(bad_input)
    assert err.value.args[0] == "Input object could not be converted to UVData object."


def test_to_uvdata_exception_nonexistent_single_file():
    bad_file = "something_not_to_be_found.uv"
    with pytest.raises(ValueError) as err:
        adjustment._to_uvdata(bad_file)
    assert err.value.args[0] == "Path to data file does not exist."
