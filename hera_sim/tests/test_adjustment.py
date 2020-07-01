"""Test the various simulation adjustment tools."""

import copy
import itertools
import os
import pytest

from astropy import units
from scipy import stats
import numpy as np

from hera_sim import adjustment
from hera_sim import antpos
from hera_sim import interpolators
from hera_sim import Simulator
from pyuvdata import UVData

# For use later when simulating foregrounds.
Tsky_mdl = interpolators.Tsky(datafile="HERA_Tsky_Reformatted.npz")
omega_p = interpolators.Beam(datafile="HERA_H1C_BEAM_POLY.npy", interpolator="poly1d")


@pytest.fixture(scope="session")
def base_config():
    Nfreqs = 100
    start_freq = 100e6
    bandwidth = 50e6
    Ntimes = 100
    start_time = 2458799.0
    integration_time = 10.7
    # Use square array with HERA antenna separation.
    base_sep = 14.6
    array_layout = {
        0: [0, 0, 0],
        1: [1, 0, 0],
        2: [0, 1, 0],
        3: [1, 1, 0],
    }
    array_layout = {ant: base_sep * np.array(pos) for ant, pos in array_layout.items()}
    config = dict(
        Nfreqs=Nfreqs,
        start_freq=start_freq,
        bandwidth=bandwidth,
        Ntimes=Ntimes,
        start_time=start_time,
        integration_time=integration_time,
        array_layout=array_layout,
    )
    return config


def antpos_equal(antpos_1, antpos_2, tol=0.1):
    return all(
        any(np.allclose(pos_1, pos_2, atol=tol) for pos_1 in antpos_1.values())
        for pos_2 in antpos_2.values()
    )


def test_basic_antenna_matching():
    # Simple test: reflected right-triangles should match exactly
    array_1 = {0: [0, 0, 0], 1: [1, 0, 0], 2: [1, 1, 0]}

    array_2 = {0: [0, 0, 0], 1: [0, 1, 0], 2: [1, 1, 0]}

    array_intersection = adjustment._get_array_intersection(array_1, array_2, tol=0)[0]
    assert antpos_equal(array_intersection, array_2, 0)

    # In general, rotated right-triangles should have at least two antennas in their
    # intersection.
    array_3 = {0: [0, 0, 0], 1: [1, 0, 0], 2: [0, 1, 0]}
    array_intersection = adjustment._get_array_intersection(array_1, array_3, tol=0)[0]
    assert len(array_intersection) == 2

    # A simple translation should just be undone
    translation = np.random.uniform(-1, 1, 3)
    array_4 = {ant: np.array(pos) - translation for ant, pos in array_1.items()}
    array_intersection = adjustment._get_array_intersection(array_1, array_4, tol=0.1)[
        0
    ]
    assert antpos_equal(array_intersection, array_4, 0.1)

    # A small hex array should be a subset of a larger hex array
    hex_array = antpos.HexArray(split_core=False, outriggers=0)
    hex_array_1 = hex_array(3)
    hex_array_2 = hex_array(2)
    array_intersection = adjustment._get_array_intersection(hex_array_1, hex_array_2)[0]
    assert antpos_equal(array_intersection, hex_array_2)
    assert len(array_intersection) == len(hex_array_2)


def test_match_antennas(base_config):
    # Prepare a config with a different array for some basic tests.
    new_config = copy.deepcopy(base_config)
    new_array = {0: [0, 0, 0], 1: [1, 0, 0], 2: [1, 1, 0], 3: [2, 1, 0], 4: [2, 2, 0]}
    new_config["array_layout"] = {
        ant: np.array(pos) * 14.6 for ant, pos in new_array.items()
    }
    # Write down the overlap between this array and the base config array.
    overlap_array = {
        ant: pos for ant, pos in new_config["array_layout"].items() if ant in (0, 1, 2)
    }
    target_to_ref_ant_map = {0: 0, 1: 1, 3: 2}
    sim = Simulator(**base_config)
    ref_sim = Simulator(**new_config)

    # First test: simple matching of a subset; check metadata makes sense.
    # Add the keyword to make sure the lines get run; but the telescope metadata
    # is the same for both data sets, so there's no point comparing the end result.
    modified_sim = adjustment.match_antennas(
        target=sim, reference=ref_sim, overwrite_telescope_metadata=True,
    )
    assert all(
        np.allclose(ref_antpos, modified_antpos)
        for ref_antpos, modified_antpos in zip(
            overlap_array.values(), modified_sim.antpos.values()
        )
    )
    Nants_overlap = len(overlap_array)
    Nbls_overlap = Nants_overlap * (Nants_overlap + 1) / 2
    assert modified_sim.data.Nbls == Nbls_overlap
    assert np.all(modified_sim.data.antenna_numbers == list(overlap_array.keys()))
    assert modified_sim.data.Nblts == sim.data.Ntimes * Nbls_overlap
    antpairs = itertools.combinations_with_replacement(overlap_array.keys(), 2)
    assert all(
        sim.data.antnums_to_baseline(*antpair) in modified_sim.data.baseline_array
        for antpair in antpairs
    )

    # Second test: repeat the first, but do not relabel antennas.
    modified_sim = adjustment.match_antennas(
        target=sim, reference=ref_sim, relabel_antennas=False,
    )
    assert antpos_equal(overlap_array, modified_sim.antpos)
    assert modified_sim.data.Nbls == Nbls_overlap
    assert np.all(
        modified_sim.data.antenna_numbers == list(target_to_ref_ant_map.keys())
    )
    assert modified_sim.data.Nblts == sim.data.Ntimes * Nbls_overlap
    antpairs = itertools.combinations_with_replacement(target_to_ref_ant_map.keys(), 2)
    assert all(
        sim.data.antnums_to_baseline(*antpair) in modified_sim.data.baseline_array
        for antpair in antpairs
    )

    # Third test: make sure antenna matching works in ENU coordinates.
    modified_sim = adjustment.match_antennas(target=sim, reference=ref_sim, ENU=True)
    assert modified_sim.data.Nbls == Nbls_overlap
    assert np.all(modified_sim.data.antenna_numbers == list(overlap_array.keys()))
    assert modified_sim.data.Nblts == sim.data.Ntimes * Nbls_overlap
    antpairs = itertools.combinations_with_replacement(overlap_array.keys(), 2)
    assert all(
        sim.data.antnums_to_baseline(*antpair) in modified_sim.data.baseline_array
        for antpair in antpairs
    )

    # Increase complexity a bit: jitter antenna positions
    jitter_radius = 0.1  # 10 cm
    new_array = {
        ant: pos + stats.uniform.rvs(-jitter_radius, jitter_radius, 3)
        for ant, pos in new_config["array_layout"].items()
    }
    new_config["array_layout"] = new_array
    overlap_array = {ant: pos for ant, pos in new_array.items() if ant in (0, 1, 2)}
    # Basically repeat first test, but use reference positions for new antpos values.
    modified_sim = adjustment.match_antennas(
        target=sim, reference=ref_sim, use_reference_positions=True,
    )
    # Allow some wiggle room appropriate for the size of the jitter added.
    assert all(
        np.allclose(
            ref_antpos, modified_antpos, atol=np.sqrt(3 * (2 * jitter_radius) ** 2)
        )
        for ref_antpos, modified_antpos in zip(
            overlap_array.values(), modified_sim.antpos.values()
        )
    )
    assert modified_sim.data.Nbls == Nbls_overlap
    assert np.all(modified_sim.data.antenna_numbers == list(overlap_array.keys()))
    assert modified_sim.data.Nblts == sim.data.Ntimes * Nbls_overlap
    antpairs = itertools.combinations_with_replacement(overlap_array.keys(), 2)
    assert all(
        sim.data.antnums_to_baseline(*antpair) in modified_sim.data.baseline_array
        for antpair in antpairs
    )

    # Now use reference positions, but keep original antenna labels.
    overlap_array = {ant: pos for ant, pos in new_array.items() if ant in (0, 1, 2)}
    overlap_array[3] = overlap_array.pop(2)
    # Basically repeat first test, but use reference positions for new antpos values.
    modified_sim = adjustment.match_antennas(
        target=sim,
        reference=ref_sim,
        use_reference_positions=True,
        relabel_antennas=False,
    )
    # Allow some wiggle room appropriate for the size of the jitter added.
    assert all(
        np.allclose(
            ref_antpos, modified_antpos, atol=np.sqrt(3 * (2 * jitter_radius) ** 2)
        )
        for ref_antpos, modified_antpos in zip(
            overlap_array.values(), modified_sim.antpos.values()
        )
    )
    assert modified_sim.data.Nbls == Nbls_overlap
    assert np.all(modified_sim.data.antenna_numbers == list(overlap_array.keys()))
    assert modified_sim.data.Nblts == sim.data.Ntimes * Nbls_overlap
    antpairs = itertools.combinations_with_replacement(overlap_array.keys(), 2)
    assert all(
        sim.data.antnums_to_baseline(*antpair) in modified_sim.data.baseline_array
        for antpair in antpairs
    )


def test_antenna_matching_conjugation(base_config):
    new_config = copy.deepcopy(base_config)
    new_antenna_numbers = (1, 0, 3, 2)
    new_config["array_layout"] = {
        new_ant: base_config["array_layout"][old_ant]
        for new_ant, old_ant in enumerate(new_antenna_numbers)
    }
    sim = Simulator(**base_config)
    ref_sim = Simulator(**new_config)

    # Simulate some foregrounds so we can actually check conjugation details.
    sim.add("diffuse_foreground", Tsky_mdl=Tsky_mdl, omega_p=omega_p, seed="redundant")

    modified_sim = adjustment.match_antennas(
        target=sim, reference=ref_sim, relabel_antennas=True
    )
    vis_1 = sim.data.get_data(0, 1, "xx")
    vis_2 = modified_sim.data.get_data(1, 0, "xx")
    assert np.allclose(vis_1, vis_2)
    vis_1 = sim.data.get_data(0, 3, "xx")
    vis_2 = modified_sim.data.get_data(1, 2, "xx")
    assert np.allclose(vis_1, vis_2)

    # Now we need to break the array symmetry...
    antpos = {
        0: [0, 0, 0],
        1: [1, 0, 0],
        2: [0, 1, 0],
        3: [1, 2, 0],
    }
    base_sep = 14.6
    antpos = {ant: base_sep * np.array(pos) for ant, pos in antpos.items()}
    ref_antpos = {
        0: [0, 0, 0],
        1: [0, -1, 0],
        2: [-1, -2, 0],
    }
    ref_antpos = {ant: base_sep * np.array(pos) for ant, pos in ref_antpos.items()}
    new_config["array_layout"] = antpos
    sim = Simulator(**new_config)
    new_config["array_layout"] = ref_antpos
    ref_sim = Simulator(**new_config)
    sim.add("diffuse_foreground", Tsky_mdl=Tsky_mdl, omega_p=omega_p)

    modified_sim = adjustment.match_antennas(
        target=sim, reference=ref_sim, relabel_antennas=True
    )
    vis_1 = sim.data.get_data(0, 3, "xx")
    vis_2 = modified_sim.data.get_data(2, 0, "xx")
    assert np.allclose(vis_1, vis_2)
    vis_1 = sim.data.get_data(3, 2, "xx")
    vis_2 = modified_sim.data.get_data(1, 2, "xx")
    assert np.allclose(vis_1, vis_2)


def test_interpolation_in_frequency(base_config):
    new_config = copy.deepcopy(base_config)
    new_config["Nfreqs"] = 200  # Increase frequency resolution.
    new_config["start_freq"] = 105e6  # Ensure reference sim freqs contained in target.
    new_config["bandwidth"] = 40e6
    sim_1 = Simulator(**base_config)
    sim_2 = Simulator(**new_config)

    # Simulate foregrounds for both.
    sim_1.add(
        "diffuse_foreground", Tsky_mdl=Tsky_mdl, omega_p=omega_p,
    )
    sim_2.add(
        "diffuse_foreground", Tsky_mdl=Tsky_mdl, omega_p=omega_p,
    )

    # Interpolate sim_1 to sim_2 along the frequency axis.
    interpolated_sim = adjustment.interpolate_to_reference(
        target=sim_1, reference=sim_2, axis="freq",
    )

    # Check that frequency metadata is updated appropriately.
    assert np.allclose(interpolated_sim.freqs, sim_2.freqs)

    # Do the same as above, but this time pass a frequency array.
    ref_freqs = sim_2.freqs * 1e9
    interpolated_sim_2 = adjustment.interpolate_to_reference(
        target=sim_1, ref_freqs=ref_freqs, axis="freq",
    )

    assert np.allclose(interpolated_sim_2.freqs, sim_2.freqs)
    assert np.allclose(
        interpolated_sim.data.data_array, interpolated_sim_2.data.data_array
    )


def test_interpolation_in_time(base_config):
    new_config = copy.deepcopy(base_config)
    new_config["Ntimes"] = 150
    offset = base_config["integration_time"] * units.s.to("day")
    new_config["start_time"] = base_config["start_time"] + offset
    new_config["integration_time"] = 5.2
    sim_1 = Simulator(**base_config)
    sim_2 = Simulator(**new_config)

    # Simulate foregrounds for both.
    sim_1.add(
        "diffuse_foreground", Tsky_mdl=Tsky_mdl, omega_p=omega_p,
    )
    sim_2.add(
        "diffuse_foreground", Tsky_mdl=Tsky_mdl, omega_p=omega_p,
    )

    # Interpolate sim_1 to sim_2 along the time axis.
    interpolated_sim = adjustment.interpolate_to_reference(
        target=sim_1, reference=sim_2, axis="time",
    )

    # Check that the metadata checks out.
    assert interpolated_sim.data.Nblts == sim_2.data.Nblts
    assert interpolated_sim.data.Ntimes == sim_2.data.Ntimes
    assert np.allclose(interpolated_sim.lsts, sim_2.lsts)
    assert np.allclose(interpolated_sim.times, sim_2.times)

    # Now do a check with passing reference times/lsts
    interpolated_sim_2 = adjustment.interpolate_to_reference(
        target=sim_1, ref_times=sim_2.times, ref_lsts=sim_2.lsts, axis="time",
    )

    # Same checks as before, but now also check interpolated data is identical.
    assert interpolated_sim_2.data.Nblts == sim_2.data.Nblts
    assert interpolated_sim_2.data.Ntimes == sim_2.data.Ntimes
    assert np.allclose(interpolated_sim_2.lsts, sim_2.lsts)
    assert np.allclose(interpolated_sim_2.times, sim_2.times)
    assert np.allclose(
        interpolated_sim.data.data_array, interpolated_sim_2.data.data_array
    )


def test_interpolation_both_axes(base_config):
    # Make same modifications done in previous two tests.
    new_config = copy.deepcopy(base_config)
    new_config["Ntimes"] = 150
    offset = base_config["integration_time"] * units.s.to("day")
    new_config["start_time"] = base_config["start_time"] + offset
    new_config["integration_time"] = 5.2
    new_config["Nfreqs"] = 200  # Increase frequency resolution.
    new_config["start_freq"] = 105e6  # Ensure reference sim freqs contained in target.
    new_config["bandwidth"] = 40e6
    sim_1 = Simulator(**base_config)
    sim_2 = Simulator(**new_config)

    # Simulate foregrounds for both.
    sim_1.add(
        "diffuse_foreground", Tsky_mdl=Tsky_mdl, omega_p=omega_p,
    )
    sim_2.add(
        "diffuse_foreground", Tsky_mdl=Tsky_mdl, omega_p=omega_p,
    )

    # Interpolate sim_1 to sim_2 along the time axis.
    interpolated_sim = adjustment.interpolate_to_reference(
        target=sim_1, reference=sim_2, axis="both",
    )

    # Check that the metadata checks out.
    assert interpolated_sim.data.Nblts == sim_2.data.Nblts
    assert interpolated_sim.data.Ntimes == sim_2.data.Ntimes
    assert interpolated_sim.data.Nfreqs == sim_2.data.Nfreqs
    assert np.allclose(interpolated_sim.lsts, sim_2.lsts)
    assert np.allclose(interpolated_sim.times, sim_2.times)
    assert np.allclose(interpolated_sim.freqs, sim_2.freqs)

    # Now do a check with passing reference arrays.
    interpolated_sim_2 = adjustment.interpolate_to_reference(
        target=sim_1,
        ref_times=sim_2.times,
        ref_lsts=sim_2.lsts,
        ref_freqs=sim_2.freqs * 1e9,
        axis="both",
    )

    # Same checks as before, but now also check interpolated data is identical.
    assert interpolated_sim_2.data.Nblts == sim_2.data.Nblts
    assert interpolated_sim_2.data.Ntimes == sim_2.data.Ntimes
    assert interpolated_sim_2.data.Nfreqs == sim_2.data.Nfreqs
    assert np.allclose(interpolated_sim_2.lsts, sim_2.lsts)
    assert np.allclose(interpolated_sim_2.times, sim_2.times)
    assert np.allclose(interpolated_sim_2.freqs, sim_2.freqs)
    assert np.allclose(
        interpolated_sim.data.data_array, interpolated_sim_2.data.data_array
    )


def test_interpolate_exception_raising(base_config):
    sim = Simulator(**base_config)

    # Bad axis parameter.
    with pytest.raises(ValueError) as err:
        adjustment.interpolate_to_reference(sim, sim, axis="nan")
    assert "axis parameter must be" in err.value.args[0]

    # Bad reference object.
    with pytest.raises(TypeError) as err:
        adjustment.interpolate_to_reference(sim, 42)
    assert "reference must be convertible" in err.value.args[0]

    # Insufficient information.
    with pytest.raises(ValueError) as err:
        adjustment.interpolate_to_reference(sim, ref_times=sim.times)
    assert "Time and LST reference information" in err.value.args[0]

    with pytest.raises(ValueError) as err:
        adjustment.interpolate_to_reference(sim, axis="freq")
    assert "Frequency reference information" in err.value.args[0]

    # Bad reference data.
    with pytest.raises(ValueError) as err:
        adjustment.interpolate_to_reference(sim, ref_times=[1, 2, 3], ref_lsts=[0, 1])
    assert "ref_times and ref_lsts must have the same length." == err.value.args[0]

    # LST wrap.
    with pytest.raises(NotImplementedError) as err:
        ref_lsts = np.linspace(7 * np.pi / 4, 5 * np.pi / 2, sim.times.size) % (
            2 * np.pi
        )
        adjustment.interpolate_to_reference(sim, ref_times=sim.times, ref_lsts=ref_lsts)
    assert "currently not supported" in err.value.args[0]

    # Only subset of frequencies overlap.
    new_config = copy.deepcopy(base_config)
    new_config["start_freq"] = 120e6
    ref_sim = Simulator(**new_config)
    with pytest.warns(UserWarning) as record:
        interpolated_sim = adjustment.interpolate_to_reference(
            target=sim, reference=ref_sim, axis="both",
        )
    assert "Reference frequencies not a subset" in record[0].message.args[0]
    overlapping_freq_channels = np.argwhere(np.isclose(sim.freqs, ref_sim.freqs))
    assert np.allclose(interpolated_sim.freqs, sim.freqs[overlapping_freq_channels])

    # Only a subset of LSTs overlap.
    new_config = copy.deepcopy(base_config)
    time_offset = 50 * base_config["integration_time"] * units.s.to("day")
    new_config["start_time"] = base_config["start_time"] + time_offset
    ref_sim = Simulator(**new_config)
    with pytest.warns(UserWarning) as record:
        interpolated_sim = adjustment.interpolate_to_reference(
            target=sim, reference=ref_sim, axis="both",
        )
    assert "Reference LSTs not a subset" in record[0].message.args[0]
    overlapping_integrations = np.argwhere(np.isclose(sim.lsts, ref_sim.lsts))
    assert np.allclose(interpolated_sim.lsts, sim.lsts[overlapping_integrations])


def test_rephase_to_reference(base_config):
    # Keep the test very simple; this function isn't intended to be used much.
    new_config = copy.deepcopy(base_config)
    time_offset = 0.1 * base_config["integration_time"] * units.s.to("day")
    new_config["start_time"] = base_config["start_time"] + time_offset
    sim = Simulator(**base_config)
    ref_sim = Simulator(**new_config)

    # Check that rephasing by a small amount works fine.
    rephased_sim = adjustment.rephase_to_reference(target=sim, reference=ref_sim)
    assert np.allclose(rephased_sim.lsts, ref_sim.lsts)
    assert np.allclose(rephased_sim.times, ref_sim.times)

    # Check that it works fine passing arrays instead.
    rephased_sim_v2 = adjustment.rephase_to_reference(
        target=sim, ref_times=ref_sim.times, ref_lsts=ref_sim.lsts
    )
    assert np.allclose(rephased_sim_v2.lsts, ref_sim.lsts)
    assert np.allclose(rephased_sim_v2.times, ref_sim.times)
    assert np.allclose(rephased_sim.data.data_array, rephased_sim_v2.data.data_array)


def test_rephasing_exception_raising(base_config):
    sim = Simulator(**base_config)

    # The basics: bad input.
    with pytest.raises(TypeError) as err:
        adjustment.rephase_to_reference(target=sim, reference=42)
    assert err.value.args[0] == "reference must be convertible to a UVData object."

    with pytest.raises(ValueError) as err:
        adjustment.rephase_to_reference(target=sim, ref_times=[1, 2, 3])
    assert "Both ref_times and ref_lsts" in err.value.args[0]

    with pytest.raises(ValueError) as err:
        adjustment.rephase_to_reference(
            target=sim, ref_times=[1, 2, 3], ref_lsts=[2, 2]
        )
    assert err.value.args[0] == "ref_times and ref_lsts must have the same length."

    dt = np.mean(np.diff(sim.times))
    dlst = np.mean(np.diff(sim.lsts))
    with pytest.warns(UserWarning) as record:
        _ = adjustment.rephase_to_reference(
            target=sim, ref_times=sim.times + 5 * dt, ref_lsts=sim.lsts + 5 * dlst
        )
    assert record[0].message.args[0] == "Some reference LSTs not near target LSTs."

    # A bit more sophisticated: uneven integration times.
    new_config = copy.deepcopy(base_config)
    new_config["integration_time"] = 10.3  # Mismatch with integration time of 10.7 s
    ref_sim = Simulator(**new_config)
    with pytest.warns(UserWarning) as record:
        _ = adjustment.rephase_to_reference(target=sim, reference=ref_sim)
    assert "Rephasing amount is discontinuous" in record[0].message.args[0]


def test_adjust_to_reference(base_config, tmp_path):
    # Update parameters for making reference metadata.
    new_config = copy.deepcopy(base_config)
    new_config["Ntimes"] = 150
    new_config["integration_time"] = 5.2
    time_offset = 4 * base_config["integration_time"] * units.s.to("day")
    new_config["start_time"] = base_config["start_time"] + time_offset
    new_config["array_layout"].update({4: np.array([0.5, 2, 0]) * 14.6})

    # Mock up some data and reference metadata; write to disk for additional tests.
    sim = Simulator(**base_config)
    ref_sim = Simulator(**new_config)
    sim.add("diffuse_foreground", Tsky_mdl=Tsky_mdl, omega_p=omega_p)
    sim.chunk_sim_and_save(save_dir=str(tmp_path), Nint_per_file=20, prefix="base")
    ref_sim.chunk_sim_and_save(save_dir=str(tmp_path), Nint_per_file=30, prefix="ref")
    sim_files = [tmp_path / f for f in os.listdir(tmp_path) if "base" in f]
    ref_files = [tmp_path / f for f in os.listdir(tmp_path) if "ref" in f]

    non_simulator_inputs = (
        (sim_input, ref_input)
        for sim_input in (sim.data, sim_files, sim_files)
        for ref_input in (ref_sim, ref_sim.data, ref_files)
    )
    simulator_inputs = [(sim, ref_input) for ref_input in (ref_sim.data, ref_files)]
    # Check all possible pairs of input types.
    modified_sim = adjustment.adjust_to_reference(target=sim, reference=ref_sim)
    for sim_input, ref_input in non_simulator_inputs:
        modified_uvd = adjustment.adjust_to_reference(
            target=sim_input, reference=ref_input
        )
        assert np.allclose(modified_sim.data.data_array, modified_uvd.data_array)
        assert np.allclose(modified_sim.data.freq_array, modified_uvd.freq_array)
        assert np.allclose(modified_sim.data.time_array, modified_uvd.time_array)
        assert np.allclose(modified_sim.data.lst_array, modified_uvd.lst_array)
        assert np.allclose(
            modified_sim.data.baseline_array, modified_uvd.baseline_array
        )
        assert np.allclose(
            modified_sim.data.antenna_positions, modified_uvd.antenna_positions
        )
        assert np.allclose(
            modified_sim.data.antenna_numbers, modified_uvd.antenna_numbers
        )

    # Same as above, but with the target input as a Simulator object.
    for sim_input, ref_input in simulator_inputs:
        modified_uvd = adjustment.adjust_to_reference(
            target=sim_input, reference=ref_input
        )
        assert np.allclose(modified_sim.data.data_array, modified_uvd.data.data_array)
        assert np.allclose(modified_sim.data.freq_array, modified_uvd.data.freq_array)
        assert np.allclose(modified_sim.data.time_array, modified_uvd.data.time_array)
        assert np.allclose(modified_sim.data.lst_array, modified_uvd.data.lst_array)
        assert np.allclose(
            modified_sim.data.baseline_array, modified_uvd.data.baseline_array
        )
        assert np.allclose(
            modified_sim.data.antenna_positions, modified_uvd.data.antenna_positions
        )
        assert np.allclose(
            modified_sim.data.antenna_numbers, modified_uvd.data.antenna_numbers
        )

    del modified_uvd

    # Check parameters are OK.
    assert np.allclose(modified_sim.freqs, ref_sim.freqs)
    assert np.allclose(modified_sim.times, ref_sim.times)
    assert np.allclose(modified_sim.lsts, ref_sim.lsts)
    assert antpos_equal(sim.antpos, modified_sim.antpos)

    # Do one check with rephasing.
    modified_sim = adjustment.adjust_to_reference(
        target=sim, reference=ref_sim, interpolate=False
    )
    rephased_sim = adjustment.rephase_to_reference(target=sim, reference=ref_sim)
    assert np.allclose(rephased_sim.times, modified_sim.times)
    assert np.allclose(rephased_sim.lsts, modified_sim.lsts)

    # One final check for conjugation.
    modified_sim = adjustment.adjust_to_reference(
        target=sim, reference=ref_sim, conjugation_convention="ant1<ant2"
    )
    assert np.all(modified_sim.data.ant_1_array <= modified_sim.data.ant_2_array)


def test_adjust_to_reference_verbosity(base_config, capsys):
    # Update parameters for making reference metadata.
    new_config = copy.deepcopy(base_config)
    new_config["Ntimes"] = 150
    new_config["integration_time"] = 5.2
    time_offset = 4 * base_config["integration_time"] * units.s.to("day")
    new_config["start_time"] = base_config["start_time"] + time_offset
    new_config["array_layout"].update({4: np.array([0.5, 2, 0]) * 14.6})
    ref_sim = Simulator(**new_config)

    # Generate a new base simulation with a bigger array; compress it by redundancy.
    new_array = {
        0: [0, 0, 0],
        1: [1, 0, 0],
        2: [2, 0, 0],
        3: [0, 1, 0],
        4: [1, 1, 0],
        5: [2, 1, 0],
    }
    new_config = copy.deepcopy(base_config)
    new_config["array_layout"] = {
        ant: 14.6 * np.array(pos) for ant, pos in new_array.items()
    }
    sim = Simulator(**new_config)
    sim.add("diffuse_foreground", Tsky_mdl=Tsky_mdl, omega_p=omega_p, seed="redundant")
    sim.data.compress_by_redundancy()

    # Do a verbosity check.
    _ = adjustment.adjust_to_reference(
        target=sim, reference=ref_sim, conjugation_convention="ant1<ant2", verbose=True
    )
    captured = capsys.readouterr()
    assert "Validating positional arguments..." in captured.out
    assert "Interpolating target data to reference data LSTs..." in captured.out
    assert "Inflating target data by baseline redundancy..." in captured.out
    assert "Conjugating target to ant1<ant2 convention..." in captured.out

    # Do a verbosity check, this time with rephasing instead of interpolating.
    _ = adjustment.adjust_to_reference(
        target=sim, reference=ref_sim, interpolate=False, verbose=True
    )
    captured = capsys.readouterr()
    assert "Validating positional arguments..." in captured.out
    assert "Rephasing target data to reference data LSTs..." in captured.out
    assert "Inflating target data by baseline redundancy..." in captured.out


def test_position_tolerance_error_raising():
    # This is the first thing that's checked, so the other input doesn't matter.
    with pytest.raises(ValueError) as err:
        adjustment.adjust_to_reference(0, 0, position_tolerance=[1, 1, 1, 1])
    assert (
        err.value.args[0] == "position_tolerance should be a scalar or length-3 array."
    )

    with pytest.raises(TypeError) as err:
        adjustment.adjust_to_reference(0, 0, position_tolerance=1j)
    assert "must be a real-valued scalar or" in err.value.args[0]


def test_uvdata_converter(base_config, tmp_path):
    # Write a simulation to disk to check some functionalities.
    sim = Simulator(**base_config)
    sim.chunk_sim_and_save(str(tmp_path), prefix="test", Nint_per_file=50)
    sim_files = [tmp_path / f for f in os.listdir(tmp_path) if "test" in f]

    # First, check that things work.
    for item in (sim, sim.data, sim_files, sim_files[0]):
        uvd = adjustment._to_uvdata(item)
        assert isinstance(uvd, UVData)

    # Now check exceptions.
    bad_files = ["not_a_file.uvh5", "another_bad_file.uvh5"]
    with pytest.raises(ValueError) as err:
        adjustment._to_uvdata(bad_files)
    assert err.value.args[0] == "At least one of the files does not exist."

    bad_input = [
        1,
    ]
    with pytest.raises(TypeError) as err:
        adjustment._to_uvdata(bad_input)
    assert err.value.args[0] == "Input object could not be converted to UVData object."

    bad_file = bad_files[0]
    with pytest.raises(ValueError) as err:
        adjustment._to_uvdata(bad_file)
    assert err.value.args[0] == "Path to data file does not exist."
