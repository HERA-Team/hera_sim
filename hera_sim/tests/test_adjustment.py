"""Test the various simulation adjustment tools."""

import os
import pytest

from astropy import units
import numpy as np

from hera_sim import adjustment
from hera_sim import antpos
from hera_sim import interpolators
from hera_sim import Simulator

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


def test_antenna_matching():
    # Simple test: reflected right-triangles should match exactly
    array_1 = {0: [0, 0, 0], 1: [1, 0, 0], 2: [1, 1, 0]}

    array_2 = {0: [0, 0, 0], 1: [0, 1, 0], 2: [1, 1, 0]}

    array_intersection = adjustment._get_array_intersection(array_1, array_2, tol=0)
    assert antpos_equal(array_intersection, array_2, 0)

    # In general, rotated right-triangles should have at least two antennas in their
    # intersection.
    array_3 = {0: [0, 0, 0], 1: [1, 0, 0], 2: [0, 1, 0]}
    array_intersection = adjustment._get_array_intersection(array_1, array_3, tol=0)
    assert len(array_intersection) == 2

    # A simple translation should just be undone
    translation = np.random.uniform(-1, 1, 3)
    array_4 = {ant: np.array(pos) - translation for ant, pos in array_1.items()}
    array_intersection = adjustment._get_array_intersection(array_1, array_4, tol=0.1)
    assert antpos_equal(array_intersection, array_4, 0.1)

    # A small hex array should be a subset of a larger hex array
    hex_array = antpos.HexArray(split_core=False, outriggers=0)
    hex_array_1 = hex_array(3)
    hex_array_2 = hex_array(2)
    array_intersection = adjustment._get_array_intersection(hex_array_1, hex_array_2)
    assert antpos_equal(array_intersection, hex_array_2)
    assert len(array_intersection) == len(hex_array_2)


def test_interpolation_in_frequency(base_config):
    new_config = base_config.copy()
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
    new_config = base_config.copy()
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
    new_config = base_config.copy()
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
