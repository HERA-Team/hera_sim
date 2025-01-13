import numpy as np
import pytest
from conftest import half_sky_model
from pyuvdata.analytic_beam import GaussianBeam

from hera_sim import io
from hera_sim.antpos import hex_array
from hera_sim.visibilities import ModelData, VisibilitySimulation

fftvis = pytest.importorskip("hera_sim.visibilities.fftvis")
FFTVis = fftvis.FFTVis


def test_fftvis_beam_error(uvdata2, sky_model):
    beams = [GaussianBeam(diameter=14.0), GaussianBeam(diameter=14.0)]
    beam_ids = [0, 1]
    simulator = FFTVis()
    data_model = ModelData(
        uvdata=uvdata2, sky_model=sky_model, beams=beams, beam_ids=beam_ids
    )
    with pytest.raises(ValueError):
        simulator.validate(data_model)


def test_stokespol(uvdata_linear, sky_model):
    uvdata_linear.polarization_array = [0, 1, 2, 3]
    with pytest.raises(ValueError):
        VisibilitySimulation(
            data_model=ModelData(uvdata=uvdata_linear, sky_model=sky_model),
            simulator=FFTVis(),
        )


def test_snap_positions():
    rng = np.random.default_rng(123)

    # Create a small hex array with perturbed positions
    ants = hex_array(5, split_core=True)
    ants = {k: v + rng.normal(scale=0.1, size=3) for k, v in ants.items()}

    uvd = io.empty_uvdata(
        Nfreqs=2,
        start_freq=50e6,
        channel_width=1e6,
        integration_time=10.0,
        Ntimes=1,
        array_layout=ants,
        start_time=2456658.5,
        conjugation="ant1<ant2",
        polarization_array=["xx", "yy", "xy", "yx"],
    )

    fftvis = FFTVis()
    sky_model = half_sky_model(uvd)

    unsnapped = VisibilitySimulation(
        simulator=fftvis,
        data_model=ModelData(uvdata=uvd.copy(), sky_model=sky_model),
        snap_antpos_to_grid=False,
    )
    snapped = VisibilitySimulation(
        simulator=fftvis,
        data_model=ModelData(uvdata=uvd.copy(), sky_model=sky_model),
        snap_antpos_to_grid=True,
        keep_snapped_antpos=True,
    )
    snapped_sneaky = VisibilitySimulation(
        simulator=fftvis,
        data_model=ModelData(uvdata=uvd.copy(), sky_model=sky_model),
        snap_antpos_to_grid=True,
        keep_snapped_antpos=False,
    )
    unsnapped.simulate()
    snapped.simulate()
    snapped_sneaky.simulate()

    # Visibilities should be the same for both snapped runs
    np.testing.assert_allclose(
        snapped.uvdata.data_array, snapped_sneaky.uvdata.data_array
    )

    # But not for the unsnapped.
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_allclose,
        snapped.uvdata.data_array,
        unsnapped.uvdata.data_array,
    )

    # The unsnapped and sneaky versions should have the same antenna positions
    np.testing.assert_allclose(
        unsnapped.uvdata.telescope.antenna_positions,
        snapped_sneaky.uvdata.telescope.antenna_positions,
    )

    # But the snapped positions should differ
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_allclose,
        snapped.uvdata.telescope.antenna_positions,
        unsnapped.uvdata.telescope.antenna_positions,
    )
