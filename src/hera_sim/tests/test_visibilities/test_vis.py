import copy
from copy import deepcopy

import numpy as np
import pytest
from astropy.units import rad, sday
from conftest import (
    NFREQ,
    NTIMES,
    create_uniform_sky,
    half_sky_model,
    horizon_sky_model,
    make_point_sky,
    twin_sky_model,
    zenith_sky_model,
)
from pyuvdata.analytic_beam import AiryBeam

from hera_sim import io
from hera_sim.beams import PolyBeam
from hera_sim.defaults import defaults
from hera_sim.visibilities import (
    SIMULATORS,
    ModelData,
    UVSim,
    VisibilitySimulation,
    VisibilitySimulator,
    load_simulator_from_yaml,
)

if "MatVis" in SIMULATORS:
    from hera_sim.visibilities.matvis import HAVE_GPU

    if HAVE_GPU:

        class MatVisGPU(SIMULATORS["MatVis"]):
            """Simple mock class to make testing MatVis with use_gpu=True easier"""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, use_gpu=True, ref_time="min", **kwargs)

        SIMULATORS = copy(SIMULATORS)
        SIMULATORS["MatVisGPU"] = MatVisGPU


def test_JD(uvdata, uvdataJD, sky_model):
    model_data = ModelData(sky_model=sky_model, uvdata=uvdata)

    vis = UVSim()

    sim1 = VisibilitySimulation(data_model=model_data, simulator=vis).simulate()

    model_data2 = ModelData(sky_model=sky_model, uvdata=uvdataJD)

    sim2 = VisibilitySimulation(data_model=model_data2, simulator=vis).simulate()

    assert sim1.shape == sim2.shape
    assert not np.allclose(sim1, sim2, atol=0.1)


@pytest.mark.parametrize("simulator", SIMULATORS.values())
def test_estimate_memory(uvdata, sky_model, simulator):
    model_data = ModelData(sky_model=sky_model, uvdata=uvdata)
    vis = simulator()
    mem = vis.estimate_memory(model_data)
    assert mem > 0


@pytest.mark.parametrize("simulator", SIMULATORS.values())
def test_shapes(uvdata, simulator):
    sky = create_uniform_sky(np.unique(uvdata.freq_array))

    sim = VisibilitySimulation(
        data_model=ModelData(uvdata=uvdata, sky_model=sky),
        simulator=simulator(),
        n_side=2**4,
    )

    assert sim.simulate().shape == (uvdata.Nblts, NFREQ, uvdata.Npols)


@pytest.mark.parametrize("precision, cdtype", [(1, np.complex64), (2, complex)])
@pytest.mark.parametrize(
    "simulator", [v for k, v in SIMULATORS.items() if k in ["MatVis", "FFTVis"]]
)
def test_dtypes(uvdata, precision, cdtype, simulator: VisibilitySimulator):
    sky = create_uniform_sky(np.unique(uvdata.freq_array))
    vis = simulator(precision=precision)

    # If data_array is empty, then we never create new vis, and the returned value
    # is literally the data array, so we should expect to get complex128 regardless.
    sim = VisibilitySimulation(
        data_model=ModelData(uvdata=uvdata, sky_model=sky), simulator=vis
    )

    v = sim.simulate()
    assert v.dtype == complex

    # Now, the uvdata array has stuff in it, so the returned v is a new array that
    # would have been added to it.
    sim = VisibilitySimulation(
        data_model=ModelData(uvdata=uvdata, sky_model=sky), simulator=vis
    )

    v = sim.simulate()
    assert v.dtype == cdtype


@pytest.mark.parametrize("simulator", SIMULATORS.values())
def test_zero_sky(uvdata, simulator):
    sky = create_uniform_sky(np.unique(uvdata.freq_array), scale=0)

    sim = VisibilitySimulation(
        data_model=ModelData(uvdata=uvdata, sky_model=sky), simulator=simulator()
    )
    v = sim.simulate()
    np.testing.assert_equal(v, 0)


@pytest.mark.parametrize("simulator", SIMULATORS.values())
def test_autocorr_flat_beam(uvdata, simulator):
    sky = create_uniform_sky(np.unique(uvdata.freq_array), nbase=6)

    sim = VisibilitySimulation(
        data_model=ModelData(uvdata=uvdata, sky_model=sky), simulator=simulator()
    )
    sim.simulate()

    v = sim.uvdata.get_data((0, 0, "xx"))

    print(v)
    # The sky is uniform and integrates to one over the full sky.
    # Thus the stokes-I component of an autocorr will be 0.5 (going to horizon)
    # Since I = XX + YY and X/Y should be equal, the xx part should be 0.25

    np.testing.assert_allclose(np.abs(v), np.mean(v), rtol=1e-4)
    np.testing.assert_almost_equal(np.abs(v), 0.25, 2)


@pytest.mark.parametrize("simulator", SIMULATORS.values())
def test_single_source_autocorr(uvdata, simulator, sky_model):
    sim = VisibilitySimulation(
        data_model=ModelData(uvdata=uvdata, sky_model=sky_model),
        simulator=simulator(),
        n_side=2**4,
    )
    sim.simulate()

    v = sim.uvdata.get_data((0, 0, "xx"))[:, 0]  # Get just one frequency

    # Make sure the source is over the horizon half the time
    # (+/- 1 because of the discreteness of the times)
    # 1e-3 on either side to account for float inaccuracies.
    assert NTIMES / 2 - 1 <= np.sum(np.abs(v) > 0) <= NTIMES / 2 + 1


@pytest.mark.parametrize("simulator", SIMULATORS.values())
def test_single_source_autocorr_past_horizon(uvdata, simulator):
    sky_model = make_point_sky(
        uvdata,
        ra=np.array([0]) * rad,
        dec=np.array(uvdata.telescope_location_lat_lon_alt[0] + 1.1 * np.pi / 2) * rad,
        align=False,
    )

    sim = VisibilitySimulation(
        data_model=ModelData(uvdata=uvdata, sky_model=sky_model),
        simulator=simulator(),
        n_side=2**4,
    )
    v = sim.simulate()

    assert np.abs(np.mean(v)) == 0


@pytest.mark.parametrize("simulator", list(SIMULATORS.values())[1:])
@pytest.mark.parametrize(
    "sky_model, beam_model",
    [
        (zenith_sky_model, None),
        (horizon_sky_model, None),
        (twin_sky_model, None),
        (half_sky_model, None),
        (half_sky_model, [AiryBeam(diameter=1.75)]),
    ],
)
def test_comparison(simulator, uvdata2, sky_model, beam_model):
    md = ModelData(uvdata=uvdata2, sky_model=sky_model(uvdata2), beams=beam_model)
    md1 = deepcopy(md)

    v0 = VisibilitySimulation(data_model=md, simulator=UVSim()).simulate()
    v1 = VisibilitySimulation(data_model=md1, simulator=simulator()).simulate()

    assert v0.shape == v1.shape
    np.testing.assert_allclose(v0, v1, rtol=0.05)


@pytest.mark.parametrize("simulator", SIMULATORS.values())
@pytest.mark.parametrize("order", ["time", "baseline", "ant1", "ant2"])
@pytest.mark.parametrize("conj", ["ant1<ant2", "ant2<ant1"])
def test_ordering(uvdata_linear, simulator, order, conj):
    uvdata_linear.reorder_blts(order=order, conj_convention=conj)

    sky_model = make_point_sky(
        uvdata_linear,
        ra=np.linspace(0, 2 * np.pi, 8) * rad,
        dec=uvdata_linear.telescope_location_lat_lon_alt[0] * np.ones(8) * rad,
        align=False,
    )

    sim = VisibilitySimulation(
        data_model=ModelData(uvdata=uvdata_linear, sky_model=sky_model),
        simulator=simulator(),
        n_side=2**4,
    )
    sim.simulate()

    sim.uvdata.reorder_blts(order="time", conj_convention="ant1<ant2")

    assert np.allclose(
        sim.uvdata.data_array[sim.uvdata.antpair2ind(0, 1), 0, 0],
        sim.uvdata.data_array[sim.uvdata.antpair2ind(1, 2), 0, 0],
    )

    assert not np.allclose(sim.uvdata.get_data((0, 1)), sim.uvdata.get_data((0, 3)))

    assert not np.allclose(
        sim.uvdata.data_array[sim.uvdata.antpair2ind(0, 1), 0, 0],
        sim.uvdata.data_array[sim.uvdata.antpair2ind(0, 3), 0, 0],
    )

    assert not np.allclose(
        sim.uvdata.data_array[sim.uvdata.antpair2ind(0, 2), 0, 0],
        sim.uvdata.data_array[sim.uvdata.antpair2ind(0, 3), 0, 0],
    )


@pytest.mark.parametrize(
    "polarization_array, xfail",
    [
        (["XX"], False),
        (["XY"], False),
        (["YY"], False),
        (["XX", "YX", "XY", "YY"], False),
    ],
)
@pytest.mark.parametrize(
    "simulator", [v for k, v in SIMULATORS.items() if k in ["MatVis", "FFTVis"]]
)
def test_pol_combos(polarization_array, xfail, simulator):
    """Test whether different combinations of input polarization array work."""

    defaults.set("h1c")
    uvdata = io.empty_uvdata(
        Nfreqs=NFREQ,
        integration_time=sday.to("s") / NTIMES,
        Ntimes=NTIMES,
        array_layout={0: (0, 0, 0)},
        start_time=2456658.5,
        conjugation="ant1<ant2",
        polarization_array=polarization_array,
    )

    sky_model = make_point_sky(
        uvdata,
        ra=np.linspace(0, 2 * np.pi, 8) * rad,
        dec=uvdata.telescope_location_lat_lon_alt[0] * np.ones(8) * rad,
        align=False,
    )

    beam = PolyBeam()
    simulator = simulator()

    if xfail:
        with pytest.raises(KeyError):
            VisibilitySimulation(
                data_model=ModelData(uvdata=uvdata, sky_model=sky_model, beams=[beam]),
                simulator=simulator,
                n_side=2**4,
            )
    else:
        VisibilitySimulation(
            data_model=ModelData(uvdata=uvdata, sky_model=sky_model, beams=[beam]),
            simulator=simulator,
            n_side=2**4,
        )


def test_bad_load(tmpdir):
    with open(tmpdir / "bad_sim.yaml", "w") as fl:
        fl.write("""simulator: nonexistent\n""")

    with pytest.raises(AttributeError, match="The given simulator"):
        load_simulator_from_yaml(tmpdir / "bad_sim.yaml")

    with open(tmpdir / "bad_sim.yaml", "w") as fl:
        fl.write("""simulator: hera_sim.foregrounds.DiffuseForeground\n""")

    with pytest.raises(ValueError, match="is not a subclass of VisibilitySimulator"):
        load_simulator_from_yaml(tmpdir / "bad_sim.yaml")

    with open(tmpdir / "bad_sim.yaml", "w") as fl:
        fl.write("""simulator: hera_sim.foregrounds.diffuse_foreground\n""")

    with pytest.raises(TypeError, match="is not a class"):
        load_simulator_from_yaml(tmpdir / "bad_sim.yaml")
