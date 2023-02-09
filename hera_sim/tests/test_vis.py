import pytest

import astropy_healpix as aph
import copy
import healvis
import numpy as np
from astropy import time as apt
from astropy import units
from astropy.coordinates.angles import Latitude, Longitude
from astropy.units import rad, sday
from pathlib import Path
from pyradiosky import SkyModel
from pyuvsim.analyticbeam import AnalyticBeam
from pyuvsim.telescope import BeamConsistencyError

from hera_sim import io
from hera_sim.beams import PolyBeam
from hera_sim.defaults import defaults
from hera_sim.visibilities import (
    ModelData,
    UVSim,
    VisCPU,
    VisibilitySimulation,
    load_simulator_from_yaml,
)
from vis_cpu import HAVE_GPU

SIMULATORS = (VisCPU, UVSim)

try:
    from hera_sim.visibilities import HealVis

    SIMULATORS = SIMULATORS + (HealVis,)
except ImportError:
    pass

if HAVE_GPU:

    class VisGPU(VisCPU):
        """Simple mock class to make testing VisCPU with use_gpu=True easier"""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, use_gpu=True, **kwargs)

    SIMULATORS = SIMULATORS + (VisGPU,)


np.random.seed(0)
NTIMES = 10
NPIX = 12 * 16**2
NFREQ = 5


@pytest.fixture
def uvdata():
    defaults.set("h1c")
    return io.empty_uvdata(
        Nfreqs=NFREQ,
        integration_time=sday.to("s") / NTIMES,
        Ntimes=NTIMES,
        array_layout={
            0: (0, 0, 0),
        },
        start_time=2456658.5,
        conjugation="ant1<ant2",
        polarization_array=["xx", "yy", "xy", "yx"],
    )


@pytest.fixture(scope="function")
def uvdata_linear():
    defaults.set("h1c")
    return io.empty_uvdata(
        Nfreqs=1,
        integration_time=sday.to("s") / NTIMES,
        Ntimes=NTIMES,
        array_layout={0: (0, 0, 0), 1: (10, 0, 0), 2: (20, 0, 0), 3: (0, 10, 0)},
        start_time=2456658.5,
        conjugation="ant1<ant2",
        polarization_array=["xx", "xy", "yx", "yy"],
    )


@pytest.fixture
def uvdataJD():
    defaults.set("h1c")
    return io.empty_uvdata(
        Nfreqs=NFREQ,
        integration_time=sday.to("s") / NTIMES,
        Ntimes=NTIMES,
        array_layout={
            0: (0, 0, 0),
        },
        start_time=2456659,
        polarization_array=["xx", "yy", "xy", "yx"],
    )


@pytest.fixture(scope="function")
def sky_model(uvdata):
    return make_point_sky(
        uvdata,
        ra=np.array([0.0]) * rad,
        dec=np.array(uvdata.telescope_location_lat_lon_alt[0]) * rad,
        align=False,
    )


@pytest.fixture
def sky_modelJD(uvdataJD):
    return make_point_sky(
        uvdataJD,
        ra=np.array([0.0]) * rad,
        dec=np.array(uvdata.telescope_location_lat_lon_alt[0]) * rad,
        align=False,
    )


def test_healvis_beam(uvdata, sky_model):
    pytest.importorskip("healvis")
    sim = VisibilitySimulation(
        simulator=HealVis(),
        data_model=ModelData(
            uvdata=uvdata,
            sky_model=sky_model,
        ),
        n_side=2**4,
    )

    assert len(sim.data_model.beams) == 1
    assert isinstance(sim.data_model.beams[0], healvis.beam_model.AnalyticBeam)


def test_healvis_beam_obsparams(tmpdir):
    # Now try creating with an obsparam file
    pytest.importorskip("healvis")
    direc = tmpdir.mkdir("test_healvis_beam")

    with open(Path(__file__).parent / "testdata" / "healvis_catalog.txt") as fl:
        txt = fl.read()

    with open(direc.join("catalog.txt"), "w") as fl:
        fl.write(txt)

    with open(direc.join("telescope_config.yml"), "w") as fl:
        fl.write(
            """
    beam_paths:
        0 : 'uniform'
    telescope_location: (-30.72152777777791, 21.428305555555557, 1073.0000000093132)
    telescope_name: MWA
    """
        )

    with open(direc.join("layout.csv"), "w") as fl:
        fl.write(
            """Name     Number   BeamID   E          N          U

    Tile061        40        0   -34.8010   -41.7365     1.5010
    Tile062        41        0   -28.0500   -28.7545     1.5060
    Tile063        42        0   -11.3650   -29.5795     1.5160
    Tile064        43        0    -9.0610   -20.7885     1.5160
    """
        )

    with open(direc.join("obsparams.yml"), "w") as fl:
        fl.write(
            """
    freq:
      Nfreqs: 1
      channel_width: 80000.0
      start_freq: 100000000.0
    sources:
      catalog: {0}/catalog.txt
    telescope:
      array_layout: {0}/layout.csv
      telescope_config_name: {0}/telescope_config.yml
    time:
      Ntimes: 1
      integration_time: 11.0
      start_time: 2458098.38824015
    """.format(
                direc.strpath
            )
        )

    sim = VisibilitySimulation(
        data_model=ModelData.from_config(direc.join("obsparams.yml").strpath),
        simulator=HealVis(),
    )
    beam = sim.data_model.beams[0]
    assert isinstance(beam, healvis.beam_model.AnalyticBeam)


def test_JD(uvdata, uvdataJD, sky_model):
    model_data = ModelData(sky_model=sky_model, uvdata=uvdata)

    vis = VisCPU()

    sim1 = VisibilitySimulation(data_model=model_data, simulator=vis).simulate()

    model_data2 = ModelData(sky_model=sky_model, uvdata=uvdataJD)

    sim2 = VisibilitySimulation(data_model=model_data2, simulator=vis).simulate()

    assert sim1.shape == sim2.shape
    assert not np.allclose(sim1, sim2, atol=0.1)


def test_vis_cpu_estimate_memory(uvdata, uvdataJD, sky_model):
    model_data = ModelData(sky_model=sky_model, uvdata=uvdata)
    vis = VisCPU()
    mem = vis.estimate_memory(model_data)
    assert mem > 0


@pytest.fixture
def uvdata2():
    defaults.set("h1c")
    return io.empty_uvdata(
        Nfreqs=NFREQ,
        integration_time=sday.to("s") / NTIMES,
        Ntimes=NTIMES,
        array_layout={0: (0, 0, 0), 1: (1, 1, 0)},
        start_time=2456658.5,
        conjugation="ant1<ant2",
        polarization_array=["xx", "yy", "xy", "yx"],
    )


def make_point_sky(uvdata, ra: np.ndarray, dec: np.ndarray, align=True):
    freqs = np.unique(uvdata.freq_array)

    # put a point source in
    point_source_flux = np.ones((len(ra), len(freqs)))

    # align to healpix center for direct comparision
    if align:
        ra, dec = align_src_to_healpix(ra * rad, dec * rad)

    return SkyModel(
        ra=Longitude(ra),
        dec=Latitude(dec),
        stokes=np.array(
            [
                point_source_flux.T,
                np.zeros((len(freqs), len(ra))),
                np.zeros((len(freqs), len(ra))),
                np.zeros((len(freqs), len(ra))),
            ]
        )
        * units.Jy,
        name=np.array(["derp"] * len(ra)),
        spectral_type="full",
        freq_array=freqs * units.Hz,
    )


def zenith_sky_model(uvdata2):
    return make_point_sky(
        uvdata2,
        ra=np.array([0.0]),
        dec=np.array([uvdata2.telescope_location_lat_lon_alt[0]]),
        align=True,
    )


def horizon_sky_model(uvdata2):
    return make_point_sky(
        uvdata2,
        ra=np.array([0.0]),
        dec=np.array([uvdata2.telescope_location_lat_lon_alt[0] + np.pi / 2]),
        align=True,
    )


def twin_sky_model(uvdata2):
    return make_point_sky(
        uvdata2,
        ra=np.array([0.0, 0.0]),
        dec=np.array(
            [
                uvdata2.telescope_location_lat_lon_alt[0] + np.pi / 4,
                uvdata2.telescope_location_lat_lon_alt[0],
            ]
        ),
        align=True,
    )


def half_sky_model(uvdata2):
    nbase = 4
    nside = 2**nbase

    sky = create_uniform_sky(
        np.unique(uvdata2.freq_array),
        nbase=nbase,
    )

    # Zero out values within pi/2 of (theta=pi/2, phi=0)
    hp = aph.HEALPix(nside=nside, order="ring")
    ipix_disc = hp.cone_search_lonlat(0 * rad, np.pi / 2 * rad, radius=np.pi / 2 * rad)
    sky.stokes[0, :, ipix_disc] = 0
    return sky


def create_uniform_sky(freq, nbase=4, scale=1) -> SkyModel:
    """Create a uniform sky with total (integrated) flux density of `scale`"""
    nfreq = len(freq)
    nside = 2**nbase
    npix = 12 * nside**2
    return SkyModel(
        nside=nside,
        hpx_inds=np.arange(npix),
        stokes=np.array(
            [
                np.ones((nfreq, npix)) * scale / (4 * np.pi),
                np.zeros((nfreq, npix)),
                np.zeros((nfreq, npix)),
                np.zeros((nfreq, npix)),
            ]
        )
        * units.Jy
        / units.sr,
        spectral_type="full",
        freq_array=freq * units.Hz,
        name=np.array([str(i) for i in range(npix)]),
    )


@pytest.mark.parametrize("simulator", SIMULATORS)
def test_shapes(uvdata, simulator):
    sky = create_uniform_sky(np.unique(uvdata.freq_array))

    sim = VisibilitySimulation(
        data_model=ModelData(uvdata=uvdata, sky_model=sky),
        simulator=simulator(),
        n_side=2**4,
    )

    assert sim.simulate().shape == (uvdata.Nblts, 1, NFREQ, uvdata.Npols)


@pytest.mark.parametrize("precision, cdtype", [(1, np.complex64), (2, complex)])
def test_dtypes(uvdata, precision, cdtype):
    sky = create_uniform_sky(np.unique(uvdata.freq_array))
    vis = VisCPU(precision=precision)

    sim = VisibilitySimulation(
        data_model=ModelData(uvdata=uvdata, sky_model=sky), simulator=vis
    )

    v = sim.simulate()
    assert v.dtype == cdtype


@pytest.mark.parametrize("simulator", SIMULATORS)
def test_zero_sky(uvdata, simulator):
    sky = create_uniform_sky(np.unique(uvdata.freq_array), scale=0)

    sim = VisibilitySimulation(
        data_model=ModelData(uvdata=uvdata, sky_model=sky), simulator=simulator()
    )
    v = sim.simulate()
    np.testing.assert_equal(v, 0)


@pytest.mark.parametrize("simulator", SIMULATORS)
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


@pytest.mark.parametrize("simulator", SIMULATORS)
def test_single_source_autocorr(uvdata, simulator, sky_model):
    sim = VisibilitySimulation(
        data_model=ModelData(
            uvdata=uvdata,
            sky_model=sky_model,
        ),
        simulator=simulator(),
        n_side=2**4,
    )
    sim.simulate()

    v = sim.uvdata.get_data((0, 0, "xx"))[:, 0]  # Get just one frequency

    # Make sure the source is over the horizon half the time
    # (+/- 1 because of the discreteness of the times)
    # 1e-3 on either side to account for float inaccuracies.
    assert NTIMES / 2 - 1 <= np.sum(np.abs(v) > 0) <= NTIMES / 2 + 1


@pytest.mark.parametrize("simulator", SIMULATORS)
def test_single_source_autocorr_past_horizon(uvdata, simulator):
    sky_model = make_point_sky(
        uvdata,
        ra=np.array([0]) * rad,
        dec=np.array(uvdata.telescope_location_lat_lon_alt[0] + 1.1 * np.pi / 2) * rad,
        align=False,
    )

    sim = VisibilitySimulation(
        data_model=ModelData(
            uvdata=uvdata,
            sky_model=sky_model,
        ),
        simulator=simulator(),
        n_side=2**4,
    )
    v = sim.simulate()

    assert np.abs(np.mean(v)) == 0


def test_viscpu_coordinate_correction(uvdata2):
    sim = VisibilitySimulation(
        data_model=ModelData(
            uvdata=uvdata2,
            sky_model=zenith_sky_model(uvdata2),
        ),
        simulator=VisCPU(
            correct_source_positions=True, ref_time="2018-08-31T04:02:30.11"
        ),
    )

    # Apply correction
    # viscpu.correct_point_source_pos(obstime="2018-08-31T04:02:30.11", frame="icrs")
    v = sim.simulate()
    assert np.all(~np.isnan(v))

    sim2 = VisibilitySimulation(
        data_model=ModelData(
            uvdata=uvdata2,
            sky_model=zenith_sky_model(uvdata2),
        ),
        simulator=VisCPU(
            correct_source_positions=True,
            ref_time=apt.Time("2018-08-31T04:02:30.11", format="isot", scale="utc"),
        ),
    )

    v2 = sim2.simulate()
    assert np.allclose(v, v2)


def align_src_to_healpix(ra, dec, nside=2**4):
    """Where the point sources will be placed when converted to healpix model

    Parameters
    ----------
    point_source_pos : ndarray
        Positions of point sources to be passed to a Simulator.
    point_source_flux : ndarray
        Corresponding fluxes of point sources at each frequency.
    nside : int
        Healpix nside parameter.


    Returns
    -------
    new_pos: ndarray
        Point sources positioned at their nearest healpix centers.
    new_flux: ndarray
        Corresponding new flux values.
    """
    # Get which pixel every point source lies in.
    pix = aph.lonlat_to_healpix(ra, dec, nside)
    ra, dec = aph.healpix_to_lonlat(pix, nside)
    return ra, dec


@pytest.mark.parametrize("simulator", SIMULATORS[1:])
@pytest.mark.parametrize(
    "sky_model, beam_model",
    [
        (zenith_sky_model, None),
        (horizon_sky_model, None),
        (twin_sky_model, None),
        (half_sky_model, None),
        (half_sky_model, [AnalyticBeam("airy", diameter=1.75)]),
    ],
)
def test_comparison(simulator, uvdata2, sky_model, beam_model):
    model_data = ModelData(
        uvdata=uvdata2, sky_model=sky_model(uvdata2), beams=beam_model
    )

    v0 = VisibilitySimulation(
        data_model=model_data, simulator=SIMULATORS[0](), n_side=2**4
    ).simulate()

    v1 = VisibilitySimulation(
        data_model=model_data, simulator=simulator(), n_side=2**4
    ).simulate()

    assert v0.shape == v1.shape

    np.testing.assert_allclose(v0, v1, rtol=0.05)


@pytest.mark.parametrize("simulator", SIMULATORS)
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
        data_model=ModelData(
            uvdata=uvdata_linear,
            sky_model=sky_model,
        ),
        simulator=simulator(),
        n_side=2**4,
    )
    sim.simulate()

    sim.uvdata.reorder_blts(order="time", conj_convention="ant1<ant2")

    assert np.allclose(
        sim.uvdata.data_array[sim.uvdata.antpair2ind(0, 1), 0, 0, 0],
        sim.uvdata.data_array[sim.uvdata.antpair2ind(1, 2), 0, 0, 0],
    )

    assert not np.allclose(sim.uvdata.get_data((0, 1)), sim.uvdata.get_data((0, 3)))

    assert not np.allclose(
        sim.uvdata.data_array[sim.uvdata.antpair2ind(0, 1), 0, 0, 0],
        sim.uvdata.data_array[sim.uvdata.antpair2ind(0, 3), 0, 0, 0],
    )

    assert not np.allclose(
        sim.uvdata.data_array[sim.uvdata.antpair2ind(0, 2), 0, 0, 0],
        sim.uvdata.data_array[sim.uvdata.antpair2ind(0, 3), 0, 0, 0],
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
def test_vis_cpu_pol(polarization_array, xfail):
    """Test whether different combinations of input polarization array work."""

    defaults.set("h1c")
    uvdata = io.empty_uvdata(
        Nfreqs=NFREQ,
        integration_time=sday.to("s") / NTIMES,
        Ntimes=NTIMES,
        array_layout={
            0: (0, 0, 0),
        },
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

    beam = PolyBeam(polarized=False)
    simulator = VisCPU()

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


def test_beam_type_consistency(uvdata, sky_model):
    beams = [AnalyticBeam("gaussian"), AnalyticBeam("airy")]
    beams[0].efield_to_power()

    with pytest.raises(BeamConsistencyError):
        ModelData(uvdata=uvdata, sky_model=sky_model, beams=beams)


def test_power_polsky(uvdata, sky_model):
    new_sky = copy.deepcopy(sky_model)
    new_sky.stokes[1:] = 1.0 * units.Jy

    beams = [AnalyticBeam("gaussian")]
    beams[0].efield_to_power()

    with pytest.raises(TypeError):
        ModelData(uvdata=uvdata, sky_model=new_sky, beams=beams)


def test_vis_cpu_stokespol(uvdata_linear, sky_model):
    uvdata_linear.polarization_array = [0, 1, 2, 3]
    with pytest.raises(ValueError):
        VisibilitySimulation(
            data_model=ModelData(uvdata=uvdata_linear, sky_model=sky_model),
            simulator=VisCPU(),
        )


def test_bad_uvdata(sky_model):
    with pytest.raises(TypeError, match="uvdata must be a UVData object"):
        ModelData(uvdata=3, sky_model=sky_model)


def test_str_uvdata(uvdata, sky_model, tmp_path):
    pth = tmp_path / "tmp_uvdata.uvh5"
    print(type(pth))
    uvdata.write_uvh5(str(pth))

    model_data = ModelData(uvdata=pth, sky_model=sky_model)
    assert model_data.uvdata.Nants_data == uvdata.Nants_data


def test_bad_healvis_skymodel(sky_model):
    pytest.importorskip("healvis")
    hv = HealVis()
    sky_model.stokes *= units.sr  # something stupid
    with pytest.raises(ValueError, match="not compatible with healvis"):
        hv.get_sky_model(sky_model)


def test_mK_healvis_skymodel(sky_model):
    pytest.importorskip("healvis")
    hv = HealVis()
    sky_model.stokes = sky_model.stokes.value * units.mK
    sky_model.nside = 2**3
    sky = hv.get_sky_model(sky_model)
    assert np.isclose(np.sum(sky.data), np.sum(sky_model.stokes[0].value / 1000))


def test_ref_time_viscpu(uvdata2):
    vc_mean = VisCPU(ref_time="mean")
    vc_min = VisCPU(ref_time="min")
    vc_max = VisCPU(ref_time="max")

    sky_model = half_sky_model(uvdata2)

    sim_mean = VisibilitySimulation(
        simulator=vc_mean, data_model=ModelData(uvdata=uvdata2, sky_model=sky_model)
    )
    sim_min = VisibilitySimulation(
        simulator=vc_min, data_model=ModelData(uvdata=uvdata2, sky_model=sky_model)
    )
    sim_max = VisibilitySimulation(
        simulator=vc_max, data_model=ModelData(uvdata=uvdata2, sky_model=sky_model)
    )

    dmean = sim_mean.simulate().copy()
    dmin = sim_min.simulate().copy()
    dmax = sim_max.simulate().copy()

    assert not np.all(dmean == dmin)
    assert not np.all(dmean == dmax)
    assert not np.all(dmax == dmin)


def test_load_from_yaml(tmpdir):
    example_dir = Path(__file__).parent.parent.parent / "config_examples"

    simulator = load_simulator_from_yaml(example_dir / "simulator.yaml")
    assert isinstance(simulator, VisCPU)
    assert simulator.ref_time == "mean"

    sim2 = VisCPU.from_yaml(example_dir / "simulator.yaml")

    assert sim2.ref_time == simulator.ref_time
    assert sim2.diffuse_ability == simulator.diffuse_ability


def test_bad_load(tmpdir):
    with open(tmpdir / "bad_sim.yaml", "w") as fl:
        fl.write("""simulator: nonexistent\n""")

    with pytest.raises(AttributeError, match="The given simulator"):
        load_simulator_from_yaml(tmpdir / "bad_sim.yaml")
