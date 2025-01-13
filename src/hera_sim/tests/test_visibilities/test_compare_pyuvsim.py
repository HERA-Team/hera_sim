"""Compare vis_cpu with pyuvsim visibilities."""

import copy

import numpy as np
import pytest
from astropy import units
from astropy.coordinates import Latitude, Longitude
from astropy.time import Time
from astropy.units import Quantity
from pyradiosky import SkyModel
from pyuvdata.analytic_beam import GaussianBeam
from pyuvsim import simsetup, uvsim
from pyuvsim.telescope import BeamList

from hera_sim import io
from hera_sim.beams import PolyBeam
from hera_sim.visibilities import SIMULATORS, ModelData, VisibilitySimulation

nfreq = 3
ntime = 20
nants = 4


def get_uvdata(pol_array=None):
    hera_lat = -30.7215
    hera_lon = 21.4283
    hera_alt = 1073.0
    obstime = Time("2018-08-31T04:02:30.11", format="isot", scale="utc")
    if pol_array is None:
        pol_array = np.array(["XX", "YY", "XY", "YX"])
    rng = np.random.default_rng(10)

    # Random antenna locations
    x = rng.random(nants) * 400.0  # Up to 400 metres
    y = rng.random(nants) * 400.0
    z = rng.random(nants) * 0.0

    ants = {i: (x[i], y[i], z[i]) for i in range(nants)}

    # Observing parameters in a UVData object
    return io.empty_uvdata(
        Nfreqs=nfreq,
        start_freq=100e6,
        channel_width=97.3e3,
        start_time=obstime.jd,
        integration_time=20.0,
        Ntimes=ntime,
        array_layout=ants,
        polarization_array=pol_array,
        telescope_location=(hera_lat, hera_lon, hera_alt),
        telescope_name="test_array",
        x_orientation="east",
        phase_type="drift",
        vis_units="Jy",
        write_files=False,
    )


@pytest.fixture(scope="function")
def uvdata_allpols():
    return get_uvdata()


def get_sky_model(uvdata, nsource):
    # One fixed source plus random other sources
    sources = [[125.7, -30.72, 2, 0]]  # Fix a single source near zenith
    rng = np.random.default_rng(0)
    if nsource > 1:  # Add random other sources
        ra = rng.uniform(low=0.0, high=360.0, size=nsource - 1)
        dec = -30.72 + rng.random(nsource - 1) * 10.0
        flux = rng.random(nsource - 1) * 4
        for i in range(nsource - 1):
            sources.append([ra[i], dec[i], flux[i], 0])
    sources = np.array(sources)

    # Source locations and frequencies
    ra_dec = np.deg2rad(sources[:, :2])
    freqs = np.unique(uvdata.freq_array)

    # Stokes for the first frequency only. Stokes for other frequencies
    # are calculated later.
    stokes = np.zeros((4, 1, ra_dec.shape[0]))
    stokes[0, 0] = sources[:, 2]
    reference_frequency = np.full(len(ra_dec), freqs[0])

    # Set up sky model
    sky_model = SkyModel(
        name=[str(i) for i in range(len(ra_dec))],
        ra=Longitude(ra_dec[:, 0], "rad"),
        dec=Latitude(ra_dec[:, 1], "rad"),
        spectral_type="spectral_index",
        spectral_index=sources[:, 3],
        stokes=stokes * units.Jy,
        reference_frequency=Quantity(reference_frequency, "Hz"),
        frame="icrs",
    )

    # Calculate stokes at all the frequencies.
    sky_model.at_frequencies(Quantity(freqs, "Hz"), inplace=True)

    return sky_model

@pytest.fixture(scope='function')
def gaussian():
    return GaussianBeam(sigma=0.103)

@pytest.fixture(scope='function')
def polybeam():
    return PolyBeam(
        ref_freq=1e8,
        spectral_index=-0.6975,
        beam_coeffs=[
            2.35088101e-01,
            -4.20162599e-01,
            2.99189140e-01,
            -1.54189057e-01,
            3.38651457e-02,
            3.46936067e-02,
            -4.98838130e-02,
            3.23054464e-02,
            -7.56006552e-03,
            -7.24620596e-03,
            7.99563166e-03,
            -2.78125602e-03,
            -8.19945835e-04,
            1.13791191e-03,
            -1.24301372e-04,
            -3.74808752e-04,
            1.93997376e-04,
            -1.72012040e-05,
        ],
    )

@pytest.mark.parametrize(
    "nsource,beam_type,polarized",
    [
        (1, "gaussian", False),
        (1, "polybeam", False),
        (1, "polybeam", True),
        (100, "gaussian", False),
        (100, "polybeam", False),
        (100, "polybeam", True),
    ],
)
@pytest.mark.parametrize("simcls", [v for k, v in SIMULATORS.items() if k != "UVSim"])
def test_compare_with_pyuvsim(
    uvdata_allpols, nsource, beam_type, polarized: bool, simcls, request
):
    """Compare simulators with pyuvsim."""
    sky_model = get_sky_model(uvdata_allpols, nsource)

    # Beam models
    beams = [request.getfixturevalue(beam_type)]
    beam_dict = {str(i): 0 for i in range(nants)}

    # ---------------------------------------------------------------------------
    # (1) Run matvis
    # ---------------------------------------------------------------------------
    # Trim unwanted polarizations
    uvdata_matvis = copy.deepcopy(uvdata_allpols)

    if not polarized:
        uvdata_matvis.select(polarizations=["ee"], inplace=True)

    # Construct simulator object and run
    simulator = simcls()

    # TODO: if we update the PolyBeam API so that it doesn't *require* 2 feeds,
    # we can get rid of this.
    vis_cpu_beams = [copy.deepcopy(beam) for beam in beams]

    sim = VisibilitySimulation(
        data_model=ModelData(
            uvdata=uvdata_matvis, sky_model=sky_model, beams=vis_cpu_beams
        ),
        simulator=simulator,
    )

    sim.simulate()
    uvd_matvis = sim.uvdata

    # ---------------------------------------------------------------------------
    # (2) Run pyuvsim
    # ---------------------------------------------------------------------------
    uvd_uvsim = uvsim.run_uvdata_uvsim(
        uvdata_allpols,
        BeamList(beams),
        beam_dict=beam_dict,
        catalog=simsetup.SkyModelData(sky_model),
        quiet=True,
    )

    # ---------------------------------------------------------------------------
    # Compare results
    # ---------------------------------------------------------------------------
    # Set relative/absolute tolerances depending on no. of sources
    # (N.B. vis_cpu source position correction approximation degrades with time)
    if nsource < 10:
        # Very stringent for a few sources
        rtol = 1e-4
        atol = 1e-7
    else:
        # Within 0.1% or so for many sources
        rtol = 1e-3
        atol = 1e-5

    for i in range(nants):
        for j in range(nants):
            print("Baseline: ", i, j)
            np.testing.assert_allclose(
                uvd_uvsim.get_data((i, j, "xx")),
                uvd_matvis.get_data((i, j, "xx")),
                atol=atol,
                rtol=rtol,
            )
