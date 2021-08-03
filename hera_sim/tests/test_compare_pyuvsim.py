"""Compare vis_cpu with pyuvsim visibilities."""
import numpy as np
import pytest

from pyuvsim import uvsim, simsetup, AnalyticBeam
from pyuvsim.telescope import BeamList

from hera_sim.beams import PolyBeam
from hera_sim.visibilities import VisCPU
from vis_cpu.conversions import equatorial_to_eci_coords

from astropy.coordinates import Latitude, Longitude, EarthLocation
from astropy.units import Quantity
from astropy.time import Time

from pyradiosky import SkyModel
import copy

nfreq = 3
ntime = 20
nants = 4
# nsource = 20


def get_beams(beam_type, nants):
    """Get a list of beam objects, one per antenna."""
    if beam_type == "gaussian":
        beams = [AnalyticBeam("gaussian", sigma=0.103)] * nants

    elif beam_type == "PolyBeam polarized":
        cfg_pol_beam = dict(
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
            polarized=True,
        )

        beams = [PolyBeam(**cfg_pol_beam)] * nants

    elif beam_type == "PolyBeam":
        cfg_pol_beam = dict(
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
            polarized=False,
        )

        beams = [PolyBeam(**cfg_pol_beam)] * nants
    else:
        raise ValueError("beam_type '%s' not recognized" % beam_type)
    return beams


@pytest.mark.parameterize(
    "nsource,beam_type",
    [
        (1, "gaussian"),
        (1, "PolyBeam"),
        (1, "PolyBeam polarized"),
        (100, "gaussian"),
        (100, "PolyBeam"),
        (100, "PolyBeam polarized"),
    ],
)
def test_compare_viscpu_with_pyuvsim(nsource, beam_type):
    """Compare vis_cpu and pyuvsim simulated visibilities."""
    hera_lat = -30.7215
    hera_lon = 21.4283
    hera_alt = 1073.0
    obstime = Time("2018-08-31T04:02:30.11", format="isot", scale="utc")

    # HERA location
    location = EarthLocation.from_geodetic(lat=hera_lat, lon=hera_lon, height=hera_alt)

    np.random.seed(10)

    # Polarization
    polarized = False
    if "polarized" in beam_type.lower():
        polarized = True

    # Random antenna locations
    x = np.random.random(nants) * 400.0  # Up to 400 metres
    y = np.random.random(nants) * 400.0
    z = np.random.random(nants) * 0.0
    ants = {}
    for i in range(nants):
        ants[i] = (x[i], y[i], z[i])

    # Observing parameters in a UVData object
    uvdata = simsetup.initialize_uvdata_from_keywords(
        Nfreqs=nfreq,
        start_freq=100e6,
        channel_width=97.3e3,
        start_time=obstime.jd,
        integration_time=20.0,
        Ntimes=ntime,
        array_layout=ants,
        polarization_array=np.array(["XX", "YY", "XY", "YX"]),
        telescope_location=(hera_lat, hera_lon, hera_alt),
        telescope_name="test_array",
        x_orientation="east",
        phase_type="drift",
        vis_units="Jy",
        complete=True,
        write_files=False,
    )
    # lsts = np.unique(uvdata.lst_array)

    # One fixed source plus random other sources
    sources = [
        [125.7, -30.72, 2, 0],  # Fix a single source near zenith
    ]
    if nsource > 1:  # Add random other sources
        ra = np.random.uniform(low=0.0, high=360.0, size=nsource - 1)
        dec = -30.72 + np.random.random(nsource - 1) * 10.0
        flux = np.random.random(nsource - 1) * 4
        for i in range(nsource - 1):
            sources.append([ra[i], dec[i], flux[i], 0])
    sources = np.array(sources)

    # Source locations and frequencies
    ra_dec = np.deg2rad(sources[:, :2])
    freqs = np.unique(uvdata.freq_array)

    # Correct source locations so that vis_cpu uses the right frame
    ra_new, dec_new = equatorial_to_eci_coords(
        ra_dec[:, 0], ra_dec[:, 1], obstime, location, unit="rad", frame="icrs"
    )

    # Calculate source fluxes for vis_cpu
    flux = (freqs[:, np.newaxis] / freqs[0]) ** sources[:, 3].T * sources[:, 2].T

    # Beam models
    beams = get_beams(beam_type=beam_type, nants=nants)
    beam_dict = {}
    for i in range(len(beams)):
        beam_dict[str(i)] = i

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
        stokes=stokes,
        reference_frequency=Quantity(reference_frequency, "Hz"),
    )

    # Calculate stokes at all the frequencies.
    sky_model.at_frequencies(Quantity(freqs, "Hz"), inplace=True)

    # ---------------------------------------------------------------------------
    # (1) Run vis_cpu
    # ---------------------------------------------------------------------------
    radec_new = np.column_stack((ra_new, dec_new))

    # Check that error is raised if polarizations exist in the UVData object
    # that cannot be calculated
    if not polarized and beam_type == "gaussian":
        _uvdata = copy.deepcopy(uvdata)

        with pytest.raises(KeyError):
            # Construct simulator object and run (expecting error)
            sim1 = VisCPU(
                uvdata=_uvdata,
                beams=beams,
                beam_ids=list(beam_dict.values()),
                sky_freqs=freqs,
                point_source_pos=radec_new,
                point_source_flux=flux,
                bm_pix=None,
                use_gpu=False,
                polarized=polarized,
                use_pixel_beams=False,
            )
            sim1.simulate()

    # Trim unwanted polarizations
    uvdata_viscpu = copy.deepcopy(uvdata)
    if not polarized:
        uvdata_viscpu.select(
            polarizations=[
                "ee",
            ],
            inplace=True,
        )

    # Construct simulator object and run
    simulator = VisCPU(
        uvdata=uvdata_viscpu,
        beams=beams,
        beam_ids=list(beam_dict.values()),
        sky_freqs=freqs,
        point_source_pos=radec_new,
        point_source_flux=flux,
        bm_pix=None,
        use_gpu=False,
        polarized=polarized,
        use_pixel_beams=False,
    )
    simulator.simulate()
    uvd_viscpu = simulator.uvdata

    # ---------------------------------------------------------------------------
    # (2) Run pyuvsim
    # ---------------------------------------------------------------------------
    uvd_uvsim = uvsim.run_uvdata_uvsim(
        uvdata,
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

    # Loop over baselines and compare
    diff_re = 0.0
    diff_im = 0.0
    for i in range(nants):
        for j in range(i, nants):

            # Get visibilities for this baseline and polarisation
            d_uvsim = uvd_uvsim.get_data((i, j, "xx")).T  # pyuvsim visibility
            d_viscpu = uvd_viscpu.get_data((i, j, "xx")).T  # vis_cpu visibility

            # Print diagnostics
            print("\nBaseline:", i, j)
            print("V_uvsim: ", d_uvsim)
            print("V_viscpu:", d_viscpu)
            print("Ratio of abs():", np.abs(d_uvsim) / np.abs(d_viscpu))
            print("Ratio of real: ", np.real(d_uvsim) / np.real(d_viscpu))
            print("Ratio of imag: ", np.imag(d_uvsim) / np.imag(d_viscpu))

            # Keep track of maximum difference
            delta = d_uvsim - d_viscpu
            if np.abs(np.max(delta.real)) > diff_re:
                diff_re = np.abs(np.max(delta.real))
            if np.abs(np.max(delta.imag)) > diff_im:
                diff_im = np.abs(np.max(delta.imag))

            # Very stringent threshold of 0.1%
            assert np.allclose(
                d_uvsim.real,
                d_viscpu.real,
                rtol=rtol,
                atol=atol,
            ), "Max. difference (re): %10.10e" % (diff_re,)
            assert np.allclose(
                d_uvsim.imag,
                d_viscpu.imag,
                rtol=rtol,
                atol=atol,
            ), "Max. difference (im): %10.10e" % (diff_im,)
