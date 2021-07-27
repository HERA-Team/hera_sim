import pytest
import numpy as np
from hera_sim.visibilities import VisCPU
from hera_sim import io
from hera_sim.beams import PerturbedPolyBeam, PolyBeam, efield_to_pstokes
from hera_sim.defaults import defaults
from astropy_healpix import healpy as hp
from vis_cpu import HAVE_GPU

np.seterr(invalid="ignore")


def antennas():
    locs = [[308, 253, 0.49], [8, 299, 0.22]]

    ants = {}
    for i in range(len(locs)):
        ants[i] = (locs[i][0], locs[i][1], locs[i][2])

    return ants


def sources():
    sources = np.array([[128, -29, 4, 0]])
    ra_dec = sources[:, :2]
    flux = sources[:, 2]
    spectral_index = sources[:, 3]

    ra_dec = np.deg2rad(ra_dec)

    return ra_dec, flux, spectral_index


def perturbed_beams(rotation, nants, polarized=False):
    """
    Elliptical PerturbedPolyBeam.

    This will also test PolyBeam, from which PerturbedPolybeam is derived.
    """
    cfg_beam = dict(
        ref_freq=1.0e8,
        spectral_index=-0.6975,
        mainlobe_width=0.3,
        beam_coeffs=[
            0.29778665,
            -0.44821433,
            0.27338272,
            -0.10030698,
            -0.01195859,
            0.06063853,
            -0.04593295,
            0.0107879,
            0.01390283,
            -0.01881641,
            -0.00177106,
            0.01265177,
            -0.00568299,
            -0.00333975,
            0.00452368,
            0.00151808,
            -0.00593812,
            0.00351559,
        ],
    )
    beams = [
        PerturbedPolyBeam(
            perturb_coeffs=np.array(
                [
                    -0.20437532,
                    -0.4864951,
                    -0.18577532,
                    -0.38053642,
                    0.08897764,
                    0.06367166,
                    0.29634711,
                    1.40277112,
                ]
            ),
            mainlobe_scale=1.0,
            xstretch=1.1,
            ystretch=0.8,
            rotation=rotation,
            polarized=polarized,
            **cfg_beam
        )
        for i in range(nants)
    ]

    return beams


class DummyMPIComm:
    """
    Exists so the MPI interface can be tested, but not run.
    """

    def Get_size(self):
        return 2  # Pretend there are 2 processes


def run_sim(
    beam_rotation,
    use_pixel_beams=True,
    use_gpu=False,
    use_pol=False,
    use_mpi=False,
    pol="xx",
):
    """
    Run a simple sim using a rotated elliptic polybeam.
    """
    defaults.set("h1c")
    pol_array = ["xx"]
    if use_pol:
        pol_array = np.array(
            ["yx", "xy", "yy", "xx"]
        )  # yx, xy, yy, xx = ne, en, nn, ee

    ants = antennas()

    # Observing parameters in a UVData object.
    uvdata = io.empty_uvdata(
        Nfreqs=1,
        start_freq=100000000.0,
        channel_width=97000.0,
        start_time=2458902.4,
        integration_time=40,
        Ntimes=1,
        array_layout=ants,
        polarization_array=pol_array,
    )
    freqs = np.unique(uvdata.freq_array)
    ra_dec, flux, spectral_index = sources()

    # calculate source fluxes for hera_sim
    flux = (freqs[:, np.newaxis] / freqs[0]) ** spectral_index * flux

    simulator = VisCPU(
        uvdata=uvdata,
        beams=perturbed_beams(beam_rotation, len(ants.keys()), polarized=use_pol),
        beam_ids=list(ants.keys()),
        sky_freqs=freqs,
        point_source_pos=ra_dec,
        point_source_flux=flux,
        use_pixel_beams=use_pixel_beams,
        use_gpu=use_gpu,
        polarized=use_pol,
        mpi_comm=DummyMPIComm() if use_mpi else None,
        bm_pix=200,
        precision=2,
    )
    simulator.simulate()
    auto = np.abs(simulator.uvdata.get_data(0, 0, pol)[0][0])

    return auto


class TestPerturbedPolyBeam:
    def test_perturbed_polybeam(self):

        # Rotate the beam from 0 to 180 degrees, and check that autocorrelation
        # of antenna 0 has approximately the same value when pixel beams are
        # used, and when pixel beams not used (direct beam calculation).
        rotations = np.zeros(180 + 1)
        pix_results = np.zeros(180 + 1)
        calc_results = np.zeros(180 + 1)
        for r in range(180 + 1):
            pix_result = run_sim(r, use_pixel_beams=True)

            # Direct beam calculation - no pixel beams
            calc_result = run_sim(r, use_pixel_beams=False)

            rotations[r] = r
            pix_results[r] = pix_result
            calc_results[r] = calc_result

        # Check that the maximum difference between pixel beams/direct calculation
        # cases is no more than 5%. This shows the direct calculation of the beam
        # tracks the pixel beam interpolation. They won't be exactly the same.
        np.testing.assert_allclose(pix_results, calc_results, rtol=0.05)

        # Check that rotations 0 and 180 produce the same values.
        assert pix_results[0] == pytest.approx(pix_results[180], abs=1e-8)
        assert calc_results[0] == pytest.approx(calc_results[180], abs=1e-8)

        # Check that the values are not all the same. Shouldn't be, due to
        # elliptic beam.
        assert np.min(pix_results) != pytest.approx(np.max(pix_results), abs=0.1)
        assert np.min(calc_results) != pytest.approx(np.max(calc_results), abs=0.1)

        # Check that attempting to use GPU with Polybeam raises an error.
        with pytest.raises(RuntimeError if HAVE_GPU else ImportError):
            run_sim(r, use_pixel_beams=False, use_gpu=True)

        # Check that attempting to use GPU with MPI raises an error.
        with pytest.raises(RuntimeError):
            run_sim(r, use_gpu=True, use_mpi=True)


class TestPolarizedPolyBeam:
    def test_all_polarized_polybeam(self):
        """
        Wrapper for all polarized PolyBeam tests.
        Instantiate and evaluate a beam (once).
        """
        pol_beam = create_polarized_polybeam()
        eval_beam, az, za, Nfreq = evaluate_polybeam(pol_beam)
        eval_beam_pStokes = convert_to_pStokes(eval_beam, az, za, Nfreq)

        # Check that the beam is normalized between 1 and 0 (± 1e-2),
        # at all polarizations and a range of selected frequencies.
        for vec in [0, 1]:
            for feed in [0, 1]:
                for freq in [0, 5, 10, 15, 20, 25]:
                    modulus = np.abs(eval_beam[vec, 0, feed, freq])
                    M = np.max(modulus)
                    m = np.min(modulus)
                    assert M <= 1 and M == pytest.approx(
                        1, rel=3e-2
                    ), "beam not properly normalized"
                    assert m >= 0 and m == pytest.approx(
                        0, abs=1e-3
                    ), "beam not properly normalized"

        # Check that neither NaNs nor Infs atre returned by the interp() method.
        assert not np.isnan(eval_beam).any(), "the beam contains NaN values"
        assert not np.isinf(eval_beam).any(), "the beam contains Inf values"

        # Check that pStokes power beams are real
        assert np.isreal(
            eval_beam_pStokes
        ).all(), "the pseudo-Stokes beams are not real"


def create_polarized_polybeam():
    """
    Create a polarized PolyBeam.

    The parameters of the beam were copied from the HERA Memo n°81:
    https://reionization.org/wp-content/uploads/2013/03/HERA081_HERA_Primary_Beam_Chebyshev_Apr2020.pdf.

    """
    # parameters
    spectral_index = -0.6975
    beam_coeffs = [
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
    ]
    ref_freq = 1e8
    # instantiate the PolyBeam object
    cfg_pol_beam = dict(
        ref_freq=ref_freq,
        spectral_index=spectral_index,
        beam_coeffs=beam_coeffs,
        polarized=True,
    )
    pol_PolyBeam = PolyBeam(**cfg_pol_beam)

    return pol_PolyBeam


def evaluate_polybeam(polybeam):
    """
    Evaluate a PolyBeam at hard-coded az and za angles, and frequencies.
    """
    n_pix_lm = 500
    L = np.linspace(-1, 1, n_pix_lm, dtype=np.float64)
    L, m = np.meshgrid(L, L)
    L = L.flatten()
    m = m.flatten()

    lsqr = L ** 2 + m ** 2
    n = np.where(lsqr < 1, np.sqrt(1 - lsqr), 0)

    # Generate azimuth and zenith angle.
    az = -np.arctan2(m, L)
    za = np.pi / 2 - np.arcsin(n)

    freqs = np.array(
        [
            1.00e08,
            1.04e08,
            1.08e08,
            1.12e08,
            1.16e08,
            1.20e08,
            1.24e08,
            1.28e08,
            1.32e08,
            1.36e08,
            1.40e08,
            1.44e08,
            1.48e08,
            1.52e08,
            1.56e08,
            1.60e08,
            1.64e08,
            1.68e08,
            1.72e08,
            1.76e08,
            1.80e08,
            1.84e08,
            1.88e08,
            1.92e08,
            1.96e08,
            2.00e08,
        ]
    )

    eval_beam = polybeam.interp(az, za, freqs)

    return (eval_beam[0], az, za, freqs.size)


def convert_to_pStokes(eval_beam, az, za, Nfreq):
    """
    Convert an E-field to its pseudo-Stokes power beam.
    """
    nside_test = 64
    pixel_indices_test = hp.ang2pix(nside_test, za, az)
    npix_test = hp.nside2npix(nside_test)

    pol_efield_beam_plot = np.zeros((2, 1, 2, Nfreq, npix_test), dtype=np.complex128)
    pol_efield_beam_plot[:, :, :, :, pixel_indices_test] = eval_beam[:, :, :, :]
    eval_beam_pStokes = efield_to_pstokes(pol_efield_beam_plot, npix_test, Nfreq)

    return eval_beam_pStokes
