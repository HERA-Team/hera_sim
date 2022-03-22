import pytest
import numpy as np
from hera_sim.visibilities import VisCPU, ModelData, VisibilitySimulation
from hera_sim import io
from hera_sim.beams import (
    PerturbedPolyBeam,
    PolyBeam,
    efield_to_pstokes,
    ZernikeBeam,
    stokes_matrix,
)
from hera_sim.defaults import defaults
from pyradiosky import SkyModel
from astropy import units
from astropy.coordinates import Longitude, Latitude
import astropy_healpix.healpy as hp
from typing import List
from vis_cpu import HAVE_GPU
import copy

np.seterr(invalid="ignore")


@pytest.fixture(scope="module")
def antennas():
    return {0: (308, 253, 0.49), 1: (8, 299, 0.22)}


@pytest.fixture(scope="module")
def sources():
    ra_dec = np.deg2rad(np.array([[128, -29]]))
    flux = np.array([[4]])
    spectral_index = np.array([[0]])
    return ra_dec, flux, spectral_index


class DummyMPIComm:
    """
    Exists so the MPI interface can be tested, but not run.
    """

    def Get_size(self):
        return 2  # Pretend there are 2 processes


def evaluate_polybeam(polybeam):
    """
    Evaluate a PolyBeam at hard-coded az and za angles, and frequencies.
    """
    n_pix_lm = 500
    L = np.linspace(-1, 1, n_pix_lm, dtype=np.float64)
    L, m = np.meshgrid(L, L)
    L = L.flatten()
    m = m.flatten()

    lsqr = L**2 + m**2
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

    # Check that calling the interp() method with wrongly sized
    # coordinates results in an error
    with pytest.raises(ValueError):
        _ = polybeam.interp(az, za[:-1], freqs)

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
    return efield_to_pstokes(pol_efield_beam_plot, npix_test, Nfreq)


def run_sim(
    ants,
    sources,
    beams,
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
    pol_array = np.array(["yx", "xy", "yy", "xx"]) if use_pol else ["xx"]

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
        x_orientation="east",
    )
    freqs = np.unique(uvdata.freq_array)
    ra_dec, flux, spectral_index = sources

    # calculate source fluxes for hera_sim
    flux = (freqs[:, np.newaxis] / freqs[0]) ** spectral_index * flux

    simulator = VisCPU(
        use_pixel_beams=use_pixel_beams,
        use_gpu=use_gpu,
        mpi_comm=DummyMPIComm() if use_mpi else None,
        bm_pix=201,
        precision=2,
    )

    data_model = ModelData(
        uvdata=uvdata,
        beams=beams,
        sky_model=SkyModel(
            freq_array=freqs,
            ra=Longitude(ra_dec[:, 0] * units.rad),
            dec=Latitude(ra_dec[:, 1] * units.rad),
            spectral_type="full",
            stokes=np.array(
                [flux, np.zeros_like(flux), np.zeros_like(flux), np.zeros_like(flux)]
            )
            * units.Jy,
            name=["derp"] * flux.shape[1],
        ),
    )
    simulation = VisibilitySimulation(
        data_model=data_model,
        simulator=simulator,
    )
    simulation.simulate()

    return np.abs(simulation.uvdata.get_data(0, 0, pol)[0][0])


class TestPerturbedPolyBeam:
    def get_perturbed_beams(
        self, rotation, polarized=False, power_beam=False
    ) -> List[PerturbedPolyBeam]:
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
        ]

        # Specify power beam if requested
        if power_beam:
            for beam in beams:
                beam.efield_to_power()

        return beams

    def test_rotations(self, antennas, sources):

        # Rotate the beam from 0 to 180 degrees, and check that autocorrelation
        # of antenna 0 has approximately the same value when pixel beams are
        # used, and when pixel beams not used (direct beam calculation).
        rvals = np.linspace(0.0, 180.0, 31, dtype=int)

        rotations = np.zeros(rvals.size)
        pix_results = np.zeros(rvals.size)
        calc_results = np.zeros(rvals.size)
        for i, r in enumerate(rvals):
            beams = self.get_perturbed_beams(r, power_beam=True)
            pix_result = run_sim(antennas, sources, beams, use_pixel_beams=True)

            # Direct beam calculation - no pixel beams
            calc_result = run_sim(antennas, sources, beams, use_pixel_beams=False)

            rotations[i] = r
            pix_results[i] = pix_result
            calc_results[i] = calc_result

        # Check that the maximum difference between pixel beams/direct calculation
        # cases is no more than 5%. This shows the direct calculation of the beam
        # tracks the pixel beam interpolation. They won't be exactly the same.
        np.testing.assert_allclose(pix_results, calc_results, rtol=0.05)

        # Check that rotations 0 and 180 produce the same values.
        assert pix_results[0] == pytest.approx(pix_results[-1], abs=1e-8)
        assert calc_results[0] == pytest.approx(calc_results[-1], abs=1e-8)

        # Check that the values are not all the same. Shouldn't be, due to
        # elliptic beam.
        assert np.min(pix_results) != pytest.approx(np.max(pix_results), abs=0.1)
        assert np.min(calc_results) != pytest.approx(np.max(calc_results), abs=0.1)

    def test_power_beam(self, antennas, sources):
        # Check that power beam calculation returns values
        beams = self.get_perturbed_beams(180.0, power_beam=True)

        calc_result = run_sim(antennas, sources, beams, use_pixel_beams=False)
        assert np.all(np.isfinite(calc_result))

    def test_beam_select(self, antennas, sources):
        # Check that PolyBeam classes have a select() method, but that it does nothing
        beams = self.get_perturbed_beams(180.0, power_beam=True)
        for beam in beams:
            beam.select(any_kwarg_should_work=1)

    def test_gpu_fails(self, antennas, sources):
        # Check that power beam calculation returns values
        beams = self.get_perturbed_beams(180.0)

        # Check that attempting to use GPU with Polybeam raises an error.
        with pytest.raises(RuntimeError if HAVE_GPU else ImportError):
            run_sim(antennas, sources, beams, use_pixel_beams=False, use_gpu=True)

        # Check that attempting to use GPU with MPI raises an error.
        with pytest.raises(RuntimeError):
            run_sim(antennas, sources, beams, use_gpu=True, use_mpi=True)

    @pytest.mark.parametrize("pol", ["ee", "nn", "en", "ne"])
    def test_polarized_validity(self, antennas, sources, pol):
        beams = self.get_perturbed_beams(12.0)
        res = run_sim(
            antennas, sources, beams, use_pixel_beams=True, use_pol=True, pol=pol
        )
        assert np.all(np.isfinite(res))

    def test_unpolarized_validity(self, antennas, sources):
        beams = self.get_perturbed_beams(12.0, power_beam=True)
        res = run_sim(
            antennas, sources, beams, use_pixel_beams=True, use_pol=False, pol="ee"
        )
        assert np.all(np.isfinite(res))

    def test_error_without_mainlobe_width(self):
        # Check that error is raised if mainlobe_width not specified
        with pytest.raises(ValueError):
            PerturbedPolyBeam(
                perturb_coeffs=np.array([-0.204, -0.486]),
                mainlobe_width=None,
                beam_coeffs=[2.35e-01, -4.2e-01, 2.99e-01],
            )

    def test_perturb_scale_greater_than_one(self):
        # Check that perturb_scale > 1 raises ValueError
        with pytest.raises(ValueError):
            PerturbedPolyBeam(
                perturb_coeffs=np.array([-0.204, -0.486]),
                mainlobe_width=None,
                beam_coeffs=[2.35e-01, -4.2e-01, 2.99e-01],
                perturb_scale=1.1,
            )

    def test_no_perturb_coeffs(self):
        # Check that specifying no perturbation coeffs works
        ppb = PerturbedPolyBeam(
            perturb_coeffs=None,
            mainlobe_width=1.0,
            beam_coeffs=[2.35e-01, -4.2e-01, 2.99e-01],
        )

        eval_beam = evaluate_polybeam(ppb)[0]
        assert np.all(np.isfinite(eval_beam))

    def test_specify_freq_perturb_coeffs(self):
        # Check that specifying freq_perturb_coeffs works
        ppb = PerturbedPolyBeam(
            perturb_coeffs=np.array([-0.204, -0.486]),
            mainlobe_width=1.0,
            beam_coeffs=[2.35e-01, -4.2e-01, 2.99e-01],
            freq_perturb_coeffs=[0.0, 0.1],
        )
        eval_beam = evaluate_polybeam(ppb)[0]
        assert np.all(np.isfinite(eval_beam))

    def test_mainlobe_scale(self):
        # Check that specifying mainlobe_scale factor works
        ppb = PerturbedPolyBeam(
            perturb_coeffs=np.array([-0.204, -0.486]),
            mainlobe_width=1.0,
            beam_coeffs=[2.35e-01, -4.2e-01, 2.99e-01],
            freq_perturb_coeffs=[0.0, 0.1],
            mainlobe_scale=1.1,
        )
        eval_beam = evaluate_polybeam(ppb)[0]
        assert np.all(np.isfinite(eval_beam))

    def test_zeropoint(self):
        ppb = PerturbedPolyBeam(
            perturb_coeffs=np.array([-0.204, -0.486]),
            mainlobe_width=1.0,
            beam_coeffs=[2.35e-01, -4.2e-01, 2.99e-01],
            freq_perturb_coeffs=[0.0, 0.1],
            perturb_zeropoint=1.0,
        )

        eval_beam = evaluate_polybeam(ppb)[0]
        assert np.all(np.isfinite(eval_beam))

    def test_bad_freq_perturb_scale(self):
        with pytest.raises(ValueError, match="must be less than 1"):
            PerturbedPolyBeam(
                perturb_coeffs=np.array([-0.204, -0.486]),
                mainlobe_width=1.0,
                beam_coeffs=[2.35e-01, -4.2e-01, 2.99e-01],
                freq_perturb_coeffs=[0.0, 0.1],
                freq_perturb_scale=2.0,
            )


class TestPolarizedPolyBeam:
    @pytest.fixture(scope="class")
    def polarized_polybeam(self):
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
        return PolyBeam(**cfg_pol_beam)

    @pytest.fixture(scope="class")
    def eval_beam(self, polarized_polybeam):
        eval_beam, az, za, Nfreq = evaluate_polybeam(polarized_polybeam)
        return eval_beam

    @pytest.fixture(scope="class")
    def eval_beam_pstokes(self, polarized_polybeam):
        eval_beam, az, za, Nfreq = evaluate_polybeam(polarized_polybeam)
        return convert_to_pStokes(eval_beam, az, za, Nfreq)

    def test_normalization(self, polarized_polybeam, eval_beam):
        """
        Wrapper for all polarized PolyBeam tests.
        Instantiate and evaluate a beam (once).
        """
        eval_beam, az, za, Nfreq = evaluate_polybeam(polarized_polybeam)

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

    def test_all_values_valid(self, eval_beam):
        # Check that neither NaNs nor Infs atre returned by the interp() method.
        assert not np.isnan(eval_beam).any(), "the beam contains NaN values"
        assert not np.isinf(eval_beam).any(), "the beam contains Inf values"

    def test_pstokes_real(self, eval_beam_pstokes):
        # Check that pStokes power beams are real
        assert np.isreal(
            eval_beam_pstokes
        ).all(), "the pseudo-Stokes beams are not real"

    def test_equality_method(self, polarized_polybeam):
        assert polarized_polybeam == polarized_polybeam

    def test_not_all_pols(self, polarized_polybeam, antennas, sources):
        ppb = copy.deepcopy(polarized_polybeam)
        ppb.feed_array = ["X", "Y"]
        with pytest.raises(
            ValueError, match="Not all polarizations in UVData object are in your beam."
        ):
            run_sim(antennas, sources, [ppb], use_pixel_beams=True, use_pol=True)


class TestZernikeBeam:
    @pytest.fixture(scope="function")
    def beams(self):
        """
        Zernike polynomial beams with some randomly-chosen coefficients.
        """
        cfg_beam = dict(
            ref_freq=1.0e8,
            spectral_index=-0.6975,
            beam_coeffs=[
                1.29778665,
                0.2,
                0.3,
                -0.10030698,
                -0.01195859,  # nofmt
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
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.01,
            ],
        )
        return [ZernikeBeam(**cfg_beam)]

    def test_sim_with_pixels(self, beams, antennas, sources):

        # Calculate visibilities using pixel and interpolated beams
        pix_result = run_sim(antennas, sources, beams, use_pixel_beams=True)
        calc_result = run_sim(antennas, sources, beams, use_pixel_beams=False)

        # Check that the maximum difference between pixel beams/direct calculation
        # cases is no more than 5%. This shows the direct calculation of the beam
        # tracks the pixel beam interpolation. They won't be exactly the same.
        np.testing.assert_allclose(pix_result, calc_result, rtol=0.05)

    def test_equality(self, beams):
        # Check basic methods
        assert beams[0] == beams[0]  # test __eq__ method
        assert beams[0] != 1

    def test_peak_normalize(self, beams):
        # Coords to evaluate at
        za = np.linspace(0.0, 0.5 * np.pi, 40)
        az = np.zeros(za.size)
        freqs = np.array([100.0e6])

        beam1 = beams[0]
        beam2 = copy.deepcopy(beams[0])

        # Check peak normalize works (beams are peak normalized by default)
        beam1.peak_normalized = False
        y1 = beam1.interp(az_array=az, za_array=za, freq_array=freqs)[0]
        y2 = beam2.interp(az_array=az, za_array=za, freq_array=freqs)[0]
        beam1.peak_normalize()
        y1a = beam1.interp(az_array=az, za_array=za, freq_array=freqs)[0]

        assert ~np.allclose(
            y1, y2
        )  # Unnormalized vs peak normalized should be different
        assert np.allclose(y1a, y2)  # Peak normalized beams should give same results

    def test_finitude(self, beams):
        # Check to make sure power beam gives finite results

        za = np.linspace(0.0, 0.5 * np.pi, 40)
        az = np.zeros(za.size)
        freqs = np.array([100.0e6])

        beams[0].beam_type = "power"
        y1b = beams[0].interp(az_array=az, za_array=za, freq_array=freqs)[0]
        assert np.all(np.isfinite(y1b))


def test_pol_stokes_bad_idx():
    with pytest.raises(ValueError, match="must be an integer between"):
        stokes_matrix(5)
