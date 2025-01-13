import numpy as np
import pytest

from hera_sim.beams import (
    PerturbedPolyBeam,
    PolyBeam,
    ZernikeBeam,
)


def evaluate_polybeam(polybeam: PolyBeam, efield: bool=True, nside: int=32):
    """
    Evaluate a PolyBeam at hard-coded az and za angles, and frequencies.
    """
    return polybeam.to_uvbeam(
        freq_array=np.arange(1e8, 2.01e8, 0.04e8),
        pixel_coordinate_system='healpix',
        nside=nside,
        beam_type='efield' if efield else 'power'
    )

def check_beam_is_finite(polybeam: PolyBeam, efield: bool = True):
    uvbeam = evaluate_polybeam(polybeam, efield)
    assert np.all(np.isfinite(uvbeam.data_array))

class TestPolyBeam:
    def get_beam(self) -> PolyBeam:
        return PolyBeam.like_fagnoni19()

    def test_beam_is_finite(self):
        beam = self.get_beam()
        check_beam_is_finite(beam)

class TestPerturbedPolyBeam:
    def get_perturbed_beam(
        self, rotation: float, polarized: bool = False
    ) -> PerturbedPolyBeam:
        """
        Elliptical PerturbedPolyBeam.

        This will also test PolyBeam, from which PerturbedPolybeam is derived.
        """
        return PerturbedPolyBeam(
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
            ref_freq=1.0e8,
            spectral_index=-0.6975,
            mainlobe_width=0.3,
            polarized=polarized,
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

    def test_rotation_180_deg(self):
        """Test that rotation by 180 degrees produces the same beam."""
        beam_unrot = self.get_perturbed_beam(rotation=0, polarized=True)
        beamrot = self.get_perturbed_beam(rotation=180.0, polarized=True)
        beam_unrot = evaluate_polybeam(beam_unrot)
        beamrot = evaluate_polybeam(beamrot)
        np.testing.assert_allclose(
            beam_unrot.data_array, -beamrot.data_array, atol=1e-8
        )

    def test_rotation_90_deg(self):
        """Test that rotation by 90 degrees produces a different beam."""
        # Rotate the beam from 0 to 180 degrees, and check that autocorrelation
        # of antenna 0 has approximately the same value when pixel beams are
        # used, and when pixel beams not used (direct beam calculation).
        #rvals = np.linspace(0.0, 180.0, 31, dtype=int)
        beam_unrot = self.get_perturbed_beam(rotation=0)
        beamrot = self.get_perturbed_beam(rotation=90.0)
        beam_unrot = evaluate_polybeam(beam_unrot)
        beamrot = evaluate_polybeam(beamrot)
        assert not np.allclose(beam_unrot.data_array, beamrot.data_array)

    @pytest.mark.parametrize("efield", [True, False])
    def test_finiteness_of_eval(self, efield):
        beam = self.get_perturbed_beam(0)
        check_beam_is_finite(beam)

    def test_perturb_scale_greater_than_one(self):
        # Check that perturb_scale > 1 raises ValueError
        with pytest.raises(ValueError):
            PerturbedPolyBeam(
                perturb_coeffs=np.array([-0.204, -0.486]),
                beam_coeffs=[2.35e-01, -4.2e-01, 2.99e-01],
                perturb_scale=1.1,
            )

    def test_no_perturb_coeffs(self):
        # Check that specifying no perturbation coeffs works
        ppb = PerturbedPolyBeam(
            mainlobe_width=1.0,
            beam_coeffs=[2.35e-01, -4.2e-01, 2.99e-01],
        )
        check_beam_is_finite(ppb)

    def test_specify_freq_perturb_coeffs(self):
        # Check that specifying freq_perturb_coeffs works
        ppb = PerturbedPolyBeam(
            perturb_coeffs=np.array([-0.204, -0.486]),
            mainlobe_width=1.0,
            beam_coeffs=[2.35e-01, -4.2e-01, 2.99e-01],
            freq_perturb_coeffs=[0.0, 0.1],
        )
        check_beam_is_finite(ppb)

    def test_mainlobe_scale(self):
        # Check that specifying mainlobe_scale factor works
        ppb = PerturbedPolyBeam(
            perturb_coeffs=np.array([-0.204, -0.486]),
            mainlobe_width=1.0,
            beam_coeffs=[2.35e-01, -4.2e-01, 2.99e-01],
            freq_perturb_coeffs=[0.0, 0.1],
            mainlobe_scale=1.1,
        )
        check_beam_is_finite(ppb)

    def test_zeropoint(self):
        ppb = PerturbedPolyBeam(
            perturb_coeffs=np.array([-0.204, -0.486]),
            mainlobe_width=1.0,
            beam_coeffs=[2.35e-01, -4.2e-01, 2.99e-01],
            freq_perturb_coeffs=[0.0, 0.1],
            perturb_zeropoint=1.0,
        )
        check_beam_is_finite(ppb)

    def test_bad_freq_perturb_scale(self):
        with pytest.raises(ValueError, match="must be less than 1"):
            PerturbedPolyBeam(
                perturb_coeffs=np.array([-0.204, -0.486]),
                mainlobe_width=1.0,
                beam_coeffs=[2.35e-01, -4.2e-01, 2.99e-01],
                freq_perturb_coeffs=[0.0, 0.1],
                freq_perturb_scale=2.0,
            )

    def test_swapped_pols(self):
        ppbn = PerturbedPolyBeam.like_fagnoni19(x_orientation='north', polarized=True)
        ppbe = PerturbedPolyBeam.like_fagnoni19(x_orientation='east', polarized=True)

        vals_n = ppbn.efield_eval(
            az_array=np.array([0, np.pi/2]),
            za_array = np.array([0, np.pi/2]),
            freq_array=np.array([150e6])
        )
        vals_e = ppbe.efield_eval(
            az_array=np.array([0, np.pi/2]),
            za_array = np.array([0, np.pi/2]),
            freq_array=np.array([150e6])
        )
        assert np.allclose(vals_n[:, 0], vals_e[:, 1])

    def test_normalization(self):
        beam = self.get_perturbed_beam(0.0, polarized=True)
        uvbeam = evaluate_polybeam(beam, nside=64)
        abs_data = np.abs(uvbeam.data_array)
        min_abs = np.min(abs_data, axis=-1)
        max_abs = np.max(abs_data, axis=-1)

        np.testing.assert_allclose(min_abs, 0, atol=1e-5)
        np.testing.assert_allclose(max_abs, 1, atol=0.08)


class TestZernikeBeam:
    def beam(self, peak_normalized: bool = True):
        """
        Zernike polynomial beams with some randomly-chosen coefficients.
        """
        return ZernikeBeam(
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
            peak_normalized=peak_normalized
        )

    def test_peak_normalize(self):
        beam_unnorm = self.beam(False)
        beam_norm = self.beam(True)

        uvbeam_unnorm = evaluate_polybeam(beam_unnorm)
        uvbeam_norm = evaluate_polybeam(beam_norm)

        assert not np.allclose(uvbeam_norm.data_array, uvbeam_unnorm.data_array)

    @pytest.mark.parametrize("peak_normalized", [False, True])
    def test_finitude(self, peak_normalized):
        beam = self.beam(peak_normalized)
        check_beam_is_finite(beam)
