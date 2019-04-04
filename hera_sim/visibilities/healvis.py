"""
Wrapper around the healvis package for producing visibilities from
healpix maps.
"""

from .simulators import VisibilitySimulator
from healvis.observatory import Observatory
from cached_property import cached_property
import numpy as np

class HealVis(VisibilitySimulator):
    def __init__(self, fov=2*np.pi, **kwargs):
        self.fov = fov
        super().__init__(**kwargs)

    @cached_property
    def observatory(self):
        """A healvis :class:`healvis.observatory.Observatory` instance"""
        obs = Observatory(latitude=self.latitude, longitude=0, freqs=self.freq)
        obs.set_pointings(self.lsts)
        obs.set_fov(self.fov)

    def simulate(self):2
        """
        Runs the cpu_vis algorithm.

        Returns:
            array_like, shape(NTIMES, NANTS, NANTS): visibilities

        Notes:
            This routine does not support negative intensity values on the sky.
        """
        return vis_cpu(
            antpos=self.antpos.astype(self._real_dtype),
            freq=self.freq,
            eq2tops=self.get_eq2tops(),
            crd_eq=self.get_crd_eq(),
            I_sky=self.sky_intensity,
            bm_cube=self.get_beam_lm(),
            real_dtype=self._real_dtype,
            complex_dtype=self._complex_dtype
        )