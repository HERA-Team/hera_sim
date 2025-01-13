from pathlib import Path

import astropy_healpix as aph
import numpy as np
import pytest
from astropy import units
from astropy.coordinates import Latitude, Longitude
from astropy.units import rad, sday
from pyradiosky import SkyModel
from pyuvdata.uvbeam import UVBeam

from hera_sim import io
from hera_sim.defaults import defaults

DATA_PATH = Path(__file__).parent.parent / "testdata" / "hera-sim-vis-config"


NTIMES = 10
NPIX = 12 * 16**2
NFREQ = 5


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
        frame="icrs",
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

    sky = create_uniform_sky(np.unique(uvdata2.freq_array), nbase=nbase)

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
        frame="icrs",
    )


@pytest.fixture
def uvdata():
    defaults.set("h1c")
    return io.empty_uvdata(
        Nfreqs=NFREQ,
        integration_time=sday.to("s") / NTIMES,
        Ntimes=NTIMES,
        array_layout={0: (0, 0, 0)},
        start_time=2456658.5,
        conjugation="ant1<ant2",
        polarization_array=["xx", "yy", "xy", "yx"],
    )


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
        array_layout={0: (0, 0, 0)},
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

@pytest.fixture(scope='module')
def uvbeam() -> UVBeam:
    return UVBeam.from_file(DATA_PATH / "NF_HERA_Dipole_small.fits")
