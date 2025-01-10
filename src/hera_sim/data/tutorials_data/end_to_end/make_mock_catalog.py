"""Script for making a mock point source catalog."""

import argparse

import numpy as np
from astropy import units
from astropy.coordinates import EarthLocation, Latitude, Longitude
from pyradiosky import SkyModel
from pyuvsim import create_mock_catalog
from pyuvsim.simsetup import initialize_uvdata_from_params


def make_mock_catalog(
    filename,
    obsparam_file,
    Nsrcs=200,
    src_prefix="mock",
    flux_cut=20,
    sigma=5,
    index_low=-3,
    index_high=-1,
    seed=None,
):
    """Generate and svae a mock point source catalog."""
    # Easiset to load the metadata this way.
    rng = np.random.default_rng(seed)

    temp_uvdata = initialize_uvdata_from_params(str(obsparam_file))[0]
    center_time = np.mean(np.unique(temp_uvdata.time_array))
    ref_freq = np.mean(np.unique(temp_uvdata.freq_array))
    array_location = EarthLocation(*temp_uvdata.telescope_location_lat_lon_alt_degrees)
    sky_model = create_mock_catalog(
        center_time, arrangement="random", Nsrcs=Nsrcs, array_location=array_location
    )[0]
    sky_model_recarray = sky_model.to_recarray()

    # Get the source positions.
    ras = np.array([row[1] for row in sky_model_recarray])
    decs = np.array([row[2] for row in sky_model_recarray])

    # Randomly assign fluxes (whether this is realistic or not).
    ref_fluxes = rng.lognormal(mean=1, sigma=sigma, size=len(ras))
    # Don't include super bright sources.
    ref_fluxes[ref_fluxes > flux_cut] = 1 / ref_fluxes[ref_fluxes > flux_cut]
    # Assign spectral indices.
    indices = rng.uniform(low=index_low, high=index_high, size=len(ras))

    # Actually add in the spectral structure.
    freqs = np.unique(temp_uvdata.freq_array)
    ref_freq = np.mean(freqs)
    scales = np.exp(np.outer(np.log(freqs / ref_freq), indices))
    fluxes = ref_fluxes.reshape(1, -1) * scales
    stokes = np.zeros((4,) + fluxes.shape, dtype=float)
    stokes[0] = fluxes  # Only Stokes-I terms.

    # Finally, make the sky model.
    sky_model = SkyModel(
        name=[f"{src_prefix}{i}" for i in range(len(ras))],
        ra=Longitude(ras, unit="deg"),
        dec=Latitude(decs, unit="deg"),
        stokes=stokes,
        spectral_type="full",
        freq_array=freqs * units.Hz,
        component_type="point",
    )

    # Save the contents as a catalog, then we're done.
    sky_model.write_text_catalog(filename)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str)
    parser.add_argument("obsparam_file", type=str)
    parser.add_argument("--Nsrcs", type=int, default=200)
    parser.add_argument("--src_prefix", type=str, default="mock")
    parser.add_argument("--flux_cut", type=float, default=20)
    parser.add_argument("--sigma", type=float, default=5)
    parser.add_argument("--index_low", type=float, default=-3)
    parser.add_argument("--index_high", type=float, default=-1)
    args = parser.parse_args()
    make_mock_catalog(
        args.filename,
        args.obsparam_file,
        Nsrcs=args.Nsrcs,
        src_prefix=args.src_prefix,
        flux_cut=args.flux_cut,
        sigma=args.sigma,
        index_low=args.index_low,
        index_high=args.index_high,
    )
