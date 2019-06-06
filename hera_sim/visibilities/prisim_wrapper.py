import os

import numpy as np
import pyuvdata
from astropy import units as u
from astropy.coordinates import SkyCoord, FK5, AltAz, EarthLocation
from astropy.io import fits
from astropy.time import Time
from cached_property import cached_property

try:
    from astroutils import catalog, geometry, mathops
    from prisim import interferometry as intf
except ImportError:
    raise ImportError("To use the PRISim wrapper, you will need to install prisim!")

from .simulators import VisibilitySimulator

PRISIM_PATH = prisim.__path__[0]


class PRISim(VisibilitySimulator):
    diffuse_ability = False

    def __init__(self, aux_file_loc='prisim_roi_file.fits', fov_radius=90.0,
                 beam_pol='x', beam_interp='cubic', precision='single', **kwargs):
        self.fov_radius = fov_radius
        self.beam_pol = beam_pol
        self.beam_interp = beam_interp
        self.aux_file_loc = aux_file_loc
        self.precision = precision

        assert self.precision.lower() in ['single', 'double']

        super(PRISim, self).__init__(**kwargs)

    @property
    def default_config_path(self):
        """This is the default path to the PRISim yaml config"""
        return os.path.join(PRISIM_PATH, "examples", "simparms", "defaultparms.yaml")

    @cached_property
    def interferometer(self):
        """
        An instance of `prisim.interferometry.InterferometerArray`
        """
        intf.InterferometerArray(
            labels=np.asarray(self.uvdata.get_ant_pairs(), dtype=[("A2", int), ("A1", int)]),
            baselines=self.uvdata.get,  # ask bryna
            channels=self.uvdata.freq_array[0],
            telescope=None,  # TODO: new class
            latitude=np.degrees(self.uvdata.telescope_lat_lon_alt[0]),
            longitude=np.degrees(self.uvdata.telescope_lat_lon_alt[1]),
            altitude=self.uvdata.telescope_lat_lon_alt[2],
            pointing_coords="altaz"
            # blgroupinfo=None,
        )

    def validate(self):
        super(PRISim, self).validate()

        assert self.n_beams == 1, "PRISim assumes a single beam for all antennae"
        assert np.all(self.uvdata.integration_time == self.uvdata.integration_time[
            0]), "PRISim assumes a single integration time for all times"

    @cached_property
    def times(self):
        return np.unique(self.uvdata.time_array)

    @cached_property
    def _sky_coords(self):
        """An astropy SkyCoord"""
        return SkyCoord(
            ra=np.degrees(self.point_source_pos[:, 0]) * u.deg,
            dec=np.degrees(self.point_source_pos[:, 1]) * u.deg,
            frame='fk5',
            equinox=Time(2000.0, format='jyear', scale='utc'),
        ).transform_to(FK5(equinox=self.times[0]))

    @cached_property
    def sky_model(self):
        """A PRISim SkyModel"""
        return catalog.SkyModel(
            name='hera_sim',
            frequency=self.sky_freqs,
            location=np.array([
                self._sky_coords.ra.deg,
                self._sky_coords.dec.deg
            ]).T,
            spec_type='spectrum',
            spectrum=self.point_source_flux.T,
            spec_parms={},  # only useful if specturm is a function.
            src_shape=np.zeros((len(self.point_source_pos), 3)),
        )

    @cached_property
    def _earth_location(self):
        """EarthLocation object for the array"""
        return EarthLocation(
            lat=self.uvdata.telescope_location_lat_lon_alt_degree[0] * u.deg,
            lon=self.uvdata.telescope_location_lat_lon_alt_degree[1] * u.deg,
            height=self.uvdata.telescope_location_lat_lon_alt[2] * u.m,
        )

    @cached_property
    def _pointing_centres_radec(self):
        radecs = [
            AltAz(
                alt=90.0 * u.deg,
                az=270 * u.deg,
                obstime=t,
                location=self._earth_location
            ).transform_to(FK5(equinox=self.times[0])) for t in self.times
        ]

        return np.array([[x.ra.deg, x.dec.deg] for x in radecs])

    @cached_property
    def get_skymodel_indices(self):
        """
        Return the indices in the SkyModel that fall within the fov_radius
        around the pointings (as function of time).

        Returns
        -------
        list of lists: each inner list corresponds to an instance of time, and
            gives the indices in the sky model in the fov for that time.
        """
        m1, m2, _ = geometry.spherematch(
            self._pointing_centres_radec[:, 0],
            self._pointing_centres_radec[:, 1],
            self._sky_coords.ra.deg,
            self._sky_coords.dec.deg,
            matchrad=self.fov_radius,
            nnearrest=0,
            maxmatches=0
        )

        m1 = np.array(m1)
        m2 = np.array(m2)

        return [m2[np.where(m1 == j)[0]] for j in range(len(self.times))]

    @cached_property
    def _prisim_telescope(self):
        if isinstance(self.beams[0], pyuvdata.UVBeam):
            shape = None
            size = None
        else:
            if self.beams[0].type == "uniform":
                shape = 'delta'
                size = 0
            elif self.beams[0].type == "airy":
                shape = 'dish'
                size = self.beams[0].diameter
            elif self.beams[0].type == 'gaussian':
                shape = 'gaussian'
                size = self.beams[0].diameter

        return {
            'shape': shape,
            "size": size,
            "orientation": np.array([90.0, 270.0]),
            "ocoords": 'altaz',
            "groundplane": None,
            'latitude': self._earth_location.lat.deg,
            'longitude': self._earth_location.lon.deg,
            'altitude': self._earth_location.height.to("m").value,
        }

    def run_region_of_interest(self):
        """Run calculations for Region of Interest, and save a file for later use"""
        roi = intf.ROI_parameters()

        indices = self.get_skymodel_indices

        for i, (ind, time) in enumerate(zip(indices, self.times)):
            if len(ind) == 0:
                roi.append_settings(
                    None, self.uvdata.freq_array[0] / 1e9,
                    telescope=self._prisim_telescope,
                    freq_scale="GHz"
                )
            else:
                src_altaz = self._sky_coords[ind].transform_to(
                    AltAz(
                        obstime=time, location=self._earth_location,
                    )
                )

                src_altaz = np.array([src_altaz.alt.deg, src_altaz.az.deg]).T
                assert src_altaz[:, 0] >= 0, "some sources are below horizon!"

                pbinfo = {}
                pbinfo['pointing_coords'] = 'altaz'
                pbinfo['pointing_centre'] = np.array([90., 270.0])

                roiinfo = {}
                roiinfo['ind'] = ind

                if isinstance(self.beams[0], pyuvdata.UVBeam):
                    beam = self.beams[0].data_array[0, 0, 0 if self.beam_pol == 'x' else 1, :, :].T
                    freqs = self.beams[0].freq_array.ravel()
                    theta_phi = np.array([
                        np.pi / 2 - np.radians(src_altaz[:, 0]),
                        np.radians(src_altaz[:, 1])
                    ])
                    interp_logbeam = mathops.healpix_interp_along_axis(
                        np.log10(beam),
                        theta_phi=theta_phi,
                        inloc_axis=freqs,
                        outloc_axis=self.uvdata.freq_array[0],
                        axis=1,
                        kind=self.beam_interp
                    )
                    interp_logbeam -= np.log10(beam.max())
                    roiinfo['pbeam'] = 10 ** interp_logbeam

                else:
                    # Analytic case
                    roiinfo['pbeam'] = None

                roiinfo['radius'] = self.fov_radius
                roiinfo['center'] = self._pointing_centres_radec[i].reshape(1, -1)
                roiinfo['center_coords'] = 'radec'

                roi.append_settings(
                    self.sky_model, self.uvdata.freq_array[0] / 1e9,
                    pinfo=pbinfo,
                    lst=Time(
                        time, format='jd', scale='utc',
                        location=self._earth_location
                    ).sidereal_time('apparent').deg,
                    time_jd=time,
                    roi_info=roiinfo,
                    telescope=self._prisim_telescope,
                    freq_scale="GHz"
                )

            roi.save(self.aux_file_loc, tabtype="BinTableHDU",
                     overwrite=True, verbose=True)

    def _simulate(self):
        """
        Runs the healvis algorithm
        """
        # First run the RegionOfInterest

        for j, (ind, time) in enumerate(zip(self.get_skymodel_indices, self.times)):
            if len(ind) > 0:
                roi_ind_snap = fits.getdata(
                    self.aux_file_loc,
                    extname="IND_{0d}".format(j),
                    memmap=False
                )

                roi_pbeam_snap = fits.getdata(
                    self.aux_file_loc,
                    extname="PB_{0d}".format(j),
                    memmap=False
                )
            else:
                roi_ind_snap = np.array([])
                roi_pbeam_snap = np.array([])

            self.interferometer.observe(
                timeobj=Time(time, format='jd'),
                Tsysinfo={"Tnet": 0},
                bandpass=self.beams[0].bandpass_array,
                pointing_center=np.array([90.0, 270.0]),
                skymodel=self.sky_model,
                t_acc=self.uvdata.integration_time[0],
                pb_info={
                    "pointing_centre": np.array([90., 270])  # hard-coded to zenith
                    "pointing_coords": 'altaz'
                },
                brightness_units="Jy",
                roi_info=dict(ind=roi_ind_snap, pbeam=roi_pbeam_snap),
                roi_radius=self.fov_radius,
                roi_center=None,
                gradient_mode=None,
                memsave=self.precision.lower() == 'single',
                vmemavail=None
            )

        # Reshape array into UVData.data_array format
        vis = self.interferometer.skyvis_freq.conj()

        # (Nbls, Nfreqs, Ntimes) -> (Ntimes, Nbls, Nfreqs) ->
        # (Nblts, Nfreqs, Nspws=1, Npols=1) ->
        # (Nblts, Nspws=1, Nfreqs, Npols=1)
        return np.transpose(
            np.transpose(vis['noiseless'], (2, 0, 1)).reshape(
                self.uvdata.Nblts, self.uvdata.Nfreqs, (0, 2, 1, 3))
        )
