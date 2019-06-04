"""
Wrapper around the healvis package for producing visibilities from
healpix maps.
"""

import numpy as np
from cached_property import cached_property
from .simulators import VisibilitySimulator
from astropy import constants as cnst
from pyuvdata.uvbeam import UVBeam
from pyuvsim import analyticbeam as ab

class PRISim(VisibilitySimulator):
    point_source_ability = False

    def __init__(self, fov=180, nprocesses=1, sky_ref_chan=0, **kwargs):
        self.fov = fov
        self._nprocs = nprocesses
        self._sky_ref_chan = sky_ref_chan

        # A bit of a hack here because healvis uses its own AnalyticBeam,
        # and doesn't check if you are using pyuvsim's one. This should be fixed.
        if "beams" not in kwargs:
            kwargs['beams'] = [AnalyticBeam("uniform")]

        super(HealVis, self).__init__(**kwargs)

    def validate(self):
        super(HealVis, self).validate()

        assert self.n_beams == 1, "HealVis assumes a single beam for all antennae"

    @cached_property
    def sky_model(self):
        """A SkyModel, compatible with healvis, constructed from the input healpix sky model"""
        sky = SkyModel()
        sky.Nside = healpy.npix2nside(self.sky_intensity.shape[1])
        sky.freqs = self.sky_freqs
        sky.Nskies = 1
        sky.ref_chan = self._sky_ref_chan

        # convert from Jy/sr to K
        intensity = 10**-26 * self.sky_intensity.T
        intensity *= (cnst.c.to("m/s").value/self.sky_freqs)**2 / (2 * cnst.k_B.value)

        sky.data = intensity[np.newaxis, :, :]
        sky._update()

        return sky

    @cached_property
    def observatory(self):
        """A healvis :class:`healvis.observatory.Observatory` instance"""
        return setup_observatory_from_uvdata(
            self.uvdata, fov=self.fov, beam=self.beams[0]
        )

    def _simulate(self):
        """
        Runs the healvis algorithm
        """
        visibility = []
        for pol in self.uvdata.get_pols():
            # calculate visibility
            visibility.append(
                self.observatory.make_visibilities(self.sky_model, Nprocs=self._nprocs, beam_pol=pol)[0]
            )

        visibility = np.moveaxis(visibility, 0, -1)

        return visibility[:, 0][:, np.newaxis, :, :]


# !python

import yaml, argparse, ast, warnings
import numpy as NP
from astropy.io import ascii
from astropy.time import Time
import prisim
from os import path
prisim_path = prisim.__path__[0] + '/'

def create_prisim_parms(prisim_project, prisim_rootdir, prisim_simid, uvdata, beams, beam_ids):
    prisim_parms = {}

    # I/O and directory structure

    prisim_parms['dirstruct']['rootdir'] = prisim_rootdir
    prisim_parms['dirstruct']['project'] = prisim_project
    prisim_parms['dirstruct']['simid'] = prisim_simid

    # Telescope parameters
    prisim_tele = prisim_parms['telescope'] = {}
    prisim_tele['latitude'], prisim_tele['longitude'], prisim_tele['altitude'] = uvdata.telescope_lat_lon_alt

    # Array parameters
    prisim_parms['array']['redundant'] = True
    prisim_parms['array']['layout'] = None # TODO: maybe this should be something different?
    prisim_parms['array']['file'] = pyuvsim_telescope_parms['array_layout']
    prisim_parms['array']['filepathtype'] = 'custom'
    prisim_parms['array']['parser']['data_start'] = 1
    prisim_parms['array']['parser']['label'] = 'Name'
    prisim_parms['array']['parser']['east'] = 'E'
    prisim_parms['array']['parser']['north'] = 'N'
    prisim_parms['array']['parser']['up'] = 'U'

    # Antenna power pattern parameters
    # TODO: is only a single beam possible?
    if isinstance(beams[0], ab.AnalyticBeam):
        if beams[0].type == "uniform":
            prisim_parms['antenna']['shape'] = 'delta'
        elif beams[0].type == "uniform":
            prisim_parms['antenna']['shape'] = 'gaussian'
            prisim_parms['antenna']['size'] = beams[0].diameter
        elif beams[0].type == "airy":
            prisim_parms['antenna']['shape'] = 'dish'
            prisim_parms['antenna']['size'] = beams[0].diameter

        prisim_parms['beam']['use_external'] = False
        prisim_parms['beam']['file'] = None
    elif isinstance(beams[0], UVBeam):
        # can it directly take a UVBeam?
        prisim_parms['beam']['use_external'] = True
        prisim_parms['beam']['file'] = pyuvsim_telescope_config['beam_paths'][0]
        prisim_parms['beam']['filepathtype'] = 'custom'
        prisim_parms['beam']['filefmt'] = 'UVBeam'

    # Bandpass parameters
    assert np.allclose(np.diff(np.diff(uvdata.freq_array[:, 0])), 0)  # ensure that uvdata has regular freqs.
    prisim_parms['bandpass']['freq_resolution'] = uvdata.freq_array[1] - uvdata.freq_array[0]
    prisim_parms['bandpass']['nchan'] = np.unique(uvdata.freq_array).size

    if prisim_parms['bandpass']['nchan'] == 1:
        warnings.warn(
            'Single channel simulation is not supported currently in PRISim. Request at least two frequency channels.')

    pyuvsim_start_freq = uvdata.freq_array.min()
    prisim_parms['bandpass']['freq'] = np.sort(np.unique(uvdata.freq_array))

    # Observing parameters

    prisim_parms['obsparm']['n_acc'] = np.unique(uvdata.lst_array)
    prisim_parms['obsparm']['t_acc'] = uvdata.integration_time
    prisim_parms['obsparm']['obs_mode'] = 'drift'

    prisim_parms['pointing']['jd_init'] = uvdata.time_array.min()
    prisim_parms['obsparm']['obs_date'] = \
    Time(prisim_parms['pointing']['jd_init'], scale='utc', format='jd').iso.split(' ')[0].replace('-', '/')

    prisim_parms['pointing']['lst_init'] = None # TODO: why is this None?
    prisim_parms['pointing']['drift_init']['alt'] = 90.0
    prisim_parms['pointing']['drift_init']['az'] = 270.0
    prisim_parms['pointing']['drift_init']['ha'] = None
    prisim_parms['pointing']['drift_init']['dec'] = None

    # TODO: Sky model

    prisim_parms['skyparm']['model'] = 'custom'
    prisim_parms['catalog']['filepathtype'] = 'custom'
    prisim_parms['catalog']['custom_file'] = pyuvsim_parms['sources']['catalog'].split('.txt')[0] + '_prisim.txt'
    pyuvsim_catalog = ascii.read(pyuvsim_parms['sources']['catalog'], comment='#', header_start=0, data_start=1)
    epoch = ''
    for colname in pyuvsim_catalog.colnames:
        if 'RA' in colname:
            ra_colname = colname
            ra_deg = pyuvsim_catalog[colname].data
            epoch = ra_colname.split('_')[1].split()[0][1:]
        if 'Dec' in colname:
            dec_colname = colname
            dec_deg = pyuvsim_catalog[colname].data
        if 'Flux' in colname:
            fint = pyuvsim_catalog[colname].data.astype(NP.float)
        if 'Frequency' in colname:
            ref_freq = pyuvsim_catalog[colname].data.astype(NP.float)

    spindex = NP.zeros(fint.size, dtype=NP.float)
    majax = NP.zeros(fint.size, dtype=NP.float)
    minax = NP.zeros(fint.size, dtype=NP.float)
    pa = NP.zeros(fint.size, dtype=NP.float)
    prisim_parms['skyparm']['epoch'] = epoch
    prisim_parms['skyparm']['flux_unit'] = 'Jy'
    prisim_parms['skyparm']['flux_min'] = None
    prisim_parms['skyparm']['flux_max'] = None
    prisim_parms['skyparm']['custom_reffreq'] = float(ref_freq[0]) / 1e9

    ascii.write([ra_deg, dec_deg, fint, spindex, majax, minax, pa], prisim_parms['catalog']['custom_file'],
                names=['RA', 'DEC', 'F_INT', 'SPINDEX', 'MAJAX', 'MINAX', 'PA'], delimiter='    ', format='fixed_width',
                formats={'RA': '%11.7f', 'DEC': '%12.7f', 'F_INT': '%10.4f', 'SPINDEX': '%8.5f', 'MAJAX': '%8.5f',
                         'MINAX': '%8.5f', 'PA': '%8.5f'}, bookend=False, overwrite=True)

    # Save format parameters
    prisim_parms['save_formats']['npz'] = False
    prisim_parms['save_formats']['uvfits'] = False
    prisim_parms['save_formats']['uvh5'] = True