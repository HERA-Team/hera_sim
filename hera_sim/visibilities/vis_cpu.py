from __future__ import division
from builtins import range
import numpy as np
from scipy.interpolate import RectBivariateSpline
import healpy

from . import conversions
from .simulators import VisibilitySimulator

from astropy.constants import c


class TransformCache(object):
    
    def __init__(self, limit=10, mode='oldest'):
        
        self.limit = limit
        self.mode = mode
        
        self.keys = []
        self.hits = []
        self.data = []
    
    def retrieve(self, func, args, key):
        if key in self.keys:
            
            # Update hit counter
            idx = self.keys.index(key)
            self.hits[idx] += 1
            
            # Retrieve from cache and return
            return self.data[idx]
        else:
            # Run function
            res = func(*args)
            
            # Store in cache
            self.keys.append(key)
            self.data.append(res)
            self.hits.append(1)
            
            # Clean up cache
            self.clean()
            
            # Return value
            return res
    
    def clean(self):
        if len(self.keys) > self.limit:
            # Remove the oldest entry
            if self.mode == 'oldest':
                del self.keys[0]
                del self.hits[0]
                del self.data[0]
    
    def clear(self):
        # Clear the entire cache
        self.keys = []
        self.data = []
        self.hits = []




class VisCPU(VisibilitySimulator):
    """
    vis_cpu visibility simulator.

    This is a fast, simple visibility simulator that is intended to be
    replaced by vis_gpu. It extends :class:`VisibilitySimulator`.
    """

    def __init__(self, bm_pix=100, use_pixel_beams=True, precision=1,
                 use_gpu=False, mpi_comm=None, split_I=False,
                 az_za_corrections=None, cache_limit=10, **kwargs):
        """
        Parameters
        ----------
        bm_pix : int, optional
            The number of pixels along a side in the beam map when
            converted to (l, m) coordinates. Defaults to 100.
        
        use_pixel_beams : bool, optional
            Whether to use primary beams that have been pixelated onto a 2D 
            grid, or directly evaluate the primary beams using the available 
            UVBeam objects. Default: True.
        
        precision : int, optional
            Which precision level to use for floats and complex numbers. 
            Allowed values:
                - 1: float32, complex64
                - 2: float64, complex128
            Default: 1.
        
        use_gpu : bool, optional
            Whether to use the GPU version of vis_cpu or not. Default: False.
        
        mpi_comm : MPI communicator, optional
            MPI communicator, for parallelization. Default: None.
        
        split_I: bool, optional
            To match pyuvsim, assume the source flux is split between
            XX and YY (half each). Since hera_sim only simulates one 
            unspecified pol, this means halving the source flux.
        
        az_za_corrections: str, optional
            Use pyuvsim/astropy to calculate source positions. Its value
            indicates which approximations to apply. Default: [] (do not use 
            astropy)
        
        cache_limit : int, optional
            Number of time-dependent transform results to cache. Increasing the 
            size of the cache increases the max. memory usage of the object. 
            Default: 10.
        
        **kwargs
            Arguments of :class:`VisibilitySimulator`.
        """
        assert precision in (1,2)
        self._precision = precision
        if precision == 1:
            self._real_dtype = np.float32
            self._complex_dtype = np.complex64
        else:
            self._real_dtype = np.float64
            self._complex_dtype = np.complex128

        if use_gpu and mpi_comm is not None and mpi_comm.Get_size() > 1:
              raise RuntimeError("Can't use multiple MPI processes with GPU (yet)")

        if use_gpu:
            if not use_pixel_beams:
                raise RuntimeError("GPU can only be used with pixel beams (use_pixel_beams=True)") 
            try:
                from hera_gpu.vis import vis_gpu
                self._vis_cpu = vis_gpu
            except ImportError:
                raise ImportError(
                    'GPU acceleration requires hera_gpu (`pip install hera_sim[gpu]`).'
                )
        else:
            self._vis_cpu = vis_cpu

        self.use_gpu = use_gpu 
        self.bm_pix = bm_pix
        self.use_pixel_beams = use_pixel_beams
        self.split_I = split_I
        self.mpi_comm = mpi_comm
        self.az_za_corrections = az_za_corrections
        self.cache_limit = cache_limit
        
        super(VisCPU, self).__init__(**kwargs)

        # Convert some arguments to simpler forms for vis_cpu.
        self.freqs = self.uvdata.freq_array[0]
        
        # Compute coordinate correction terms
        if az_za_corrections:
            # Julian times will be needed
            pol = self.uvdata.get_pols()[0]
            self.jd_times=np.unique(self.uvdata.get_times(pol))
            
            # Set up object 
            self.az_za_transforms = conversions.AzZaTransforms(
                        obstimes=self.jd_times,
                        ra=self.point_source_pos[:, 0],
                        dec=self.point_source_pos[:, 1],
                        precompute=("precompute" in az_za_corrections),
                        use_central_time_values=("level_0" in az_za_corrections),
                        uvbeam_az_correction=("uvbeam_az" in az_za_corrections),
                        astropy=("level_2" in az_za_corrections)
                        )
            
        else: 
            self.az_za_transforms = None
        
        # Get antpos for active antennas only
        #self.antpos = self.uvdata.get_ENU_antpos()[0].astype(self._real_dtype)
        self.ant_list = self.uvdata.get_ants() # ordered list of active ants
        self.antpos = []
        _antpos = self.uvdata.get_ENU_antpos()[0].astype(self._real_dtype)
        for ant in self.ant_list:
            # uvdata.get_ENU_antpos() and uvdata.antenna_numbers have entries 
            # for all telescope antennas, even ones that aren't included in the 
            # data_array. This extracts only the data antennas.
            idx = np.where(ant == self.uvdata.antenna_numbers)
            self.antpos.append(_antpos[idx].flatten())
        self.antpos = np.array(self.antpos)
        

    @property
    def lsts(self):
        """
        Sets LSTs from uvdata if not already set.

        Returns
        -------
        array_like
            LSTs of observations. Shape=(NTIMES,).
        """
        try:
            return self.__lsts
        except AttributeError:
            self.__lsts = self.uvdata.lst_array[::self.uvdata.Nbls]

            return self.__lsts

    def validate(self):
        """Checks for correct input format."""
        super(VisCPU, self).validate()

        # This one in particular requires that every baseline is used!
        N = len(self.uvdata.get_ants())
        # N(N-1)/2 unique cross-correlations + N autocorrelations.
        if len(self.uvdata.get_antpairs()) != N * (N + 1) / 2:
            raise ValueError("VisCPU requires using every pair of antennas, "
                             "but the UVData object does not comply.")

        if (len(self.uvdata.data_array) != len(self.uvdata.get_antpairs())
                * len(self.lsts)):
            raise ValueError("VisCPU requires that every baseline uses the "
                             "same LSTS.")

        # Parse az_za_corrections. 
        if self.az_za_corrections:
            if isinstance(self.az_za_corrections, str):
                self.az_za_corrections = [ self.az_za_corrections ]
            allowed = [ "level_0", "level_1", "level_2" ]
            extra = [ "uvbeam_az", "precompute" ]
            for azt in self.az_za_corrections:
                if azt not in allowed+extra:
                    raise ValueError("Invalid az_za_correction option: \""+str(azt)+"\"")
            
            # Check only one of min/max/astropy
            num = 0
            for a in allowed:
                if a in self.az_za_corrections: num += 1
            if num > 1:
                raise RuntimeError("Only one of "+str(allowed)+" can be specified in az_za_corrections")
            elif num == 0:
                raise RuntimeError("One of "+str(allowed)+" must be specified in az_za_corrections")
        
        # Check to make sure enough beams are specified
        if not self.use_pixel_beams:
            for ant in self.uvdata.get_ants():
                assert len(np.where(self.beam_ids == ant)[0]), \
                       "No beam found for antenna %d" % ant
        
    def get_beam_lm(self):
        """
        Obtain the beam pattern in (l,m) co-ordinates for each antenna.

        Returns
        -------
        array_like
            The beam pattern in (l,m) for each antenna.
            Shape=(NANT, BM_PIX, BM_PIX).

        Notes
        -----
            Due to using the verbatim :func:`vis_cpu` function, the beam
            cube must have an entry for each antenna, which is a bit of
            a waste of memory in some cases. If this is changed in the
            future, this method can be modified to only return one
            matrix for each beam.
        """
        return np.asarray([
            conversions.uvbeam_to_lm(
                self.beams[np.where(self.beam_ids == ant)[0][0]], self.freqs, self.bm_pix
            ) for ant in self.ant_list
        ])

    def get_diffuse_crd_eq(self):
        """
        Calculate equatorial coords of HEALPix sky pixels (Cartesian).

        Returns
        -------
        array_like of self._real_dtype
            The equatorial co-ordinates of each pixel.
            Shape=(12*NPIX^2, 3).
        """
        diffuse_eq = conversions.healpix_to_crd_eq(self.sky_intensity[0])
        return diffuse_eq.astype(self._real_dtype)

    def get_point_source_crd_eq(self):
        """
        Calculate approximate HEALPix map of point sources.

        Returns
        -------
        array_like
            equatorial coordinates of Healpix pixels, in Cartesian
            system. Shape=(3, NPIX).
        """
        ra, dec = self.point_source_pos.T
        return np.asarray([np.cos(ra)*np.cos(dec), np.cos(dec)*np.sin(ra),
                         np.sin(dec)])

    def get_eq2tops(self):
        """
        Calculate transformations from equatorial to topocentric coords.

        Returns
        -------
        array_like of self._real_dtype
            The set of 3x3 transformation matrices converting equatorial
            to topocenteric co-ordinates at each LST.
            Shape=(NTIMES, 3, 3).
        """

        sid_time = self.lsts
        eq2tops = np.empty((len(sid_time), 3, 3), dtype=self._real_dtype)

        for i, st in enumerate(sid_time):
            dec = self.uvdata.telescope_location_lat_lon_alt[0]
            eq2tops[i] = conversions.eq2top_m(-st, dec)

        return eq2tops

    def _base_simulate(self, crd_eq, I):
        """
        Calls :func:vis_cpu to perform the visibility calculation.

        Returns
        -------
        array_like of self._complex_dtype
            Visibilities. Shape=self.uvdata.data_array.shape.
        """
        # Setup MPI info if enabled
        if self.mpi_comm is not None:
            myid = self.mpi_comm.Get_rank()
            nproc = self.mpi_comm.Get_size()
        
        # Convert equatorial to topocentric coords
        eq2tops = self.get_eq2tops()
        
        # Get pixelized beams if required
        if self.use_pixel_beams:
            beam_lm = self.get_beam_lm()
        else:
            beam_list = [self.beams[np.where(self.beam_ids == ant)[0][0]] 
                         for ant in self.ant_list]
        
        # Match pyuvsim where pol XX/YY is half I
        if self.split_I: I /= 2.
            
        visfull = np.zeros_like(self.uvdata.data_array,
                                dtype=self._complex_dtype)
        
        # Create cache for coordinate transforms
        # N.B. There will be one cache per MPI worker
        cache = TransformCache(limit=self.cache_limit)
        
        for i, freq in enumerate(self.freqs):
            
            # Divide tasks between MPI workers if needed
            if self.mpi_comm is not None:
                if i % nproc != myid: continue
            
            if self.use_pixel_beams:
                # Use pixelized primary beams
                vis = self._vis_cpu(
                    antpos=self.antpos,
                    freq=freq,
                    eq2tops=eq2tops,
                    crd_eq=crd_eq,
                    I_sky=I[i],
                    bm_cube=beam_lm[:, i],
                    precision=self._precision
                )
            else:
                # Use UVBeam objects directly
                vis = self._vis_cpu(
                    antpos=self.antpos,
                    freq=freq,
                    eq2tops=eq2tops,
                    crd_eq=crd_eq,
                    I_sky=I[i],
                    beam_list=beam_list,
                    precision=self._precision,
                    # These for astropy az/za calcs if used
                    point_source_pos=self.point_source_pos,
                    az_za_transforms=self.az_za_transforms,
                    transform_cache=cache
                )
            
            # Extract upper triangle (i.e. get each bl only once)
            indices = np.triu_indices(vis.shape[1])
            vis_upper_tri = vis[:, indices[0], indices[1]]
            
            # Unpack into the same array ordering as the target UVData array
            order = 'F' if np.isfortran(visfull) else 'C'
            visfull[:, 0, i, 0] = vis_upper_tri.flatten(order=order)
        
        # Reduce visfull array if in MPI mode
        if self.mpi_comm is not None:
            from mpi4py.MPI import SUM
            _visfull = np.zeros(visfull.shape, dtype=visfull.dtype)
            self.mpi_comm.Reduce(visfull, _visfull, op=SUM, root=0)
            if myid == 0:
                return _visfull
            else:
                return 0 # workers return 0
            
        return visfull

    def _simulate_diffuse(self):
        """
        Simulate diffuse sources.

        Returns
        -------
        array_like
            Visibility from point sources.
            Shape=self.uvdata.data_array.shape.
        """
        crd_eq = self.get_diffuse_crd_eq()
        # Multiply intensity by pix area because the algorithm doesn't.
        return self._base_simulate(
            crd_eq,
            self.sky_intensity * healpy.nside2pixarea(self.nside)
        )

    def _simulate_points(self):
        """
        Simulate point sources.

        Returns
        -------
        array_like
            Visibility from diffuse sources.
            Shape=self.uvdata.data_array.shape.
        """
        crd_eq = self.get_point_source_crd_eq()
        return self._base_simulate(crd_eq, self.point_source_flux)

    def _simulate(self):
        """
        Simulate diffuse and point sources.

        Returns
        -------
        array_like
            Visibility from all sources.
            Shape=self.uvdata.data_array.shape.
        """
        vis = 0
        if self.sky_intensity is not None:
            vis += self._simulate_diffuse()
        if self.point_source_flux is not None:
            vis += self._simulate_points()
        return vis


def vis_cpu(antpos, freq, eq2tops, crd_eq, I_sky, bm_cube=None, beam_list=None,
            precision=1, point_source_pos=None, az_za_transforms=None, 
            transform_cache=None):
    """
    Calculate visibility from an input intensity map and beam model.

    Provided as a standalone function.

    Parameters
    ----------
    antpos : array_like
        Antenna position array. Shape=(NANT, 3).
    
    freq : float
        Frequency to evaluate the visibilities at [GHz].
    
    eq2tops : array_like
        Set of 3x3 transformation matrices converting equatorial
        coordinates to topocentric at each
        hour angle (and declination) in the dataset.
        Shape=(NTIMES, 3, 3).
    
    crd_eq : array_like
        Equatorial coordinates of Healpix pixels, in Cartesian system.
        Shape=(3, NPIX).
    
    I_sky : array_like
        Intensity distribution on the sky,
        stored as array of Healpix pixels. Shape=(NPIX,).
    
    bm_cube : array_like, optional
        Pixelized beam maps for each antenna. Shape=(NANT, BM_PIX, BM_PIX).
    
    beam_list : list of UVBeam, optional
        If specified, evaluate primary beam values directly using UVBeam 
        objects instead of using pixelized beam maps (`bm_cube` will be ignored 
        if `beam_list` is not None).
    
    precision : int, optional
        Which precision level to use for floats and complex numbers. 
        Allowed values:
            - 1: float32, complex64
            - 2: float64, complex128
        Default: 1.
    
    point_source_pos: 2-D ndarray
        Source catalog ra/dec needed for az_za_transforms. Shape (nsource, 2)
        Not used if az_za_transforms is None.
    
    az_za_transforms: an AzZaTransforms object (conversions.py)
        If not None, used to produce more accurate az/za and crd_top values.
        point_source_pos must also be present.
    
    transform_cache : TransformCache object, optional
        Cache the results of the angular coordinate transforms. Default: None.
    
    Returns
    -------
    array_like
        Visibilities. Shape=(NTIMES, NANTS, NANTS).
    """
    assert precision in (1,2)
    if precision == 1:
        real_dtype=np.float32
        complex_dtype=np.complex64
    else:
        real_dtype=np.float64
        complex_dtype=np.complex128
    
    if bm_cube is None and beam_list is None:
        raise RuntimeError("One of bm_cube/beam_list must be specified")
    if bm_cube is not None and beam_list is not None:
        raise RuntimeError("Cannot specify both bm_cube and beam_list")

    nant, ncrd = antpos.shape
    assert ncrd == 3, "antpos must have shape (NANTS, 3)."
    ntimes, ncrd1, ncrd2 = eq2tops.shape
    assert ncrd1 == 3 and ncrd2 == 3, "eq2tops must have shape (NTIMES, 3, 3)."
    ncrd, npix = crd_eq.shape
    assert ncrd == 3, "crd_eq must have shape (3, NPIX)."
    assert I_sky.ndim == 1 and I_sky.shape[0] == npix, \
        "I_sky must have shape (NPIX,)."
    
    if beam_list is None:
        bm_pix = bm_cube.shape[-1]
        assert bm_cube.shape == (
            nant,
            bm_pix,
            bm_pix,
        ), "bm_cube must have shape (NANTS, BM_PIX, BM_PIX)."
    else:
        assert len(beam_list) == nant, "beam_list must have length nant"

    # Intensity distribution (sqrt) and antenna positions. Does not support
    # negative sky.
    Isqrt = np.sqrt(I_sky).astype(real_dtype)
    antpos = antpos.astype(real_dtype)

    ang_freq = 2 * np.pi * freq

    # Empty arrays: beam pattern, visibilities, delays, complex voltages.
    A_s = np.empty((nant, npix), dtype=real_dtype)
    vis = np.empty((ntimes, nant, nant), dtype=complex_dtype)
    tau = np.empty((nant, npix), dtype=real_dtype)
    v = np.empty((nant, npix), dtype=complex_dtype)
    crd_eq = crd_eq.astype(real_dtype)
    
    # Precompute splines is using pixelized beams
    if beam_list is None:
        bm_pix_x = np.linspace(-1, 1, bm_pix)
        bm_pix_y = np.linspace(-1, 1, bm_pix)
    
        splines = []
        for i in range(nant):
            # Linear interpolation of primary beam pattern.
            spl = RectBivariateSpline(bm_pix_y, bm_pix_x, bm_cube[i], kx=1, ky=1)
            splines.append(spl)
            
    # Loop over time samples
    for t, eq2top in enumerate(eq2tops.astype(real_dtype)):
        
        # Primary beam response
        if beam_list is None:
            tx, ty, tz = crd_top = np.dot(eq2top, crd_eq)
            
            # Primary beam pattern using pixelized primary beam
            for i in range(nant):
                A_s[i] = splines[i](ty, tx, grid=False)
                # TODO: Try using a log-space beam for accuracy!
        else:
            # Primary beam pattern using direct interpolation of UVBeam object
            if az_za_transforms:
                # Supplies more accurate az/za and crd_top (overwrites crd_top)
                # Use cached results, or compute directly
                if transform_cache is not None:
                    if t == 0:
                        print("Cache in use")
                    az, za, crd_top = transform_cache.retrieve(
                                               func=az_za_transforms.transform, 
                                               args=(point_source_pos[:, 0], 
                                                     point_source_pos[:, 1], 
                                                     t),
                                               key=t)
                else:
                    az, za, crd_top = az_za_transforms.transform(
                                                        point_source_pos[:, 0], 
                                                        point_source_pos[:, 1], 
                                                        t )
                tz = crd_top[2] # pick out 3rd column
            else:
                tx, ty, tz = crd_top = np.dot(eq2top, crd_eq)
                az, za = conversions.lm_to_az_za(ty, tx) # FIXME: Order of tx, ty
            
            for i in range(nant):
                interp_beam = beam_list[i].interp(az, za, np.atleast_1d(freq))[0]
                A_s[i] = interp_beam[0,0,1] # FIXME: assumes xx pol for now
        
        A_s = np.where(tz > 0, A_s, 0)

        # Calculate delays, where tau = (b * s) / c
        np.dot(antpos, crd_top, out=tau)
        tau /= c.value
        
        # Component of complex phase factor for one antenna 
        # (actually, b = (antpos1 - antpos2) * crd_top / c; need dot product 
        # below to build full phase factor for a given baseline)
        np.exp(1.j * (ang_freq * tau), out=v)

        # Complex voltages.
        v *= A_s * Isqrt

        # Compute visibilities using product of complex voltages (upper triangle).
        for i in range(len(antpos)):
            np.dot(v[i:i+1].conj(), v[i:].T, out=vis[t, i:i+1, i:])

    return vis
