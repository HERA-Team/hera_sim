"""
A number of mappings which may be useful for visibility simulators.
"""
import healpy
import numpy as np


def uvbeam_to_lm(uvbeam, freqs, n_pix_lm=63, **kwargs):
    """
    Convert a UVbeam to a uniform (l,m) grid

    Parameters
    ----------
    uvbeam : UVBeam object
        Beam to convert to an (l, m) grid.
    freqs : array_like
        Frequencies to interpolate to in [Hz]. Shape=(NFREQS,).
    n_npix_lm : int, optional
        Number of pixels for each side of the beam grid. Default is 63.

    Returns
    -------
    ndarray
        The beam map cube. Shape=(NFREQS, BEAM_PIX, BEAM_PIX).
    """

    L = np.linspace(-1, 1, n_pix_lm, dtype=np.float32)
    L, m = np.meshgrid(L, L)
    L = L.flatten()
    m = m.flatten()

    lsqr = L ** 2 + m ** 2
    n = np.where(lsqr < 1, np.sqrt(1 - lsqr), 0)

    az = -np.arctan2(m, L)
    za = np.pi/2 - np.arcsin(n)

    efield_beam = uvbeam.interp(az, za, freqs, **kwargs)[0]
    efieldXX = efield_beam[0, 0, 1]

    # Get the relevant indices of res
    bm = np.zeros((len(freqs), len(L)))

    bm = efieldXX

    if np.max(bm) > 0:
        bm /= np.max(bm)

    return bm.reshape((len(freqs), n_pix_lm, n_pix_lm))


def eq2top_m(ha, dec):
    """
    Calculates the equatorial to topocentric conversion matrix.
    
    Conversion at a given hour angle (ha) and declination (dec). Ripped 
    straight from aipy.
    
    Parameters
    ----------
    ha : float
        Hour angle [rad].
    dec : float
        Declination [rad].
    Returns
    -------
    ndarray
        Coordinate transform matrix converting equatorial coordinates to
        topocentric coordinates. Shape=(3, 3).
    """
    sin_H, cos_H = np.sin(ha), np.cos(ha)
    sin_d, cos_d = np.sin(dec), np.cos(dec)
    zero = np.zeros_like(ha)

    map = np.array([[sin_H, cos_H, zero],
                    [-sin_d * cos_H, sin_d * sin_H, cos_d],
                    [cos_d * cos_H, -cos_d * sin_H, sin_d]])

    if len(map.shape) == 3:
        map = map.transpose([2, 0, 1])

    return map


def healpix_to_crd_eq(h, nest=False):
    """
    Determine equatorial co-ordinates of a healpix map's pixels.

    Parameters
    ----------
    h : array_like
        The HEALPix array. Shape=(12*N^2,) for integer N.
    nest : bool, optional
        Whether to use the NEST configuration for the HEALPix array.
    Returns
    -------
    ndarray
       The equatorial coordinates of each HEALPix pixel. 
       Shape=(12*N^2, 3) for integer N.
    """
    assert h.ndim == 1, "h must be a 1D array."

    px = np.arange(len(h))
    crd_eq = np.array(healpy.pix2vec(healpy.get_nside(h), px, nest=nest),
                      dtype=np.float32)
    return crd_eq


def lm_to_az_za(l, m):
    """
    Convert l and m (on intervals -1, +1) to azimuth and zenith angle.
    
    Parameters
    ----------
    l, m : array_like
        Normalized angular coordinates on the interval (-1, +1).
    
    Returns
    -------
    az, za : array_like
        Corresponding azimuth and zenith angles (in radians).
    """
    lsqr = l**2. + m**2.
    n = np.where(lsqr < 1., np.sqrt(1. - lsqr), 0.)
    
    az = -np.arctan2(m, l)
    za = np.pi/2. - np.arcsin(n)
    return az, za
    
import time
class AzZaTransforms:
    """
    A class for converting source ra/dec to az/za/lmn based on 
    multiple observing times (Julian dates), using the same operations and code 
    that pyuvsim/astropy uses.  Once initialized, the main entry point 
    is the transform() method.

    An attempt is made to calculate all values on initialization if
    precomput=True. If this fails (likely due to insufficient memory), 
    values will be computed when they are needed.
    """


    def __init__(self, obstimes=None, ra=None, dec=None, use_central_time_values=False, 
            precompute=True, uvbeam_az_correction=False, astropy=False):
        """
        Initialize the object, precomputing values for later use.

        Parameters
        ----------
        obstimes: 1-D ndarray
            The observation times expressed as Julian dates.
        ra, dec : 1-D ndarrays
            Source ra/dec from catalogue, in radians.
        use_central_time_values : bool
            If True, calculate some internal values based on the central (mid) observing time. 
            This speeds computation. Assumes that these values do not change significantly
            over the time range (user beware). The values are: CIRS ra/dec, polar motion, and
            delta ut1 -> utc. Cannot be True if astropy=True.
        precompute: bool
            Attempt to compute all the angles for all the times, on initialization.
            This will fail if it takes a lot of memory, in which case execution will
            proceed without precomputation. The advantage is that the values
            are not recomputed for every frequency i.e. in the hera_sim frequency
            loop.
        uvbeam_az_correction: bool
            If True, apply the correction to the azimuth that pyuvsim applies to
            satisfy the UVBeam convention. Done by altaz_to_zenithangle_azimuth().
        astropy: bool
            If True, call astropy to do the calculations. If False, use code in this object
            that has been extracted from astropy, and stripped of the astropy type system,
            meta-transform handling, and other astropy layers.
            
            Cannot be True if use_central_time_values=True.
        """
        
        print("Initializing AzZaTransforms for az_za_corrections")
        self.obstimes = obstimes
        self.use_central_time_values = use_central_time_values
        self.uvbeam_az_correction = uvbeam_az_correction
        if not astropy:
            from astropy import _erfa as erfa
            self.erfa = erfa    
            from astropy.utils.iers \
                import earth_orientation_table, TIME_BEFORE_IERS_RANGE, TIME_BEYOND_IERS_RANGE
            self.iers_info = (earth_orientation_table.get(), TIME_BEFORE_IERS_RANGE, TIME_BEYOND_IERS_RANGE)
            self.astropy = False
        else:
            if use_central_time_values:
                raise RuntimeError("use_central_time_values=True doesn't make sense when using astropy")
            from astropy.coordinates import SkyCoord, EarthLocation, Longitude, Latitude
            from astropy.coordinates.builtin_frames import AltAz
            from pyuvsim import Telescope
            from pyuvsim.astropy_interface import Time
            self.SkyCoord = SkyCoord
            self.EarthLocation = EarthLocation
            self.Longitude = Longitude
            self.Latitude = Latitude
            self.Telescope = Telescope
            self.AltAz = AltAz
            self.Time = Time
            self.erfa = None
            self.iers_info = None
            self.astropy = True
        
        self.precomputed = False
        
        if precompute:         
            # Some functions cannot be vectorized, hence the loops.
            # TODO: See if can vectorize them by flattening the arrays.
            print("Pre-computing az/za for all times.")
            start = time.time()
            try:
                if astropy:
                    self.az = np.zeros((len(obstimes), len(ra)))
                    self.za = np.zeros((len(obstimes), len(ra)))
                    for i in range(len(obstimes)):
                        self.az[i], self.za[i] = self.call_astropy(ra, dec, obstimes[i])
                    
                elif use_central_time_values:
                    polar_x, polar_y = self.get_polar_motion(obstimes[len(obstimes)//2])
                    dut1utc = self.get_dut1utc(obstimes[len(obstimes)//2])
        
                    cirs_ra, cirs_dec = self.icrs_to_cirs(ra, dec, obstimes[len(obstimes)//2])
                    self.az, self.za = \
                        self.cirs_to_az_za(cirs_ra, cirs_dec, obstimes[len(obstimes)//2],
                                    polar_x, polar_y, dut1utc)
                    self.az = np.zeros((len(obstimes), len(ra)))
                    self.za = np.zeros((len(obstimes), len(ra)))
                    for i in range(len(obstimes)):      # az, za must use the correct time, so obstimes[i]
                        self.az[i], self.za[i] = \
                            self.cirs_to_az_za(cirs_ra, cirs_dec, obstimes[i], polar_x, polar_y, dut1utc)
        
                else:
                    cirs_ra = np.zeros((len(obstimes), len(ra)))
                    cirs_dec = np.zeros((len(obstimes), len(ra)))
                    # Might be able to vectorize this if flattened
                    for i in range(len(obstimes)):
                        cirs_ra[i], cirs_dec[i] = self.icrs_to_cirs(ra, dec, obstimes[i])
                    polar_x, polar_y = self.get_polar_motion(obstimes)
                    dut1utc = self.get_dut1utc(obstimes)
                    self.az = np.zeros((len(obstimes), len(ra)))
                    self.za = np.zeros((len(obstimes), len(ra)))
                    for i in range(len(obstimes)):
                        self.az[i], self.za[i] = \
                            self.cirs_to_az_za(cirs_ra[i], cirs_dec[i], obstimes[i], polar_x[i], polar_y[i], dut1utc[i])
    
                self.crd_top = self.calc_crd_top(self.az, self.za)
                # pyuvsim does a UVBeam correction for az, za used for beam interpolation only
                if self.uvbeam_az_correction: 
                    self.za, self.az = self.altaz_to_zenithangle_azimuth(np.pi/2-self.za, self.az) # alt, az
                self.precomputed = True
    
            except:
                self.precomputed = False
                print("<<< AzZaTransforms precomputation failed. >>>")
                print("<<< Try increasing memory. Execution will be slow. >>>")

            if self.precomputed:
                end = time.time()
                print("Finished pre-computing. Execution time: "+'{:.6f}'.format(end-start))

        if use_central_time_values and not self.precomputed:
            # Save these for later use in transform()
            self.polar_x_centre, self.polar_y_centre = self.get_polar_motion(obstimes[len(obstimes)//2])
            self.dut1utc_centre = self.get_dut1utc(obstimes[len(obstimes)//2])
            self.cirs_ra_centre, self.cirs_dec_centre = self.icrs_to_cirs(ra, dec, obstimes[len(obstimes)//2])

    def transform(self, ra, dec, obstime_index):
        """
        Convert source ra/dec to az/za and lmn positions.

        Parameters
        ----------
        ra, dec : 1-D ndarrays
            Source ra/dec from catalogue.
        obstime_index: int
            Used as an index into precomputed values, if precomputation
            was successful.

        Returns
        -------
        az, za, crd_top
            az and za are 1-D ndarrays of shape (nsource,)
            crd_top is the lmn locations of the sources, an ndarray
            of shape (3, nsource).
        """

        # If precomputation was successful, the values are already calculated
        # for all obstimes, so just return the appropriate ones.
        if self.precomputed:
            return self.az[obstime_index], self.za[obstime_index], self.crd_top[:, obstime_index, :]

        # Otherwise we must calculate the values now.
        if self.astropy:
            az, za = self.call_astropy(ra, dec, self.obstimes[obstime_index])
        else:
            if self.use_central_time_values:
                az, za = self.cirs_to_az_za(self.cirs_ra_centre, self.cirs_dec_centre, self.obstimes[obstime_index], 
                        self.polar_x_centre, self.polar_y_centre, self.dut1utc_centre)
            else:
                cirs_ra, cirs_dec = self.icrs_to_cirs(ra, dec, self.obstimes[obstime_index])
                polar_x, polar_y = self.get_polar_motion(self.obstimes[obstime_index])
                dut1utc = self.get_dut1utc(self.obstimes[obstime_index])
                az, za = self.cirs_to_az_za(cirs_ra, cirs_dec, self.obstimes[obstime_index], 
                        polar_x, polar_y, dut1utc)

        crd_top = self.calc_crd_top(az, za)
        if self.uvbeam_az_correction:
            za, az = self.altaz_to_zenithangle_azimuth(np.pi/2-za, az)  # alt, az
        return az, za, crd_top


    def calc_crd_top(self, az, za):
        """
        Calculate the lmn source positions that pyuvsim calculates.
        These are the crd_top values used by hera_sim.

        Parameters
        ----------
        az, za : 1-D ndarrays of shape(nsource,) OR 2-D arrays of shape (ntimes, nsource).
            Source az/za.

        Returns
        -------
        crd_top
            ndarray of shape (3, nsource) or (3, ntimes, nsource)
            The lmn locations of the sources.
        """

        alt = np.pi/2-za
        if len(az.shape) == 2: crd_top = np.empty((3, az.shape[0], az.shape[1]))  # Multiple times
        else: crd_top = np.empty((3, az.shape[0]))
        crd_top[0] = np.sin(az) * np.cos(alt)  # (obstime, nsources)
        crd_top[1] = np.cos(az) * np.cos(alt)
        crd_top[2] = np.sin(alt)
        return crd_top

    # ---- All the following functions are extracted from astropy or pyuvsim. ----

    def icrs_to_cirs(self, ra, dec, obstime):
        """
        icrs_to_cirs in astropy/coordinates/builtin_frames/icrs_cirs_transforms.py
        Stripped down to what is executed. 
        """
        jd1, jd2 = self.get_jd12(obstime, "tt") # utc -> tai -> tt
        x, y, s = self.get_cip(jd1, jd2)
        earth_pv, earth_heliocentric = self.prepare_earth_position_vel(obstime)
        astrom = self.erfa.apci(jd1, jd2, earth_pv, earth_heliocentric, x, y, s)
        cirs_ra, cirs_dec = self.atciqz(ra, dec, astrom)
        return cirs_ra, cirs_dec

    
    def cirs_to_az_za(self, cirs_ra, cirs_dec, obstime, polar_x, polar_y, dut1utc):
        """
        Do cirs_to_altaz in astropy/coordinates/builtin_frames/cirs_observed_transforms.py
        """
        lon, lat, height = self.to_geodetic()
        jd1, jd2 = self.get_jd12(obstime, 'utc')
        pressure_value = temperature_value = relative_humidity_value = 0.0
        obswl_value = 1.0
        astrom = self.erfa.apio13(jd1, jd2,
                         dut1utc,
                         lon, lat,
                         height,
                         polar_x, polar_y, # polar motion
                         # all below are already in correct units because they are QuantityFrameAttribues
                         pressure_value,
                         temperature_value,
                         relative_humidity_value,
                         obswl_value)
        az, zen, _, _, _ = self.erfa.atioq(cirs_ra, cirs_dec, astrom)

        return az, zen

    
    def two_sum(self, a, b):
        """
        From astropy/time/utils.py, exact copy.
        Add ``a`` and ``b`` exactly, returning the result as two float64s.
        The first is the approximate sum (with some floating point error)
        and the second is the error of the float64 sum.
        Using the procedure of Shewchuk, 1997,
        Discrete & Computational Geometry 18(3):305-363
        http://www.cs.berkeley.edu/~jrs/papers/robustr.pdf
        Returns
        -------
        sum, err : float64
            Approximate sum of a + b and the exact floating point error
        """
        x = a + b
        eb = x - a  # bvirtual in Shewchuk
        ea = x - eb  # avirtual in Shewchuk
        eb = b - eb  # broundoff in Shewchuk
        ea = a - ea  # aroundoff in Shewchuk
        return x, ea + eb

    def day_frac(self, val1, val2, factor=None, divisor=None):
        """
        From astropy/time/utils.py, exact copy.
        Return the sum of ``val1`` and ``val2`` as two float64s.
        The returned floats are an integer part and the fractional remainder,
        with the latter guaranteed to be within -0.5 and 0.5 (inclusive on
        either side, as the integer is rounded to even).
        The arithmetic is all done with exact floating point operations so no
        precision is lost to rounding error.  It is assumed the sum is less
        than about 1e16, otherwise the remainder will be greater than 1.0.
        Parameters
        ----------
        val1, val2 : array of float
            Values to be summed.
        factor : float, optional
            If given, multiply the sum by it.
        divisor : float, optional
            If given, divide the sum by it.
        Returns
        -------
        day, frac : float64
            Integer and fractional part of val1 + val2.
        """
        # Add val1 and val2 exactly, returning the result as two float64s.
        # The first is the approximate sum (with some floating point error)
        # and the second is the error of the float64 sum.
        sum12, err12 = self.two_sum(val1, val2)
    
        if factor is not None:
            sum12, carry = two_product(sum12, factor)
            carry += err12 * factor
            sum12, err12 = two_sum(sum12, carry)
    
        if divisor is not None:
            q1 = sum12 / divisor
            p1, p2 = two_product(q1, divisor)
            d1, d2 = two_sum(sum12, -p1)
            d2 += err12
            d2 -= p2
            q2 = (d1 + d2) / divisor  # 3-part float fine here; nothing can be lost
            sum12, err12 = two_sum(q1, q2)
    
        # get integer fraction
        day = np.round(sum12)
        extra, frac = self.two_sum(sum12, -day)
        frac += extra + err12
        # Our fraction can now have gotten >0.5 or <-0.5, which means we would
        # loose one bit of precision. So, correct for that.
        excess = np.round(frac)
        day += excess
        extra, frac = self.two_sum(sum12, -day)
        frac += extra + err12
        return day, frac


    def _get_delta_tdb_tt(self, jd1, jd2):
        """
        From astropy/time/core.py, modified.
        
        # First go from the current input time (which is either
        # TDB or TT) to an approximate UT1.  Since TT and TDB are
        # pretty close (few msec?), assume TT.  Similarly, since the
        # UT1 terms are very small, use UTC instead of UT1.
        """
        njd1, njd2 = self.erfa.tttai(jd1, jd2)
        njd1, njd2 = self.erfa.taiutc(njd1, njd2)
        # subtract 0.5, so UT is fraction of the day from midnight
        ut = self.day_frac(njd1 - 0.5, njd2)[1]

        # Geodetic params needed for d_tdb_tt()
        lon_rad, rxy_km, z_km = 0.3739944696510935, 5488.7502182535945, -3239.939404619622
        return self.erfa.dtdb(
                jd1, jd2, ut, lon_rad,
                rxy_km, z_km)

    def get_jd12(self, obstime, coord_frame):
        """
        From astropy/coordinates/builtin_frames/utils.py
        """
        jd1, jd2 = self.day_frac(obstime, 0.0)

        if coord_frame == "tt":   # utc -> tai -> tt
            jd1, jd2 = self.erfa.core.utctai(jd1, jd2)
            jd1, jd2 = self.erfa.core.taitt(jd1, jd2)
        elif coord_frame == "tdb": # utc -> tai -> tt -> tdb
            jd1, jd2 = self.erfa.core.utctai(jd1, jd2)
            jd1, jd2 = self.erfa.core.taitt(jd1, jd2)
            jd1, jd2 = self.erfa.core.tttdb(jd1, jd2, self._get_delta_tdb_tt(jd1, jd2))
        elif coord_frame == "utc":
            pass
        else:
            raise RuntimeError("Unimplemented time frame: "+coord_frame)
        return jd1, jd2

    def get_cip(self, jd1, jd2):
        """
        From astropy/coordinates/builtin_frames/utils.py, exact copy
        Find the X, Y coordinates of the CIP and the CIO locator, s.

        Parameters
        ----------
        jd1 : float or `np.ndarray`
            First part of two part Julian date (TDB)
        jd2 : float or `np.ndarray`
            Second part of two part Julian date (TDB)

        Returns
        --------
        x : float or `np.ndarray`
            x coordinate of the CIP
        y : float or `np.ndarray`
            y coordinate of the CIP
        s : float or `np.ndarray`
            CIO locator, s
        """
        # classical NPB matrix, IAU 2006/2000A
        rpnb = self.erfa.pnm06a(jd1, jd2)
        # CIP X, Y coordinates from array
        x, y = self.erfa.bpn2xy(rpnb)
        # CIO locator, s
        s = self.erfa.s06(jd1, jd2, x, y)
        return x, y, s

    def get_body_barycentric(self, name, time, get_velocity=False):
        """
        From astropy/coordinates/solar_system.py 
        Stripped down to what is executed for pyuvsim
        """
        jd1, jd2 = self.get_jd12(time, "tdb")    # utc -> tdb
        earth_pv_helio, earth_pv_bary = self.erfa.epv00(jd1, jd2)
        if name == "earth": body = earth_pv_bary
        if name == "sun":
           body = self.erfa.pvmpv(earth_pv_bary, earth_pv_helio)

        if get_velocity: return body['p'], body['v']
        else: return body['p']

    def get_body_barycentric_posvel(self, name, time):
        """
        From astropy/coordinates/solar_system.py 
        Stripped down to what is executed for pyuvsim
        """
       
        return self.get_body_barycentric(name, time, True)

    def prepare_earth_position_vel(self, obstime):
        """
        From astropy/coordinates/builtin_frames/utils.py
        Almost exact copy, the get_xyz() stripped out
        because we have native types.

        Get barycentric position and velocity, and heliocentric position of Earth
    
        Parameters
        -----------
        time : `~astropy.time.Time`
            time at which to calculate position and velocity of Earth
    
        Returns
        --------
        earth_pv : `np.ndarray`
            Barycentric position and velocity of Earth, in au and au/day
        earth_helio : `np.ndarray`
            Heliocentric position of Earth in au
        """
        # get barycentric position and velocity of earth
        earth_p, earth_v = self.get_body_barycentric_posvel('earth', obstime)
    
        # get heliocentric position of earth, preparing it for passing to erfa.
        sun = self.get_body_barycentric('sun', obstime)
        earth_heliocentric = (earth_p - sun)
    
        # Also prepare earth_pv for passing to erfa, which wants it as
        # a structured dtype.
        earth_pv = self.erfa.pav2pv(earth_p, earth_v)
        return earth_pv, earth_heliocentric
    
    
    def atciqz(self, rc, dc, astrom):
        """
        From astropy/coordinates/builtin_frames/utils.py, exact copy
        
        A slightly modified version of the ERFA function ``eraAtciqz``.

        ``eraAtciqz`` performs the transformations between two coordinate systems,
        with the details of the transformation being encoded into the ``astrom`` array.

        The companion function ``eraAticq`` is meant to be its inverse. However, this
        is not true for directions close to the Solar centre, since the light deflection
        calculations are numerically unstable and therefore not reversible.

        This version sidesteps that problem by artificially reducing the light deflection
        for directions which are within 90 arcseconds of the Sun's position. This is the
        same approach used by the ERFA functions above, except that they use a threshold of
        9 arcseconds.

        Parameters
        ----------
        rc : float or `~numpy.ndarray`
            right ascension, radians
        dc : float or `~numpy.ndarray`
            declination, radians
        astrom : eraASTROM array
            ERFA astrometry context, as produced by, e.g. ``eraApci13`` or ``eraApcs13``

        Returns
        --------
        ri : float or `~numpy.ndarray`
        di : float or `~numpy.ndarray`
        """
        # BCRS coordinate direction (unit vector).
        pco = self.erfa.s2c(rc, dc)

        # Light deflection by the Sun, giving BCRS natural direction.
        pnat = self.erfa.ld(1.0, pco, pco, astrom['eh'], astrom['em'], 5e-8)

        # Aberration, giving GCRS proper direction.
        ppr = self.erfa.ab(pnat, astrom['v'], astrom['em'], astrom['bm1'])

        # Bias-precession-nutation, giving CIRS proper direction.
        # Has no effect if matrix is identity matrix, in which case gives GCRS ppr.
        pi = self.erfa.rxp(astrom['bpn'], ppr)

        # CIRS (GCRS) RA, Dec
        ri, di = self.erfa.c2s(pi)
        return self.erfa.anp(ri), di


    def to_geodetic(self):
        """
        From astropy/coordinates/earth.py
        Hardwired to HERA and lat/lon converted to radians for later use
        It does not change if the obs time changes.
        """
        return 0.3739944696510935, -0.5361917991288502, 1073.000000006672 


    def get_polar_motion(self, time):
        """
        From astropy/coordinates/builtin_frames/utils.py, slightly modified
        gets the two polar motion components in radians for use with apio13
        """

        iers_table, TIME_BEFORE_IERS_RANGE, TIME_BEYOND_IERS_RANGE = self.iers_info
        
        # Get the polar motion from the IERS table
        xp, yp, status = iers_table.pm_xy(time, return_status=True)
        if np.any(status == TIME_BEFORE_IERS_RANGE):
            wmsg = ('Tried to get polar motions for times before IERS data is '
                'valid. Defaulting to polar motion from the 50-yr mean for those. '
                'This may affect precision at the 10s of arcsec level')
            xp[status == TIME_BEFORE_IERS_RANGE] = _DEFAULT_PM[0]
            yp[status == TIME_BEFORE_IERS_RANGE] = _DEFAULT_PM[1]

            print(wmsg)

        if np.any(status == TIME_BEYOND_IERS_RANGE):
            wmsg = ('Tried to get polar motions for times after IERS data is '
                'valid. Defaulting to polar motion from the 50-yr mean for those. '
                'This may affect precision at the 10s of arcsec level')

            xp[status == TIME_BEYOND_IERS_RANGE] = _DEFAULT_PM[0]
            yp[status == TIME_BEYOND_IERS_RANGE] = _DEFAULT_PM[1]
        # Convert to radian from arcsec
        xp *= np.pi/180/60/60
        yp *= np.pi/180/60/60
        return xp.value, yp.value

    def _get_delta_ut1_utc(self, jd1, jd2, iers_table):
        """
        astropy/time/core.py, line 1776, modified
        
        Get ERFA DUT arg = UT1 - UTC.  This getter takes optional jd1 and
        jd2 args because it gets called that way when converting time scales.
        If delta_ut1_utc is not yet set, this will interpolate them from the
        the IERS table.
        """
        # Sec. 4.3.1: the arg DUT is the quantity delta_UT1 = UT1 - UTC in
        # seconds. It is obtained from tables published by the IERS.
        # interpolate UT1-UTC in IERS table
        return self.iers_info[0].ut1_utc(jd1, jd2)

    def get_dut1utc(self, obstime):
        """
        astropy/coordinates/builtin_frames/utils.py
        
        Calls delta_ut1_utc and needs IERS table
        """
        jd1, jd2 = self.day_frac(obstime, 0.0)
        return self._get_delta_ut1_utc(jd1, jd2, self.iers_info[0]).value
       
    def altaz_to_zenithangle_azimuth(self, altitude, azimuth):
        """
        From pyuvsim/utils.py, exact copy
        
        Convert from astropy altaz convention to UVBeam az/za convention.
        Parameters
        ----------
        altitude, azimuth: float or array of float
            altitude above horizon
            azimuth in radians in astropy convention: East of North (N=0, E=pi/2)
        Returns
        -------
        zenith_angle: float or array of float
            In radians
        azimuth: float or array of float
            In radians in uvbeam convention: North of East(East=0, North=pi/2)
        """
        input_alt = np.asarray(altitude)
        input_az = np.asarray(azimuth)
        if input_alt.size != input_az.size:
            raise ValueError('number of altitude and azimuth values must match.')

        zenith_angle = np.pi / 2 - input_alt
        new_azimuth = np.pi / 2 - input_az

        if new_azimuth.size > 1:
            wh_neg = np.where(new_azimuth < -1e-9)
            if wh_neg[0].size > 0:
                new_azimuth[wh_neg] = new_azimuth[wh_neg] + np.pi * 2
        elif new_azimuth.size == 1:
            if new_azimuth < -1e-9:
                new_azimuth = new_azimuth + np.pi * 2

        return zenith_angle, new_azimuth

    def call_astropy(self, ra, dec, obstime):
        """
        Extracted from update_positions() in pyradiosky/skymodel.py.
        Some astropy types need to be setup, then use SkyCoord.
        Does the work necessary to call astropy to convert ra/dec to az/za (AltAz).
        """
       
        tloc = [5109342.82705015, 2005241.839292723, -3239939.404619622]  # HERA
        location = self.EarthLocation.from_geocentric(*tloc, unit='m')
        telescope = self.Telescope("HERA", location, [])
        time_array = self.Time(obstime, scale='utc', format='jd', location=telescope.location)
        ra = self.Longitude(ra, "rad"); dec = self.Latitude(dec, "rad")
        skycoord_use = self.SkyCoord(ra, dec, frame="icrs")
        # Now we go into astropy
        source_altaz = skycoord_use.transform_to(
            self.AltAz(obstime=time_array, location=telescope.location))
        alt_az = np.array([source_altaz.alt.rad, source_altaz.az.rad])
        return alt_az[1], np.pi/2-alt_az[0] 

