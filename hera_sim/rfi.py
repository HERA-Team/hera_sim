'''A module for generating realistic HERA noise.'''

import numpy as np
import aipy

class RfiStation:
    def __init__(self, fq0, duty_cycle=1., strength=100., std=10., timescale=100.):
        self.fq0 = fq0
        self.duty_cycle = duty_cycle
        self.std = std
        self.strength = strength
        self.timescale = timescale
    def gen_rfi(self, fqs, lsts, rfi=None):
        sdf = np.average(fqs[1:] - fqs[:-1])
        if rfi is None: rfi = np.zeros((lsts.size,fqs.size), dtype=np.complex)
        assert(rfi.shape == (lsts.size, fqs.size))
        try: ch1 = np.argwhere(np.abs(fqs - self.fq0) < sdf)[0,0]
        except(IndexError): return rfi
        if self.fq0 > fqs[ch1]: ch2 = ch1+1
        else: ch2 = ch1-1
        phs1,phs2 = np.random.uniform(0,2*np.pi,size=2)
        signal = .999*np.cos(lsts*aipy.const.sidereal_day / self.timescale + phs1) + 2*(self.duty_cycle - .5)
        signal = np.where(signal > 0, np.random.normal(self.strength,self.std) * np.exp(1j*phs2), 0)
        rfi[:,ch1] += signal * (1 - np.abs(fqs[ch1] - self.fq0)/sdf).clip(0,1)
        rfi[:,ch2] += signal * (1 - np.abs(fqs[ch2] - self.fq0)/sdf).clip(0,1)
        return rfi

HERA_RFI_STATIONS = [
    # FM Stations
    (.1007, 1., 1000 * 100., 10., 100.), 
    (.1016, 1., 1000 * 100., 10., 100.), 
    (.1024, 1., 1000 * 100., 10., 100.),
    (.1028, 1., 1000 *   3.,  1., 100.),
    (.1043, 1., 1000 * 100., 10., 100.),
    (.1050, 1., 1000 *  10.,  3., 100.),
    (.1052, 1., 1000 * 100., 10., 100.),
    (.1061, 1., 1000 * 100., 10., 100.),
    (.1064, 1., 1000 *  10.,  3., 100.),
    # Orbcomm
    (.1371, .2, 1000 *  100.,  3., 600.),
    (.1372, .2, 1000 *  100.,  3., 600.),
    (.1373, .2, 1000 *  100.,  3., 600.),
    (.1374, .2, 1000 *  100.,  3., 600.),
    (.1375, .2, 1000 *  100.,  3., 600.),
    # Other
    (.1831, 1., 1000 * 100., 30., 1000),
    (.1891, 1., 1000 *   2.,  1., 1000),
    (.1911, 1., 1000 * 100., 30., 1000),
    (.1972, 1., 1000 * 100., 30., 1000),
    # DC Offset from ADCs
    #(.2000, 1., 100., 0., 10000),
]

def rfi_stations(fqs, lsts, stations=HERA_RFI_STATIONS, rfi=None):
    for s in stations:
        s = RfiStation(*s)
        rfi = s.gen_rfi(fqs, lsts, rfi=rfi)
    return rfi

def rfi_impulse(fqs, lsts, rfi=None, chance=.001, strength=20.):
    if rfi is None: rfi = np.zeros((lsts.size,fqs.size), dtype=np.complex)
    assert(rfi.shape == (lsts.size, fqs.size))
    impulse_times = np.where(np.random.uniform(size=lsts.size) <= chance)[0]
    dlys = np.random.uniform(-300,300, size=impulse_times.size) # ns
    impulses = strength * np.array([np.exp(2j*np.pi*dly*fqs) for dly in dlys])
    if impulses.size > 0: rfi[impulse_times] += impulses
    return rfi
    
def rfi_scatter(fqs, lsts, rfi=None, chance=.0001, strength=10, std=10):
    if rfi is None: rfi = np.zeros((lsts.size,fqs.size), dtype=np.complex)
    assert(rfi.shape == (lsts.size, fqs.size))
    rfis = np.where(np.random.uniform(size=rfi.size) <= chance)[0]
    rfi.flat[rfis] += np.random.normal(strength,std) * np.exp(1j*np.random.uniform(size=rfis.size)) 
    return rfi
    
