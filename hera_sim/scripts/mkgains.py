#!/usr/bin/env python3
'''
mkgains.py: Creates fake gains
This takes some UVData-compatible data file/folder, and creates gains
based on the shape of the data. It filters the gains using a boxcar
filter over the entire dataset. The user must select the size of the
boxcars and the standard deviation of the distribution of the gains.
'''
from pyuvdata import UVData, UVCal
import numpy as np
import argparse as argp
import os
import sys
import subprocess as sub

def bc(car_size, data_shape):
    isOdd = car_size & 1
    radius = car_size >> 1
    center = data_shape >> 1
    rect = np.zeros(data_shape)
    rect[center - radius:center + radius + isOdd] = 1
    return rect

def sinc(fwhm, shape): #, outTuple = True):
    k = 3.79 / (fwhm * np.pi)

    sincran = np.linspace(-np.pi * fwhm, np.pi * fwhm, shape)
    sincout = np.sinc(sincran * k)

    #if outTuple:
    #    return (sincout, sincran)
    #else:
    #    return sincout
    return sincout

# Start parsing arguments
parse = argp.ArgumentParser()
parse.add_argument("path", help='Path to data file', type=str)
parse.add_argument('-f', '--frequency', help='Amount of frequency (in Hertz) used for the size of the boxcar' + \
        ' for the boxcar smoothing.', type=float)
parse.add_argument('-t', '--time', help='Amount of time (in seconds) used for the size of the boxcar' + \
        ' for the boxcar smoothing.', type=float)
parse.add_argument('-a', '--ampstdev', help='The standard deviation used to generate the amplitude' + \
        ' of the gains', type=np.float64)
parse.add_argument('-p', '--phasestdev', help='The standard deviation used to generate the phase of' + \
        ' the gains', type=np.float64)
parse.add_argument('-j', '--timefourier', help='Changes the filter done over time so that it is done in' + \
        ' Fourier space instead', action='store_true', default=False)
parse.add_argument('-g', '--freqfourier', help='Changes the filter done over frequency so that it is done' + \
        ' in Fourier space instead', action='store_true', default=False)
args = parse.parse_args()

# Check args
fullpath = os.path.abspath(os.path.expanduser(args.path))
if fullpath.endswith('/'):
    fullpath = fullpath[:-1]

argsdict = vars(args)
for imp in list(argsdict.keys())[1:]:
    if argsdict[imp] == None:
        raise ValueError('--' + imp + ' must be used')

# Load data
uvd = UVData()
readfuncts = [getattr(uvd, n) for n in dir(uvd) if n.startswith('read')]
numfailed = 0
for read in readfuncts:
    try:
        read(fullpath)
    except:
        numfailed += 1
    else:
        print('%s used to read data' % (read.__name__))
        break

if numfailed == len(readfuncts):
    raise IOError('Data could not be read using any read function in UVData')

# Prepare cal file
uvc = UVCal()

# Automatically import as much as we can for the data
required = [r[1:] for r in uvc.required()]
for var in required:
    try:
        attr = getattr(uvd, var)
    except:
        pass
    else:
        setattr(uvc, var, attr)

# Change what's left manually
uvc.cal_style = 'redundant'
uvc.cal_type = 'gain'
uvc.gain_convention = 'multiply'

command_given = sys.argv[1:]
command_given = ["'" + x + "'" if ' ' in x else x for x in command_given]
command_given = ' '.join(command_given)
script_path = os.path.dirname(os.path.abspath(__file__))
try:
    git_hash = sub.check_output(['git', '-C', script_path, 'rev-parse', 'HEAD']).strip().decode('UTF-8')
except:
    git_hash = ''
    print('Error: Cannot get git hash')

uvc.history = 'Created using mkgains.py\nCommand run: mkgains.py %s\nmkgains.py Git Hash: %s' % (command_given, git_hash)

uvc.Njones = uvd.Npols
uvc.jones_array = uvd.polarization_array
uvc.ant_array = np.arange(uvc.Nants_data)
uvc.integration_time = np.mean(uvc.integration_time)
uvc.time_range = [uvc.time_array[0] - (uvc.integration_time / 172800.), 
                  uvc.time_array[-1] + (uvc.integration_time / 172800.)]
uvc.quality_array = np.zeros((uvc.Nants_data, uvc.Nspws, uvc.Nfreqs, uvc.Ntimes, uvc.Njones), dtype='float64')
uvc.flag_array = np.zeros((uvc.Nants_data, uvc.Nspws, uvc.Nfreqs, uvc.Ntimes, uvc.Njones), dtype='bool')
uvc.time_array = np.unique(uvd.time_array)
if uvc.x_orientation == None:
    uvc.x_orientation = 'East'

# Create random gains
gain_shape = (uvc.Nants_data, uvc.Nspws, uvc.Nfreqs, uvc.Ntimes, uvc.Njones)
gains = np.random.normal(1.0, args.ampstdev, gain_shape) * \
        np.exp(np.random.normal(0.0, args.phasestdev, gain_shape) * 1j) 

# Preparing variables for transformations
timesize = int(args.time / uvc.integration_time)
freqsize = int(args.frequency / np.abs(uvc.freq_array[0, -1] - uvc.freq_array[0, -2]))
if timesize == 0:
    raise ValueError('--time is too small (Needs to be bigger than %f)' % (uvc.integration_time))
elif timesize > uvc.Ntimes:
    raise ValueError('--time is too big (Needs to be smaller than %f)' % (uvc.Ntimes * uvc.integration_time))
if freqsize == 0:
    raise ValueError('--frequency is too small (Needs to be bigger than %f)' % (np.abs(uvc.freq_array[0, -1] - uvc.freq_array[0, -2])))
elif freqsize > uvc.Nfreqs:
    raise ValueError('--frequency is too big (Needs to be smaller than %f)' % (uvc.Nfreqs * np.abs(uvc.freq_array[0, -1] - uvc.freq_array[0, -2])))

if args.timefourier:
    mod_size = 3.79 / (timesize * np.pi)
    timefilter = bc(mod_size, uvc.Ntimes) / timesize
    timefilter = timefilter.astype('complex')

else:
    mod_size = 3.79 / (timesize * np.pi)
    timefilter = sinc(mod_size, uvc.Ntimes)
    timefilter = timefilter.astype('complex')

if args.freqfourier:
    mod_size = 3.79 / (freqsize * np.pi)
    freqfilter = bc(mod_size, uvc.Nfreqs) / freqsize
    freqfilter = freqfilter.astype('complex')

else:
    mod_size = 3.79 / (freqsize * np.pi)
    freqfilter = sinc(mod_size, uvc.Nfreqs)
    freqfilter = freqfilter.astype('complex')

# Transform random gains
for ant in range(uvc.Nants_data):
    for spw in range(uvc.Nspws):
        for polar in range(uvc.Njones):
            # Boxcar smoothing
            for freq in range(uvc.Nfreqs):
                gains[ant, spw, freq, :, polar] = np.fft.ifft(np.fft.fft(gains[ant, spw, freq, :, polar]) * timefilter)

            # Fouier space low pass filter
            for time in range(uvc.Ntimes):
                gains[ant, spw, :, time, polar] = np.fft.ifft(np.fft.fft(gains[ant, spw, :, time, polar]) * freqfilter)
          
uvc.gain_array = gains

# Write out calfits file in same directory as data file
i = 0
toappend = ''
while toappend != 'stop':
    try:
        uvc.write_calfits(fullpath + toappend + '.cal')
    except IOError as err:
        if str(err).endswith('already exists.'):
            i += 1
            toappend = "_%d" % (i)
        else:
            raise NotImplementedError("Cannot write to location '%s'" % (fullpath))
    else:
        print("calfits file written to '%s'" % (fullpath + toappend + '.cal'))
        toappend = 'stop'
