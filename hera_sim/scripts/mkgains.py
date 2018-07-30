#!/usr/bin/env python3
'''
mkgains.py: Creates fake gains
This takes some UVData-compatible data file/folder, and creates gains
based on the shape of the data. It filters the gains in two ways:
    1: Boxcar filter over # of integrations
    2: Low pass filter over frequency
The user must select the size of the boxcar, the size used for the
low pass filter and the standard deviation of the distribution of the
gains.
'''
from pyuvdata import UVData, UVCal
import numpy as np
import argparse as argp
import os
import sys
import subprocess as sub

def timeboxcarFFT(data, conv, width):
    datafft = np.fft.fft(data)
    convfft = np.fft.fft(conv/uvc.Ntimes)
    return np.fft.ifft(datafft * convfft)

def timeboxcarREAL(data, conv, width):
    datafft = np.fft.fft(data)
    convfft = np.fft.fft(conv)
    convfft /= np.max(convfft) * width
    return np.fft.ifft(datafft * convfft)

def freqlowpassFFT(data, conv):
    datafft = np.fft.fft(data)
    convfft = np.fft.fft(conv)
    convfft /= np.max(convfft)
    return np.fft.ifft(datafft * convfft)

def freqlowpassREAL(data, conv):
    return data * conv

# Start parsing arguments
parse = argp.ArgumentParser()
parse.add_argument("path", help='Path to data file', type=str)
parse.add_argument('-f', '--frequency', help='Without -g (--freqfourier), this specifies the range of' + \
        ' frequencies to zero out, starting from the last frequency (in Hertz). With -g (--freqfourier), this' + \
        ' specifies the Full Width Half Maximum of the sinc used to create the boxcar in fourier space (in Hertz).'
        , type=np.float64)
parse.add_argument('-t', '--time', help='Amount of time (in seconds) used for the size of the boxcar' + \
        ' for the boxcar smoothing.', type=int)
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
git_hash = sub.check_output(['git', '-C', script_path, 'rev-parse', 'HEAD']).strip().decode('UTF-8')
uvc.history = 'Created using mkgains.py\nCommand run: mkgains.py %s\nmkgains.py Git Hash: %s' % (command_given, git_hash)

uvc.Njones = uvd.Npols
uvc.jones_array = uvd.polarization_array
uvc.ant_array = np.arange(uvc.Nants_data)
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
if timesize == 0:
    raise ValueError('--time is too small')

if args.timefourier:
    if timesize > gain_shape[3]:
        raise ValueError('--time is too big')

    sincran = np.linspace(-np.pi * args.time, np.pi * args.time, gain_shape[3])
    timefilter = np.sinc(sincran * timesize)
    timetransform = timeboxcarFFT
    width = None

else:
    timefilter = np.zeros(uvc.Ntimes)
    sizeodd = timesize & 1
    radius = timesize >> 1
    center = timefilter.size >> 1
    timefilter[center - radius:center + radius + sizeodd] = 1
    timefilter /= np.float64(timesize)
    timetransform = timeboxcarREAL
    width = (radius * 2) + sizeodd

if args.freqfourier:
    sincran = np.linspace(-args.frequency, args.frequency, gain_shape[2])
    freqfilter = np.sinc(sincran * 3.79 / args.frequency)
    freqtransform = freqlowpassFFT

else:
    if args.frequency > np.abs(uvc.freq_array[0, -1] - uvc.freq_array[0, 0]):
        raise ValueError('--frequency is too big')
    
    freqsize = int(args.frequency / np.abs(uvc.freq_array[0, -1] - uvc.freq_array[0, -2])) + 1
    freqfilter = np.ones(gain_shape[2], dtype=complex)
    if uvc.freq_array[0, -1] > uvc.freq_array[0, -2]:
        freqfilter[-freqsize:] = 0 + 0j
    else:
        freqfilter[:freqsize] = 0 + 0j
    freqtransform = freqlowpassREAL

# Transform random gains
for ant in range(uvc.Nants_data):
    for spw in range(uvc.Nspws):
        for polar in range(uvc.Njones):
            # Boxcar smoothing
            for freq in range(uvc.Nfreqs):
                gains[ant, spw, freq, :, polar] = timetransform(gains[ant, spw, freq, :, polar], timefilter, width)

            # Fouier space low pass filter
            for time in range(uvc.Ntimes):
                gains[ant, spw, :, time, polar] = freqtransform(gains[ant, spw, :, time, polar], freqfilter)
          
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
