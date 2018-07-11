#!/usr/bin/env python3
# mkgains: Create fake gains

from pyuvdata import UVData, UVCal
import numpy as np
import argparse as argp
import os
import sys
import subprocess as sub

# Start parsing arguments
parse = argp.ArgumentParser()
parse.add_argument("path", help='Path to data file', type=str)
parse.add_argument('-n', '--integrations',  help='Number of integrations used for boxcar smoothing', type=int)
parse.add_argument('-f', '--frequency', help='Minimum frequency to start zeroing out (in Hz)', type=np.float64)
parse.add_argument('-t', '--time', help='Amount of time (in seconds) used for boxcar smoothing', type=int)
parse.add_argument('-a', '--amplitudephase', help='Use the Amplitude-Phase form for calculating fake gains',
        action='store_true')
parse.add_argument('-d', '--distribution', help='Select the type of random distribution used for finding the' + \
        ' real and imaginary parts (or the amplitude and phase if -a is used) for the creation of the gains',
        nargs=2, default=['normal', 'normal'], choices=['normal', 'uniform'])
parse.add_argument('-r', '--realrange', help='If uniform distribution is selected for the real part' + \
        ' (or the amplitude if -a is used), then this selects the minimum and maximum values for the' + \
        ' distribution, in that order. Otherwise, it selects the mean and standard deviation of the' + \
        ' normal distribution, in that order.', nargs=2, type=np.float64, default=[None, None])
parse.add_argument('-i', '--imagrange', help='If uniform distribution is selected for the imaginary part' + \
        ' (or the phase if -a is used), then this selects the minimum and maximum values for the' + \
        ' distribution, in that order. Otherwise, it selects the mean and standard deviation of the' + \
        ' normal distribution, in that order.', nargs=2, type=np.float64, default=[None, None])
args = parse.parse_args()

# Check and populate args
fullpath = os.path.abspath(os.path.expanduser(args.path))
if fullpath.endswith('/'):
    fullpath = fullpath[:-1]

if bool(args.time) == bool(args.integrations):
    raise ValueError('Either -i or -t must be used')

if args.frequency == None:
    raise ValueError('-f must be used')

rand_input = []
for count, check in enumerate([args.realrange, args.imagrange]):
    if check == [None, None]:
        if args.distribution[0] == 'normal':
            rand_input += [count ^ 1, .01]
        elif args.distribution[0] == 'uniform':
            if count == 0:
                rand_input += [1, 10]
            elif count == 1:
                rand_input += [-np.pi, np.pi]
    elif args.distribution[count] == 'uniform':
        if check[0] >= check[1]:
            raise ValueError('The first value must be smaller than the second' + \
                    'value if uniform distribution was chosen for that input')

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

if args.integrations > uvd.Ntimes and not bool(args.time):
    raise ValueError('Number of integrations is too big (Must be <= Ntimes in your data)')

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

if args.amplitudephase:
    names = ['Amplitude', 'Phase']
else:
    names = ['Real', 'Imaginary']

distfuncts = []
for count, dist in enumerate(args.distribution):
    if dist == 'normal':
        distfuncts += [np.random.normal]
        descriptions = ['Mean', 'Standard Deviation']
    elif dist == 'uniform':
        distfuncts += [np.random.uniform]
        descriptions = ['Minimum', 'Maximum']
    print('%s %s: %f' % (names[count], descriptions[0], rand_input[count * 2]))
    print('%s %s: %f' % (names[count], descriptions[1], rand_input[count * 2 + 1]))

if args.amplitudephase:
    gains = distfuncts[0](rand_input[0], rand_input[1], gain_shape) * \
            np.exp(distfuncts[1](rand_input[2], rand_input[3], gain_shape) * 1j)
else:
    gains = distfuncts[0](rand_input[0], rand_input[1], gain_shape) + \
            distfuncts[1](rand_input[2], rand_input[3], gain_shape) * 1j

# Preparing variables for transformations
if args.time:
    lengthday = np.float64(args.time) / 86400
    size = np.abs(uvc.time_array - (uvc.time_array.min() + lengthday)).argmin() + 1
    print('boxcar size: %d integrations' % (size))
    if lengthday + uvc.time_array[0] > uvc.time_range[1]:
        print('Warning: The given time range is bigger than the time range of the data set')
else:
    size = args.integrations

if size < 2:
    raise ValueError('Specificed integraions or time value is too small')

bc = np.zeros(uvc.Ntimes)
sizeodd = size & 1
radius = size >> 1
center = bc.size >> 1
bc[center - radius:center + radius + sizeodd] = 1
bc /= np.float64(size)

f0index = np.argmax(uvc.freq_array >= args.frequency)
if f0index == 0:
    if np.any(np.amin(uvc.freq_array, axis = 0) > args.f0):
        raise ValueError('f0 is too small (f0 must be between freq_array[0] and freq_array[-1])')
    if np.any(np.amax(uvc.freq_array, axis = 0) < args.f0):
        raise ValueError('f0 is too big (f0 must be between freq_array[0] and freq_array[-1])')

# Transform random gains
for ant in range(uvc.Nants_data):
    for spw in range(uvc.Nspws):
        for polar in range(uvc.Njones):
            # Boxcar smoothing
            for freq in range(uvc.Nfreqs):
                timefft = np.fft.fft(gains[ant, spw, freq, :, polar], n = uvc.Nfreqs * 2 - 1)
                bcfft = np.fft.fft(bc, n = uvc.Nfreqs * 2 - 1)
                smoothed = np.fft.ifft(timefft * bcfft, n = uvc.Nfreqs * 2 - 1)
                smoothed_center = smoothed.size >> 1
                smoothed_radius = uvc.Ntimes >> 1
                gains[ant, spw, freq, :, polar] = smoothed[smoothed_center - smoothed_radius:smoothed_center + smoothed_radius]

            # Fouier space low pass filter
            for time in range(uvc.Ntimes):
                freqsfft = np.fft.fft(gains[ant, spw, :, time, polar])
                freqsfft[f0index:] = 0 + 0j
                gains[ant, spw, :, time, polar] = np.fft.ifft(freqsfft)

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
