#===============================================================================
#                                       MAIN
#===============================================================================

import os
import numpy as np
from pylab import *
from scipy.signal import butter, lfilter, filtfilt, freqz
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import skimage
from skimage import data, io, filters, measure
from skimage import img_as_float, img_as_uint

from Inizialization_images import drawShape, Make_a_rectangoular_crop, Find_a_mask, Apply_a_mask, Inside_outside_check
from frequency_analysis import bandpass_filter, multiplot
from minimum_analysis import find_minimum, min_interpol, parab_fit, min_analysis

#------------------------------PARAMETER DEFINITION-----------------------------

# Filter requirements.
order = 6
fs = 25.0           # sample rate, Hz
lowcut = 0.3        # desired cutoff frequency of the filter, Hz
highcut = 2.5

SAMPLING_TIME = 40. # ms (unit)
DIM_X_REDUCED = 40         #image dimension
DIM_Y_REDUCED = 50

MACRO_PIXEL_DIM = 2 #pixel dimention of our new 'macro-pixel'

ZOOM = 4            #minimum fit parameter
t_min = 0           #time range for minimum research
t_max = 2

#-----------------------------IMAGE LOADING-------------------------------------

Images2D = np.loadtxt('inizialized_images.txt')
print Images2D.shape
print Images2D

img_collection_reduced = np.reshape(Images2D, ((1000, DIM_X_REDUCED, DIM_Y_REDUCED)),order='C')
print img_collection_reduced.shape
#plt.matshow(img_collection_reduced[0])
#plt.show()

#--------------------------------Spectrum analysis------------------------------

# Next we take the fourier transformation of the input along the temporal axis
img_spectrum = np.fft.rfftn(img_collection_reduced, axes = [0])
img_spectrum_freq = np.fft.rfftfreq(img_collection_reduced.shape[0], d = 1. / SAMPLING_TIME)

# We take the mean of the spectrum
spectrum_mean = measure.block_reduce(img_spectrum, (1, np.size(img_collection_reduced,1), np.size(img_collection_reduced,2)), np.mean)

xmax = np.amax(img_collection_reduced[:, 5, 7])
xmin = np.amin(img_collection_reduced[:, 5, 7])

plt.plot(img_collection_reduced[300:800, 5, 7])
#plt.show()

#Clean the signal through a bandpass-pass filter

clean_signals = bandpass_filter(img_collection_reduced)

#To have an idea we choose randomly the pixel 5,7
plt.plot(clean_signals[20, 20,:])
#plt.show()

#plt.hist(clean_signals[5, 7, :], bins='auto', range = (xmin , xmax))
#plt.yscale('log')
#plt.show()


#Multiplot of different pixels
#plt.plot = multiplot(10, 40, 10, 10, 40, 10, 650, 750)
#plt.show()


#----------------------------------MINIMUM ANALYSIS-------------------------------

#Identify min and max values
minimum_signal = []
minimum_signal = min_analysis(clean_signals, ZOOM, t_min, t_max)
