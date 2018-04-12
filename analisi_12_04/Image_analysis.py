#===============================================================================
#                                       MAIN
#===============================================================================

import os
import numpy as np
from pylab import *
from scipy.signal import butter, lfilter, filtfilt, freqz, hilbert
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
lowcut = 0.1        # desired cutoff frequency of the filter, Hz
highcut = 1.5

SAMPLING_TIME = 40. # ms (unit)
DIM_X_REDUCED = 40         #image dimension
DIM_Y_REDUCED = 50

MACRO_PIXEL_DIM = 2 #pixel dimention of our new 'macro-pixel'

ZOOM = 4            #minimum fit parameter
t_min = 3           #time range for minimum research
t_max = 997
points = 5

#-----------------------------IMAGE LOADING-------------------------------------

Images2D = np.loadtxt('inizialized_images_NB.txt')
#print Images2D.shape
#print Images2D

fig, ax = plt.subplots()
img_collection_reduced = np.reshape(Images2D, ((1000, DIM_X_REDUCED, DIM_Y_REDUCED)),order='C')

#print img_collection_reduced.shape

#plt.matshow(img_collection_reduced[0])
#plt.colorbar()
#plt.show()

#--------------------------------Spectrum analysis------------------------------


# Next we take the fourier transformation of the input along the temporal axis
img_spectrum = np.fft.rfftn(img_collection_reduced, axes = [0])
img_spectrum_freq = np.fft.rfftfreq(img_collection_reduced.shape[0], d = 1. / SAMPLING_TIME)

# We take the mean of the spectrum
spectrum_mean = measure.block_reduce(img_spectrum, (1, np.size(img_collection_reduced,1), np.size(img_collection_reduced,2)), np.nanmean)

#print type(spectrum_mean)
print spectrum_mean.shape

xmax = np.amax(img_collection_reduced[:, 20, 21])
xmin = np.amin(img_collection_reduced[:, 20, 21])

#plt.plot(img_spectrum_freq[0:100], (spectrum_mean[0:100, 0, 0].real)**2)
#plt.show()

#Clean the signal through a bandpass-pass filter

clean_signals = bandpass_filter(img_collection_reduced)

#To have an idea we choose randomly the pixel 5,7
#plt.plot(clean_signals[20, 20,:])
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
minimum_signal = min_analysis(clean_signals, ZOOM, points, t_min, t_max)

'''
#save frame images
for t in range(2015,2060):

    masked_minimum_signal = np.ma.array(minimum_signal[:,:,t], mask=np.isnan(minimum_signal[:,:,t]))
    cmap = plt.cm.jet
    cmap.set_bad(color='white', alpha=1.)
    #current_cmap = plt.cm.get_cmap()
    #current_cmap.set_bad(color='red')
    plt.imshow(masked_minimum_signal, interpolation='nearest', shape=(40,50), cmap=cmap)
    #plt.show()
    img_name = 'frame' + str(t) + '.png'
    plt.savefig(img_name)


#create a gif

import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np

# Set up formatting for the movie files

fig = plt.figure(figsize=(40,50))
#plt.show()
ax = plt.axes()
img_collection=[]
for t in range(2025,2080):
    masked_minimum_signal = np.ma.array(minimum_signal[:,:,t], mask=np.isnan(minimum_signal[:,:,t]))
    cmap = plt.cm.jet
    cmap.set_bad(color='white', alpha=1.)
    frame = plt.imshow(masked_minimum_signal, interpolation='nearest', shape=(40,50), cmap=cmap)
    #plt.show()
    #img_collection.add_subplot(frame)
    img_collection.append([frame])
    #print len(img_collection)
    #plt.show(img_collection)

plt.show(img_collection[3])
ani = matplotlib.animation.ArtistAnimation(fig, img_collection, interval=400)
Writer = matplotlib.animation.writers['ffmpeg']
writer = Writer(fps=3, metadata=dict(artist='Me'))

ani.save('/Users/Mac/Desktop/Slow_Students-master/analisi_separata/frame/GIF_NUOVA.mp4', writer=writer)

'''

#-------------------------------Hilbert transformation

minimum_signal_Transformed = (hilbert(minimum_signal[10,10, :]))
minimum_phase=np.angle(minimum_signal_Transformed)

plt.plot(minimum_phase[:])
plt.show()

clean_signal_Transformed = (hilbert(clean_signals[10,10, :]))
clean_phase=np.angle(clean_signal_Transformed)


plt.plot(clean_phase[:])
plt.show()
