import os
import numpy as np
from pylab import *
from scipy.signal import butter, lfilter, filtfilt, freqz
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
# np.set_printoptions(threshold=np.inf)

import skimage
from skimage import data, io, filters, measure
from skimage import img_as_float, img_as_uint


# Filter requirements.
order = 6
fs = 25.0       # sample rate, Hz
lowcut = 0.3  # desired cutoff frequency of the filter, Hz
highcut = 2.5

SAMPLING_TIME = 40. # ms (unit)

path = 'C:/Users/Marco/Desktop/WaveScalES/Topo1/171111/mouse_3/deep/t1/provevideo'

img_path_list = []
for i in range(1, 1001):
    img_path = '1_' + str(i) + '.tif'
    filename = os.path.join(skimage.data_dir, path + img_path)

    img_path_list.append(filename)

# Load all the collection of the images
img_collection_float = img_as_float(io.imread_collection(img_path_list))

img_background = np.zeros((100, 100), np.float64)

# Evaluate the background of the images as the mean over the whole set
for i in img_collection_float:
    # Convert all images to float, range -1 to 1, to avoid dangerous overflows
    img_background += i

img_background /= len(img_collection_float)

# Substract from each image the background
for i in img_collection_float:
    i -= img_background

# Now we need to reduce the noise from the images by performing a spatial smoothing
img_collection_reduced = measure.block_reduce(img_collection_float, (1, 3, 3), np.mean)
print img_collection_reduced.shape

# Next we take the fourier transformation of the input along the temporal axis
img_spectrum = np.fft.rfftn(img_collection_reduced, axes = [0])
img_spectrum_freq = np.fft.rfftfreq(img_collection_reduced.shape[0], d = 1. / SAMPLING_TIME)

# We take the mean of the spectrum
spectrum_mean = measure.block_reduce(img_spectrum, (1, 34, 34), np.mean)


xmax = np.amax(img_collection_reduced[:, 5, 7])
xmin = np.amin(img_collection_reduced[:, 5, 7])

#plt.hist(img_collection_reduced[:, 5,7], bins='auto', range = (xmin , xmax))
#plt.show()

plt.plot(img_collection_reduced[300:800, 5, 7])
plt.show()

#plt.plot(img_spectrum_freq[0:100], (img_spectrum[0:100, 5, 7].real)**2)
#plt.show()

#spectrum = ft.real**2
#max_freq = spectrum.argmax()

#t = np.arange(1000)
#plt.plot(t, img_collection_reduced[:, 5, 7])
#plt.show()

#Clean the signal through a low-pass filter 
from signal_filter import bandpass_filter

clean_signals = bandpass_filter(img_collection_reduced)

plt.plot(clean_signals[5, 7,:])
plt.show()

plt.hist(clean_signals[5, 7, :], bins='auto', range = (xmin , xmax))
#plt.yscale('log')
plt.show()

from subplot import multiplot

plt.plot = multiplot(10, 40, 10, 10, 40, 10, 650, 750)
plt.show()
