
#===============================================================================
#                           FUNCTION DEFINITION
#===============================================================================

#-------------------------------------------------------------------------------

def multiplot(xstart, xstop, xsteps, ystart, ystop, ysteps, tmin, tmax):

    import matplotlib.pyplot as plt
    import numpy as np

    xsize = len(np.arange(xstart, xstop, xsteps))
    ysize = len(np.arange(ystart, ystop, ysteps))

    # Multiplot construction
    f, axarr = plt.subplots(xsize, ysize, figsize=(5, 5))
    k = -1
    for i in range (xstart, xstop, xsteps):
        k += 1
        l = -1
        for j in range (ystart, ystop, ysteps):
            l += 1
            axarr[k, l].plot(clean_signals[i, j, tmin:tmax])
            axarr[k, l].set_title('Pixel [%s,%s]' % (i, j), fontsize = 8)


    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    plt.setp([a.get_xticklabels() for a in axarr[1, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:, 2]], visible=False)
    plt.show();

#=====================================SIGNAL FILTER=============================
def bandpass_filter(data): # data has to be a 3D array

    from scipy.signal import butter, lfilter
    from numpy import reshape

    # Filter requirements.
    order = 6     # order of the Butterworth filter
    fs = 25.0       # sample rate, Hz
    lowcut = 0.3  # desired cutoff frequency of the high-pass filter, Hz
    highcut = 2.5  # desired cutoff frequency of the low-pass filter, Hz

    #Definition of the Butterworth bandpass filter

    def butter_bandpass_filter(data, lowcut, highcut, fs, order):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq

        b, a = butter(order, [low, high], btype='band')
        y = lfilter(b, a, data)
        return y

    #Application of the filter to the dataset

    clean_signals = []

    for i in range (len(data[1])):
        for j in range (len(data[2])):
            clean_signals.append([butter_bandpass_filter(data[:, i, j], lowcut, highcut, fs, order)])

    clean_signals = reshape(clean_signals, (34, 34, 1000))

    return clean_signals

#-------------------------------------------------------------------------------
#===============================================================================
#                               MAIN
#===============================================================================

# coding=utf-8
import os
import numpy as np
# np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
import skimage
from skimage import data, io, filters, measure
from skimage import img_as_float, img_as_uint
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt

SAMPLING_TIME = 40. # ms (unit)

    #IMAGE LOADING

path = "/Users/Mac/Desktop/Uni/LabII/data/t1/provevideo"

img_path_list = []
for i in range(1, 1001):
    img_path = '1_' + str(i) + '.tif'
    filename = os.path.join(skimage.data_dir, path + img_path)

    img_path_list.append(filename)

    #APPLY A MASK

# Load all the collection of the images
img_collection_float = img_as_float(io.imread_collection(img_path_list))

#Transform the array to a 2D matrix 100x100
img_collection_float = img_collection_float.reshape(1000,100,100)

    #BACKGROUND CLEANING

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


#---------------------------------ANALISI DEL SEGNALE----------------------
# Next we take the fourier transformation of the input along the temporal axis

img_spectrum = np.fft.rfftn(img_collection_reduced, axes = [0])
img_spectrum_freq = np.fft.rfftfreq(img_collection_reduced.shape[0], d = 1. / SAMPLING_TIME)

# We take the mean of the spectrum
spectrum_mean = measure.block_reduce(img_spectrum, (1, 34, 34), np.mean)


xmax = np.amax(img_collection_reduced[:, 20, 10])
xmin = np.amin(img_collection_reduced[:, 20, 10])

#LOW Filter

#Clean the signal through a low-pass filter
clean_signals = bandpass_filter(img_collection_reduced)


#----------------------------------PHASE ANALYSIS-------------------------------

#Identify min and max values


#--------------------------------------------------
def minimum_sorted(clean_signals):
    min = []
    #iter on time
    for t in range(650,750):

        if ((clean_signals[t-1]> clean_signals[t]) and (clean_signals[t]< clean_signals[t+1])):
            #min[] contains indexes and clean_signals values where it has a minimum
            min.append([t, clean_signals[t]])

    #Let's sort now min
    min_sorted = sorted(min, key=itemgetter(1), reverse=True)

    return min_sorted
#----------------------------------------------------

from operator import itemgetter

min_collection = []
#clean_signals is a 34x34x1000 list
#iter on macro-pixel
for x in range(0,34):
    for y in range(0,34):

        time = clean_signals[x][y]
        min_sorted = minimum_sorted(time)
        min_collection.append([x, y, min_collection])
#min_collection contains (x, y, [array 2d of time and min sorted for each position])
#print min_collection
