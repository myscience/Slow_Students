# coding=utf-8

import os
import numpy as np
# np.set_printoptions(threshold=np.inf)

import matplotlib.pyplot as plt
import skimage
from skimage import data, io, filters, measure
from skimage import img_as_float, img_as_uint

SAMPLING_TIME = 40. # ms (unit)

path = "/home/paolo/Scrivania/Università/Human Brain Project/Slow Waves Project/data/t1/provevideo"

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


xmax = np.amax(img_collection_reduced[:, 20, 10])
xmin = np.amin(img_collection_reduced[:, 20, 10])

#plt.hist(img_collection_reduced[:, 20,10], bins='auto', range = (xmin , xmax))
#plt.show()

#plt.plot(img_collection_reduced[:, 15, 17])
#plt.show()



plt.plot(img_spectrum_freq[0:100], (spectrum_mean[0:100, 0, 0].real)**2)
plt.show()

#spectrum = ft.real**2
#max_freq = spectrum.argmax()

#t = np.arange(1000)
#plt.plot(t, img_collection_reduced[:, 5, 7])
#plt.show()
