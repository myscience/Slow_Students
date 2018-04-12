
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

#------------------------------PARAMETER DEFINITION-----------------------------

# Filter requirements.
order = 6
fs = 25.0           # sample rate, Hz
lowcut = 0.3        # desired cutoff frequency of the filter, Hz
highcut = 2.5

SAMPLING_TIME = 40. # ms (unit)
DIM_X = 100         #image dimension
DIM_Y = 100

MACRO_PIXEL_DIM = 2 #pixel dimention of our new 'macro-pixel'

ZOOM = 4            #minimum fit parameter
t_min = 0           #time range for minimum research
t_max = 2

#-----------------------------IMAGE LOADING-------------------------------------

path = "/Users/Mac/Desktop/Uni/LabII/data/t1/provevideo"


img_path_list = []
for i in range(1, 2):
    img_path = '1_' + str(i) + '.tif'
    filename = os.path.join(skimage.data_dir, path + img_path)

    img_path_list.append(filename)

# Load all the collection of the images
img_collection_float = img_as_float(io.imread_collection(img_path_list))
print img_collection_float[0,0,0]

img_collection_UP = []
img_collection_DOWN = []

img_collection_UP = Make_a_rectangoular_crop(img_collection_float, 0, 0, 19, 100)
img_collection_DOWN = Make_a_rectangoular_crop(img_collection_float, 20, 0, 100, 100)


#-----------------------Analysis of Down images
DIM_X=80
DIM_Y=100

Contours_DOWN = Find_a_mask(img_collection_DOWN)

'''
#DA VELOCIZZARE QUESTO PUNTO
'''
img_collection_DOWN = Apply_a_mask(img_collection_DOWN, Contours_DOWN, DIM_X, DIM_Y)


img_background = np.zeros((DIM_X, DIM_Y), np.float64)

# Evaluate the background of the images as the mean over the whole set
for i in img_collection_DOWN:
    # Convert all images to float, range -1 to 1, to avoid dangerous overflows
    img_background += i

img_background /= len(img_collection_DOWN)

# Substract from each image the background
for i in img_collection_DOWN:
    i -= img_background

# Now we need to reduce the noise from the images by performing a spatial smoothing
img_collection_reduced = []
img_collection_reduced = measure.block_reduce(img_collection_DOWN, (1, MACRO_PIXEL_DIM, MACRO_PIXEL_DIM), np.mean)
print img_collection_reduced

#------------------------Write on a txt filename

Images_2D = np.reshape(img_collection_reduced, (np.size(img_collection_reduced,0),(np.size(img_collection_reduced,1)*np.size(img_collection_reduced,2))), order='C')
img_collection_reduced.shape
np.savetxt('inizialized_images_PROVA.txt', Images_2D, delimiter = ' ', header ='#numpy 2d array containing image reduced in sequences', newline='\n')
