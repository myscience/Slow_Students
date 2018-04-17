# -*- coding: utf-8 -*-

from minimum_analysis import find_minimum

def find_maximum(clean_signal, t_min, t_max):

    min = []
    #iter on time
    for t in range(t_min,t_max):

        if ((clean_signal[t-1]< clean_signal[t]) and (clean_signal[t]> clean_signal[t+1])):
            #min[] contains indexes and clean_signals values where it has a maximum
            min.append(t)

    return min

#--------------------------------------------------
    
# This function could be useful both to study how the slow-waves 
# amplitude evolves in time (it would be interesting to observe their amplitude decreasing)
# or to clean min/max collections from spurious inflection points
# arising from the signal's filtration
    
#Returns an array of jump amplitudes for all clean_signals
# minima/maxima has to be a 1D array of lists like [[x1, y1, [t_minima_1]], [x2, y2, [t_minima_2]], ecc...]

def amplitudes(clean_signals, min_collection, max_collection):
    
    amplitudes = []
    
    # Iteration over pixels of the image
    for i in range(0, len(clean_signals[:, 0, 0])):
        for j in range(0, len(clean_signals[0, :, 0])):
            pixel = i*len(clean_signals[0, :, 0]) + j
            
            # Checking if the pixel has any minima
            if min_collection[pixel][2] != []:
                
                # Cleaning min_collection and max_collection
                #of the pixel so that they have equal lenght
                if max_collection[pixel][2][0] < min_collection[pixel][2][0]:
                    a = max_collection[pixel][2][0]
                    max_collection[pixel][2].remove(a)
        
                if max_collection[pixel][2][-1] < min_collection[pixel][2][-1]:
                    a = min_collection[pixel][2][-1]
                    min_collection[pixel][2].remove(a)
        
               #Print error if, after cleaning, min_collection and max_collection
               #of the pixel still have different lenght
                if len(min_collection[pixel][2]) != len(max_collection[pixel][2]):
                    print('Error occured, number of maxima and minima differs in pixel [%s, %s]' % (i, j))
        
        # Creating an array of down-up amplitudes for the pixel (i, j)
            amplitudes_ij = []
        
            for t in range(0, len(min_collection[pixel][2])):
                amplitudes_ij.append(clean_signals[i, j, max_collection[pixel][2][t]] - clean_signals[i, j, min_collection[pixel][2][t]])
        
            #print (pixel, i, j, amplitudes_ij)
        
            # Filling an array with the single pixel amplitude's array as a list
            amplitudes.append(amplitudes_ij)
        
    return amplitudes