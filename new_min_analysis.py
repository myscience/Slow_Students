# -*- coding: utf-8 -*-

#Library with all the functions to detect signal's minima and interpolate around them.

# Main differences from the previous version:
# Now "min_analysis" returns a dictionary whose elements are
# 1) 'min_time': a list of lists, eacn one containing the 
# un-quantized minima time for a particoular pixel.
# 2) 'params': a list of arrays containing the quadratic 
# fit parameters of every minimum in each pixel.
# eg. min_analysis['min_time'][998][3] would be the time at wich the
# third minimum of the 998th (19th row, 48th column) pixel occurs
# while min_analysis['params'][998][3] would be its quadratic fit parameters


#Sets output values of divisions as floating points, as in Python 3
from __future__ import division

#----------------------------------------------------

def find_minimum(clean_signal, t_min, t_max):

    min = []
    #iter on time
    for t in range(t_min,t_max):

        if ((clean_signal[t-1]> clean_signal[t]) and (clean_signal[t]< clean_signal[t+1])):
            #min[] contains indexes and clean_signals values where it has a minimum
            min.append(t)

    return min

#-----------------------------------------------------

def parabola(x, a, b, c):
        return a*x**2 + b*x + c
    
#-----------------------------------------------------

def min_interpol(clean_signals, min_collection, points):

    import numpy as np

    around_min = np.zeros(points)
    t = np.zeros(points)
    interpol_min_collection = []
    quadratic_params = []
    
    # Iterates
    for i in range(0, len(min_collection)):
        k = int(i/np.size(clean_signals,1))
        l = i%(np.size(clean_signals,1))
        pixel_minima = []
        pixel_params = []
        
        for j in min_collection[i][2]:
    
            #Sets t as an array of lenght 'points' around the local minimum time and ar_min as it's relative intensities
            t0 = j-int(points/2)  
            
            for n in range(0, points):
                t[n] = t0 + n

                around_min[n] = clean_signals[k, l, int(t0 + n)]

                #Fits the local minimum with a parabola

            params = np.polyfit(t, around_min, 2)
            t_min = -params[1]/(2*params[0])
            
            # Adds parabola's minimum and parameters to the proper pixel's list
            
            pixel_params.append(params)
            pixel_minima.append(t_min)
            
        #Adds each pixel's list to a list of lists
        quadratic_params.append(pixel_params)
        interpol_min_collection.append(pixel_minima)
        
    #Returns a dictionary of two elements
        
    return {'min_time' : interpol_min_collection, 'params' : quadratic_params}


#------------------------------------------
    
def min_analysis(clean_signals, points, t_min, t_max):
#Suggested values for a quadratic fit on a 1000 frames image set are
# point = 5, t_min = 3, t_max = 997         

    import numpy as np

    min_collection = []
   
    #iter on macro-pixels

    for x in range(0,np.size(clean_signals,0)):
        for y in range(0,np.size(clean_signals,1)):

            clean_signals_xy = clean_signals[x][y]
            time = find_minimum(clean_signals_xy, t_min, t_max)

            min_collection.append([x, y, time])
        
    #Be careful the function returns a DICTIONARY in which the element
    # 'min_time' is a collection of all the interpolated minima
    # 'params' are the parameters of each parabola
    return min_interpol(clean_signals, min_collection, points)

#-----------------------------------------------
    
#Creates a .mat variable containing min_collection on chosen path

def min_to_matlab(min_collection):

    import scipy.io
            
    print('Choose the name of the variable you want to create: ')
    name = raw_input()
    #REMEMBER to change the path manually here!
    path = 'C:/Users/Marco/Desktop/' + name + '.mat'
    scipy.io.savemat(path, mdict={name: min_collection})
 
#-------------------------------------------------------

#Produces a plot of the minima's wave at a particular time t
#only considering the pixels for which a minimum occurs in a given 
#range 'dist' around t
#The value of each pixel, if a minimum is near enough, is 
#the value that the interpolated parabola takes at time t
def plot_wave(min_time, params, t, dist, rows = 40, columns = 50):  #ridondante passare clean_signals
# Typically params will be something as " min_analysis['params'] "
# min_time something as " min_analysis['min_time'] "
    import matplotlib.pyplot as plt
    import numpy as np
    
    min_signal = np.zeros((rows, columns))
    
    for x in range(0, rows):
        for y in range(0, columns):
            
            index = x*columns + y
            
            if params[index] == []:
                min_signal[x, y] = np.nan
            
            else:
                time_dist = [abs(z - t) for z in min_time[index]]
                i = np.argmin(time_dist)
                
                if time_dist[i] < dist:
                    #print(parabola(t, *params[index][i]), i, index)
                    min_signal[x, y] = parabola(t, *params[index][i])
                else:
                    min_signal[x, y] = 0
                    
    plt.imshow(min_signal, interpolation = 'nearest')
    plt.colorbar()
