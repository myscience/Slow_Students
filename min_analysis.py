# -*- coding: utf-8 -*-

#----------------------------------PHASE ANALYSIS-------------------------------

#Library with all the functions to detects minima of the signals and track their propagation.

#Sets output values of divisions as floating points, as in Python 3
from __future__ import division

#--------------------------------------------------
def minimum_sorted(clean_signals):
    from operator import itemgetter
    min = []
    time = []
    #iter on time
    for t in range(0,900):

        if ((clean_signals[t-1]> clean_signals[t]) and (clean_signals[t]< clean_signals[t+1])):
            #min[] contains indexes and clean_signals values where it has a minimum
            min.append([t, clean_signals[t]])
            time.append(t)
    #Let's sort now min
    min_sorted = sorted(min, key=itemgetter(1), reverse=True)

    return time

#------------------------------------------------------
        
#Makes a parabolic fit of the array data vs t. 
#Returns an array 4x zoomed in t, with parabolic fitted values.

def parab_fit(t, data, zoom):
    import numpy as np
    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt
   
    
    def parabola(x, a, b, c):
        return a*x**2 + b*x + c
    
    params, cov = curve_fit(parabola, t, data)
    t1 = np.arange(t[0], t[-1] + (1/zoom), 1/zoom)
    
    #plt.plot(t, data, '.', t1, parabola(t1, *params), '-', color = 'red')

    return parabola(t1, *params)

#----------------------------------------------------------

#Parabolic interpolation of all the local minimums of clean_signals
#Returns a 3D array with just the local minumum parabolic interpolations different from 0 and 4x zoomed on time  

def min_interpol(clean_signals, min_collection, zoom):
    import numpy as np
    
    points = 5
    
    around_min = np.zeros(points)
    t = np.zeros(points)
    minimum_signal = np.zeros((len(clean_signals[1]), len(clean_signals[2]), len(clean_signals[0, 0, :])*zoom))
    
    #Creates a new 3D array containing only the parabolic fit around local minimums, with a 4*time zoom
    for i in range(0, len(min_collection)):
        k = int(i/len(clean_signals[0]))
        l = i%len(clean_signals[0])
        for j in min_collection[i][2]:
            #Set t as an array of P points around the local minimum time and ar_min as it's relative intensities
            t0 = j - 2  #attento qui dovrebbe dipendere da P
            for n in range(0, points):
                t[n] = t0 + n
                # QUI devi aggiustare il fatto che se sei in prossimità di 1000 non puoi interpolare o esci dall'array
                around_min[n] = clean_signals[k, l, int(t0 + n)]
            #Fit the local minimum with a parabola
            y = parab_fit(t, around_min, zoom)
            #Put parabulas values in minimum signal (zoomed in time)
            for n in range(0, len(y)):
                minimum_signal[k, l, int(zoom*t0 + n)] = y[n]
                #print(k, l, int(zoom*t0 + n))
                #print(minimum_signal[k, l, int(zoom*t0 + n)])
                    
    return minimum_signal

#-----------------------------------------------------------

#Final function that uses all the previous functions
#to return a 3D array (x, y, t) of all the interpolated 
#local minima signals

def min_analysis(clean_signals, zoom):
    from operator import itemgetter

    min_collection = []
    #clean_signals is a 34x34x1000 list
    #iter on macro-pixel
    for x in range(0,len(clean_signals[0])):
        for y in range(0,len(clean_signals[0])):

            t = clean_signals[x][y]
            time = minimum_sorted(t)
            min_collection.append([x, y, time])
            
    #min_collection contains (x, y, [array 2d of time and min sorted for each position])
    #print min_collection
            
    return min_interpol(clean_signals, min_collection, zoom)

#----------------------------------------------------------
    
#Function (still to be fixed) that plots 3 different
# gradient fields of signals, at a given time t
    
def min_grad(min_signals, t):
    
    import numpy as np
    import matplotlib.pyplot as plt

    # Set limits and number of points in grid
    y, x = np.mgrid[0:len(min_signals[0]), 0:len(min_signals[0])]  
    p = min_signals[:, :, t]
    
    #Compute gradients of min_signals map
    dx, dy = np.gradient(p)
    
    #--------------------------
    # First plot: gradient module 
    
    fulgrad = np.sqrt(dx**2 + dy**2)
    plt.imshow(fulgrad, vmin = np.amin(fulgrad),vmax = np.amax(fulgrad))  
    plt.colorbar()
    plt.show()
    
    #--------------------------
    #Second plot: simple gradient field plot
    
    fig, ax = plt.subplots()
    ax.quiver(x, y, dx, dy, p)
    ax.set(aspect=1, title='Gradient field at time %s' %t)
    plt.show()
    
    #---------------------------
    #Third plot: gradient field plot over original image
    #Grid is reduced to half of the original points
    
    skip = (slice(None, None, 2), slice(None, None, 2))

    fig, ax = plt.subplots()
    im = ax.imshow(p, extent=[x.min(), x.max(), y.min(), y.max()])
    ax.quiver(x[skip], y[skip], dx[skip], dy[skip])

    fig.colorbar(im)
    ax.set(aspect=1, title='Gradient field at time %s' % t )
    plt.show()