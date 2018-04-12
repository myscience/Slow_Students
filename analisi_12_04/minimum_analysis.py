# -*- coding: utf-8 -*-

#--------------------------------------------------
def find_minimum(clean_signals, t_min, t_max):

    min = []
    #iter on time
    for t in range(t_min,t_max):

        if ((clean_signals[t-1]> clean_signals[t]) and (clean_signals[t]< clean_signals[t+1])):
            #min[] contains indexes and clean_signals values where it has a minimum
            min.append(t)

    return min

#--------------------------------------------------

#Makes a parabolic fit of the array data vs t.
#Returns an array 4x zoomed in t, with parabolic fitted values.

def parab_fit(t, data, zoom):

    import numpy as np
    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt

    def parabola(x, a, b, c):
        return a*x**2 + b*x + c

    params, cov = curve_fit(parabola, t, data)
    t1 = np.arange(t[0], t[-1] + 0.25, 0.25)

    #plt.plot(t, data, '.', t1, parabola(t1, *params), '-', color = 'red')

    return parabola(t1, *params)


#---------------------------------------------
#Parabolic interpolation of all the local minimums of clean_signals
#Returns a 3D array with just the local minumum parabolic interpolations different from 0 and 4x zoomed on time

def min_interpol(clean_signals, min_collection, points, zoom):

    import numpy as np

    around_min = np.zeros(points)
    t = np.zeros(points)

    minimum_signal = np.zeros((np.size(clean_signals,0), np.size(clean_signals,1), len(clean_signals[0, 0, :])*4))

    #set the mask for minimum_signal
    #print 'inizio!'
    for x in range(0,np.size(clean_signals,0)):
            for y in range(0,np.size(clean_signals,1)):
                #print clean_signals[x,y,0]
                if np.isnan(clean_signals[x,y, 0]):
                    #print (x,y)
                    minimum_signal[x,y,:] = 'nan'


    #Creates a new 3D array containing only the parabolic fit around local minimums, with a 4*time zoom
    for i in range(0, len(min_collection)):
        k = int(i/np.size(clean_signals,1))
        l = i%(np.size(clean_signals,1))

        for j in min_collection[i][2]:
            #print i
            #Set t as an array of P points around the local minimum time and ar_min as it's relative intensities
            t0 = j-2  #attento qui dovrebbe dipendere da P

            for n in range(0, points):
                t[n] = t0 + n

                # QUI devi aggiustare il fatto che se sei in prossimità di 1000 non puoi interpolare o esci dall'array

                around_min[n] = clean_signals[k, l, int(t0 + n)]
            #print around_min

            #Fit the local minimum with a parabola

            y = parab_fit(t, around_min, zoom)
            #Put parabulas values in minimum signal (4x zoomed in time)
            for n in range(0, len(y)):
                minimum_signal[k, l, int(4*t0 + n)] = y[n]
                #print(k, l, int(4*t0 + n))
                #print(minimum_signal[k, l, int(4*t0 + n)])



    return minimum_signal


#---------------------------------------------

def min_analysis(clean_signals, zoom, points, t_min, t_max):

    from operator import itemgetter
    import numpy as np

    min_collection = []
    #clean_signals is a 34x34x1000 list
    #iter on macro-pixel

    for x in range(0,np.size(clean_signals,0)):
        for y in range(0,np.size(clean_signals,1)):

            clean_signals_xy = clean_signals[x][y]
            time = find_minimum(clean_signals_xy, t_min, t_max)

            min_collection.append([x, y, time])

    #min_collection contains (x, y, [array 2d of time and min sorted for each position])
    #print len(min_collection)
    return min_interpol(clean_signals, min_collection, points, zoom)
