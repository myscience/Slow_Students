# -*- coding: utf-8 -*-

# Minimum squared error between two images, not useful
def mse(image1, image2):
    import numpy as np
    # the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
    err = np.nansum((image1 - image2) ** 2)
    err /= float(len(image1)* len(image1[0]))
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
    return err


def min_collection_reshape(min_collection):
    new_collection = []
    for i in range(0, len(min_collection)):
        for j in range(0, len(min_collection[i][2])):
            new_collection.append([min_collection[i][0], min_collection[i][1], min_collection[i][2][j]])
            
    return new_collection
        
def min_distance(min1, min2):
    ds = 1 # weight of spatial distance
    dt = 1 # weight of temporal distance
    
    #Calculating distance between two minima as a weighted Manhattan distance
    dist = ds*(abs(min1[0] - min2[0]) + abs(min1[1] - min2[1])) + dt*(abs(min1[2] - min2[2]))
    
    return dist

# 'wave' has to be a collection of minima as 'new_collection'
def waves_distance(waveA, waveB, penality):
    import numpy as np
    
    if len(waveA) < len(waveB):
        wave1 = waveA
        wave2 = waveB
    else:
        wave1 = waveB
        wave2 = waveA
    
    distances = np.zeros((len(wave1), len(wave2)))
    
    #ID = []
    # Iterating over every pair of minima (N1*N2)
    for i in range(0, len(wave1)):
        for j in range(0, len(wave2)):
            # Create a list of min_distances with associated IDs
            distances[i, j] =  min_distance(wave1[i], wave2[j])
            #ID.append([i, j])
            
    N1 = min(len(wave1), len(wave2))
    N2 = max(len(wave1), len(wave2))
     
    #iteration until matched_dist remains unchenged
    matched_pair = np.empty(N1)
    matched_pair[:] = np.nan
    matched_dist = np.zeros(N1)
    matched_dist[:] = np.nan
    unmatched_dist = np.empty(N2-N1)
    unmatched_dist[:] = np.nan
    
    dist_tot_m = dist_tot_matched(distances, matched_dist, matched_pair, N1, N2)
    dist_tot_u = (distances, unmatched_dist, matched_pair, N2-N1)
    
    return dist_tot_m + penality*dist_tot_u

def dist_tot_matched(distances, matched_dist, matched_pair, N1, N2):
    import numpy as np
    count = 1

    while count != 0:
        count = 0
        # itero su tutti i minimi che voglio matchare
        for i in range(0,N1):
            k = 0
            while k < N1:
                #print(i, k)
                # trovo minimo in wave2 a cui vogli o accoppiare il mio
                pair = np.argpartition(distances[:,i], k)[k]
                #se è già accoppiato confronto
                if pair in matched_pair:
                    #print('matched')
                    #trovo minimo di wave 1 a cui è già accoppiato
                    j = list(matched_pair).index(pair)
                    # se è proprio lui sono contento
                    if i == j:
                        k = N1
                    #se è un match migliore lo sostituisco
                    elif np.partition(distances[:,i], k)[k] < matched_dist[j]:
                        matched_dist[i] = np.partition(distances[:,i], k)[k]
                        matched_pair[i] = int(pair)
                        matched_pair[j] = np.nan
                        k = N1
                        count += 1
                    #altimenti cerco un altro match
                    else:
                        k += 1
                # se minimo in wave2 non accoppiato lo accoppio subito a quello di wave1
                else:
                    #print('unmatched')
                    matched_dist[i] = np.partition(distances[:,i], k)[k]
                    matched_pair[i] = int(pair)
                    k = N1
                    count += 1
            #print(count)
    return np.sum(matched_dist)
        

def dist_tot_unmatched(distances, unmatched_dist, matched_pair, N2, N1):
    import numpy as np
    #Per i minimi in eccesso cerco semplicemnte il minimo più vicini in wave1
    k = 0
    for i in range(0, N2-N1):
        if i not in list(matched_pair): 
            unmatched_dist[k](np.min(distances[i,:])
            k += 1
    return np.sum(unmatched_dist)
    
