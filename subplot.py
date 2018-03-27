# xstart = pixel on x axis from which plots will begin

def multiplot(xstart, xstop, xsteps, ystart, ystop, ysteps, tmin, tmax):
    
    import matplotlib.pyplot as plt
    import numpy as np
    from SlowWaves1_0 import clean_signals
    
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
    
