def  write_gif(minimum_signal):

import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np

# Set up formatting for the movie files
fig = plt.figure(figsize=(len(minimum_signal[0]),len(minimum_signal[1])))
#plt.show()
ax = plt.axes()
img_collection=[]
for t in range(2900,2950):
    frame = plt.imshow(minimum_signal[:,:,t], interpolation='nearest', shape=(34,34), cmap='Blues')
    #plt.show()
    #img_collection.add_subplot(frame)
    img_collection.append([frame])
    #print len(img_collection)
    #plt.show(img_collection)
plt.show(img_collection[3])
ani = matplotlib.animation.ArtistAnimation(fig, img_collection, interval=400)
Writer = matplotlib.animation.writers['ffmpeg']
writer = Writer(fps=3, metadata=dict(artist='Me'))

ani.save('/Users/Mac/Desktop/Uni/LabII/gif/gif.mp4', writer=writer)
