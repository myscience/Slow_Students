def  write_gif(minimum_signal):
    import cv2

    import matplotlib.pyplot as plt
    import matplotlib.animation
    import numpy as np

    from PIL import Image

    import heatmap
    import seaborn as sns

    import imageio

    clouster = np.zeros((34,34))
    #images = []
    minimum = np.amin(minimum_signal)
    maximum = np.amax(minimum_signal)
    hm = heatmap.Heatmap()

    images = []

    #from moviepy.editor import VideoClip
    import cv2

    import matplotlib.pyplot as plt
    import matplotlib.animation
    import numpy as np

    # Set up formatting for the movie files

    fig = plt.figure(figsize=(34,34))
    #plt.show()
    ax = plt.axes()
    X, Y = np.meshgrid(33, 33)
    img_collection=[]
    for t in range(2900,2950):
        frame = plt.imshow(minimum_signal[:,:,t], interpolation='nearest', shape=(34,34), cmap='Blues')
        #plt.show()
        #img_collection.add_subplot(frame)
        img_collection.append([frame])
        #print len(img_collection)
        #plt.show(img_collection)
    #plt.show(img_collection[3])
    ani = matplotlib.animation.ArtistAnimation(fig, img_collection, interval=400)
    Writer = matplotlib.animation.writers['ffmpeg']
    writer = Writer(fps=3, metadata=dict(artist='Me'))

    ani.save('/Users/Mac/Desktop/Uni/LabII/gif/gif.mp4', writer=writer)
