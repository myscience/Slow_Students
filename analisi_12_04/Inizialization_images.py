
#-------------------------------FUNCTION DEFINITION-----------------------------
def Make_a_rectangoular_crop(img_collection, topx, topy, bottomx, bottomy):

    #import matplotlib.pyplot as plt
    from PIL import Image

    img_collection_cropped = []
    #img = ((100,100))

    for i in range(len(img_collection)):
        img = img_collection[i]

        cropped  = img[topx:bottomx, topy:bottomy]
        img_collection_cropped.append(cropped)
        #plt.matshow(cropped)
        #plt.show()

    return(img_collection_cropped)


#--------------------------Contours plotting------------------------------------


def drawShape(img, coordinates, color):
        #This function drows all contours founded on the mask
        # In order to draw our line in red
    img = skimage.color.gray2rgb(img)

        # Make sure the coordinates are expressed as integers
    coordinates = coordinates.astype(int)

    img[coordinates[:, 0], coordinates[:, 1]] = color

    return img


def Contours_printing(img_float, contours):
        # Display the image and plot all contours found
    fig, ax = plt.subplots()
    ax.imshow(img_float, interpolation='nearest', cmap=plt.cm.gray)

    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

        #Create a Black mask of the same dimentions of the image
    mask = np.zeros_like(img_float) # Create mask where white is what we want, black otherwise

    #Drow contours on 'mask'
    for contour in contours:
        mask = drawShape(mask, contour, [255, 0, 0])

#------------------------Find best contour--------------------------------------

def Find_Contours(img):

    import numpy as np
    from PIL import Image
    import skimage
    from skimage import data, io, filters, measure, img_as_float, img_as_uint
    from matplotlib import pyplot as plt

    Contour_Limit = 0.05

    contour = []

    while (len(contour) != 1):
        contour = measure.find_contours(img, Contour_Limit)
        Contour_Limit += 0.001
        #print Contour_Limit

    return contour

def Find_a_mask(img_collection):
    #This function finds and drows contures of an image given.
    #It returns the cropped image as a numpy array of float

        # Find contours the best Contour_Limit for the first image of the serie
    contours = Find_Contours(img_collection[0])

    return(contours)

#-------------------------------------------------------------------------------


def Inside_outside_check(point, contours):
    #Mi raccomando Contours lo si deve passare come Contours[0]

    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon

    point = Point(point)
    polygon = Polygon(contours)
    return(polygon.contains(point))

#-------------------------------------------------------------------------------

def Apply_a_mask(img_collection, Contours, DIM_X, DIM_Y):

    import numpy as np
    #import matplotlib.pyplot as plt

    img_collection=np.asarray(img_collection)

    #print len(img_collection)

    for t in range(len(img_collection)):

        for x in range(DIM_X):
            for y in range(DIM_Y):
                point = (x,y)
                if not(Inside_outside_check(point, Contours[0])):
                    #se il punto e' nel contorno
                    img_collection[t,x,y] = 'nan'
        #plt.matshow(img_collection[t])
        #plt.show()

    return img_collection
