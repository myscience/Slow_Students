
def Inside_outside_check(point, contours):
    #Mi raccomando Contours lo si deve passare come Contours[0]

    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon

    point = Point(point)
    polygon = Polygon(contours)
    return(polygon.contains(point))

#-------------------------------------------------------------------------------

def Make_a_rectangoular_crop(img, topx, topy, bottomx, bottomy):
    cropped  = img[topx:bottomx, topy:bottomy]
    #plt.matshow(cropped)
    #plt.show()
    return(cropped)

#-------------------------------FUNCTION DEFINITION-----------------------------
        #This function drows all contours founded on the mask
def drawShape(img, coordinates, color):
        # In order to draw our line in red
    img = skimage.color.gray2rgb(img)

        # Make sure the coordinates are expressed as integers
    coordinates = coordinates.astype(int)

    img[coordinates[:, 0], coordinates[:, 1]] = color

    return img
#-------------------------------------------------------------------------------


def Apply_a_mask(img_float, Contours_Limit):
    #This function finds and drows contures of an image given.
    #It returns the cropped image as a numpy array of float
    import numpy as np
    from PIL import Image
    import skimage
    from skimage import data, io, filters, measure, img_as_float, img_as_uint
    from matplotlib import pyplot as plt

        #Transform the array to a 2D matrix 100x100
    #img_float = img_float.reshape(100,100)
    #plt.matshow(img_float)
    #plt.show()

        # Find contours at a constant value of Contours_Limit (diciamo 0.45)
    contours = measure.find_contours(img_float, Contours_Limit)

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

    return(contours)

#-------------------------------------------------------------------------------

'''
import numpy as np
from PIL import Image
import skimage
from skimage import data, io, filters, measure, img_as_float, img_as_uint
from matplotlib import pyplot as plt

    #Import the image as a 1D array type float
img_float = img_as_float(io.imread_collection('a.tif'))
#print img_float.size

    #Transform the array to a 2D matrix 100x100
img_float = img_float.reshape(100,100)

img_UP = Make_a_rectangoular_crop(img_float, 0, 0, 19, 100)
img_DOWN = Make_a_rectangoular_crop(img_float, 19, 0, 100, 100)

Contours_UP = Apply_a_mask(img_UP, 0.1635)
Contours_DOWN = Apply_a_mask(img_DOWN, 0.197)

#point = (30,30)
#answer=Inside_outside_check(point, Contours_DOWN[0)
#print answer
'''
