# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Ellipse
from matplotlib import cm

import imageio

# Needed to handle exits properly
import sys, traceback

from CustomException import ExpiredException, StyleException

class EllipticalWave:

    # We initialize an EllipticalWave by giving a starting position, a point on
    # a grid, and two vector k: k_1, k_2; defining the two direction of propagation
    def __init__(self, width, height, pivot = [0, 0], start_radius = [1., 1.],\
                    start_angle = 0., velocity = [1., 1.], acceleration = [0., 0.],\
                    lifespan = None, style = 'random', display = False, saveToGif = False):

        # Initialize the grid dimentions
        self.width = width
        self.height = height

        # Creating the grid on which wave will propagate
        self.grid = np.zeros((width, height))

        # Set the pivot, radius and velocity
        self.pivot = pivot
        self.a = start_radius[0]
        self.b = start_radius[1]
        self.angle = start_angle

        self.velocity = velocity
        self.acceleration = acceleration

        self.time = 0
        self.lifespan = lifespan

        self.style = style
        self.display = display

        # Parameters needed for the .gif saving
        self.savetoGif = saveToGif
        self.img_colletion = []

    def updateWave(self, dt = 1.):
        self.a += (dt * self.velocity[0] + 0.5 * dt * dt * self.acceleration[0])
        self.b += (dt * self.velocity[1] + 0.5 * dt * dt * self.acceleration[1])

        self.time += 1

        if self.lifespan != None:
            if self.time > self.lifespan:
                raise ExpiredException

    def resetWave(self, style = 'random'):
        # THe wave is out of bounds, we need to reset it
        if style == 'random':
            self.pivot[0] = np.random.uniform(0, self.width)
            self.pivot[1] = np.random.uniform(0, self.height)

            self.a = np.random.uniform(0.51, 3.)
            self.b = np.random.uniform(0.51, 3.)

            self.velocity[0] = np.random.uniform(0.1, 2.)
            self.velocity[1] = np.random.uniform(0.1, 2.)

        else:
            raise StyleException('Error in EllipticalWave: style %s not supported', (style))

    def isWaveOn(self, pivot = [0, 0], radius = [1., 1.]):
        if pivot != [0, 0]:
            self.pivot = pivot

        if radius[0] != 1. or y_radius[0] != 1.:
            self.a = radius[0]
            self.b = radius[1]

        if self.a == 0 and self.b == 0:
            self.a = 0.01
            self.b = 0.01


        angle = self.angle * 2 * np.pi / 360.

        x = int(math.floor((pivot[0] + self.a * math.cos(angle))))
        y = int(math.floor((pivot[1] + self.a * math.sin(angle))))

        temp_flag = False
        if (x in range(self.width) and y in range(self.height)):
            temp = np.array([[x, y]])
            temp_flag = True

        self.isOn = np.ones((1, 2)) * -1

        key = "%d, %d" % (x, y)
        seen = {key : True}

        stopFlag = True
        oneFound = False

        cross = [[0, -1], [1, 0], [0, 1], [-1, 0]]

        while stopFlag:
            # Our step, we built the supercover proceding with crosses
            # We are South-West of pivot
            if y - pivot[1] < 0 and x - pivot[0] < 0:
                #         Left      Down   Right    Up
                #print "SW", x, y, self.a, self.b
                cross = [[-1, 0], [0, -1], [1, 0], [0, 1]]

            # We are South-Est
            elif y - pivot[1] < 0 and x - pivot[0] > 0:
                #          Down    Right    Up     Left
                #print "SE", x, y, self.a, self.b
                cross = [[0, -1], [1, 0], [0, 1], [-1, 0]]

            # We are North-West
            elif y - pivot[1] > 0 and x - pivot[0] < 0:
                #         Up      Left     Down    Right
                #print "NW", x, y, self.a, self.b
                cross = [[0, 1], [-1, 0], [0, -1], [1, 0]]

            # We are North-Est
            else:
                #         Right    Up       Left    Down
                #print "NE", x, y, self.a, self.b
                cross = [[1, 0], [0, 1], [-1, 0], [0, -1]]

            for cross_ in cross:

                x_ = x + cross_[0]
                y_ = y + cross_[1]

                tmp = 0

                # Check the square in [x, y]
                for i in [0, 1]:
                    for j in [0, 1]:
                        tmp += -1 if (((x_ - pivot[0] + i) * math.cos(angle) +\
                                       (y_ - pivot[1] + j) * math.sin(angle))**2 / self.a**2 +\
                                      ((x_ - pivot[0] + i) * math.sin(angle) -\
                                       (y_ - pivot[1] + j) * math.cos(angle))**2 / self.b**2) < 1. else 1

                key = "%d, %d" % (x_, y_)
                if  key in seen:
                    stopFlag = False

                else:
                    # Check if ellipse has crossed this square
                    if tmp != -4 and tmp != 4:
                        x = x_
                        y = y_
                        seen[key] = True
                        stopFlag = True

                        if (x in range(self.width) and y in range(self.height)):
                            self.isOn = np.append(self.isOn, [[x, y]], axis = 0)
                            oneFound = True

                        # Bring the found direction to the top
                        #cross.insert(0, cross.pop(cross.index(cross_)))

                        break

                    else:
                        stopFlag = False

        if temp_flag:
            self.isOn = np.append(self.isOn, temp, axis = 0)

        if not oneFound:
            raise IndexError

        return self.isOn[1:].astype(int)


    def printWave(self, figure, axis):

        ellipse = Ellipse(xy = self.pivot, width = self.a * 2, height = self.b * 2,\
                            angle = self.angle, color = 'g', fill = False)

        axis.add_artist(ellipse)

        return figure, axis

    def _printWave(self):

        figure, axis = plt.subplots()

        plt.imshow(self.grid, extent = (0, self.width, self.height, 0), interpolation='nearest', cmap = cm.coolwarm)
        plt.colorbar()

        figure, axis = self.printWave(figure, axis)

        plt.xlim(0, self.width)
        plt.ylim(0, self.height)

        axis.set_xticks(range(0, self.width + 1))
        axis.set_yticks(range(0, self.height + 1))
        plt.grid()
        plt.show()

    def run(self, t):

        temp = []

        for dt in range(t):

            try:
                self.updateWave(1)
                temp = self.isWaveOn(pivot = self.pivot, radius = [self.a, self.b])

            except ExpiredException:
                # We re-raise the exception
                raise ExpiredException("EllipticalWave %s has expired is lifespan." % self)

            except IndexError:
                if self.style != None:
                    try:
                        self.resetWave(self.style)

                    except StyleException:
                        # We re-raise the exception
                        raise StyleException("EllipticallWave %s has no style: %s" % (self, self.style))

                    else:
                        temp = self.isWaveOn(pivot = self.pivot, radius = [self.a, self.b])

                        self.grid[:, :] = 0
                        self.grid[temp[:, 0], temp[:, 1]] = 1

                else:
                    raise ExpiredException("EllipticalWave has exit grid borders.")

            else:
                self.grid[:, :] = 0
                self.grid[temp[:, 0], temp[:, 1]] = 1

                if self.display:
                    self._printWave()

                if self.savetoGif:
                    img = self._printWave()
                    filename = "LinearWave/frames/frame_" + str(self.time) + ".png"
                    img.savefig(filename)
                    plt.close(img)
                    self.image_colletion.append(imageio.imread(filename))


        return temp

    # Create a gif of the hystory of the wave propagation
    def saveAsGif(self, filename):
        imageio.mimsave(filename, self.image_colletion)


#width = 15
#height = 10

#wave = EllipticalWave(width, height, pivot = [5, 5], start_radius = [3.5, 5.7], start_angle = 150, display = True)
#wave.run(50)
