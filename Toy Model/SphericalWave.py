# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib import cm

import math
import imageio

# Needed to handle exits properly
import sys, traceback
from CustomException import ExpiredException, StyleException

class SphericalWave:

    # We initialize a SphericalWave by giving a starting position, a point on
    # a grid. Optionally a velocity and a resolution can be passed

    def __init__(self, width, height, pivot = [0, 0], start_radius = 0., \
                    velocity = .1, acceleration = 0., lifespan = None,\
                    style = 'random', display = False, saveToGif = False):

        # Initialize the grid dimentions
        self.width = width
        self.height = height

        # Creating the grid on which wave will propagate
        self.grid = np.zeros((width, height))

        # Set the pivot, radius and velocity
        self.pivot = pivot
        self.radius = start_radius
        self.velocity = velocity
        self.acceleration = acceleration

        self.time = 0
        self.lifespan = lifespan

        self.style = style
        self.display = display

        # Parameters needed for the .gif saving
        self.savetoGif = saveToGif
        self.img_colletion = []


    def updateWave(self, dt, style = 'random'):
        self.radius += (dt * self.velocity + 0.5 *dt * dt * self.acceleration)

        self.time += 1

        if self.lifespan != None:
            if self.time > self.lifespan:
                raise ExpiredException

    def resetWave(self, style = 'random'):
        # The wave is out of bounds, we need to reset it
        if style == 'random':
            self.pivot[0] = np.random.uniform(0, self.width)
            self.pivot[1] = np.random.uniform(0, self.height)

            self.radius = np.random.uniform(0.51, 3.)

            self.velocity = np.random.uniform(0.1, 2.)

        else:
            raise StyleException('Error in SphericalWave: style %s not supported', (style))


    def isWaveOn(self, pivot = [0, 0], radius = 1.):
        if pivot != [0, 0]:
            self.pivot = pivot

        if radius != 1.:
            self.radius = radius

        if self.radius == 0:
            self.radius = 0.01

        x = int(math.floor(pivot[0] + self.radius))
        y = int(math.floor(pivot[1]))

        r2 = self.radius * self.radius

        temp_flag = False
        if (x in range(self.width) and y in range(self.height)):
            temp = np.array([[x, y]])
            temp_flag = True

        self.isOn = np.ones((1, 2)) * -1

        key = "%d, %d" % (x, y)
        seen = {key : True}

        stopFlag = True
        oneFound = False

        while stopFlag:

            # Our step, we built the supercover proceding with crosses
            # We are South-West of pivot
            if y - pivot[1] < 0 and x - pivot[0] < 0:
                #         Left      Down   Right    Up
                cross = [[-1, 0], [0, -1], [1, 0], [0, 1]]

            # We are South-Est
            elif y - pivot[1] < 0 and x - pivot[0] > 0:
                #          Down    Right    Up     Left
                cross = [[0, -1], [1, 0], [0, 1], [-1, 0]]

            # We are North-West
            elif y - pivot[1] > 0 and x - pivot[0] < 0:
                #         Up      Left     Down    Right
                cross = [[0, 1], [-1, 0], [0, -1], [1, 0]]

            # We are North-Est
            else:
                #         Right    Up       Left    Down
                cross = [[1, 0], [0, 1], [-1, 0], [0, -1]]

            for cross_ in cross:

                x_ = x + cross_[0]
                y_ = y + cross_[1]

                tmp = 0

                # Check the square in [x, y]
                for i in [0, 1]:
                    for j in [0, 1]:
                        tmp += -1 if ((x_ - pivot[0] + i)**2 + (y_ - pivot[1] + j)**2) < r2 else 1

                key = "%d, %d" % (x_, y_)
                if  key in seen:
                    stopFlag = False

                else:
                    # Check if circle has crossed this square
                    if tmp != -4 and tmp != 4:
                        x = x_
                        y = y_
                        seen[key] = True
                        stopFlag = True

                        if (x in range(self.width) and y in range(self.height)):
                            self.isOn = np.append(self.isOn, [[x, y]], axis = 0)
                            oneFound = True

                        break

                    else:
                        stopFlag = False

        if temp_flag:
            self.isOn = np.append(self.isOn, temp, axis = 0)

        if not oneFound:
            raise IndexError

        return self.isOn[1:].astype(int)

    def printWave(self, figure, axis):

        circle = plt.Circle(self.pivot, self.radius, color = 'g', fill = False)
        axis.add_artist(circle)

        return figure, axis

    def _printWave(self):

        figure, axis = plt.subplots()

        plt.imshow(self.grid, extent = (0, self.width, self.height, 0), interpolation='nearest', cmap = cm.coolwarm)
        plt.colorbar()

        figure, axis = self.printWave(figure, axis)

        plt.xlim(0, self.width)
        plt.ylim(0, self.height)

        spacing = 1
        minorLocator = MultipleLocator(spacing)
        axis.xaxis.set_minor_locator(minorLocator)
        axis.yaxis.set_minor_locator(minorLocator)
        plt.grid(True, which = 'minor')
        plt.show()

    def run(self, t):

        temp = []

        for dt in range(t):
            try:
                self.updateWave(1)
                temp = self.isWaveOn(pivot = self.pivot, radius = self.radius)

            except ExpiredException:
                # We re-raise the exception
                raise ExpiredException("SphericalWave %s has expired is lifespan." % self)

            except IndexError:
                if self.style != None:
                    try:
                        self.resetWave(self.style)

                    except StyleException:
                        # We re-raise the exception
                        raise StyleException("SphericalWave %s has no style: %s" % (self, self.style))

                    else:
                        temp = self.isWaveOn(pivot = self.pivot, radius = self.radius)

                        self.grid[:, :] = 0
                        self.grid[temp[:, 0], temp[:, 1]] = 1

                else:
                    raise ExpiredException("SphericalWave %s has exit grid borders." % self)

            else:
                self.grid[:, :] = 0
                self.grid[temp[:, 0], temp[:, 1]] = 1

                if self.display:
                    self._printWave()

                if self.savetoGif:
                    img = self._printWave()
                    filename = "SphericalWave/frames/frame_" + str(self.time) + ".png"
                    img.savefig(filename)
                    plt.close(img)
                    self.image_colletion.append(imageio.imread(filename))

        return temp

    # Create a gif of the hystory of the wave propagation
    def saveAsGif(self, filename):
        imageio.mimsave(filename, self.image_colletion)
