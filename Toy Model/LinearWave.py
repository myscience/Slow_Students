# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.ticker import MultipleLocator
from matplotlib import cm

import imageio

# Needed to handle exits properly
import sys, traceback
from CustomException import ExpiredException, StyleException

class LinearWave:

    # We initialize a LinearWave by giving a starting position, a point on the
    # grid, and a vector k defining the normal to the linear wave front and the
    # direction of propagation. Optionally a velocity can be passed
    def __init__(self, width, height, pivot=[0, 0],  angle = 0., velocity = .1,\
                    acceleration = 0., style = 'random', lifespan = None, \
                    display = False, saveToGif = False):

        # Initialize the grid dimentions
        self.width = width
        self.height = height

        # Creating the grid on which wave will propagate
        self.grid = np.zeros((width, height))

        # Set the k vector
        self.k = [math.cos(angle), math.sin(angle)]

        # Set the pivot and velocity
        self.pivot = pivot
        self.velocity = velocity
        self.acceleration = acceleration

        self.wave = np.zeros(1)

        # The extremum of the wave from on the grid
        self.x0 = 0.
        self.x1 = 0.
        self.y0 = 0.
        self.y1 = 0.

        self.lifespan = lifespan
        self.time = 0

        self.style = style
        self.display = display

        # Parameters needed for the .gif saving
        self.savetoGif = saveToGif
        self.img_colletion = []

    def computeWave(self, pivot, k):
        self.pivot = pivot
        self.k = k

        # First we compute the extremum of out wave given a pivot on the grid
        # and a wave vector k

        # The equation for the line is the following
        if k[1] != 0:
            self.alert = False
            m = - (k[0] / k[1])

            self.x = np.arange(0, self.width, 0.01)
            self.y = pivot[1] + m * (self.x - pivot[0])

        else:
            self.alert = True
            self.y = np.arange(0, self.height, 0.01)
            self.x = np.ones(len(self.y)) * pivot[0]

        mask = (self.y < self.height) & (self.y > 0)

        # We check if it's not the case k[1] == 0
        if ((len(self.x[mask]) > 0) and (not (self.alert))):
            self.x0 = min(self.x[mask])
            self.x1 = max(self.x[mask])

            idx_x0 = np.ma.array(self.x, mask = ~mask).argmin()
            idx_x1 = np.ma.array(self.x, mask = ~mask).argmax()

            self.y0 = self.y[idx_x0]
            self.y1 = self.y[idx_x1]

        # This is the case k[1] == 0
        elif ((len(self.x[mask]) > 0) and (self.alert)):
            self.x0 = self.x1 = self.pivot[0]

            # We manually set the y extremum
            self.y0 = 0
            self.y1 = self.height - 1e-5

        else:
            self.x0 = self.y0 = self.x1 = self.y1 = 0.


    def isWaveOn(self, x0 = 0, x1 = 0, y0 = 0, y1 = 0):
        if x0 != 0 or x1 != 0 or y0 != 0 or y1 != 0:
            self.x0 = x0
            self.x1 = x1
            self.y0 = y0
            self.y1 = y1

        dx = abs(self.x1 - self.x0)
        dy = abs(self.y1 - self.y0)

        x = int(math.floor(self.x0))
        y = int(math.floor(self.y0))

        n = 1

        self.isOn = np.zeros((1, 2))

        if (dx == 0):
            x_inc = 0
            error = 1e10

        elif (self.x1 > self.x0):
            x_inc = 1
            n += int(math.floor(self.x1)) - x
            error = (math.floor(self.x0) + 1 - self.x0) * dy

        else:
            x_inc = -1
            n += x - int(math.floor(self.x1))
            error = (self.x0 - math.floor(self.x0)) * dy

        if (dy == 0):
            y_inc = 0
            error -= 1e10

        elif (self.y1 > self.y0):
            y_inc = 1
            n += int(math.floor(self.y1)) - y
            error -= (math.floor(self.y0) + 1 - self.y0) * dx

        else:
            y_inc = -1
            n += y - int(math.floor(self.y1))
            error -= (self.y0 - math.floor(self.y0)) * dx

        if (dx == 0) and (dy == 0):
            n = 0

        oneFound = False
        while n > 0:
            self.isOn = np.append(self.isOn, [[x, y]], axis = 0)
            oneFound = True

            if (error > 0):
                y += y_inc
                error -= dx
            else:
                x += x_inc
                error += dy

            n -= 1

        if not oneFound:
            raise IndexError

        return self.isOn[1:].astype(int)

    def updateWave(self, dt):
        self.pivot[0] += self.k[0] * (dt * self.velocity + 0.5 * dt * dt * self.acceleration)
        self.pivot[1] += self.k[1] * (dt * self.velocity + 0.5 * dt * dt * self.acceleration)

        self.time += 1

        if self.lifespan != None:
            if self.time > self.lifespan:
                raise ExpiredException

    def resetWave(self, style = 'random'):
        # The wave is out of bounds, we need to reset it
       if style == 'random':
           self.pivot[0] = np.random.uniform(0, self.width)
           self.pivot[1] = np.random.uniform(0, self.height)
           self.k[0] = np.random.uniform(-2, 2)
           self.k[1] = np.random.uniform(-2, 2)

       elif style == 'v_linear':
           self.pivot[0] = 0
           self.pivot[1] = self.height / 2
           self.k[0] = 1.
           self.k[1] = 0.

       elif style == 'h_linear':
           self.pivot[0] = self.width / 2
           self.pivot[1] = 0.
           self.k[0] = 0.
           self.k[1] = 1.

       else:
           raise StyleException('Error in LinearWave: style %s not supported"', (style))

    def printWave(self, figure, axis):

        axis.plot(self.x, self.y)
        axis.plot(self.pivot[0], self.pivot[1], marker='o', markersize = 3, color = "red")
        axis.plot(self.x0, self.y0, marker='o', markersize = 4, color = "blue")
        axis.plot(self.x1, self.y1, marker='o', markersize = 4, color = "blue")
        axis.quiver(self.pivot[0], self.pivot[1], self.k[0], self.k[1], color='g', scale_units = 'xy')

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
                self.computeWave(self.pivot, self.k)

                temp = self.isWaveOn()

            except ExpiredException:
                raise ExpiredException("LinearWave %s has expired is lifespan." % self)

            except IndexError:
                if self.style != None:
                    try:
                        self.resetWave(self.style)

                    except StyleException:
                        # We re-raise the exception
                        raise StyleException("LinearWave %s has no style: %s" % (self, self.style))

                    else:
                        self.computeWave(self.pivot, self.k)
                        temp = self.isWaveOn()

                        self.grid[:, :] = 0
                        self.grid[temp[:, 0], temp[:, 1]] = 1

                else:
                    raise ExpiredException("LinearWave %s has exit grid borders." % self)
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

#width = 10
#height = 10
#wave = LinearWave(width, height, \
#    pivot = [np.random.uniform(0, width), np.random.uniform(0, height)],\
#    k = [np.random.uniform(-2, 2), np.random.uniform(-2, 2)], \
#    velocity = np.random.uniform(0.1, 2), display = True )

#wave.run(50)
