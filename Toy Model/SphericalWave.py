# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib import cm

import math
import imageio


# Needed to handle exits properly
import sys, traceback

class SphericalWave:

    # We initialize a SphericalWave by giving a starting position, a point on
    # a grid. Optionally a velocity and a resolution can be passed

    def __init__(self, width, height, pivot = [0, 0], start_radius = 0., velocity = .1):

        # Initialize the grid dimentions
        self.width = width
        self.height = height

        # Creating the grid on which wave will propagate
        self.grid = np.zeros((width, height))

        # Set the pivot, radius and velocity
        self.pivot = pivot
        self.radius = start_radius
        self.velocity = velocity

    def updateWave(self, dt, style = 'random'):
        self.radius += dt * self.velocity

        # Check if the wave is out of bound
        # Evaluate the distances from the corners
        c_1 = (self.pivot[0]**2 + self.pivot[1]**2)
        c_2 = (self.width - self.pivot[0])**2 + self.pivot[1]**2
        c_3 = self.pivot[0]**2 + (self.height - self.pivot[1])**2
        c_4 = (self.width - self.pivot[0])**2 + (self.height - self.pivot[1])**2

        if (self.radius * self.radius) > max(c_1, c_2, c_3, c_4):
            try:
                if style == 'random':
                    self.pivot[0] = np.random.uniform(0, self.width)
                    self.pivot[1] = np.random.uniform(0, self.height)

                    self.radius = np.random.uniform(0.1, 3.)

                else:
                    raise ValueError('Error in SphericalWave: style %s not supported', (style))

            except ValueError as err:
                traceback.print_exc(file=sys.stdout)
                sys.exit(0)

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

        flag = True
        while flag:
            # Our step, we built the supercover proceding with crosses
            if y - pivot[1] <= 0:
                cross = [[0, -1], [-1, 0], [1, 0], [0, 1]]
            else:
                cross = [[0, 1], [1, 0], [-1, 0], [0, -1]]

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
                    flag = False

                else:
                    # Check if circle has crossed this square
                    if tmp != -4 and tmp != 4:
                        x = x_
                        y = y_
                        seen[key] = True
                        flag = True

                        if (x in range(self.width) and y in range(self.height)):
                            self.isOn = np.append(self.isOn, [[x, y]], axis = 0)

                        break

                    else:
                        flag = False

        if temp_flag:
            self.isOn = np.append(self.isOn, temp, axis = 0)

        return self.isOn[1:]


    def printWave(self, figure, axis):

        circle = plt.Circle(self.pivot, self.radius, color = 'g', fill = False)
        axis.add_artist(circle)

        return figure, axis

    def run(self, t, style = 'random'):

        for dt in range(t):
            self.updateWave(1, style)

            self.grid[:, :] = 0
            temp = self.isWaveOn(pivot = self.pivot, radius = self.radius).astype(int)
            self.grid[temp[:, 1], temp[:, 0]] = 1

        return temp

#wave = SphericalWave(10, 10, pivot = [7., 2.], start_radius = 3.5, velocity = 2)
#wave.run(100)
