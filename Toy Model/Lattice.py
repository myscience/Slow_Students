# coding=utf-8

# We define a Lattice class to store the position of Pixels and process the
# consequances of a wave passing upon the neuro cortex
from Pixel import Pixel
import numpy as np
import matplotlib.pyplot as plt


class Lattice :

    def __init__(self, width, height, density = 1.):
        # Width and Height of the Lattice
        self.width = width;
        self.height = height;

        self.densities = density * np.ones(width * height)

        # The list of Pixels to be stored
        self.pixels = [Pixel(self.densities[i]) for  i in range(width * height)];

        self.time = 1;

        # The array for signals to be stored
        self.signals = np.zeros((width, height, self.time))

    def getPixelsDensities(self):
        return self.densities

    def setPixelsDensities(self, densities):
        for p, i in zip(self.pixels, range(densities.size)):
            p.setDensity(densities[i])
            self.densities[i] = densities[i]

    def getPixelDensity(self, i):
        return self.densities[i]

    def setPixelActive(self, i, j):
        self.pixels[j + i * self.width].setActive()

    def getPixelSignal(self, i, j):
        self.signals = np.zeros((self.width, self.height, self.time))
        self.signals[i][j] = self.pixels[j + i * self.width].getSignal()

        return self.signals[i][j]

    def stepSimulation(self, dt):
        for i in range(dt):
            self.time = self.time + 1;

            for p in self.pixels:
                p.stepSimulation()

    def printLattice(self):
        for p, i in zip(self.pixels, range(self.densities.size)):
            if p.active:
                self.densities[i] = -1.

        img = plt.imshow(self.densities.reshape(self.width, self.height))
        plt.colorbar()
        plt.show()
