# coding=utf-8

# We define a Lattice class to store the position of Pixels and process the
# consequances of a wave passing upon the neuro cortex
from Pixel import Pixel
from LinearWave import LinearWave
from SphericalWave import SphericalWave

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib import cm

# Needed for proper terminal output
from sys import stdout, stderr

# Needed to handle exits properly
import sys, traceback

class Lattice:

    def __init__(self, width, height, density = 1., num_wave = 1, \
            wave_type = ['Spherical'], wave_mode = ['random'], printFrame = False):

        # Width and Height of the Lattice
        self.width = width;
        self.height = height;

        # Initialize waves number and 're-birth' mode
        self.num_waves = num_wave
        self.wave_mode = wave_mode
        self.wave_type = wave_type

        # Flag for frame-per-frame printing of lattice status
        self.printFrame = printFrame

        # Current time of the simulation
        self.time = 1

        try:
            if len(self.wave_mode) != self.num_waves:
                print len(self.wave_mode), self.num_waves
                raise ValueError('Lenght of wave_mode differs from the number of waves passed')
            if len(self.wave_type) != self.num_waves:
                raise ValueError('Length of wave_type differs from the number of waves passed')
        except ValueError as err:
            traceback.print_exc(file=sys.stdout)
            sys.exit(0)

        # We print the Initialization parameters of the simulation
        status = "\nINITIALIZING SIMULATION.\n" +\
                 "Width: %d" % self.width + "\tHeight: %d" % self.height +\
                 "\nNumber of waves: %d" %self.num_waves +\
                 "\nWave types: %s" % self.wave_type +\
                 "\nWaves rebirth mode: %s\n" % self.wave_mode

        print status

        self.densities = density * np.ones(width * height)

        # The wave that will propagate along the lattice
        self.waves = []
        for num_wave in range(self.num_waves):
            try:
                if self.wave_type[num_wave] == 'Linear':
                    self.waves.append(LinearWave(self.width, self.height, \
                        pivot = [np.random.uniform(0, self.width), np.random.uniform(0, self.height)],\
                        k = [np.random.uniform(-2, 2), np.random.uniform(-2, 2)], velocity = np.random.uniform(0.1, 2)))

                elif self.wave_type[num_wave] == 'Spherical':
                    self.waves.append(SphericalWave(self.width, self.height, \
                        pivot = [np.random.uniform(0, self.width), np.random.uniform(0, self.height)],\
                        start_radius = np.random.uniform(0, 4), velocity = np.random.uniform(0.1, 2)))

                else:
                    raise ValueError('Wave type %s is not supported' % self.wave_type[num_wave])
            except ValueError as err:
                traceback.print_exc(file=sys.stdout)
                sys.exit(0)

        # The list of Pixels to be stored
        self.pixels = [Pixel(self.densities[i]) for  i in range(width * height)];

    def getPixelsDensities(self):
        return self.densities

    def setPixelsDensities(self, densities):
        for p, i in zip(self.pixels, range(densities.size)):
            p.setDensity(densities[i])
            self.densities[i] = densities[i]

    def getPixelDensity(self, i):
        return self.densities[i]

    def setPixelActive(self, i, j):
        self.pixels[i + j * self.width].setActive()

    def setPixelsActive(self, indices):
        for i, j in zip(indices[:, 0], indices[:, 1]):
            self.setPixelActive(i, j)

    def getPixelSignal(self, i, j):
        return self.pixels[i + j * self.width].getSignal()

    def collectPixelsSignal(self):
        # The array for signals to be stored
        self.signals = np.zeros((self.width, self.height, 2 * self.time - 1))

        for i in range(self.width):
            for j in range(self.height):
                self.signals[i][j] = self.getPixelSignal(i, j)

    def runSimulation(self, total_dt):
        for i in range(total_dt):
            self.time += 1

            for i in range(self.num_waves):
                # We collect the indices of active pixels
                idx_actives = self.waves[i].run(1, self.wave_mode[i])
                self.setPixelsActive(idx_actives)

            for p in self.pixels:
                p.stepSimulation()

            if (self.printFrame):
                self.printLattice()

    def printPixelSignal(self, i, j):
        plt.plot(self.signals[i][j])
        plt.show()

    def printLattice(self):
        figure, axis = plt.subplots()

        intensities = np.zeros(self.width * self.height)
        for p, i in zip(self.pixels, range(self.width * self.height)):
            intensities[i] = int(p.active) * p.density

        temp = self.waves[0].grid
        for i in range(1, self.num_waves):
            temp += self.waves[i].grid

        img_grid = temp + intensities.reshape(self.width, self.height)
        plt.imshow(img_grid, extent = (0, self.width, self.height, 0), interpolation='nearest', cmap = cm.coolwarm)
        plt.colorbar()

        for i in range(self.num_waves):
            figure, axis = self.waves[i].printWave(figure, axis)

        plt.xlim(0, self.width)
        plt.ylim(0, self.height)

        spacing = 1
        minorLocator = MultipleLocator(spacing)
        axis.xaxis.set_minor_locator(minorLocator)
        axis.yaxis.set_minor_locator(minorLocator)
        plt.grid(True, which = 'minor')
        plt.show()

# Control need for Windows support
if __name__ == '__main__':
    width = 15
    height = 15
    lattice = Lattice(width, height, num_wave = 2,  wave_type = ['Spherical', 'Spherical'],\
                        wave_mode = ['random', 'random'], printFrame = False)
    lattice.setPixelsDensities(np.random.normal(loc = 5, scale = 2, size = (width * height)))

    lattice.runSimulation(100)
    lattice.collectPixelsSignal()

    lattice.printPixelSignal(3, 7)

    lattice.pixels[0].Neurons[4].printSignal()
