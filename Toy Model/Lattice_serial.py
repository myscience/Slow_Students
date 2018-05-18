# coding=utf-8

# We define a Lattice class to store the position of Pixels and process the
# consequances of a wave passing upon the neuro cortex
from Pixel import Pixel
from LinearWave import LinearWave
from SphericalWave import SphericalWave
from EllipticalWave import EllipticalWave

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib import cm

from collections import OrderedDict

# Needed for proper terminal output
from sys import stdout, stderr

# Needed to handle exits properly
import sys, traceback
from datetime import datetime

from CustomException import ExpiredException, StyleException

class Lattice:

    width = 0
    height = 0

    # Default inizialize method, used for bult-in wave generation
    def __init__(self, density = 1., printFrame = False):
        # Flag for frame-per-frame printing of lattice status on screen
        self.printFrame = printFrame

        # Flag for frame-per-frame print of lattice status on .tif image
        self.savetoTif = False

        # Current time of the simulation
        self.time = 1

        # Matrix storing the density of neurons for each pixel
        self.densities = density * np.ones(width * height)

        # The list of Pixels to be stored
        self.pixels = [Pixel(self.densities[i]) for  i in range(width * height)];

    @classmethod
    def initWithWaves(cls, width, height, wave_dict, density = 1., printFrame = False):
        # Width and Height of the Lattice
        cls.width = width
        cls.height = height

        # Initialize waves number and 're-birth' mode
        cls.num_waves = wave_dict['num_wave']
        cls.wave_mode = wave_dict['wave_mode']
        cls.wave_type = wave_dict['wave_type']
        cls.wave_lifespan = wave_dict['wave_lifespan']

        try:
            if len(cls.wave_mode) != cls.num_waves:
                print len(cls.wave_mode), cls.num_waves
                raise ValueError('Lenght of wave_mode differs from the number of waves passed')

            if len(cls.wave_type) != cls.num_waves:
                raise ValueError('Length of wave_type differs from the number of waves passed')

            if len(cls.wave_lifespan) != cls.num_waves:
                raise ValueError('Length of wave_lifespan differs from the number of waves passed')

        except ValueError as err:
            traceback.print_exc(file=sys.stdout)
            sys.exit(0)

        # We print the Initialization parameters of the simulation
        status = "\nINITIALIZING SIMULATION.\n" +\
                 "Width: %d" % cls.width + "\tHeight: %d" % cls.height +\
                 "\nNumber of waves: %d" % cls.num_waves +\
                 "\nWave types: %s" % cls.wave_type +\
                 "\nWaves rebirth mode: %s" % cls.wave_mode +\
                 "\nWaves lifespan: %s\n" % cls.wave_lifespan

        print status

        # The wave that will propagate along the lattice
        cls.waves = []
        for wave_type, wave_mode, wave_lifespan in zip(cls.wave_type, cls.wave_mode, cls.wave_lifespan):
            try:
                if wave_type == 'Linear':
                    cls.waves.append(LinearWave(cls.width, cls.height, \
                        pivot = [np.random.uniform(0, cls.width), np.random.uniform(0, cls.height)],\
                        angle = np.random.uniform(0., 2. * np.pi), velocity = np.random.uniform(0.1, 2),\
                        acceleration = np.random.uniform(-0.1, 0.1), style =  wave_mode,\
                        lifespan = wave_lifespan))

                elif wave_type == 'Spherical':
                    cls.waves.append(SphericalWave(cls.width, cls.height, \
                        pivot = [np.random.uniform(0, cls.width), np.random.uniform(0, cls.height)],\
                        start_radius = np.random.uniform(0, 4), \
                        velocity = np.random.uniform(0.1, 2), \
                        acceleration = np.random.uniform(-0.1, 0.1),\
                        lifespan = wave_lifespan, style =  wave_mode))

                elif wave_type == 'Elliptical':
                    cls.waves.append(EllipticalWave(cls.width, cls.height,\
                        pivot = [np.random.uniform(0, cls.width), np.random.uniform(0, cls.height)], \
                        start_radius = [np.random.uniform(0, 2.), np.random.uniform(0, 2.)], \
                        start_angle = np.random.uniform(0, 360),
                        velocity = [np.random.uniform(0.1, 1.), np.random.uniform(0., 1.)],\
                        acceleration = [np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1)],\
                        lifespan = wave_lifespan, style = wave_mode))

                else:
                    raise ValueError('Wave type %s is not supported' % wave_type)
            except ValueError as err:
                traceback.print_exc(file=sys.stdout)
                sys.exit(0)

        cls.has_waves = True
        cls.loadedFile = False

        return cls(density = density, printFrame = printFrame)

    @classmethod
    def initiFromFile(cls, width, height, filename, delimiter = '[', density = 1., printFrame = False):
        cls.width = width
        cls.height = height

        print "\n\tINITIALIZING SIMULATION FROM FILE\n"
        print "Loading...",
        # Here we load the file
        with open(filename) as f:
            # We built is as a dict with key = pixel-ID ; value = activation-time
            cls.trans_dict = OrderedDict()
            cls.last_up_time = -1
            cls.first_up_time = 1E9
            last_id = -1
            first_id = -1

            for line in f:
                pos = line.find('[')
                key = int(line[:pos])
                try:
                    value = [int(float(x)) for x in line[pos + 1:-3].split(",")]

                    # We keep track of the first and last up transition
                    if cls.last_up_time < max(value):
                        cls.last_up_time = max(value)
                        last_id = np.argmax(value)
                        key_last = key

                    if cls.first_up_time > min(value):
                        cls.first_up_time = min(value)
                        first_id = np.argmin(value)
                        key_first = key



                except ValueError as err:
                    value = []

                cls.trans_dict[key] = value

        print "Completed!"

        if cls.trans_dict.keys()[-1] + 1 != cls.width * cls.height:
            print "\nError on initiFromFile: pixels-IDs do not math simulation dimentions"
            print "Max pixels-ID is %d while Width x Height is %d\n" % (cls.trans_dict.keys()[-1] + 1, cls.width * cls.height)
            sys.exit(-1)

        print "First up transition detected at time T = %d for ID = %d" % (cls.first_up_time, key_first)
        print "Last up transition detected at time T = %d for ID = %d" % (cls.last_up_time, key_last)

        cls.has_waves = False
        cls.loadedFile = True

        return cls(density = density, printFrame = printFrame)

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

    def runWaveSimulation(self, total_dt):
        if self.has_waves:
            print "STARTING SIMULATION"
            for i in range(total_dt):
                self.time += 1

                for wave in self.waves:
                    # We collect the indices of active pixels
                    try:
                        idx_actives = wave.run(1)

                    except ExpiredException:
                        # We need to remove wave from our list
                        self.waves.remove(wave)

                    except StyleException:
                        # Report the problem
                        traceback.print_exc(file = sys.stderr)

                        # Now we can perform a clean exit with error
                        return True

                    else:
                        self.setPixelsActive(idx_actives)

                for p in self.pixels:
                    p.stepSimulation()

                if (self.printFrame):
                    self.printLattice()

                if self.savetoTif:
                    path = "/home/paolo/Scrivania/Universita\'/Human Brain Project/Slow Waves Project/data/ToyTiff/"
                    filename = "frame_" + str(i) + ".tif"

                    path = path + filename
                    self.printToTif(path)

            return False

        else:
            print "Cannot run Wave Simulation. No wave initialized."
            print "Please try the Lattice.initWithWaves classmethod"

            # Returning with error
            return True

        return True

    def runFromFileSimulation(self, total_dt = None):
        if self.loadedFile:
            # We need to define the up_schedules for all the Neurons

            # We iterate over the keys of the dictionary which are the pixels-ID
            for key in self.trans_dict:
                # We check if the list is not empty
                if self.trans_dict[key]:
                    # Now we set the schedule for all the neurons in this pixel
                    for neuron in self.pixels[key].Neurons:
                        neuron.setActiveAt(self.trans_dict[key])

            # Default behaviour is to simulate the whole file-induced simulation
            if total_dt == None:
                total_dt = self.last_up_time

            print "STARTING SIMULATION of Total Time: %d" % total_dt
            t_start = datetime.now()

            for i in range(total_dt):
                t_current = datetime.now()
                t_diff = t_current - t_start

                expected_time = datetime.utcfromtimestamp((total_dt - (i + 1)) / ((i + 1) / (t_diff.total_seconds())))

                msg = "CURRENT ITERATION: %d ESTIMATED TIME: %s\r" % (i + 1, expected_time.time())
                sys.stdout.write(msg)
                sys.stdout.flush()

                # Now we step the simulation
                for p in self.pixels:
                    p.stepSimulation()

                if (self.printFrame):
                    self.printLattice()

                if self.savetoTif:
                    path = "/home/paolo/Scrivania/Universita\'/Human Brain Project/Slow Waves Project/data/ToyTiff/"
                    filename = "frame_" + str(i) + ".tif"

                    path = path + filename
                    self.printToTif(path)

                self.time += 1

            # Returning with no error
            return False

        else:
            print "Cannot run FromFile Simulation. No File loaded."
            print "Please try the Lattice.initiFromFile classmethod"

            #Returning with error
            return True

        return True

    def printPixelSignal(self, i, j):
        plt.plot(self.signals[i][j])
        plt.show()

    def printLattice(self):
        figure, axis = plt.subplots()

        intensities = np.zeros(self.width * self.height)
        for p, i in zip(self.pixels, range(self.width * self.height)):
            intensities[i] = int(p.active) * p.density

        # Create a rectangular matrix
        intensities = intensities.reshape(self.height, self.width)

        img_grid = intensities

        if self.has_waves:
            temp = np.zeros((self.width, self.height))
            for wave in self.waves:
                temp += wave.grid
                figure, axis = wave.printWave(figure, axis)

                img_grid += zip(*temp)

        plt.imshow(img_grid, extent = (0, self.width, self.height, 0), interpolation='nearest', cmap = 'coolwarm')
        plt.colorbar()

        plt.xlim(0, self.width)
        plt.ylim(0, self.height)

        axis.set_xticks(range(0, self.width + 1))
        axis.set_yticks(range(0, self.height + 1))

        # Turn off tick labels
        axis.set_yticklabels([])
        axis.set_xticklabels([])

        plt.title("Time t = %d" % self.time)

        plt.grid()
        plt.show()

    def printToTif(self, path):
        dpi = 80
        height, width = np.array((self.width, self.height), dtype=float) / dpi
        figure = plt.figure(figsize = (height, width), dpi = dpi)

        axis = figure.add_axes([0, 0, 1, 1])
        axis.axis('off')

        intensities = np.zeros(self.width * self.height)
        for p, i in zip(self.pixels, range(self.width * self.height)):
            intensities[i] = int(p.active) * p.density

        # Create a rectangular matrix
        intensities = intensities.reshape(self.height, self.width)

        img_grid = intensities

        if self.has_waves:
            temp = np.zeros((self.width, self.height))
            for wave in self.waves:
                temp += wave.grid
                figure, axis = wave.printWave(figure, axis)

                temp = zip(*temp)
                img_grid += temp

        axis.imshow(img_grid, interpolation = 'none')
        figure.savefig(path, dpi = dpi)

# Control need for Windows support
if __name__ == '__main__':
    width = 50
    height = 40

    wave_dict = { 'num_wave' : 2, 'wave_type' : ['Elliptical', 'Spherical'], 'wave_mode' : ['random', 'random'], 'wave_lifespan' : ['25', 'None'] }

    filename = "/home/paolo/Scrivania/Universita'/Human Brain Project/Slow Waves Project/data/min_time.txt"

    lattice = Lattice.initiFromFile(width, height, filename, printFrame = False)

    lattice.setPixelsDensities(np.random.normal(loc = 10, scale = 2, size = (width * height)))

    failure = lattice.runFromFileSimulation(total_dt = 10)
    print "SIMULATION COMPLETED.\n"
    if not failure:
        print "Collecting Pixels Signal..."
        lattice.collectPixelsSignal()

        #lattice.printPixelSignal(3, 4)

        lattice.pixels[401].Neurons[0].printSignal()

        print "\nLATTICE SIMULATION HAS BEEN SUCCESSFUL.\n"

    else:
        print "\nLATTICE SIMULATION HAS BEEN UNSUCCESSFUL.\n"
