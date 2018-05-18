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
from CustomException import ExpiredException, StyleException

# Needed for parallelization
import multiprocessing
from multiprocessing import Process, Queue
from Queue import Empty

class Worker(Process):
    def __init__(self, pix, start_code, height, width, in_q, out_q):
        super(Worker, self).__init__()

        self.pixels = pix
        self.start_code = start_code

        self.in_queue = in_q
        self.out_queue = out_q

        self.width = width
        self.height = height

        print "Worker %s is now active" % self

    def run(self):
        while True:
            try:
                # Get a new message
                msg = self.in_queue.get()

                # this is the 'TERM' signal
                if msg is None:
                    self.out_queue.put((self.start_code, self.pixels))
                    break

                # We need to perform a step
                elif msg[0] == "step":
                    for i in range(len(msg[1])):
                        self.pixels[msg[1][i]].setActive()

                    for p in self.pixels:
                        p.stepSimulation()

                elif msg[0] == "init":
                    for key in msg[1]:
                        # Now we set the schedule for all the neurons in this pixel
                        for neuron in self.pixels[key].Neurons:
                            neuron.setActiveAt((msg[1])[key])

                    print self, "Has finished Initialization"

                elif msg[0] == "run":
                    print self, "Is Starting Simulation"
                    for dt in range(msg[1]):
                        for p in self.pixels:
                            p.stepSimulation()

                elif msg[0] == "just_step":
                    for p in self.pixels:
                        p.stepSimulation()

                else:
                    print "Warning: message parsing failed"
                    pass

            except Exception, e:
                print "Error on Worker run: ", e
                break

        print self, "has finished"

class Lattice:

    def __init__(self, density = 1.):

        # Width and Height of the Lattice
        self.width = width;
        self.height = height;

        # Matrix storing the density of neurons for each pixel
        self.densities = density * np.ones(width * height)

        # Current time of the simulation
        self.time = 1

        # Setting up number of cores for multiprocessing
        self.num_cores = multiprocessing.cpu_count()

        # The list of Pixels to be stored
        self.pixels = [Pixel(self.densities[i]) for  i in range(width * height)];


        # We split the image into chuncks accordingly to the number of processing
        # unit for better performance
        self.chunck_size = (len(self.pixels) / self.num_cores)
        split_pixels = [self.pixels[x : x + self.chunck_size]\
                        for x in xrange(0, len(self.pixels), self.chunck_size)]

        in_queues  = [Queue() for i in range(self.num_cores)]
        out_queues = [Queue() for i in range(self.num_cores)]

        # We create workers for our simulation
        self.workers = [Worker(s, i, self.height, self.width, in_q, out_q) for s, i, in_q, out_q in \
             zip(split_pixels, range(len(split_pixels)), in_queues, out_queues)]

        # We start our workers
        for w in self.workers:
            w.start()

    @classmethod
    def initWithWaves(cls, width, height, wave_dict, density = 1.):
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
                 "\nNumber of waves: %d" %cls.num_waves +\
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

        return cls(density = density)

    @classmethod
    def initiFromFile(cls, width, height, filename, delimiter = '[', density = 1.):
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

        return cls(density = density)

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
            for i in range(total_dt):
                self.time += 1

                idx = [[] for i in range(self.num_cores)]

                for wave in self.waves:
                    # We collect the indices of active pixels
                    try:
                        idx_actives = wave.run(1)

                    except ExpiredException:
                        # We need to remove wave from our list
                        self.waves.remove(wave)

                    except StyleException as err:
                        # We signal the Halt signal to the workers and collect the results
                        for w in self.workers:
                            w.in_queue.put(None)

                        # We terminate the processes
                        for w in self.workers:
                            w.terminate()

                        for w in self.workers:
                            w.join(5)

                        print "PROCESSES TERMINATED"

                        # Report the problem
                        traceback.print_exc(file = sys.stderr)

                        # Now we can perform a clean exit
                        return True

                    else:
                        # We prepare the indices to pass to the workers, splitting based on
                        # the workers indices sections
                        for i in range(len(idx_actives)):
                            tmp = idx_actives[i][0] + idx_actives[i][1] * self.width

                            for k in range(self.num_cores):
                                if tmp < (k + 1) * self.chunck_size:
                                    tmp -= k * self.chunck_size
                                    idx[k].append(tmp)
                                    break

                for i, w in enumerate(self.workers):
                    w.in_queue.put(["step", idx[i]])

            # We signal the Halt signal to the workers and collect the results
            for w in self.workers:
                w.in_queue.put(None)
                # We collect the results
                res = w.out_queue.get()
                self.pixels[res[0] * self.chunck_size : (res[0] + 1) * self.chunck_size] = res[1]

            # We join the processes
            for w in self.workers:
                w.join(5)

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

            idx = [{} for i in range(self.num_cores)]

            # We iterate over the keys of the dictionary which are the pixels-ID
            for key in self.trans_dict:
                # We check if the list is not empty
                if self.trans_dict[key]:
                    # We prepare the indices to pass to the workers, splitting based on
                    # the workers indices sections
                    tmp = key

                    for k in range(self.num_cores):
                        if tmp < (k + 1) * self.chunck_size:
                            tmp -= k * self.chunck_size
                            (idx[k])[tmp] = self.trans_dict[key]
                            break

            # We signal the set-up configuration
            for i, w in enumerate(self.workers):
                w.in_queue.put(["init", idx[i]])

            # Default behaviour is to simulate the whole file-induced simulation
            if total_dt == None:
                total_dt = self.last_up_time


            # We signal the run simulation
            for w in self.workers:
                w.in_queue.put(["run", total_dt])

            self.time += total_dt

            # We signal the Halt signal to the workers and collect the results
            for w in self.workers:
                w.in_queue.put(None)
                # We collect the results
                res = w.out_queue.get()
                self.pixels[res[0] * self.chunck_size : (res[0] + 1) * self.chunck_size] = res[1]

            # We join the processes
            for w in self.workers:
                w.join(5)

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

# Control need for Windows support
if __name__ == '__main__':
    width = 50
    height = 40

    wave_dict = { 'num_wave' : 2, 'wave_type' : ['Elliptical', 'Spherical'], 'wave_mode' : ['random', 'random'], 'wave_lifespan' : ['25', 'None'] }

    filename = "/home/paolo/Scrivania/Universita'/Human Brain Project/Slow Waves Project/data/min_time.txt"

    lattice = Lattice.initiFromFile(width, height, filename)

    lattice.setPixelsDensities(np.random.normal(loc = 10, scale = 2, size = (width * height)))

    failure = lattice.runFromFileSimulation(total_dt = 700)
    print "SIMULATION COMPLETED.\n"
    if not failure:
        lattice.collectPixelsSignal()

        lattice.printPixelSignal(25, 20)

        lattice.pixels[1000].Neurons[4].printSignal()

        print "\nLATTICE SIMULATION HAS BEEN SUCCESSFUL.\n"

    else:
        print "\nLATTICE SIMULATION HAS BEEN UNSUCCESSFUL.\n"
