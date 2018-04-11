# coding=utf-8

# We define a Lattice class to store the position of Pixels and process the
# consequances of a wave passing upon the neuro cortex
from Pixel import Pixel
from LinearWave import LinearWave
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib import cm

# Needed for proper terminal output
from sys import stdout, stderr

# Needed to handle exits properly
import sys, traceback

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

                else:
                    print "Warning: message parsing failed"
                    pass

            except Exception, e:
                print "Error on Worker run: ", e
                break

        print self, "has finished"

class Lattice:

    def __init__(self, width, height, density = 1., num_wave = 1, wave_mode = ['random'], printFrame = False):

        # Width and Height of the Lattice
        self.width = width;
        self.height = height;

        # Initialize waves number and 're-birth' mode
        self.num_waves = num_wave
        self.wave_mode = wave_mode

        # Flag for frame-per-frame printing of lattice status
        self.printFrame = printFrame

        # Current time of the simulation
        self.time = 1

        try:
            if len(self.wave_mode) != self.num_waves:
                print len(self.wave_mode), self.num_waves
                raise ValueError('Lenght of wave_mode differs from the number of waves passed')
        except ValueError as err:
            traceback.print_exc(file=sys.stdout)
            sys.exit(0)

        # We print the Initialization parameters of the simulation
        status = "\nINITIALIZING SIMULATION.\n" +\
                 "Width: %d" % self.width + "\tHeight: %d" % self.height +\
                 "\nNumber of waves: %d" %self.num_waves +\
                 "\nWaves rebirth mode: %s\n" % self.wave_mode

        print status

        self.densities = density * np.ones(width * height)

        # The wave that will propagate along the lattice
        self.waves = [LinearWave(self.width, self.height, \
            pivot = [np.random.uniform(0, self.width), np.random.uniform(0, self.height)],
            k = [np.random.uniform(-2, 2), np.random.uniform(-2, 2)], velocity = .7) \
            for i in range(num_wave)]

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

            idx = [[] for i in range(self.num_cores)]

            for i in range(self.num_waves):
                # We collect the indices of active pixels
                idx_actives = self.waves[i].run(1, self.wave_mode[i])

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

            if (self.printFrame):
                self.printLattice()

        # We signal the Halt signal to the workers and collect the results
        for w in self.workers:
            w.in_queue.put(None)
            # We collect the results
            res = w.out_queue.get()
            self.pixels[res[0] * self.chunck_size : (res[0] + 1) * self.chunck_size] = res[1]

        # We join the processes
        for w in self.workers:
            w.join(5)

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
            plt.plot(self.waves[i].x, self.waves[i].y)
            plt.plot(self.waves[i].pivot[0], self.waves[i].pivot[1], marker='o', markersize = 3, color = "red")
            plt.plot(self.waves[i].x0, self.waves[i].y0, marker='o', markersize = 4, color = "blue")
            plt.plot(self.waves[i].x1, self.waves[i].y1, marker='o', markersize = 4, color = "blue")
            plt.quiver(self.waves[i].pivot[0], self.waves[i].pivot[1], self.waves[i].k[0],
                                    self.waves[i].k[1], color='g', scale_units = 'xy')
            plt.xlim(0, self.waves[i].width)
            plt.ylim(0, self.waves[i].height)

        spacing = 1
        minorLocator = MultipleLocator(spacing)
        axis.xaxis.set_minor_locator(minorLocator)
        axis.yaxis.set_minor_locator(minorLocator)
        plt.grid(True, which = 'minor')
        plt.show()

# Control need for Windows support
if __name__ == '__main__':
    width = 10
    height = 10
    lattice = Lattice(width, height, num_wave = 3, wave_mode = ['random', 'v_linear', 'h_linear'])
    lattice.setPixelsDensities(np.random.normal(loc = 5, scale = 2, size = (width * height)))

    lattice.runSimulation(300)
    lattice.collectPixelsSignal()

    lattice.printPixelSignal(3, 7)

    lattice.pixels[0].Neurons[4].printSignal()
