# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt

class Neuron :

    # We keep track of time elapsed of the simulation for each neuron
    max_up_time = 5
    rise_time = 5

    def __init__(self, depth):
        # Here we set the values of the Neuron main attributes

        # The state of the neuron: False = Inactive, True = Active
        self.state = False;

        # The depth at whichH the neuron is created: it affects the overall signal
        self.depth = depth;

        # The offset from the zero of the action potential
        self.zero_off = 0;

        # Time of the up states, in milliseconds
        self.up_time = 0;

        # The variance of the value of the up state action potential
        self.up_sigma = 0;

        # The rate time for the up and down states
        self.up_rate = 15.;
        self.down_rate = 2.;

        # The height of the response function
        self.resp_heigh = 1.;

        # The raw signal the Neuron produce during the simulation
        self.raw_signal = np.zeros(1)

        # The time of the simulation
        self.time = 1

    def setActive(self):
        self.state = True
        self.up_time = self.time

    def updateState(self):
        # We check if it's time to turn off the neuron
        if self.state:
            if self.time - self.up_time > self.max_up_time:
                self.state = False;

    def stepSimulation(self):
        # We increase the time
        self.time = self.time + 1
        self.updateState()

    def getTime(self):
        return Neuron.time

    def updateRawSignal(self):
        rate = self.state * self.up_rate + (1 - self.state) * self.down_rate
        self.raw_signal = np.append(self.raw_signal, [np.random.poisson(rate) * self.depth * self.depth])

    def evaluateSignal(self):
        # The response function of the singol neuron
        self.response = self.resp_heigh  * LogNorm(self.time, self.time / 10, sigma = 1.)

        self.signal = np.convolve(self.raw_signal, self.response[:self.time], mode = "full")

        return self.signal

    def run(self, dt, eval = False, print_signal = False):
        for i in range(dt):
            self.stepSimulation()
            self.updateRawSignal()

        if eval :
            self.evaluateSignal()

        if print_signal:
            self.printSignal()

    def printSignal(self):
        f, axarr = plt.subplots(3)
        axarr[0].plot(self.raw_signal)
        axarr[0].set_title('Neuron Raw Signal')
        axarr[1].plot(self.response[:100])
        axarr[1].set_title('Neuron Response Function')
        axarr[2].plot(self.signal)
        axarr[2].set_title('Neuron Signal Convolution')

        plt.show()


def Theta(rise_time, up_time, points):
    return ((np.arange(points) < up_time) & (np.arange(points) > rise_time)).astype(int)

def Landau(time, points):
    return 1. / np.sqrt(2 * np.pi) * np.exp(-1. * ((np.linspace(-5, time, num = points) +
                        np.exp(-1. * (np.linspace(-5, 15, num = points)))) / 2.))

def LogNorm(points, time, mu = 0., sigma = 1.):
    return np.reciprocal((np.linspace(1e-5, time, num = points))) / (sigma * np.sqrt(2 * np.pi)) *\
     np.exp(-1. * (np.log(np.linspace(1e-5, time, num = points)) - mu)**2 / (2. * sigma * sigma))
