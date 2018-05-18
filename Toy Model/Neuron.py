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

        # Lookup table for up-transition, initialize to None
        self.up_schedule = None

        # The time of the simulation
        self.time = 1

    def setActive(self):
        self.state = True
        self.up_time = self.time

    def setActiveAt(self, up_trans):
        self.up_schedule = up_trans
        self.next_up_tran = self.up_schedule[0]

    def updateState(self):
        # We check if it's time to turn off the neuron
        if self.state:
            if self.time - self.up_time > self.max_up_time:
                self.state = False;


    def stepSimulation(self):
        # We increase the time
        self.time = self.time + 1

        # We check if it's time to switch on the neuron based on up_schedule
        if self.up_schedule != None:
            if self.time < self.next_up_tran:
                # Keep waiting
                pass

            else:
                # It's time to activate the Neuron
                self.setActive()

                # We try to grab the next_up_tran
                try:
                    if self.up_schedule:
                        self.next_up_tran = self.up_schedule.pop(0)
                    else:
                        raise StopIteration

                # We empty the up_schedule
                except StopIteration:
                    self.next_up_tran = None
                    self.up_schedule = None

        self.updateState()

    def getTime(self):
        return Neuron.time

    def updateRawSignal(self):
        rate = self.state * self.up_rate + (1 - self.state) * self.down_rate
        self.raw_signal = np.append(self.raw_signal, [np.random.poisson(rate) * self.depth * self.depth])

    def evaluateSignal(self):
        # The response function of the singol neuron

        # ATTENZIONE: Nel settaggio della funzione di risposta di è scelto 20 come parametro
        #             ma è più per provare che per cognizione di causa
        self.response = self.resp_heigh  * LogNorm(self.time, tmax = 20, sigma = 1.)

        self.signal = np.convolve(self.raw_signal, self.response, mode = "full")

        return self.signal

    def run(self, dt, eval_ = False, print_signal = False):
        for i in range(dt):
            self.stepSimulation()
            self.updateRawSignal()

        if eval_:
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

def LogNorm(points, tmax = 20, mu = 0., sigma = 1.):
    #print points, time
    return np.reciprocal((np.linspace(1e-5, tmax, num = points))) / (sigma * np.sqrt(2 * np.pi)) *\
     np.exp(-1. * (np.log(np.linspace(1e-5, tmax, num = points)) - mu)**2 / (2. * sigma * sigma))
