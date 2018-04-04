# coding=utf-8

import numpy as np

class Neuron :

    def __init__(self, depth):
        # Here we set the values of the Neuron main attributes

        # The state of the neuron: False = Inactive, True = Active
        self.state = False;

        # The depth at with the neuron is created: it affects the overall signal
        self.depth = depth;

        # The offset from the zero of the action potential
        self.zero_off = 0;

        # Time of the up and down states, in milliseconds
        self.up_time = 20;
        self.down_time = 100;

        # We keep track of time elapsed for each neuron
        self.time = 1;

        # The variance of the value of the up state action potential
        self.up_sigma = 0;

        # The rate time for the up and down states
        self.up_rate = 10;
        self.down_rate = 1;

        # The height of the response function
        self.resp_heigh = 1.;

        # The response function of the singol neuron
        self.response = self.resp_heigh * Theta(self.up_time, self.up_time + self.down_time)

    def updateState(self):
        if self.state:
            if self.time > self.up_time:
                self.state = False;

        # We increase the time
        self.time = self.time + 1

        # We reset the time
        if self.time > self.down_time + self.up_time:
            self.time = 1;

    def getTime(self):
        return self.time


    def getSignal(self):
        rate = self.state * self.up_rate + (1 - self.state) * self.down_rate
        raw_signal = np.random.poisson(rate, self.time) * self.depth

        return np.convolve(raw_signal, self.response[:self.time], mode='same')

def Theta(up_time, points):
    return (np.arange(points) < up_time).astype(int)
