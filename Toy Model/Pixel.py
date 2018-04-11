# coding=utf-8


# We define a Pixel class whose main gaol is to store a collection of neuron
# and collect their global signal base on the presence of a wave.

from Neuron import Neuron
import numpy as np
import random

class Pixel :

    max_up_time = 1

    def __init__(self, density, volume = 10):

        # The cortical volume the pixel is representing
        self.volume = volume;

        # The density of neurons in this pixel
        self.density = density

        # The number of Neurons is the volume times the density
        self.Neurons = [Neuron(random.random()) for i in range(int(density * volume + 1))]

        # Flag to tell if a wave is currently upon this Pixel
        self.active = False

        # The value of signal coming from this Pixel
        self.signal = np.zeros((len(self.Neurons), 1))

        self.time = 1

    def getSignal(self):
        self.signal = np.zeros((len(self.Neurons), 2 * self.time - 1))

        # We iterate over the Neurons list to activate each neuron
        for neuron, i in zip(self.Neurons, range(len(self.Neurons))):
            self.signal[i] = neuron.evaluateSignal()

        return np.sum(self.signal, axis=0)

    def getDensity(self):
        return self.density;

    def setDensity(self, density):
        if density < 0:
            density = 0

        self.density = density
        self.Neurons = [Neuron(random.random() * (0.99) + 0.01) for i in range(int(self.density * self.volume + 1))]

    def setActive(self):
        self.active = True;

        for neuron in self.Neurons:
            neuron.setActive()

    def stepSimulation(self):
        self.time = self.time + 1

        for neuron in self.Neurons:
            neuron.run(1)

        if self.Neurons[0].state == False:
            self.active = False
