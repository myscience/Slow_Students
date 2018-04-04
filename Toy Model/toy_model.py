# coding=utf-8

from Lattice import Lattice
from LinearWave import LinearWave
import numpy as np

wave = LinearWave(10, 10, np.array(1, 0), np.array(0, 0))
wave.printWave()

#width = 10
#height = 10
#lattice = Lattice(width, height)
#lattice.setPixelsDensities(np.random.rand(width * height))

#print lattice.getPixelSignal(4, 6)

#lattice.setPixelActive(4, 6)

#print lattice.getPixelSignal(4, 6)

#lattice.stepSimulation(25)

#print lattice.getPixelSignal(4, 6)

#lattice.printLattice()
