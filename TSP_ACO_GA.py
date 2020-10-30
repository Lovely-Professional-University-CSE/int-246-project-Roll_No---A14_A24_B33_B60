from tkinter import *
from tkinter.ttk import *
import time
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from matplotlib.animation import FuncAnimation
#import matplotlib.animation as animation


import numpy
import string
import matplotlib.pyplot as plt
import matplotlib.animation
#import matplotlib
import random

numpy.random.seed(1)

N = 25

_nodes = [(random.uniform(-400, 400), random.uniform(-400, 400)) for _ in range(0, 25)]
xx = [i[0] for i in _nodes]
yy = [i[1] for i in _nodes]

# Generation of labels and random coordinates for N cities
CITY_LABELS = list(range(N))
#CITY_COORD = numpy.random.randint(0, 200, (N, 2))
#CITY_COORD=numpy.array([(random.uniform(-400, 400), random.uniform(-400, 400)) for _ in range(0, 25)])
CITY_COORD=numpy.array(_nodes)
CITY_DICT = {label: coord for (label, coord) in zip(CITY_LABELS, CITY_COORD)}

# Population initialization function
def init(pop_size):
    def random_permutation():
        population = list()
        for _ in range(pop_size):
            # Each individual is a random permutation of the set of cities
            individual = list(numpy.random.permutation(CITY_LABELS))
            population.append(individual)
        return population
    
    return random_permutation()
