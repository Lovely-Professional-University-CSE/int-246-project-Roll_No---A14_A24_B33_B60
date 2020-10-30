# GENETIC ALGORITHM
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


#fitness function
# Calculates the fitness of all individuals in the population

def fit(population):
    fitness = list()
    for individual in population:
        distance = 0
        for i, city in enumerate(individual):
            s = CITY_DICT[individual[i-1]]
            t = CITY_DICT[individual[i]]
            distance += numpy.linalg.norm(s-t)
        fitness.append(1/distance)
    return fitness

# Selection function
def selection(population, fitness, n):
    def roulette():
        # Obtaining the indices for each individual in the population
        idx = numpy.arange(0, len(population))
        # Calculation of selection probabilities based on individuals' aptitude
        probabilities = fitness/numpy.sum(fitness)
        # Choice of parent indexes
        parents_idx = numpy.random.choice(idx, size=n, p=probabilities)
        # Choice of parents based on selected indexes
        parents = numpy.take(population, parents_idx, axis=0)
        parents = [(parents[i], parents[i+1])
                   for i in range(0, len(parents)-1, 2)]
        return parents
    return roulette()

# Crossover function
def crossover(parents, crossover_rate=0.9):
    def ordered():
        children = list()
        # Iteration by all pairs of parents
        for pair in parents:
            if numpy.random.random() < crossover_rate:
                for (parent1, parent2) in [(pair[0], pair[1]), (pair[1], pair[0])]:
                    # Cut segment definition
                    points = numpy.random.randint(0, len(parent1), 2)
                    start = min(points)
                    end = max(points)
                    segment1 = [x for x in parent1[start:end]]
                    segment2 = [x for x in parent2[end:] if x not in segment1]
                    segment3 = [x for x in parent2[:end] if x not in segment1]
                    child = segment3 + segment1 + segment2
                    children.append(child)
            else:
                # If the crossing does not occur, the parents remain in the next generation
                children.append(pair[0])
                children.append(pair[1])
        return children
    return ordered()


