"""
File: hw5_predprey.py
Description: An extension of the rabbit artificial life simulation which includes foxes.

"""
import copy
import random as rnd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import argparse
from matplotlib.colors import ListedColormap
mpl.use('macosx')

# choose colors for each living organism
colors = ['white', 'green', 'blue', 'red']
cmap = ListedColormap(colors)

WRAP = False # when moving beyond the border, do we wrap around to the other side
SIZE = None # x/y dimensions of the field
R_OFFSPRING = 2 # the number of offspring when a rabbit reproduces
F_OFFSPRING = 1 # the number of offspring when a fox reproduces
GRASS_RATE = None # probability of grass growing at any given location (i.e. 2%)
INIT_RABBITS = None # number of starting rabbits
INIT_FOXES = None # number of starting foxes
SPEED = 5 # number of generations per frame
LIFECYCLES = None # number of generations foxes can live without food

class Animal:
    def __init__(self, type):
        self.x = rnd.randrange(0, SIZE)
        self.y = rnd.randrange(0, SIZE)
        self.type = type
        self.eaten = 0
        self.lifespan = 0

    def reproduce(self):
        self.eaten = 0
        self.lifespan = 0
        return copy.deepcopy(self)

    def eat(self, amount):
        self.eaten += amount

    def move(self):
        if self.type == 'rabbit':
            if WRAP:
                self.x = (self.x + rnd.choice([-1, 0, 1])) % SIZE
                self.y = (self.y + rnd.choice([-1, 0, 1])) % SIZE
            else:
                self.x = min(SIZE-1, max(0, (self.x + rnd.choice([-1, 0, 1]))))
                self.y = min(SIZE-1, max(0, (self.y + rnd.choice([-1, 0, 1]))))
        if self.type == 'fox':
            if WRAP:
                self.x = (self.x + rnd.choice([-2, 1, 0, 1, 2])) % SIZE
                self.y = (self.y + rnd.choice([-2, -1, 0, 1, 2])) % SIZE
            else:
                self.x = min(SIZE-1, max(0, (self.x + rnd.choice([-2, -1, 0, 1, 2]))))
                self.y = min(SIZE-1, max(0, (self.y + rnd.choice([-2, -1, 0, 1, 2]))))
            self.lifespan += 1


class Field:
    """ a field is a patch of grass with 0 or more rabbits hopping around in search of grass
        and 0 or more foxes in search of rabbits to eat """

    def __init__(self):
        self.rabbits = []
        self.foxes = []
        self.field = np.ones(shape=(SIZE, SIZE), dtype=int)

    def add_rabbit(self, rabbit):
        self.rabbits.append(rabbit)

    def add_fox(self, fox):
        self.foxes.append(fox)

    def move(self):
        for r in self.rabbits:
            r.move()
        for f in self.foxes:
            f.move()

    def eat(self):
        """ all rabbits try to eat grass at their current location and all foxes
            try to eat rabbits at their current location """
        for r in self.rabbits:
            r.eat(self.field[r.x, r.y])
            self.field[r.x, r.y] = 0

        # Create a dictionary mapping positions to rabbits
        rabbit_pos = {(r.x, r.y): r for r in self.rabbits}

        # Track rabbits to be removed
        remove_r = []

        for f in self.foxes:
            rabbit = rabbit_pos.get((f.x, f.y))
            if rabbit:
                f.eat(1)
                remove_r.append(rabbit)
            else:
                f.eat(0)

        # Remove the eaten rabbits
        for rabbit in remove_r:
            if rabbit in self.rabbits:  # Ensure the rabbit is still in the list
                self.rabbits.remove(rabbit)


    def survive(self):
        """ rabbits that have not eaten die, otherwise they live """
        self.rabbits = [r for r in self.rabbits if r.eaten > 0]
        self.foxes = [f for f in self.foxes if f.eaten > 0 or f.lifespan <= LIFECYCLES]


    def reproduce(self):
        r_born = []
        for r in self.rabbits:
            for _ in range(rnd.randint(1, R_OFFSPRING)):
                r_born.append(r.reproduce())
        self.rabbits += r_born

        f_born = []
        for f in self.foxes:
            if f.eaten > 0:
                for _ in range(rnd.randint(1, F_OFFSPRING)):
                    f_born.append(f.reproduce())
        self.foxes += f_born

    def grow(self):
        growloc = (np.random.rand(SIZE, SIZE) < GRASS_RATE) * 1
        self.field = np.maximum(self.field, growloc)

    def generation(self):
        """ Run one generation of rabbit actions"""
        self.move()
        self.eat()
        self.survive()
        self.reproduce()
        self.grow()


def animate(i, field, im):
    for _ in range(SPEED):
        field.generation()

    # array of ones to show grass
    grass = field.field

    # first an array of zeros, then replacing zeros in rabbits' position to twos
    rabbits = np.zeros((SIZE, SIZE), dtype=int)
    for r in field.rabbits:
        rabbits[r.x, r.y] = 2

    # first an array of zeros, then replacing zeros in foxes' position to threes
    foxes = np.zeros((SIZE, SIZE), dtype=int)
    for f in field.foxes:
        foxes[f.x, f.y] = 3

    im.set_array(np.maximum(np.maximum(grass, rabbits), foxes))
    plt.title('Generation: ' + str(i * SPEED) + ' Rabbits: ' + str(len(field.rabbits))
              + ' Foxes: ' + str(len(field.foxes)))
    return im,

def main():

    # initialize the parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('-g', '--grass_rate', type=float, default=0.05,
                        help='probability of grass growing at any given location')
    parser.add_argument('-k', '--fox_lifespan', type=int, default=40,
                        help='number of generations foxes can live without food')
    parser.add_argument('-s', '--size', type=int, default=400,
                        help='size of the field')
    parser.add_argument('-ir', '--init_rabbits', type=int, default=50,
                        help='number of rabbits to begin simulation')
    parser.add_argument('-if', '--init_foxes', type=int, default=50,
                        help='number of rabbits to begin simulation')

    # parse the arguments
    args = parser.parse_args()

    # create global variables
    global GRASS_RATE, LIFECYCLES, SIZE, INIT_RABBITS, INIT_FOXES
    GRASS_RATE = args.grass_rate
    LIFECYCLES = args.fox_lifespan
    SIZE = args.size
    INIT_RABBITS = args.init_rabbits
    INIT_FOXES = args.init_foxes

    # create the ecosystem for the plot
    field = Field()

    # initialize with some rabbits
    for _ in range(INIT_RABBITS):
        field.add_rabbit(Animal(type='rabbit'))

    # initialize with some foxes
    for _ in range(INIT_FOXES):
        field.add_fox(Animal(type='fox'))

    # initialize population list objects
    gens = range(1001)
    rabbit_pop = [len(field.rabbits)]
    fox_pop = [len(field.foxes)]
    grass_pop = [np.sum(field.field)]

    # plot at 1000 generations
    for _ in range(1000):
        field.generation()
        rabbit_pop.append(len(field.rabbits))
        fox_pop.append(len(field.foxes))
        grass_pop.append(np.sum(field.field))

    plt.plot(gens, rabbit_pop, label='Rabbits')
    plt.plot(gens, fox_pop, label='Foxes')
    plt.plot(gens, grass_pop, label='Grass')
    plt.xlabel('Generations')
    plt.ylabel('Organisms')
    plt.title('Rabbit, Fox, and Grass Population over the course of 1000 Generations')
    plt.legend()
    plt.savefig('alife_simulation.png')
    plt.show()


    # set up another field object for the animation
    # create the ecosystem
    field = Field()

    # initialize with some rabbits
    for _ in range(INIT_RABBITS):
        field.add_rabbit(Animal(type='rabbit'))

    # initialize with some foxes
    for _ in range(INIT_FOXES):
        field.add_fox(Animal(type='fox'))

    # set up the image object
    array = np.ones(shape=(SIZE, SIZE), dtype=int)
    fig = plt.figure(figsize=(10, 10))
    im = plt.imshow(array, cmap=cmap, interpolation='hamming', aspect='auto', vmin=0, vmax=3)
    anim = animation.FuncAnimation(
        fig,
        animate,
        fargs=(field, im),
        frames = 10 ** 100,
        interval = 1,
        repeat = True
    )
    plt.show()

if __name__ == '__main__':
    main()