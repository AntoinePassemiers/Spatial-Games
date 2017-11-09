# -*- coding: utf-8 -*-
# sim.py
# author : Antoine Passemiers

import numpy as np
import random, sys
from scipy.misc import imread
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Import markovian stuff
# https://github.com/AntoinePassemiers/ArchMM
try:
    from archmm.mrf import MarkovRandomField, clique_2nd_order
    USE_ARCHMM = True
except ImportError:
    USE_ARCHMM = False

from utils import *

import pyximport; pyximport.install()
pyximport.install(setup_args = {'include_dirs': np.get_include()})
import simulation


if __name__ == "__main__":
    SEED = 1494
    np.random.seed(SEED)
    random.seed(SEED)

    n_iter = 50

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    """ Part 1 """

    PDP_MAT = np.asarray([[ 7, 0], [10, 0]]) # payoff matrix

    # Plot evolution of the cooperation level for a 50x50 lattice
    history50, coop50 = simulation.run(n_iter, 50, True, True, PDP_MAT)
    plot_cooperation([coop50], "coop1", ["50x50"])

    # Plot history of the world state using a Morre neighborhood
    plot_history(history50, "history1")

    # Plot evolution of the cooperation level for different lattice sizes
    historyXX, coopXX = list(), list()
    for l in [4, 8, 12, 20, 50]:
        avg_coop = np.zeros(n_iter, dtype = np.float)
        for i in range(100):
            history, coop = simulation.run(n_iter, l, True, True, PDP_MAT)
            avg_coop += coop
        avg_coop / 100.0
        historyXX.append(history)
        coopXX.append(avg_coop)
    plot_cooperation(coopXX, "coop1XX", ["4x4", "8x8", "12x12", "20x20", "50x50"])

    # Plot history of the world state using a Von neumann neighborhood
    history50, coop50 = simulation.run(n_iter, 50, False, True, PDP_MAT)
    plot_cooperation([coop50], "coop1vonneumann", ["50x50"])
    plot_history(history50, "history1vonneumann")


    """ Part 2 """

    PDP_MAT = np.asarray([[ 7, 3], [10, 0]]) # payoff matrix

    # Plot evolution of the cooperation level for a 50x50 lattice
    history50, coop50 = simulation.run(n_iter, 50, True, False, PDP_MAT)
    plot_cooperation([coop50], "coop2", ["50x50"])

    # Plot history of the world state using a Morre neighborhood
    plot_history(history50, "history2")

    # Plot evolution of the cooperation level for different lattice sizes
    historyXX, coopXX = list(), list()
    for l in [4, 8, 12, 20, 50]:
        avg_coop = np.zeros(n_iter, dtype = np.float)
        for i in range(100):
            history, coop = simulation.run(n_iter, l, True, False, PDP_MAT)
            avg_coop += coop
        avg_coop / 100.0
        historyXX.append(history)
        coopXX.append(avg_coop)
    plot_cooperation(coopXX, "coop2XX", ["4x4", "8x8", "12x12", "20x20", "50x50"])

    # Plot history of the world state using a Von neumann neighborhood
    history50, coop50 = simulation.run(n_iter, 50, False, False, PDP_MAT)
    plot_cooperation([coop50], "coop2", ["50x50"])
    plot_history(history50, "history2vonneumann")


    """ Part 3 """

    if USE_ARCHMM:
        n_classes = 2
        targets = [imread("imgs/mrf/%i.png" % i) for i in range(0, 2)]
        X = [x[:, :, :3] for x in targets]
        Y = [np.full(x.shape[:3], i, dtype = np.int) for i, x in enumerate(X)]

        print(X[0])

        mrf = MarkovRandomField(n_classes, save_history = True, clique = clique_2nd_order)
        mrf.fit(X, Y)
        img = imread("imgs/mrf/bear.png")
        img = np.asarray(img, dtype = np.uint8)

        print(mrf.parameters)

        history = mrf.simulated_annealing(img, T0 = 10.0, dq = 0.95, beta = 5.0, eta = 10.0, max_n_iter = 50, fast = True)


        plot_history(history, "history3", timestamps = [0, 1, 2, 3, 5, 50])