# -*- coding: utf-8 -*-
# utils.py
# author : Antoine Passemiers

import os, random
import numpy as np
import matplotlib.pyplot as plt


SAVE_FIG = True
OUTPUT_FOLDER = "imgs"


def plot_cooperation(coops, name, labels):
    for label, coop in zip(labels, coops):
        plt.plot(coop, label = label)
    plt.legend()
    if not SAVE_FIG:
        plt.show()
    else:
        plt.savefig(os.path.join(OUTPUT_FOLDER, "%s.png" % name))
        plt.clf()
        plt.close()

def plot_history(history, name, timestamps = [0, 1, 5, 10, 20, 50]):
    f, axarr = plt.subplots(2, 3)
    axarr[0, 0].imshow(history[timestamps[0]])
    axarr[0, 0].set_title("t = %i" % timestamps[0])
    axarr[0, 1].imshow(history[timestamps[1]])
    axarr[0, 1].set_title("t = %i" % timestamps[1])
    axarr[0, 2].imshow(history[timestamps[2]])
    axarr[0, 2].set_title("t = %i" % timestamps[2])
    axarr[1, 0].imshow(history[timestamps[3]])
    axarr[1, 0].set_title("t = %i" % timestamps[3])
    axarr[1, 1].imshow(history[timestamps[4]])
    axarr[1, 1].set_title("t = %i" % timestamps[4])
    axarr[1, 2].imshow(history[timestamps[5]])
    axarr[1, 2].set_title("t = %i" % timestamps[5])

    if not SAVE_FIG:
        plt.show()
    else:
        plt.savefig(os.path.join(OUTPUT_FOLDER, "%s.png" % name))
        plt.clf()
        plt.close()