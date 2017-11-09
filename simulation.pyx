# -*- coding: utf-8 -*-
# simulation.pyx
# author : Antoine Passemiers
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as cnp
cnp.import_array()

cimport libc.stdlib
from cython.parallel import parallel, prange

import random


COOPERATE = 0
DEFECT    = 1

cdef cnp.int_t[:, :] moore_indices = np.asarray([[0,1],[1,0],[-1,0],[0,-1],[-1,-1],[1,-1],[-1,1],[1,1]], dtype = np.int)
cdef cnp.int_t[:, :] von_neumann_indices = np.asarray([[0,1],[1,0],[-1,0],[0,-1]], dtype = np.int)

cdef void get_neighborhood(Py_ssize_t L, Py_ssize_t i, Py_ssize_t j, cnp.int_t[:, :] indices, cnp.int_t[:, :] abs_indices) nogil:
    cdef Py_ssize_t di, dj, k
    for k in range(indices.shape[0]):
        di, dj = indices[k, 0], indices[k, 1]
        abs_indices[k, 0] = (i + di + L) % L
        abs_indices[k, 1] = (j + dj + L) % L

def init_matrices(Py_ssize_t L):
    lattice = np.random.randint(2, size = (L, L))
    lattice = np.asarray(lattice, dtype = np.uint8)
    old_lattice = np.copy(lattice)
    earnings = np.zeros((L, L), dtype = np.int)
    return old_lattice, lattice, earnings

def run(Py_ssize_t n_iter, Py_ssize_t L, bint use_moore, bint random_neighbor, object payoff_matrix):
    print("Run simulation with %i iterations on a (%i x %i) matrix" % (n_iter, L, L))
    cooperation_level = list()
    history = list()
    old_lattice, lattice, earnings = init_matrices(L)
    history.append(np.copy(lattice))

    payoff_matrix = np.asarray(payoff_matrix, dtype = np.int)
    cdef cnp.int_t[:, :] payoff_mat = payoff_matrix
    cdef cnp.int_t[:, :] earnings_buf = earnings
    cdef cnp.uint8_t[:, :] lattice_buf = lattice
    cdef cnp.uint8_t[:, :] old_lattice_buf = old_lattice
    cdef cnp.int_t[:, :] rel_indices = moore_indices if use_moore else von_neumann_indices
    cdef cnp.int_t[:, :] abs_indices = np.copy(moore_indices if use_moore else von_neumann_indices)

    cdef float Pij
    cdef int best_earnings, Wi, Wj, max_payoff = payoff_matrix.max(), min_payoff = payoff_matrix.min()
    cdef Py_ssize_t k, l, i, j, ni, nj, best_k
    for k in range(n_iter):
        with nogil:
            # Playing game with neighbours
            for i in range(L):
                for j in range(L):
                    earnings_buf[i, j] = 0
                    get_neighborhood(L, i, j, rel_indices, abs_indices)
                    for l in range(abs_indices.shape[0]):
                        ni, nj = abs_indices[l, 0], abs_indices[l, 1]
                        earnings_buf[i, j] += payoff_mat[old_lattice_buf[i, j], old_lattice_buf[ni, nj]]
        with nogil:
            # Update strategies
            for i in range(L):
                for j in range(L):
                    get_neighborhood(L, i, j, rel_indices, abs_indices)

                    if random_neighbor:
                        best_earnings, best_k = earnings_buf[i, j], -1
                        for l in range(abs_indices.shape[0]):
                            ni, nj = abs_indices[l, 0], abs_indices[l, 1]
                            if earnings_buf[ni, nj] > best_earnings:
                                best_earnings, best_k = earnings_buf[ni, nj], l
                        if best_k != -1:
                            ni, nj = abs_indices[best_k, 0], abs_indices[best_k, 1]
                            lattice_buf[i, j] = old_lattice_buf[ni, nj]
                    else:
                        best_k = libc.stdlib.rand() % rel_indices.shape[0]
                        ni, nj = abs_indices[best_k, 0], abs_indices[best_k, 1]
                        Wi = earnings_buf[i, j]   # earnings of current players
                        Wj = earnings_buf[ni, nj] # earnings of the random neighbour
                        Pij = (1.0 + (Wj - Wi) / (<float>rel_indices.shape[0] * (max_payoff - min_payoff))) / 2.0
                        if <double>libc.stdlib.rand() / <double>libc.stdlib.RAND_MAX < Pij: # Replicator rule
                            lattice_buf[i, j] = old_lattice_buf[ni, nj]


        old_lattice_buf[:, :] = lattice_buf[:, :]
        lattice = np.asarray(lattice_buf)
        history.append(np.copy(lattice))
        cooperation_level.append(float((lattice == COOPERATE).sum()) / float(L ** 2))

    return np.asarray(history), np.asarray(cooperation_level)