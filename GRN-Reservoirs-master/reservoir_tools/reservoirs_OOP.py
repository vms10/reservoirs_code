# -*- coding: utf-8 -*-
"""Tools to simulate reservoir dynamics."""

import numpy as np


class Reservoir(object):

    def __init__(self, weights, input_weights, initial_state):
        nnodes = weights.shape[0]
        nstreams = 1 if input_signal.ndim == 1 else input_signal.shape[0]
        tsteps = input_signal.shape[-1]

        if weights.dims != 2 or nnodes != weights.shape[1]:
            raise

        input_weights = np.atleast_2d(input_weights)
        if (nstreams != input_weights.shape[0]
                or nnodes != input_weights.shape[1]):
            raise

        self._nnodes = nnodes
        self._input_dim = None #TODO
        self.weights = weights
        self.input_weights = input_weights
        self.initial_state = initial_state
        self.state = initial_state.copy()
        self.node_function = node_function or np.tanh

    def randomize_weights(self):
        self.weights = (self.weights
                        * np.random.uniform(0, 1, size=self.weights.shape))

    def randomize_input_weights(self):
        self.input_weights = self.input_weights * (np.random.randint(0, 2)*2-1)

    def scale_spectral_radius(self, spectral_radius=0.9):
        current_sr = rt.get_spectral_radius(self.weights)
        if current_sr != 0:
            self.weights *= spectral_radius/current_sr

    def __exec__(input_signal):
        if input_signal.ndim != 1 and input_signal.shape[0] != self._input_dim:
            raise
        tsteps = input_signal.shape[-1]
        matrix_product = np.dot
        dynamics = np.zeros((tsteps+1, self._nnodes))

        dynamics[0, :] = self.state

        input_per_node = matrix_product(input_signal, self.input_weights)

        for t in range(0, input_signal.shape[0]):
            dynamics[t+1, :] = self.node_function(
                                   matrix_product(dynamics[t, :], self.weights)
                                   + input_per_node[t])

        self.state = dynamics[-1, :]
        return dynamics[1:, :]
