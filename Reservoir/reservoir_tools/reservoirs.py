#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 10:27:24 2022

@author: leila
"""

# -*- coding: utf-8 -*-
"""Tools to simulate reservoir dynamics."""

import warnings

import numpy as np
from scipy.integrate import odeint 
from scipy.stats import linregress

from utils import get_spectral_radius

import matplotlib.pyplot as plt

def simulate_reservoir_dynamics(weights, input_weights, phase, freq, input_type,sample_len,
                                init_state=None, node_function=None):
    """Simulate the dynamics of a reservoir.

    Given the internal weights that define the reservoir, the input weights,
    the input signal stream, the initial state and the integrative function
    that the nodes apply, it computes the dynamics of the reservoir.

    Parameters
    ----------
    weights : numpy array
        Weighted adjacency matrix that defines the connectivity of the
        reservoir. Must be squared.
    input_weights : numpy array
        Weights that determine the effect of the input signal stream on each of
        the nodes of the reservoir. Rows are input streams (there might be more
        than one) and columns correspond to reservoir nodes (equal to number of
        rows and columns of `weights`).
    init_state : None, numpy array, optional (default=None)
        Value of the nodes of the reservoir at the initial time step. It must
        be a 1-dimensional vector of the length of the reservoir size. If
        `None` all nodes are initialized at `0`.
    node_function : None, Function, optional (default=None)
        Function applied to the weigted sum of all the incomming connections of
        a node to compute its state. It defaults to the hiperbolic tangent.
    """
    nnodes = weights.shape[0]
    k=0.05
    d=0.05
    amp=5
    
    def sigmoid(x):
     return (((x-0.5)/(np.sqrt(((x-0.5)**2)+0.1)))+1)/2
    
    def input_signal(t, phase, freq, input_type):
        if input_type == 0:
            return 0*t+5
        else :
            return amp*np.sin(freq*(t+phase))+10
        
    def dyn(x, t, phase, freq, input_type):
        
        dxdt = np.zeros(nnodes)
        for i in range(nnodes):
            r = 0
            for j in range(nnodes):
                r += weights[j][i] * x[j]
            dxdt[i] = k * sigmoid(input_weights[i]*input_signal(t, phase, freq, input_type) + r) - d * x[i] # dinamica de los genes del reservorio, ecuacion 1 del TFM
         
        return dxdt


    x0 = np.repeat(0,nnodes)
    if init_state is not None:
        x0 = init_state 
    
    t = np.linspace(0,sample_len,sample_len)

    x = odeint(dyn, x0, t, args=(phase,freq, input_type))
    
    np.savetxt('output.csv', x)
    
    
    
    return x
                               



def remove_node(node, weights, input_weights, list_nodes=None,
                spectral_radius=None):
    """Remove one node from the reservoir.

    Given the internal weights that define the reservoir and the input weights,
    remove one node by eliminating all the connections that go from or to it.

    Parameters
    ----------
    node : int or other.
        Index of the node that must be eliminated. If `list_nodes` is an
        iterable, the index at which the first instance of `node` is found in
        `list_nodes` will be the eliminated node.
    weights : numpy array
        Weighted adjacency matrix that defines the connectivity of the
        reservoir. Must be squared.
    input_weights : numpy array
        Weights that determine the effect of the input signal stream on each of
        the nodes of the reservoir. Columns are input streams (there might be
        more than one) and rows correspond to reservoir nodes (equal to number
        of rows and columns of `weights`).
    list_nodes : None, iterable, optional (default=None)
        List of nodes in which `node` is searched. If not specified `node` is
        assumed to be the index of the node to be eliminated.
    spectral_radius : None, float, optional (default=None)
        If specified the spectral radius of the weighted connectivity matrix is
        scaled to that value after removing the node.
    """
    if list_nodes is not None:
        node = next((i for i, x in enumerate(list_nodes) if x == node))

    weights = np.delete(np.delete(weights, node, axis=0), node, axis=1)

    if spectral_radius:
        try:
            weights *= spectral_radius/get_spectral_radius(weights)
        except ZeroDivisionError:
            warnings.warn("Could not normalize spectral radius because "
                          "it is 0")

    input_weights = np.delete(np.atleast_2d(input_weights), node, axis=1)

    return weights, input_weights
