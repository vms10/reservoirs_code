# -*- coding: utf-8 -*-
"""Tools to simulate reservoir dynamics."""

import warnings
import numpy as np
from scipy.integrate import odeint 

def get_spectral_radius(W):
    """
    Return the spectral radius (largest absolute eigenvalue) of the matrix W.
    """
    return np.amax(np.absolute(np.linalg.eigvals(W)))

def simulate_reservoir_dynamics(weights, input_weights, input_der, sample_len,
                                init_state=None):
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
    k=0.05 # protein production
    d=0.05 # protein degradation
    
    def sigmoid(x):
        return 0.5*((x-0.5)/np.sqrt((x-0.5)**2+0.1)+1)
    
    def input_signal(t, input_der):
        return input_der*t + 1
        
    def dyn(x, t, input_der):
        
        dxdt = np.zeros(nnodes)
        for i in range(nnodes):
            r = 0
            for j in range(nnodes):
                r += weights[j][i] * x[j]
            dxdt[i] = k * sigmoid(input_weights[i]*input_signal(t, input_der) + r) - d * x[i] # dinamica de los genes del reservorio, ecuacion 1 del TFM
         
        return dxdt


    x0 = np.repeat(0,nnodes)
    if init_state is not None:
        x0 = init_state 
    
    t = np.linspace(0,sample_len,sample_len)

    x = odeint(dyn, x0, t, args=(input_der,))
    
    #np.savetxt('output.csv', x)
    return x
                               

