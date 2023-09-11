#!usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import datasets_sol as datasets
from reservoir_tools.readouts import RidgeRegression
from reservoirs import simulate_reservoir_dynamics
import random

def get_spectral_radius(W):
    """
    Return the spectral radius (largest absolute eigenvalue) of the matrix W.
    """
    return np.amax(np.absolute(np.linalg.eigvals(W)))

def training(data_source,in_weight,weights):
    # Generate dataset 
    sample_len=1000
    [x, y, x_der] = data_source.func(n_samples=1000,sample_len=sample_len)
    
    res_dynamics = [simulate_reservoir_dynamics(weights, in_weight,
                                                input_der, sample_len)
                    for input_der in x_der]
    
    with_bias = False ################## NO BIAS #################
    rregr = RidgeRegression(use_bias=with_bias)
    
    [rregr.train(x_train, y_train) for x_train, y_train in zip(res_dynamics, y)]

    rregr.finish_training()

    return(rregr)
        



def reservoir_performance(data_source, adj_matrix, input_weight=None,
                          spectral_radius_scale=0.9, with_bias=True):

    if hasattr(adj_matrix, "todense"):
        adj_matrix = adj_matrix.todense()
    adj_matrix = np.asarray(adj_matrix)

    weights = adj_matrix * (np.random.random(adj_matrix.shape)*2-1) ### RANDOM wei

    if spectral_radius_scale:
        spectral_radius = get_spectral_radius(weights)
        if spectral_radius == 0:
            raise RuntimeError("Nilpotent adjacency matrix matrix")
        weights = weights *(spectral_radius_scale / spectral_radius)
    #np.savetxt('weights.txt', weights)
    #weights=np.loadtxt('weights.txt')

    in_scaling = 0.05
    in_weight = input_weight * in_scaling
    #np.savetxt('in_weight.txt', in_weight)	
    #in_weight =np.loadtxt('in_weight.txt')
                    
    rregr =  training(data_source,in_weight, weights)
        
    #np.savetxt('beta.txt', rregr.beta)
    #rregr.beta =np.loadtxt('beta.txt')
    
    results_2 = []
    results_3 = []
    results_4 = []
    
    results = []
    sample_len=1000
    n_samples=100
    

    [x, y, x_der] = data_source.func(n_samples=n_samples,sample_len=sample_len)
    
    res_dynamics = [simulate_reservoir_dynamics(weights, in_weight, input_der, sample_len) for input_der in x_der]

    
    for i in range(n_samples):

        pred = rregr(res_dynamics[i])
    
        n_out = 3  ### FOURTH GENE
        error = 0
        
        for outs in range(n_out):
            steady = np.mean(pred[outs][(sample_len-200):])
            steady_target = np.mean(y[i][(sample_len-200):,outs])
            error += (np.absolute(steady_target - steady))


        results.append(error)
        results_2.append(steady)
        results_3.append(pred)
        results_4.append(x[i][:])
            
    return((results), (results_2), (results_3), (results_4))


class _data_source():
    def __init__(self, name, func):
        self.name = name
        self.func = func


data_sources = [_data_source("30th order NARMA", datasets.narma30),
                _data_source("Gene activation 3", datasets.ga3)]

