#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 18:51:40 2022

@author: leila
"""

# -*- coding: utf-8 -*-
"""Datasets used to test the performance of Reservoir Computing setups."""

import functools
import warnings
import numpy as np
from scipy.integrate import odeint 




def keep_bounded(dataset_func, max_trials=100, threshold=1e5):    
    """Wrapper function to regenerate datasets when they get unstable."""
    @functools.wraps(dataset_func)
    def stable_dataset(n_samples=100, sample_len=1000):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "overflow", RuntimeWarning)
            for i in range(max_trials):
                [x, y, z] = dataset_func(n_samples=n_samples, sample_len=sample_len)
                if np.max((x)) < threshold:
                    return [x, y, z]
            else:
                errMsg = ("It was not possible to generate dataseries with {} "
                          "bounded by {} in {} trials.".format(
                              dataset_func.__name__, threshold, max_trials))
                raise RuntimeError(errMsg)

    return stable_dataset


@keep_bounded
def ga3(n_samples=100, sample_len=1000):

    inputs, outputs = [], []
    inputs_derivative = []
    
    for sample in range(n_samples):

        kd = 0.05

        rand = np.random.randint(0,3)

        if rand == 0:
            a = np.random.uniform(1*((1/float(sample_len))),0.8*((1/float(sample_len))))
            k_A = 0.05
            k_B = 0
            k_C = 0
        elif rand == 1:
            a = np.random.uniform(-0.1*((1/float(sample_len))),0.1*((1/float(sample_len))))
            k_A = 0
            k_B = 0.05
            k_C = 0
        else:
            a = np.random.uniform(-0.8*((1/float(sample_len))),-1*((1/float(sample_len))))
            k_A = 0
            k_B = 0
            k_C = 0.05

        inputs_derivative.append(a)

        def gen_act(x,t):

            z, A, B, C = x 
            dzdt = a # input
            dAdt = (k_A)-(kd*A) # gen A output
            dBdt = (k_B)-(kd*B) # gen B output
            dCdt = (k_C)-(kd*C) # gen C output

            return(dzdt,dAdt,dBdt,dCdt)

        x0 = [1,0,0,0] # Input is 1 at t=0

        t = np.linspace(0,sample_len,num=1000)

        x = odeint(gen_act,x0,t)

        x[:,0] += np.random.uniform(-0.1,0.1,len(x[:,0]))

        inputs.append(x[:,0])
        inputs[sample].shape = (-1, 1)
        outputs.append(x[:,1:])
        
    return inputs, outputs, inputs_derivative


@keep_bounded
def narma30(n_samples=10, sample_len=1000):
    """
    Return data for the 30th order NARMA task.

    Generate a dataset with the 30th order Non-linear AutoRegressive Moving
    Average.

    Parameters
    ----------
    n_samples : int, optional (default=10)
        number of example timeseries to be generated.
    sample_len : int, optional (default=1000)
        length of the time-series in timesteps.

    Returns
    -------
    inputs : list (len `n_samples`) of arrays (shape `(sample_len, 1)`)
        Random input used for each sample in the dataset.
    outputs : list (len `n_samples`) of arrays (shape `(sample_len, 1)`)
        Output of the 30th order NARMA dataset for the input used.
    """
    system_order = 30
    inputs, outputs = [], []
    for sample in range(n_samples):
        inputs.append(np.random.rand(sample_len, 1) * .5)
        inputs[sample].shape = (-1, 1)
        outputs.append(np.zeros((sample_len, 1)))
        for k in range(system_order-1, sample_len - 1):
            outputs[sample][k + 1] = .2 * outputs[sample][k] +          \
                .04 * outputs[sample][k] *                              \
                np.sum(outputs[sample][k - (system_order-1):k+1]) +     \
                1.5 * inputs[sample][k - 29] * inputs[sample][k] + .001
    return inputs, outputs, outputs