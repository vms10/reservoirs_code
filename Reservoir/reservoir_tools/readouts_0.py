#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 09:58:40 2022

@author: leila
"""

# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.integrate import odeint
from scipy.integrate import ode 
from scipy.interpolate import interp2d
import pandas as pd


def sigmoid(x):
    #return (1/(1+np.exp(-x)))
    #return ((x/(np.sqrt((x**2)+1)))+1)/2
    return (((x-0.5)/(np.sqrt(((x-0.5)**2)+0.1)))+1)/2


def _add_constant(x):
    """Add a constant term to the vector 'x'.
    x -> [1 x]
    """
    return np.concatenate((np.ones((x.shape[0], 1), dtype=x.dtype), x), axis=1)
    


class LinearRegression(object):
    
    def __init__(self,  train_x=None, train_y=None, use_bias=True,
                 use_pinv=True, finish_training=False):
        self.use_bias = use_bias
        self.use_pinv = use_pinv

        # Set two temporary variables for the linear regression estimation
        self._xTx = None
        self._xTy = None

        # keep track of how many data points have been sent
        self._tlen = 0

        # final regression coefficients
        # if with_bias=True, beta includes the bias term in the first column


        self.beta = None

        if train_x is not None and train_y is not None:
            self.train(train_x, train_y, finish_training=finish_training)

    def train(self, x, y, finish_training=False):

        if x.shape[0] != y.shape[0]:
            raise ValueError("X and Y do not describe the same number of "
                             "points ({} vs {})".format(x.shape, y.shape))
        # Forget current regression coefficients
        self.beta = None

        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if self.use_bias:
            x = _add_constant(x)


        # initialize internal vars if necessary
        if self._xTx is None:
            x_size = x.shape[1]
            self._xTx = np.zeros((x_size, x_size), x.dtype)
            self._xTy = np.zeros((x_size, y.shape[1]), x.dtype)


        # update internal variables
        self._xTx += (np.dot(x.T,x))
        self._xTy += (np.dot(x.T,y))
        self._tlen += x.shape[0]

        #print(self._xTy)
        invfun = np.linalg.pinv if self.use_pinv else np.linalg.inv
        inv_xTx = invfun(self._xTx)


        if finish_training:
            self.finish_training()

    def finish_training(self):
        if self._xTx is None:
            raise RuntimeError("The LinearRegression instance was not "
                               "trained!")
        invfun = np.linalg.pinv if self.use_pinv else np.linalg.inv
        inv_xTx = invfun(self._xTx)
        self.beta = np.dot(inv_xTx, self._xTy)





    def __call__(self, x, freq, phase, input_type):
        

        
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        if self.beta is None:
            self.finish_training()
        
        if self.use_bias:
            x = _add_constant(x)

        n_outs = 2
        
        def sigmoid(x):
         return (((x-0.5)/(np.sqrt(((x-0.5)**2)+0.1)))+1)/2
        
        def input_signal(t, types):
            if input_type == 0:
                return 0*t+5
            else :
                return amp*np.sin(freq*(t+phase))+10
            
        def dyn(x, t, k, d, types):
            
            dxdt = np.zeros(len(y0))
            for i in range(len(y0)):
                r = 0
                for j in range(len(w_in)):
                    r += W[j][i] * x[j]
                if i < len(w_res):
                    dxdt[i] = k * sigmoid(w_in[i]*input_signal(t, input_type) + r) - d * x[i] # dinamica de los genes del reservorio, ecuacion 1 del TFM
                else:
                    dxdt[i] = k * sigmoid(r)-d*x[i] # dinamica de B y D, ecuacion 5 del TFM
                 
            return dxdt
        
        
        w_res=np.loadtxt('weights.txt')
        w_in=np.loadtxt('in_weight.txt')        
        W =  np.append(w_res.transpose(), self.beta, axis=1)
        t = np.linspace(0,x.shape[0],num=len(x))
        amp=5
        k=0.05
        d=0.05
        y0=[0 for _ in range(len(w_in)+n_outs)]
        res_dyn = x
        output = odeint(dyn, y0, t, args=(k,d,res_dyn))
        
        return output[:,len(w_in):]


class RidgeRegression(LinearRegression):

    def __init__(self,  train_x=None, train_y=None, ridge_param=0,
                 use_bias=True, use_pinv=True, finish_training=False):

        self.ridge_param = ridge_param
        self.ridge_param = 0

        super(RidgeRegression, self).__init__(
            train_x=train_x, train_y=train_y, use_bias=use_bias,
            use_pinv=use_pinv, finish_training=finish_training)

    def finish_training(self):
        if self._xTx is None:
            raise RuntimeError("The LinearRegression instance was not "
                               "trained!")


        invfun = np.linalg.pinv if self.use_pinv else np.linalg.inv

        #print(-np.log((1/(self._xTy))-1))

        inv_xTx = invfun(self._xTx + self.ridge_param*np.eye(*self._xTx.shape))

        #constraint = (np.sinh(self._xTx)**2 + np.cosh((self._xTx)**2))/(np.sinh(self._xTx) * np.cosh((self._xTx)))
        self.beta = (np.dot(inv_xTx,self._xTy))

        #print(self._xTx)
        #print(self._xTy)
        print("LOS PESOS O ALGO")
        print(self.beta)
