# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.integrate import ode 

def sigmoid(x):
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



        if finish_training:
            self.finish_training()

    def finish_training(self):
        if self._xTx is None:
            raise RuntimeError("The LinearRegression instance was not "
                               "trained!")
        invfun = np.linalg.pinv if self.use_pinv else np.linalg.inv
        inv_xTx = invfun(self._xTx)
        self.beta = np.dot(inv_xTx, self._xTy)



    def __call__(self, x):

        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        if self.beta is None:
            self.finish_training()
        
        if self.use_bias:
            x = _add_constant(x)


        decay = 0.05
        n_outs = 3

        def readout(t,y,res_dyn):

            equations = ((0.05*sigmoid(np.dot(res_dyn,self.beta[:,out])))-(decay*y))

            return equations

        x0 = np.repeat(0,n_outs)

        t = np.linspace(0,x.shape[0],num=len(x))

        dt = t[1]-t[0]

        output = [np.empty_like(t),np.empty_like(t)]

        res_dyn = x

        for out in range(0,n_outs):
            r = ode(readout).set_integrator("dop853")
            r.set_initial_value(x0[out],0).set_f_params(res_dyn[0:])
        
            for i in xrange(len(t)):
                
                r.set_f_params(res_dyn[i,:])
                r.integrate(r.t+dt)

                output[out][i] = r.y

        return (output)



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
        inv_xTx = invfun(self._xTx + self.ridge_param*np.eye(*self._xTx.shape))
        self.beta = (np.dot(inv_xTx,self._xTy))

    