# -*- coding: utf-8 -*-
"""Tasks used to test RC performance."""

import warnings

import numpy as np

from readouts import RidgeRegression
from utils import bisection


def k_memory_capacity(k, Ut, Xt, U, X):
    """Compute the k-delay memory capacity."""
    lr = RidgeRegression(use_bias=True, use_pinv=True)
    [lr.train(x=xt[k:, :], y=ut[:-k]) for ut, xt in zip(Ut, Xt)]

    Y = np.squeeze(lr(X[k:, :]))
    (var_u, cov_uy), (cov_uy, var_y) = np.cov(np.squeeze(U[:-k]), Y)
    return cov_uy**2/(var_u*var_y)


def memory_capacity(U, X, rtol=0.001, max_delay=10000):
    """Compute the short-term memory capacity."""
    memCapacity = 0.
    for k in xrange(1, max_delay):
        MCk = k_memory_capacity(k, Ut=U[:-1], Xt=X[:-1], U=U[-1], X=X[-1])
        memCapacity += MCk

        if MCk/memCapacity < rtol:
            break
    else:
        warnings.warn("Limit iterations exceeded without reaching the desired"
                      " precision in memory capacity.", RuntimeWarning)
    return memCapacity


def critical_memory_capacity(U, X, threshold=0.5):
    """Compute the critical memory capacity."""
    def f(k, Ut, Xt, U, X):
        return threshold - k_memory_capacity(k, Ut, Xt, U, X)

    return bisection(f, low=0, high=100, integer=True, skipchecks=True,
                     Ut=U[:-1], Xt=X[:-1], U=U[-1], X=X[-1])
