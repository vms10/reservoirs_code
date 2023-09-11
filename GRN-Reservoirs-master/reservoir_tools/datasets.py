# -*- coding: utf-8 -*-
"""Datasets used to test the performance of Reservoir Computing setups."""

import functools
import warnings

import numpy as np


def keep_bounded(dataset_func, max_trials=100, threshold=1e5):
    """Wrapper function to regenerate datasets when they get unstable."""
    @functools.wraps(dataset_func)
    def stable_dataset(n_samples=10, sample_len=1000):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "overflow", RuntimeWarning)
            for i in range(max_trials):
                [x, y] = dataset_func(n_samples=n_samples,
                                      sample_len=sample_len)
                if np.max((x, y)) < threshold:
                    return [x, y]
            else:
                errMsg = ("It was not possible to generate dataseries with {} "
                          "bounded by {} in {} trials.".format(
                              dataset_func.__name__, threshold, max_trials))
                raise RuntimeError(errMsg)

    return stable_dataset


@keep_bounded
def narma10(n_samples=10, sample_len=1000):
    """
    Return data for the 10th order NARMA task.

    Generate a dataset with the 10th order Non-linear AutoRegressive Moving
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

    WARNING: this is an unstable dataset. There is a small chance the system
    becomes unstable, leading to an unusable dataset. It is better to use
    NARMA30 which where this problem happens less often.
    """
    system_order = 10
    inputs, outputs = [], []
    for sample in range(n_samples):
        inputs.append(np.random.rand(sample_len, 1) * .5)
        inputs[sample].shape = (-1, 1)
        outputs.append(np.zeros((sample_len, 1)))
        for k in range(system_order-1, sample_len - 1):
            outputs[sample][k + 1] = .3 * outputs[sample][k] +         \
                .05 * outputs[sample][k] *                             \
                np.sum(outputs[sample][k - (system_order-1):k+1]) +    \
                1.5 * inputs[sample][k - 9] * inputs[sample][k] + .1
    return inputs, outputs


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
    return inputs, outputs
