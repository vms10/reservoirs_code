import scipy as sp
import numpy as np


def get_spectral_radius(W):
    """
    Return the spectral radius (largest absolute eigenvalue) of the matrix W.
    """
    return np.amax(np.absolute(np.linalg.eigvals(W)))


def nrmse(input_signal, target_signal):
    """
    Compute the Normalized Root Mean Square Error between two signals.

    Calculates the normalized root mean square error (NRMSE) of the input
    signal compared to the target signal.
    Parameters:
        - input_signal : array
        - target_signal : array
    """
    if input_signal.shape != target_signal.shape:
        raise RuntimeError("Input shape (%s) and target_signal shape"
                           " (%s) should be the same." % (input_signal.shape,
                                                          target_signal.shape))

    input_signal = input_signal.flatten()
    target_signal = target_signal.flatten()

    # Use normalization with N-1, as in matlab
    var = target_signal.std(ddof=1) ** 2

    error = (target_signal - input_signal) ** 2
    return sp.sqrt(error.mean() / var)


def bisection(function, low, high, threshold=0, integer=False, atol=1e-5,
              skipchecks=False, **kwargs):
    """Flexible bisection method to find when a function crosses a threshold.

    Find the value of `x` for which `function(x, **kwargs) == theshold` using
    the bisection method. This algorithm limits the search of `x` within the
    interval `[low, high)`. The interval must fulfill
    `function(low, **kwargs) < threshold` and
    `function(high, **kwargs) > threshold`, or biceversa.

    Parameters:
        - threshold : float (0)
            Traditionay, the bisection algorithm finds the root of a function,
            that the value of `x` for which `function(x) == 0`. This
            implementation finds the value of `x` for which
            `function(x) == theshold`.
        - integer : boolean (False)
            Whether to limit the domain of `function` to integer values.
        - atol : float (1e-5)
            Absolute tolerance used as termination criterium to stop the
            iteration.
        - skipchecks : boolean (False)
            If `True`, `function(low)` and `function(high)` are not evaluated
            and the former is assumed to be smaller or equal than threshold and
            the latter is assumed to be larger than `threshold`. This is to be
            used when the execution of `function(x)` is computationally
            expensive and the system is known.
    """
    # TODO: if integer=True and function(x)==0 for a non-integer x, the
    # criterium to chose the returned value its not clear. In fact, right now,
    # the value returned for function(x)=0 and -function(x)=0 would not be the
    # same.
    if not skipchecks:
        y1, y2 = map(function, (low, high))
        if (y1-threshold)*(y2-threshold) > 0:
            raise RuntimeError(
                "Bisection algorithm requires that the two starting points "
                "have an image with different sign (f({x1})={y1}, "
                "f({x2})={y2}).".format(x1=low, y1=y1, x2=high, y2=y2))
        if y2-threshold <= 0:
            low, high = high, low

    atol = (abs(atol)//1 or 1) if integer else abs(atol)
    while abs(high-low) > atol:
        mid = (low+high)//2 if integer else (low+high)/2.
        if function(mid, **kwargs) <= threshold:
            low = mid
        else:
            high = mid
    return low

def is_number(s):
    """Check if an input string is a number or not"""
    try:
        float(s)
        return True
    except ValueError:
        return False

def get_args_index(num, steps):
    res = []
    for step in steps:
        res.append(num % step)
        num //= step
    res.append(num)
    return res
