# -*- coding: utf-8 -*-
"""Package to simulate and test Reservoir Computing setups."""

from reservoirs import simulate_reservoir_dynamics, remove_node
from readouts import LinearRegression, RidgeRegression
from datasets import narma10, narma30
from tasks import memory_capacity, critical_memory_capacity
from utils import get_spectral_radius, nrmse, bisection, get_args_index

from Reservoir_tools import *
