# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np

from reservoir_tools import utils
from reservoir_tools.control_graphs import control_graphs, randomize_network
from reservoir_tools import datasets
from reservoir_tools import grn_networks
from reservoir_tools import network_tools as nt
from reservoir_tools.network_statistics import average_degree, connectivity_fraction
from reservoir_tools.readouts import RidgeRegression
from reservoir_tools.reservoirs import simulate_reservoir_dynamics


def reservoir_performance(data_source, adj_matrix, input_weight=None,
                          spectral_radius_scale=0.9, with_bias=True):

    if hasattr(adj_matrix, "todense"):
        adj_matrix = adj_matrix.todense()
    adj_matrix = np.asarray(adj_matrix)

    # Generate dataset
    [x, y] = data_source.func(sample_len=1000)

    weights = adj_matrix * (np.random.random(adj_matrix.shape)*2-1)

    if spectral_radius_scale:
        spectral_radius = utils.get_spectral_radius(weights)
        if spectral_radius == 0:
            raise RuntimeError("Nilpotent adjacency matrix matrix")
        weights *= spectral_radius_scale / spectral_radius

    in_scaling = 0.05
    in_weight = input_weight * in_scaling

    res_dynamics = [simulate_reservoir_dynamics(weights.T, in_weight.T,
                                                i_sig.ravel())
                    for i_sig in x]
    rregr = RidgeRegression(use_bias=with_bias)
    [rregr.train(x_train, y_train)
     for x_train, y_train in zip(res_dynamics[:-1], y[:-1])]
    pred = rregr(res_dynamics[-1])
    nrmse = utils.nrmse(pred, y[-1])

    if np.isnan(nrmse) or np.isinf(nrmse):
        raise RuntimeError("The NRMSE value obtained is not finite.")

    return nrmse


class _data_source():
    def __init__(self, name, func):
        self.name = name
        self.func = func


data_sources = [_data_source("10th order NARMA", datasets.narma10),
                _data_source("30th order NARMA", datasets.narma30)]

cp_file_layout = "multy_{task}NARMA_{system}_{num_trials}trials.cp"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run NARMA test on a given "
                                                 "network")
    g_basic = parser.add_argument_group('Basic arguments')
    g_basic.add_argument("--network", nargs="*", required=True)
    g_basic.add_argument("--whole-network", action="store_true")
    g_basic.add_argument("-t", "--task", type=int, choices=(0, 1),
                         required=True)
    g_basic.add_argument("-n", "--num_trials", type=int, default=1000)
    g_basic.add_argument("-s", "--seed", type=int, default=None)

    g_randomize = parser.add_argument_group('Simulating GRN topologies')
    g_randomize.add_argument('-r', '--randomize', type=int, choices=(0, 1, 2),
                             default=0, help=("0 -> no randomization; "
                                              "1 -> degree conservative; "
                                              "2 -> size & order conservative"))

    g_control = parser.add_argument_group('Simulating control topologies')
    g_control.add_argument("--size", type=int, default=0)
    g_control.add_argument("--controls", nargs="*", default=[None])
    g_control.add_argument("--number_edges", type=int, default=-1)
#    parser.add_argument("--avrg_degree", type=float)
#    parser.add_argument("--conn_fraction", type=float)

    g_input = parser.add_argument_group('Signal input connections')
    g_input.add_argument('--inputs', nargs="*", default=['0.66'])
    g_input.add_argument('--inputs_file', type=argparse.FileType('r'))
    g_input.add_argument('--randomize_inputs', action='store_true')

    g_output = parser.add_argument_group('Output control arguments')
    g_output.add_argument("-o", "--override_output_file", action="store_true")
    g_output.add_argument("-v", "--verbose", action='count')
    args = parser.parse_args()

    import Reservoir_tools as rt
    import sys
    import os
    import warnings
    import cPickle as cP
    import networkx as nx
    import pandas as pd

    import random

    args.network = map(str.lower, args.network)
    # TODO: size < 0
    if args.randomize and hasattr(args, 'controls'):
        raise Exception
    if args.size is not 0 and args.controls[0] is None:
        raise Exception
    # TODO: test for valid int values and valid float values
    if not all((utils.is_number(i) for i in args.inputs)):
        if not hasattr(args, 'inputs_file'):
            raise Exception
        else:
            inputs_data = pd.read_csv(args.inputs_file, delim_whitespace=True)
            if not all((utils.is_number(i) or i in inputs_data.columns or i == "None"
                        for i in args.inputs)):
                raise Exception
        if not all((net.lower() in ['ecocyc', '1'] for net in args.network)):
            raise Exception

    if args.seed is None:
        args.seed = random.randint(1, 10000000)

    if args.verbose > 0:
        warnings.simplefilter("always")

    size = args.size
    num_trials = args.num_trials
    spectral_radius = 0.9

    GRN_systems = [grn for index, grn in enumerate(grn_networks.grn_names)
                   if str(index) in args.network or grn.lower() in args.network]

    if not GRN_systems:
        warnings.warn("Oops! No GRN was selected!")
    # TODO: make possible to add ask for a control and the GRN at the same call
    if args.controls[0] is None:
        control_topologies = [('None', None, None)]
    else:
        control_topologies = [topo for index, topo in enumerate(control_graphs)
                              if str(index) in args.controls or
                              topo[0] in args.controls]

    for grn_name in GRN_systems:
        graph = grn_networks.load_network(grn_name)

        if not args.whole_network:
            graph = nt.prune_graph(graph, verbose=args.verbose)
        else:
            grn_name += "_WN"

        if args.size is 0:
            size = graph.number_of_nodes()

        # avrg_degree = 2 * num_edges / num_nodes
        avrg_degree = average_degree(graph)
        # conn_fraction = num_edges / total_possible_edges
        #               = num_edges / num_nodes^2
        conn_fraction = connectivity_fraction(graph)

        number_nodes = graph.number_of_nodes()
        if args.number_edges > 0:
            if args.number_edges > number_nodes**2:
                warnings.warn(('WARN: The network used has {} nodes, and thus '
                               'it can have up to {} edges. It is not possible'
                               ' to make it have {}'
                               ' edges.').format(number_nodes, number_nodes**2,
                                                 args.number_edges))
                sys.exit()
            avrg_degree = 2*args.number_edges / number_nodes
            conn_fraction = args.number_edges / number_nodes**2

        for control_name, control_topo, name_func in control_topologies:

            if control_name.startswith('FFW'):
                SR_scale = 0
            else:
                SR_scale = spectral_radius
            for inputs_set in args.inputs:
                random.seed(args.seed)
                np.random.seed(args.seed)

                if not utils.is_number(inputs_set):
                    if inputs_set == "None":
                        inputs_arr = None
                    else:
                        inputs_arr = inputs_data[inputs_set].values.reshape(-1, 1)
                    if inputs_set == "Any":
                        inputs_arr *= np.random.randint(0, 2, (size, 1))*2 - 1
                    if args.randomize_inputs:
                        rand_inputs = True
                        inputs_set = inputs_set
                        if control_name == 'None':
                            control_name = 'randIN'
                    else:
                        rand_inputs = False
                else:
                    num_inputs = float(inputs_set)
                    if num_inputs < 1:  # it cannot be the number of inputs
                        num_inputs *= size  # so it must mean the proportion
                    num_inputs = int(num_inputs)
                    inputs_arr = np.sign(np.random.rand(size, 1) - 0.5)
                    inputs_arr[num_inputs:] = 0
                    rand_inputs = True

                file_prefix = 'seed{seed}_'.format(seed=args.seed)

                if args.randomize:
                    file_prefix += "rand{}_".format(args.randomize)

                if name_func is not None:
                    file_prefix += name_func(reference=grn_name,
                                             size=args.size,
                                             conn_fraction=conn_fraction,
                                             avrg_degree=avrg_degree,
                                             expected_edges=args.number_edges)
                    file_prefix += '_'
                elif control_name != 'None':
                    file_prefix += 'control={}_'.format(control_name)

                file_prefix += 'inputs={inputs}_'.format(inputs=inputs_set)

                filename = cp_file_layout.format(
                                task=data_sources[args.task].name[:4],
                                system=grn_name, num_trials=num_trials)
                filename = file_prefix + filename

                if not args.override_output_file and os.path.isfile(filename):
                    warnings.warn("Output file already exists. Use the "
                                  "argument 'override_output_file' if you "
                                  "want to override it")
                    continue

                noNilpotent_trials = 0
                results = []

                while len(results) < num_trials:
                    #print("trial ->", len(results))

                    if control_topo is not None:
                        adj_matrix = control_topo(graph=graph,
                                                  size=size,
                                                  conn_fraction=conn_fraction,
                                                  avrg_degree=avrg_degree,
                                                  expected_edges=args.number_edges)

                    else:  # TODO: randomize options as normal controls
                        if args.randomize == 1:
                            adj_matrix = randomize_network(
                                graph.copy(), degree_conservative=True)
                        elif args.randomize == 2:
                            adj_matrix = randomize_network(
                                graph.copy(), degree_conservative=False)
                        else:
                            adj_matrix = nx.adjacency_matrix(graph).todense()
                        #print("Randomized!")

                    if isinstance(adj_matrix, nx.Graph):
                        adj_matrix = nx.adjacency_matrix(adj_matrix).todense()
                    adj_matrix = np.asarray(adj_matrix)

                    if rand_inputs:
                        np.random.shuffle(inputs_arr)

                    if SR_scale and utils.get_spectral_radius(adj_matrix) == 0:
                        #print("Spectral radius --> 0")
                        noNilpotent_trials += 1
                        if noNilpotent_trials > 10000:
                            print("NO-NILPOTENT MATRIX NOT FOUND!")
                            raise ValueError
                        continue

                    results.append(reservoir_performance(
                                        data_sources[args.task], adj_matrix,
                                        input_weight=inputs_arr,
                                        spectral_radius_scale=SR_scale))
                    #print(results[-1])

                data = {"system": grn_name,
                        "task": args.task,
                        "seed": args.seed,
                        "randomized": args.randomize,
                        "control": control_name,
                        "results": results,
                        "inputs": inputs_set,
                        "avrg_degree": average_degree(adj_matrix),
                        "size": size,
                        "conn_fraction": connectivity_fraction(adj_matrix)}
                print(data)
                #cP.dump(data, file(filename,"w"), protocol=2)
