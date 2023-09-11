"""Measure the effect of the weight scalling in MC for RNN and FFW."""

from __future__ import division, print_function
import warnings
import sys
import os

import numpy as np
import networkx as nx

from reservoir_tools import grn_networks
from reservoir_tools.control_graphs import random_topology
from reservoir_tools import network_tools as nt
from reservoir_tools import (get_spectral_radius, simulate_reservoir_dynamics,
                             memory_capacity, critical_memory_capacity)


def rand_FFN(nodes, edges, layers, maxdistance, prng=None):
    # If only the number of layers is given assume they have the same size
    if isinstance(layers, int):
        layers = [nodes//layers + (nodes % layers > i) for i in range(layers)]

    # let layer_of be a list where layer_of[i] is the layer of the ith node
    layer_of = [l for l in range(len(layers)) for node in range(layers[l])]

    nlayers = len(layers)
    if 1 > maxdistance:
        maxdistance = nlayers

    # Check if it is even possible
    maxedges = sum((layers[i]*layers[j]
                    for i in range(nlayers)
                    for j in range(i+1, min(i+1+maxdistance, nlayers))))
    if maxedges < edges:
        raise RuntimeError("{} are too many edges for a network with layers of"
                           " size {}".format(edges, layers))

    if not isinstance(prng, np.random.RandomState):
        prng = np.random.RandomState(prng)

    edgesset = set()
    for i in range(100000):
        # genera parella de nombres, ordenats i no iguals
        e = tuple(np.sort(prng.randint(nodes, size=2)))
        steps = layer_of[e[1]] - layer_of[e[0]]
        if 0 < steps < maxdistance and e not in edgesset:
            edgesset.add(e)
            if len(edgesset) == edges:
                break
    else:
        raise RuntimeError("Could not build the complete network after "
                           "reaching the maximum number of iterations allowed "
                           "({} edges created)".format(len(edgesset)))
    ffw_graph = nx.DiGraph()
    ffw_graph.add_nodes_from([(n, {"layer": layer_of[n]})
                              for n in range(nodes)])
    ffw_graph.add_edges_from(edgesset)
    return ffw_graph


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file_prefix", "-o", dest="outfile",
                        default=None)
    parser.add_argument("seed_weights", type=int)
    parser.add_argument("seed_KO", type=int)
    parser.add_argument("--seed_input", "-si", type=int, default=None)
    parser.add_argument("--seed_control", "-sc", type=int, default=None)
    parser.add_argument("--test", default="memory_capacity",
                        choices=["memory_capacity",
                                 "critical_memory_capacity"])
    parser.add_argument('--control', choices=['WT', 'FFW', 'ErdosRenyi'],
                        default="WT")
    parser.add_argument("--num_knockouts", "-nko", type=int, required=True)
    parser.add_argument("--spectral_radius", "-sp", type=float, default=0.9)
    parser.set_defaults(FFW=False)
    args = parser.parse_args()
    control = args.control
    prngW = np.random.RandomState(args.seed_weights)
    prngK = np.random.RandomState(args.seed_KO)
    prngI = (prngW if args.seed_input is None
             else np.random.RandomState(args.seed_input))
    prngC = (prngK if args.seed_control is None
             else np.random.RandomState(args.seed_control))
    layers = 7
    maxdistance = 2

    # Load network (and remove selfloops)
    graph = grn_networks.load_network("EcoCyc")
    graph = nt.prune_graph(graph, target="io", depth=0, copy=True,
                           verbose=0, ignore_selfloops=True)
    graph.remove_edges_from(graph.selfloop_edges())

    nnodes = graph.number_of_nodes()
    nedges = graph.number_of_edges()

    # Select index of edges that will be knocked out (total of 317)
    if args.num_knockouts == 1:
        ko_ith_edges = [args.seed_KO % nedges]
        # [ko_ith_edges.append(args.seed_KO // nedges**i)
        #  for i in range(1, args.num_knockouts)]
    else:
        ko_ith_edges = prngK.randint(nedges, size=args.num_knockouts)

    # Generate filename and terminate script if it already exists
    if args.outfile is not None:
        filename = (args.outfile + "_contorl={}_seedW={}_seedKO={}_KO={}"
                    ".csv".format(control, args.seed_weights,
                                  args.seed_KO, ko_ith_edges).replace(" ", ""))

        if os.path.isfile(filename):
            warnings.warn("Output file {} already exists. Terminating script",
                          category=RuntimeWarning)
            sys.exit()

    # if FFW, generate random network
    if control == "FFW":
        graph = rand_FFN(nodes=graph.number_of_nodes(),
                         edges=graph.number_of_edges(),
                         layers=layers, maxdistance=maxdistance, prng=prngC)
        is_input_node = np.array([graph.in_degree(n) == 0
                                  for n in graph.nodes()])
        is_output_node = np.array([graph.out_degree(n) == 0
                                   for n in graph.nodes()])
    elif control == "ErdosRenyi":
        graph = nx.DiGraph(random_topology(nodes=graph.number_of_nodes(),
                                           edges=graph.number_of_edges(),
                                           prng=prngC))

    # Generate weights (and scale spectral radius if appropriate)
    nnodes = graph.number_of_nodes()
    sortednodes = sorted(graph.nodes())
    weights = (nx.adj_matrix(graph, nodelist=sortednodes).toarray()
               * (prngW.rand(nnodes, nnodes)*2 - 1))

    w_input = (prngW.rand(1, nnodes)*2-1) * 0.1

    if control == "FFW":
        # Set w_input to 0 for nodes with in-degree != 0
        w_input *= is_input_node
    else:
        weights *= args.spectral_radius / get_spectral_radius(weights)

    # generate dataset
    sample_len = 5000
    nsamples = 10
    input_signal = [(prngI.rand(sample_len, 1)*2-1) for i in xrange(nsamples)]

    # Run WT dynamics (for training)
    res_dynamics = [simulate_reservoir_dynamics(weights, w_input, i_sig)
                    for i_sig in input_signal]

    # remove links
    sortededges = sorted(graph.edges())
    ko_weights = weights.copy()
    for ith_e in ko_ith_edges:
        i, j = map(sortednodes.index, sortededges[ith_e])
        ko_weights[i, j] = 0

    # run KO dynamics (only testing phase)
    res_dynamics[-1] = simulate_reservoir_dynamics(ko_weights, w_input,
                                                   input_signal[-1])
    if control == "FFW":
        # Hide dynamics for nodes with out-degree != 0
        res_dynamics *= is_output_node

    # Train and test MC task
    if args.test == "memory_capacity":
        ko_mc = memory_capacity(input_signal, res_dynamics)
    elif args.test == "critical_memory_capacity":
        ko_mc = critical_memory_capacity(input_signal, res_dynamics)

    # Save results to output file
    out = open(filename, "w") if args.outfile is not None else sys.stdout
    print (args.seed_weights, args.seed_KO, args.seed_input, args.seed_control,
           control, args.spectral_radius, ko_mc, args.test,
           *ko_ith_edges, sep=",", end="\n", file=out)
    out.flush()
