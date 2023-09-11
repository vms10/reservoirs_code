# -*- coding: utf-8 -*-
"""Network statistics."""
from __future__ import division

from collections import Counter

import numpy as np
import networkx as nx


def average_degree(system):
    """Return the average degree of a graph or adjacency matrix."""
    # avrg_degree = 2 * num_edges / num_nodes
    if isinstance(system, nx.Graph):
        return system.size() * 2. / system.order()
    elif isinstance(system, np.ndarray) and system.shape[0] == system.shape[1]:
        return system.nonzero()[0].size * 2. / system.shape[0]
    else:
        raise ValueError("system is not a networkX graph nor a squared "
                         "numpy.array")


def connectivity_fraction(system):
    """Return the fraction of existing edges over the possible ones."""
    # conn_fraction = num_edges / total_possible_edges
    #               = num_edges / num_nodes^2
    if isinstance(system, nx.Graph):
        return 1. * system.size() / (system.order()**2)
    elif isinstance(system, np.ndarray) and system.shape[0] == system.shape[1]:
        return 1. * system.nonzero()[0].size / np.prod(system.shape)
    else:
        raise ValueError("system is not a networkX graph nor a squared "
                         "numpy.array")


def pf_eigenval(adj_matrix):
    """Compute the Perron-Frobenius eigenvalue."""
    if not isinstance(adj_matrix, np.array):
        adj_matrix = nx.adj_matrix(adj_matrix, weight=None).todense()
    return max(filter(np.isreal, np.linalg.eigvals(adj_matrix)))


def path_entropy(graph, backward=False):
    u"""Compute the average path entropy of a graph.

    There is a more efficient algorithm defined in the paper. It is only
    partially implemented so far. This is the naive aproach.

    Corominas-Murtra, B., Goñi, J., Solé, R. V, & Rodríguez-Caso, C. (2013).
    "On the origins of hierarchy in complex networks". PNAS, 110(33), 13316–21.
    """
    gc = nx.condensation(graph)
    L = nx.dag_longest_path_length(gc)
    B = nx.adj_matrix(gc)
    return _path_entropy(gc, B, L, backward=backward)


def _path_entropy(graph, B, L, backward=False):
    u"""Compute the average path entropy of a graph.

    CURRENTLY NOT WORKING!

    This is an auxiliary function for called by `treeness` function.
    An public wrapper function could be created easily. It should condensate
    the SCC in `graph`, search the longest path `L`, and generate the
    unweighted adjacency matrix `B`.

    Corominas-Murtra, B., Goñi, J., Solé, R. V, & Rodríguez-Caso, C. (2013).
    "On the origins of hierarchy in complex networks". PNAS, 110(33), 13316–21.
    http://doi.org/10.1073/pnas.1300832110
    """
    if backward:
        terminals_degree_iter = graph.in_degree_iter
        sources = [n for n, d in graph.out_degree_iter() if d == 0]
        Bsum = B.sum(axis=0).astype(float)
        B = np.where(Bsum != 0, B/Bsum, 0).T  # To avoid dividing by 0
    else:
        terminals_degree_iter = graph.out_degree_iter
        sources = [n for n, d in graph.in_degree_iter() if d == 0]
        Bsum = B.sum(axis=1).astype(float)
        B = np.where(Bsum != 0, B/Bsum, 0)  # To avoid dividing by 0

    # Bs = [B**k for k in range(1, L+1)]
    Bs = sum([np.linalg.matrix_power(B, k) for k in range(1, L+1)])
    # Bs = (B[:, :, None]**np.arange(1, L+1)).sum(axis=2)

    res = 0
    for i in sources:
        for j, d in terminals_degree_iter():
            if d == 0:
                continue
            # res += np.log(d) * sum([b[i, j] for b in Bs])
            res += np.log(d) * Bs[i, j]
            # print res, np.log(d), d,  [b[i, j] for b in Bs]
    return res/len(sources)


def path_entropy(graph, backward=False, condensed=False):
    u"""Compute the average path entropy of a graph.

    There is a more efficient algorithm defined in the paper. It is only
    partially implemented so far. This is the naive aproach.

    Corominas-Murtra, B., Goñi, J., Solé, R. V, & Rodríguez-Caso, C. (2013).
    "On the origins of hierarchy in complex networks". PNAS, 110(33), 13316–21.
    http://doi.org/10.1073/pnas.1300832110
    """
    gc = graph if condensed else nx.condensation(graph)
    maximals = [n for n, d in gc.in_degree_iter() if d == 0]
    minimals = [n for n, d in gc.out_degree_iter() if d == 0]
    degrees_func = gc.in_degree if backward else gc.out_degree

    entropy = 0
    for source in maximals:
        for target in minimals:
            for path in nx.all_simple_paths(gc, source, target):
                d = 1
                degs = [degrees_func(n) for n in path]
                degs = degs if backward else degs[::-1]
                _degs = 1/np.cumprod(degs[1:], dtype=float)
                entropy += (_degs*np.log(_degs)).sum()
    return -1 * entropy


def treeness(graph, condensed=False):
    u"""Compute the treeness of a graph.

    Treeness (T, where −1 ≤ T ≤ 1) weights how pyramidal is the structure and
    how unambiguous is its chain of command. This measure covers the range from
    hierarchical (T > 0) to antihierarchical (T < 0) graphs, including those
    structures that do not exhibit any pyramidal behavior (T = 0).

    Corominas-Murtra, B., Goñi, J., Solé, R. V, & Rodríguez-Caso, C. (2013).
    "On the origins of hierarchy in complex networks". PNAS, 110(33), 13316–21.
    http://doi.org/10.1073/pnas.1300832110
    """
    def f(G):
        res = 0
        Gorder = G.order()
        for wcc in nx.weakly_connected_component_subgraphs(G):
            if wcc.size() == 0:
                continue
            # Hf = _path_entropy(wcc, B, L, backward=False)
            Hf = path_entropy(wcc, backward=False)
            # Hb = _path_entropy(wcc, B, L, backward=True)
            Hb = path_entropy(wcc, backward=True)
            if Hf != 0 or Hb != 0:
                res += float(Hf - Hb)/max(Hf, Hb) * wcc.order() / Gorder
        return res

    gc = graph if condensed else nx.condensation(graph)
    L = nx.dag_longest_path_length(gc)
    # B = nx.adj_matrix(gc)

    treeness = f(gc)
    gkr, gkl = gc.copy(), gc.copy()
    for i in range(L):
        gkr.remove_nodes_from([n for n, d in gkr.in_degree_iter() if d == 0])
        gkl.remove_nodes_from([n for n, d in gkl.out_degree_iter() if d == 0])
        treeness += f(gkr) + f(gkl)

    return float(treeness)/(2*L-1)


def feedforwardness(graph, condensed=False):
    u"""Compute the feedforwardness of a graph.

    Feedforwardness captures the the impact of the non-orderable regions of the
    graph over the potential causal paths described by it. In raw words,
    where, within the causal flow, we find the non-orderable regions.

    Corominas-Murtra, B., Goñi, J., Solé, R. V, & Rodríguez-Caso, C. (2013).
    "On the origins of hierarchy in complex networks". PNAS, 110(33), 13316–21.
    http://doi.org/10.1073/pnas.1300832110
    """
    gc = graph if condensed else nx.condensation(graph)
    nweight = Counter(gc.graph['mapping'].itervalues())
    maximal = [n for n, d in gc.in_degree_iter() if d == 0]

    F = [float(len(path)) / reduce(lambda x, y: x+nweight[y], path, 0)
         for source in maximal
         for target in gc.nodes()
         if source != target
         for path in nx.all_simple_paths(gc, source, target)]

    return sum(F) / float(len(F)) if F else 0


def orderability(graph):
    u"""Compute the orderability of a graph.

    The Orderability, O, of the graph G is the fraction of nodes not belonging
    to any cycle -which are, by definition, non-orderable structures.

    Corominas-Murtra, B., Goñi, J., Solé, R. V, & Rodríguez-Caso, C. (2013).
    "On the origins of hierarchy in complex networks". PNAS, 110(33), 13316–21.
    http://doi.org/10.1073/pnas.1300832110
    """
    ffw_nodes = sum([1 for scc in nx.strongly_connected_components(graph)
                     if len(scc) == 1])
    return float(ffw_nodes)/graph.order()


def edges_in_cycles(graph, min_nodes_per_cycle=2):
    """Return the number of edges in cycles and the number of cycles."""
    # graph = graph.copy()
    graph.remove_edges_from(graph.selfloop_edges())
    cycles = nx.simple_cycles(graph)
    num_cycles = 0
    cyc_edges = set()
    add_edge = cyc_edges.add
    for cycle in cycles:
        if len(cycle) < min_nodes_per_cycle:
            continue
        num_cycles += 1
        for i in xrange(len(cycle)):
            add_edge((cycle[i-1], cycle[i]))
    return cyc_edges, num_cycles
