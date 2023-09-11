# -*- coding: utf-8 -*-
"""Various generators of control graphs."""
from __future__ import division, absolute_import, print_function

import itertools as it
import random
import sys

import numpy as np
import networkx as nx

# TODO: randomize options as normal controls


def SCR_control(size, **kwargs):
    """Simple Circular Reservoir control (directed cycle) with the given number
    of nodes
    """
    g = nx.cycle_graph(size, create_using=nx.DiGraph())
    return np.asarray(nx.adj_matrix(g).todense())


def fESN_control(size, conn_fraction, **kwargs):
    """Erdos-Renyi random network with the given number of nodes and the given
    proportion of existing edges with respect of the total number of possible
    edges
    """
    # num_edges = conn_fraction * total_possible_edges
    num_edges = int(conn_fraction * size**2 + 0.5)  # Rounding
    return random_topology(nodes=size, edges=num_edges)


def kESN_control(size, avrg_degree, **kwargs):
    """Erdos-Renyi random network with the given number of nodes and the given
    mean degree
    """
    num_edges = int((size * avrg_degree)/2. + 0.5)  # Rounding
    return random_topology(nodes=size, edges=num_edges)


def FFW_control_remove(graph, **kwargs):
    """Transformation of the given graph to remove all recursivities so that
    it becomes a pure feedforward graph. It is achieved by removing edges from
    the cycles
    """
    g = feedforwardize_network(graph.copy(), mode="remove",
                               remove_selfloops=True,
                               remove_bidirectionals=False, max_steps=200)
    return np.asarray(nx.adj_matrix(g).todense())


def FFW_control_swap(graph, **kwargs):
    """Transformation of the given graph to remove all recursivities so that
    it becomes a pure feedforward graph. It is achieved by changing the
    direction of edges in the cycles
    """
    g = feedforwardize_network(graph.copy(), mode="swap",
                               remove_selfloops=True,
                               remove_bidirectionals=True, max_steps=200)
    return np.asarray(nx.adj_matrix(g).todense())


def FFW_control_move(graph, **kwargs):
    """Transformation of the given graph to remove all recursivities so that
    it becomes a pure feedforward graph. It is achieved by moving edges from
    the cycles somewhere else in the graph.
    """
    g = feedforwardize_network(graph.copy(), mode="move",
                               remove_selfloops=True,
                               remove_bidirectionals=False, max_steps=200)
    return np.asarray(nx.adj_matrix(g).todense())


def edges_num_control(graph, expected_edges, **kwargs):
    if isinstance(graph, nx.Graph):
        graph = nx.adjacency_matrix(graph).todense()
    adj_matrix = np.asarray(graph)

    edges_offset = expected_edges - np.abs(adj_matrix).sum()
    if edges_offset < 0:
        sources, targets = np.nonzero(adj_matrix)
        rand_selection = np.random.choice(sources.size,
                                          size=np.abs(edges_offset),
                                          replace=False)
        adj_matrix[[sources[rand_selection], targets[rand_selection]]] = 0
    elif edges_offset > 0:
        sources, targets = np.where(adj_matrix == 0)
        rand_selection = np.random.choice(sources.size,
                                          size=edges_offset,
                                          replace=False)
        adj_matrix[[sources[rand_selection], targets[rand_selection]]] = \
            np.random.randint(0, 2, size=edges_offset)*2 - 1
    return adj_matrix


def _lazy_loader(func, *args, **kwargs):
    def lazy_loader(**late_kwargs):
        # Only the kwargs that where specified et the _lazy_loader declaration
        # can be modified when executing it. Any other kwarg is ignored.
        call_kwargs = {key: (late_kwargs[key] if key in late_kwargs else value)
                       for key, value in kwargs.iteritems()}
        return func(*args, **call_kwargs)
    return lazy_loader


# Generic name, network generator, specific name generator
control_graphs = [["SCR", SCR_control,
                   _lazy_loader(str.format, "SCR-s{size}", size=None)],
                  ["fESN", fESN_control,
                   _lazy_loader(str.format, "ESN-s{size}-f{conn_fraction:.3f}",
                                size=None, conn_fraction=None)],
                  ["kESN", kESN_control,
                   _lazy_loader(str.format, "ESN-s{size}-k{avrg_degree:.3f}",
                                size=None, avrg_degree=None)],
                  ['FFW_move', FFW_control_move,
                   _lazy_loader(str.format, "FFWmove-ref={reference}",
                                reference=None)],
                  ['FFW_swap', FFW_control_swap,
                   _lazy_loader(str.format, "FFWswap-ref={reference}",
                                reference=None)],
                  ['FFW_remove', FFW_control_remove,
                   _lazy_loader(str.format, "FFWremove-ref={reference}",
                                reference=None)],
                  ['edges', edges_num_control,
                   _lazy_loader(str.format, "EdgeControl={expected_edges}",
                                expected_edges=None)]]


def random_topology(nodes, edges, prng=None):
    # It would be easier to use a binomial random, but for small networks
    # deviations are not irrelevant
    prng = prng or np.random
    matrix = np.zeros(nodes**2)
    matrix[:edges] = 1
    prng.shuffle(matrix)
    return matrix.reshape((nodes, nodes))


def randomize_network(graph, degree_conservative=False):
    nnodes = graph.number_of_nodes()
    nedges = graph.number_of_edges()
    nselfloops = graph.number_of_selfloops()

    if degree_conservative:
        steps = 5 * graph.number_of_edges()
        random_graph = _randomize_degree_conservative(graph, steps)
    else:
        random_graph = _randomize_size_conservative(graph)

    if isinstance(random_graph, nx.Graph):
        assert nnodes == random_graph.number_of_nodes()
        assert nedges == random_graph.number_of_edges()
        assert nselfloops == random_graph.number_of_selfloops()
    else:
        assert nnodes == random_graph.shape[0]
        assert nedges == random_graph.nonzero()[0].size
        # assert nselfloops == random_graph.number_of_selfloops()
    return random_graph


def _randomize_size_conservative(graph):
    nnodes = graph.number_of_nodes()
    nselfloops = graph.number_of_selfloops()
    nedges = graph.number_of_edges() - nselfloops
    new_graph = nx.gnm_random_graph(nnodes, nedges, directed=True)
    [new_graph.add_edge(nod, nod)
     for nod in random.sample(new_graph.nodes(), nselfloops)]
    return new_graph


def _randomize_degree_conservative(graph, steps):
    new_graph = graph.copy()
    step_count = 0
    edge_sampling_size = 2
    nedges = new_graph.number_of_edges()
    while step_count < steps:
        step_advanced = False
        for (source1, target1), (source2, target2) in (
                it.combinations(
                    random.sample(new_graph.edges(), edge_sampling_size), 2)):
            # Abort if the change implies an self-loop generation/removal
            # or any of the two new edges already exists.
            if (source1 in (target1, target2) or
                    source2 in (target1, target2) or
                    new_graph.has_edge(source1, target2) or
                    new_graph.has_edge(source2, target1)):
                continue
            else:
                # If it's not a trivial change, swap edges
                if source1 != source2 and target1 != target2:
                    new_graph.remove_edges_from([(source1, target1),
                                                 (source2, target2)])
                    new_graph.add_edges_from([(source1, target2),
                                              (source2, target1)])
                step_advanced = True
                break
        if step_advanced:
            step_count += 1
        elif edge_sampling_size < nedges:
            edge_sampling_size += 1

    return new_graph


def _remove(graph, node1, node2):
    """Remove the edge between the two nodes
    """
    graph.remove_edge(node1, node2)


def _swap(graph, node1, node2):
    """Change the direction of the link between the two nodes.
    If it already exists raises a ValueError exception
    """
    if node1 != node2 and not graph.has_edge(node2, node1):
        graph.remove_edge(node1, node2)
        graph.add_edge(node2, node1)
    else:
        raise ValueError


def _move(graph, node1, node2):
    """Remove the edge between the two nodes and create
    a new edge between two random nodes that are not already connected
    (in that direction)
    """
    for i in range(30):
        n1, n2 = random.sample(graph.nodes(), 2)
        if not graph.has_edge(n1, n2):
            graph.add_edge(n1, n2)
            graph.remove_edge(node1, node2)
            break


def _crosslink(graph, node1, node2):
    """Remove the edge between the two nodes. Look for another
    existing edge and remove it. Create a link from the first source to
    the second target and another from the second source to the first
    target.

    WARNING: not usable, the number of cycles explotes easily
    WARNING: if any of the two edges it has to create already exsist it
             does nothing
    """
    donor1, donor2 = random.sample(graph.edges(), 1)[0]
    if (not graph.has_edge(donor1, node2) and
            not graph.has_edge(node1, donor2)):
        graph.remove_edge(node1, node2)
        graph.remove_edge(donor1, donor2)
        graph.add_edge(node1, donor2)
        graph.add_edge(donor1, node2)


def feedforwardize_network(graph, mode, remove_selfloops=True,
                           remove_bidirectionals=False, max_steps=200,
                           verbose=False):
    funcs = {"remove": _remove,
             "swap": _swap,
             "move": _move,
             "crosslink": _crosslink}
    func = funcs[mode]
    # 1) remove self-loops
    num_selfloops = 0
    if remove_selfloops:
        num_selfloops = len(graph.selfloop_edges())
        graph.remove_edges_from(graph.selfloop_edges())
    # 2) bidirectional links to unidirectional ones (randomly)
    num_bidirectionals = 0
    if remove_bidirectionals:
        for n1, n2 in graph.edges():
            if graph.has_edge(n1, n2) and graph.has_edge(n2, n1):
                num_bidirectionals += 1
                if random.random() < 0.5:
                    graph.remove_edge(n1, n2)
                else:
                    graph.remove_edge(n2, n1)
    # 3) Cycles
    steps = 0
    while steps < max_steps:
        if verbose > 1:
            print (steps, graph.number_of_edges(),
                   len(list(nx.simple_cycles(graph))))
            sys.stdout.flush()
        steps += 1
        list_cycles = list(nx.simple_cycles(graph))
        if len(list_cycles) == 0:
            break
        cycle = random.sample(list_cycles, 1)[0]
        i = random.randint(0, len(cycle)-1)
        func(graph, cycle[i-1], cycle[i])
    # BONUS) Final report?
    if verbose:
        msg = ("STEPS: {}\tSELFLOOPS REMOVED: {}\tBIDIRECTIONAL REMOVED:{}"
               "\tFINAL EDGES: {}\tFINAL CYCLES: {}")
        print(msg.format(steps, num_selfloops, num_bidirectionals,
                         graph.number_of_edges(),
                         len(list(nx.simple_cycles(graph)))))
        sys.stdout.flush()
    return graph
