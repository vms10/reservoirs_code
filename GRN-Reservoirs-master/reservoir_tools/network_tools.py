# -*- coding: utf-8 -*-
"""Util functions to work with networks (as graphs or adjacency matrices)."""
from __future__ import division, absolute_import, print_function

from collections import OrderedDict
import itertools as it

import networkx as nx


def prune_graph(graph, target="io", depth=0, to_keep=[], copy=True, verbose=0,
                ignore_selfloops=True):
    """Removes non-recurrent parts of a graph deleting lonely nodes, nodes with
    no input and/or nodes with no output.
    Self loops are not considered as connections.

    PARAMETERS:

    graph       a networkx directed graph (DiGraph) that is going to be pruned.

    target      a string specifying the type of nodes to be removed. If the
                string contains a given letter that kind of node will be
                deleted. Multiple letters imply additive criteria. Not
                matching letters will be ignored.
                    "i" - nodes with no input
                    "o" - nodes with no output
                    "a" - final search against nodes with neither input or
                            output, i. e. isolated nodes

    depth       Searches against no input nodes and no output nodes can
                be applied an specified number of iterations. If a value of 0
                is given unlimited iterations will be applied until no new
                nodes are removed.
                If specified, final search against isolated nodes will be done
                just once, at the end of the iterations

    to_keep     List of nodes that will be kept regardless they fulfill or not
                the criteria specified in the target argument

    verbose     controls the level of printed output

    ingnore_selfloops
                If set to True, the self edges are ignored when checking
                the in- and out- degree of a node.
    """

    to_keep = set(to_keep)
    if to_keep.difference(graph.nodes()):
        raise RuntimeError

    graph = graph.copy() if copy else graph

    root = "i" in target
    leaf = "o" in target
    lonely = "a" in target
    selfloops = (set(graph.nodes_with_selfloops()) if ignore_selfloops
                 else set())

    rounds = 0
    while rounds < depth or depth == 0:
        rounds += 1
        remove, nremoved = [], []
        for strategy, degree_iter in [(root, graph.in_degree_iter),
                                      (leaf, graph.out_degree_iter)]:
            if strategy:
                remove += [n for n, d in degree_iter()
                           if n not in to_keep
                           if d == 0 or
                           (ignore_selfloops and d == 1 and n in selfloops)]
            nremoved.append(len(remove))

        if not remove:
            break

        graph.remove_nodes_from(remove)
        if verbose > 0:
            print("Deleted nodes: {} ({} roots, {} leaves)"
                  .format(nremoved[-1], nremoved[0], nremoved[-1]-nremoved[0]))

    if lonely:
        remove = [n for n, d in graph.degree_iter() if d == 0]
        graph.remove_nodes_from(remove)
        if verbose > 0:
            print("Deleted nodes: {} (isolated)".format(len(remove)))

    return graph


def bow_tie_structure(graph, network_name=None):
    """
    Classify the nodes of `graph` in the parts of a bow tie network scheme.

    Classify the nodes of `graph` using a version of the bow tie network
    structure: Input, Reservoir, Readout, in-Tendrils, out-Tendrils and Tubes.
    Note that this is a slight variation of the original bow tie structure,
    as the SCC is reinterpreted as the Reservoir (which is not necessarily
    equivalent). Additionally, In and Out categories are renamed as Input and
    Readout.
    Returns an `OrderedDict` with the name of each category as key and a set
    with the node names as value.

    WARNING:
        It does not check if the reservoir found has one or more components!"""
    graph = graph.copy()
    graph.remove_edges_from(graph.selfloop_edges())
    # TODO: select main component??
    # RESERVOIR nodes
    nreservoir = set(prune_graph(graph, target="io", depth=0))
    nothers = set(graph.nodes()).difference(nreservoir)
    # input nodes
    ninput = set((n for n in nothers
                  if any((nx.has_path(graph, n, rn) for rn in nreservoir))))
    # readout nodes
    noutput = set((n for n in nothers
                   if any((nx.has_path(graph, rn, n) for rn in nreservoir))))
    # Terminal reservoir nodes
    out_terminal = set((nod for nod in noutput if graph.out_degree(nod) == 0))

    # in-tendrils, out-tendrils and tube nodes
    nothers.difference_update(noutput, ninput)
    in_tendrils = set((n for n in nothers
                       if any((nx.has_path(graph, rn, n) for rn in ninput))))
    out_tendrils = set((n for n in nothers
                        if any((nx.has_path(graph, n, rn) for rn in noutput))))
    tubes = in_tendrils.intersection(out_tendrils)
    in_tendrils.difference_update(tubes)
    out_tendrils.difference_update(tubes)
    nothers.difference_update(in_tendrils, out_tendrils, tubes)

    ngroups = [] if network_name is None else [("Network", network_name)]
    ngroups += [("Input", ninput), ("Reservoir", nreservoir),
                ("Readout", noutput), ("Terminal", out_terminal),
                ("Other", nothers), ("in-Tendrils", in_tendrils),
                ("out-Tendrils", out_tendrils), ("Tubes", tubes)]

    return OrderedDict(ngroups)


def remove_small_attractors(graph, min_size=3):
    """
    Remove all attractors smaller than a given size.

    Return a copy of `graph` with all nodes that were part of attractors
    (scc without outflux) smaller than `min_size` nodes removed.
    """
    graph = graph.copy()
    attractors = (comp for comp in nx.attracting_components(graph)
                  if len(comp) < min_size)
    graph.remove_nodes_from(it.chain.from_iterable(attractors))
    return graph


def condense_small_attractors(graph, min_size=3):
    """
    Condense all attractors smaller than a given size.

    Return a copy of `graph` with all attractors (scc without outflux) smaller
    than `min_size` nodes condensed.
    """
    attractors = [comp for comp in nx.attracting_components(graph)
                  if len(comp) < min_size]
    to_condense = set(it.chain.from_iterable(attractors))
    attractors += [{n} for n in graph.nodes() if n not in to_condense]
    return (nx.condensation(graph, scc=attractors)
            if len(attractors) > 0 else graph.copy())
