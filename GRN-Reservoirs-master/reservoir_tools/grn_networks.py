# -*- coding: utf-8 -*-
"""Paths and functions to load the Gene Regulatory Networks."""
from __future__ import division, absolute_import, print_function

from collections import OrderedDict
import os

import networkx as nx

_RELATIVE_GRN_PATHS = OrderedDict(
    (("DBTBS", "../networks/network_edge_list_DBTBS.csv"),
     ("EcoCyc", "../networks/network_edge_list_EcoCyc.csv"),
     ("YEASTRACT", "../networks/network_edge_list_YEASTRACT.csv"),
     ("modENCODE", "../networks/network_edge_list_modENCODE.csv"),
     ("ENCODE", "../networks/network_edge_list_ENCODE.csv")))

grn_names = _RELATIVE_GRN_PATHS.keys()


def load_network(network):
    if network in _RELATIVE_GRN_PATHS:
        path = os.path.join(os.path.dirname(__file__),
                            _RELATIVE_GRN_PATHS[network])
    elif os.path.isfile(network):
        path = network
    else:
        raise ValueError("'{}' is not a known network, nor a path to a valid "
                         "file.".format(network))
    edge_list = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or line in ["Source\tTarget\tMode", ""]:
                continue
            edge = line.split("\t")
            source, target = edge[:2]
            if len(edge) == 2:
                attr = {}
            elif len(edge) == 3:
                attr = {"mode": edge[-1]}
            else:
                raise ValueError("Each row should have 2 or 3 columns. '{}' "
                                 "seems wrong.".format(line))
            edge_list.append([source, target, attr])
    return nx.DiGraph(edge_list)
