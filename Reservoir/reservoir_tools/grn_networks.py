# -*- coding: utf-8 -*-
"""Paths and functions to load the Gene Regulatory Networks."""
from __future__ import division, absolute_import, print_function

from reservoir_tools import utils

from collections import OrderedDict
import os

import time
import numpy as np
import networkx as nx

_RELATIVE_GRN_PATHS = OrderedDict(
    (("DBTBS", "../networks/network_edge_list_DBTBS.csv"),
     ("EcoCyc", "../networks/network_edge_list_EcoCyc.csv"),
     ("YEASTRACT", "../networks/network_edge_list_YEASTRACT.csv"),
     ("modENCODE", "../networks/network_edge_list_modENCODE.csv"),
     ("ENCODE", "../networks/network_edge_list_ENCODE.csv"),
     ("Vohradsky", "../networks/network_edge_list_Vohradsky.csv"),
     ("Vohradsky4", "../networks/network_edge_list_Vohradsky4.csv"),
      ("Test", "../networks/network_edge_list_Test.csv"),
      ("Test15", "../networks/network_edge_list_Test15.csv"),
      ("Test25", "../networks/network_edge_list_Test25.csv")))

grn_names = _RELATIVE_GRN_PATHS.keys()

def generate_rand_network(network,n_genes = 3):
    
    #wm = np.empty((n_genes,n_genes))

    #for line in range(n_genes):
    #    for gene in range(n_genes):
    #        wm[line][gene] = np.random.randint(-1,2)
    
    #np.random.seed(None)
    wm = np.zeros((n_genes,n_genes))
    #wm = np.random.randint(-1,2,size=(n_genes,n_genes))
    
    # Constant number of connections
    #n_conn = 2

    #Random number of connections
    #n_conn = np.random.randint(0.998*n_genes*n_genes,0.999*n_genes*n_genes)

    # 2 connections per gene
    n_conn = 2*n_genes

    while(utils.get_spectral_radius(wm) == 0):
        wm = np.zeros((n_genes,n_genes))
        count = 0
        while count < n_conn:
            x = np.random.randint(-1,n_genes-1)
            y = np.random.randint(-1,n_genes-1)
            if wm[x][y] == 0:
                wm[x][y] = np.random.choice([-1,1])
                #wm[x][y] = 1
                count += 1

    print(wm)



    #if SR_scale and utils.get_spectral_radius(adj_matrix) == 0:
        #print("Spectral radius --> 0")
    #    noNilpotent_trials += 1
    #    if noNilpotent_trials > 10000:
     #       noNilpotent_matrix = False

    path = os.path.join(os.path.dirname(__file__),
                    _RELATIVE_GRN_PATHS[network])

    f = open(path,"w")

    for line in range(n_genes):
        for gene in range(n_genes):
            if wm[line][gene] != 0:
                if wm[line][gene] == 1:
                    f.write("g"+str(line)+"\t"+"g"+str(gene)+"\t"+"+"+"\n")
                elif wm[line][gene] == -1:
                    f.write("g"+str(line)+"\t"+"g"+str(gene)+"\t"+"-"+"\n")


    f.close()


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

    return nx.OrderedDiGraph(edge_list)
