#!usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from reservoir_tools import utils
from reservoir_tools.control_graphs import control_graphs, randomize_network
from reservoir_tools import datasets_1 as datasets
from reservoir_tools import grn_networks
from reservoir_tools import network_tools as nt
from reservoir_tools.network_statistics import average_degree, connectivity_fraction
from reservoir_tools.readouts import RidgeRegression
from reservoir_tools.reservoirs import simulate_reservoir_dynamics
import matplotlib.pyplot as plt
import random

plt.rcParams.update({"font.size":32})
plt.rc('legend',fontsize=24)
plt.rc('xtick',labelsize=28)
plt.rc('ytick',labelsize=28)
plt.rc('lines',linewidth=5)
plt.rc('figure',figsize=(12,9))

import Tkinter

def sigmoid(x):
    #return ((x/(np.sqrt((x**2)+1)))+1)/2
    return (((x-0.5)/(np.sqrt(((x-0.5)**2)+0.1)))+1)/2
    #return (1/(1+np.exp(-x)))

def training(data_source,train_freq,train_amp,train_pha,in_weight,weights,train="ALL"):

        # Generate dataset
        print("TRAINING:")
        print(train)
        sample_len=1000
        [x, y] = data_source.func(n_samples=1000,sample_len=sample_len,freq=train_freq,amp=train_amp,phase=train_pha, train = train, noise=0)
        phase_list = np.loadtxt("phase_list.csv")
        freq_list = np.loadtxt("freq_list.csv")
        rand_list = np.loadtxt("rand_list.csv")

        res_dynamics = [simulate_reservoir_dynamics(weights, in_weight,
                                                    phase_list[count], freq_list[count], rand_list[count],sample_len, node_function=sigmoid)
                        for count, i_sig in enumerate(x)]
        
         

        with_bias = False ################## NO BIAS #################
        rregr = RidgeRegression(use_bias=with_bias)
        
        [rregr.train(x_train, y_train) for x_train, y_train in zip(res_dynamics, y)]

        rregr.finish_training()

        return(rregr)
        



def reservoir_performance(data_source, adj_matrix, input_weight=None,
                          spectral_radius_scale=0.9, with_bias=True):

    if hasattr(adj_matrix, "todense"):
        adj_matrix = adj_matrix.todense()
    adj_matrix = np.asarray(adj_matrix)

    num_train_sets = 1
    
    
    num_sets = 1
    
        

    #freq_list = [(np.random.randint(16,24))*0.001 for _ in range(2000)]
    #freq_list = [random.uniform(0.018, 0.024) for _ in range(1000)]
    #phase_list=[random.uniform(0, np.pi*100) for _ in range(1000)]

    train_amp = np.linspace(5,5,num=num_train_sets)
    phase_list=[random.uniform(0, np.pi*100) for _ in range(1000)]
    freq_list=[0.02 for _ in range(1000)]

    np.savetxt("phase_list.csv", phase_list, delimiter =", ", fmt ='% s')
    np.savetxt("freq_list.csv", freq_list, delimiter =", ", fmt ='% s')
    
    twod_plot = np.zeros((num_train_sets,num_sets))
    twod_plot_B = np.zeros((num_train_sets,num_sets))
    twod_plot_D = np.zeros((num_train_sets,num_sets))

    for train_set in range(0,len(train_amp)):


        weights = adj_matrix * (np.random.random(adj_matrix.shape)*2-1) ### RANDOM wei


        if spectral_radius_scale:
            spectral_radius = utils.get_spectral_radius(weights)
            if spectral_radius == 0:
                raise RuntimeError("Nilpotent adjacency matrix matrix")
            weights = weights *(spectral_radius_scale / spectral_radius)
        np.savetxt('weights.txt', weights)
        #weights=np.loadtxt('weights.txt')


        in_scaling = 0.05
        in_weight = input_weight * in_scaling
        np.savetxt('in_weight.txt', in_weight)	
        #in_weight =np.loadtxt('in_weight.txt')
        
        scal=0
        if scal == 0:                
            rregr =  training(data_source,freq_list,train_amp[train_set],phase_list,in_weight, weights,"BD")
        else:
            rregr_D =  training(data_source,freq_list,train_amp[train_set],phase_list,in_weight, weights,"D")
            rregr_B =  training(data_source,freq_list,train_amp[train_set],phase_list,in_weight, weights,"B")
            
            rregr = RidgeRegression(use_bias=False)
            
            rregr.beta = np.hstack((rregr_B.beta,rregr_D.beta))
            
            
        np.savetxt('beta.txt', rregr.beta)
        #rregr.beta =np.loadtxt('beta.txt')
        
        osc_ev_res = []
        init_res = []
        amp = np.linspace(5,5,num=num_sets)
        noise = np.linspace(0,0,num_sets)
        results_2 = []
        results_3 = []
        results_4 = []
        print('EMPIEZA PRUEBAAAAAAA')


        for osc_ev in range(0,len(amp)):

            results = []
            osc_ev_res_B = []
            osc_ev_res_D = []
            sample_len=1000
            n_samples=1000
            

            [x, y] = data_source.func(n_samples=n_samples,sample_len=sample_len,freq=freq_list,amp=amp[osc_ev],phase=phase_list, noise=noise[osc_ev])
            rand_list = np.loadtxt("rand_list.csv")
            res_dynamics = [simulate_reservoir_dynamics(weights, in_weight, phase_list[count], freq_list[count], rand_list[count], sample_len, node_function=sigmoid) for count, i_sig in enumerate(x)]

            
            for count, i in enumerate(range(0,n_samples)):

                pred = rregr(res_dynamics[i])
            

                n_out = 2  ### FOURTH GENE
                error = 0
                nrmse = 0
                
                for outs in range(n_out):
                    steady = np.mean(pred[outs][(sample_len-200):])
                    steady_target = np.mean(y[i][(sample_len-200):,outs])
                    error += (np.absolute(steady_target - steady))


                results.append(error)
                results_2.append(steady)
                results_3.append(pred)
                results_4.append(x[i][:])
                
                if y[i][(sample_len-200),1] > 0:
                    osc_ev_res_B.append(np.mean(pred[0][(sample_len-200):]))
                    osc_ev_res_D.append(np.mean(pred[1][(sample_len-200):]))
                    try:
                        init_res.append((np.where((x[i]<1.05) & (x[i]>0.95))[0][0],np.mean(pred[0][(sample_len-200):]),np.mean(pred[1][(sample_len-200):])))
                    except:
                        pass


            osc_ev_res.append([np.mean(osc_ev_res_B),np.mean(osc_ev_res_D)])
            

            twod_plot[train_set][osc_ev] = (np.mean(osc_ev_res_D)/np.mean(osc_ev_res_B))
            twod_plot_B[train_set][osc_ev] = np.mean(osc_ev_res_B)
            twod_plot_D[train_set][osc_ev] = np.mean(osc_ev_res_D)
            
            print("THESE ARE THE VALUES")
            print("VALUE "+str(amp[osc_ev]))
            print("Total: " + str(twod_plot[train_set][osc_ev]))
            print("B: " + str(np.mean(osc_ev_res_B)))
            print("D: " + str(np.mean(osc_ev_res_D)))
        
        print("OCILLATORY EVALUATION RESULTS")
        print(osc_ev_res)

        osc_ev_res = np.array(osc_ev_res)



        init_res = np.array(init_res)
    



    if np.isnan(nrmse) or np.isinf(nrmse):
        raise RuntimeError("The NRMSE value obtained is not finite.")
        
    return((results), (osc_ev_res), (freq_list), (results_3), (results_4))



class _data_source():
    def __init__(self, name, func):
        self.name = name
        self.func = func


data_sources = [_data_source("10th order NARMA", datasets.narma10),
                _data_source("30th order NARMA", datasets.narma30),
                _data_source("Gene activation 3", datasets.ga3),
                _data_source("Gene activation 4", datasets.ga4),
                _data_source("Gene activation absolute", datasets.ga3_osc),
                _data_source("Gene activation 3", datasets.ga_abs),
                _data_source("Gene activation function", datasets.ga_func)]

cp_file_layout = "multy_{task}NARMA_{system}_{num_trials}trials.cp"


if __name__ == "__main__":


    import argparse
    parser = argparse.ArgumentParser(description="Run NARMA test on a given "
                                                 "network")
    g_basic = parser.add_argument_group('Basic arguments')
    g_basic.add_argument("--network", nargs="*", required=True)
    g_basic.add_argument("--whole-network", action="store_true")
    g_basic.add_argument("-t", "--task", type=int, choices=(0,1,2,3,4,5),
                         required=True)
    g_basic.add_argument("-n", "--num_trials", type=int, default=40)
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

    g_input = parser.add_argument_group('Signal input connections')
    g_input.add_argument('--inputs', nargs="*", default=['0.66'])
    g_input.add_argument('--inputs_file', type=argparse.FileType('r'))
    g_input.add_argument('--randomize_inputs', action='store_true', default=False)

    g_output = parser.add_argument_group('Output control arguments')
    g_output.add_argument("-o", "--override_output_file", action="store_true")
    g_output.add_argument("-v", "--verbose", action='count')
    args = parser.parse_args()

    import reservoir_tools as rt
    import sys
    import os
    import warnings
    import cPickle as cP
    import networkx as nx
    import pandas as pd

    import random

    fout = open("res.csv","w") 
    bar = open("bar.csv","w")

    rand_results = []
    rand_results_2 =[]
    pre_network = []
    post_network = []
    test_network = [15]
    count=0
    
    
    matrix = []
    self_loop=[]
    cylcles_g0=[]
    #cylcles_g1=[]
    #cylcles_g2=[]
    

        
for rand_net in test_network:

    args.network = map(str.lower, args.network)
    if args.randomize and hasattr(args, 'controls'):
        raise Exception
    if args.size != 0 and args.controls[0] is None:
        raise Exception
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
    if args.controls[0] is None:
        control_topologies = [('None', None, None)]
    else:
        control_topologies = [topo for index, topo in enumerate(control_graphs)
                              if str(index) in args.controls or
                              topo[0] in args.controls] 

    for grn_name in GRN_systems:
        if grn_name == "Test":
            pre_network.append(rand_net)
            grn_networks.generate_rand_network(grn_name, rand_net)
            print('red de genes:', rand_net)
        graph = grn_networks.load_network(grn_name)
        

        if args.whole_network:
            graph = nt.prune_graph(graph, verbose=args.verbose)
            post_network.append(len(graph))
            matrix.append(graph.edges.data('mode'))
            print(graph.edges.data('mode'))
            
        else:
            grn_name += "_WN"
            post_network.append(len(graph))
            matrix.append(graph.edges.data('mode'))
            print(graph.edges.data('mode'))


        if args.size == 0:
            size = graph.number_of_nodes()
            
        

        print("EDGES")
        print(graph.number_of_edges())
        print("NODES")
        print(graph.number_of_nodes())
        print("AVERAGE DEGREE")
        print(np.mean(dict(graph.degree).values()))
        print("SELF LOOPS")
        print(graph.number_of_selfloops())

        print(graph.size())
        print(graph.order())
        #print(len(list(nx.simple_cycles(graph))))
        G = nx.Graph(graph)
        #print('CYLES')         
        #print(nx.cycle_basis(G, 'g0'))
        #print(nx.cycle_basis(G, 'g1'))
        #print(nx.cycle_basis(G, 'g2'))
        
        self_loop.append(graph.number_of_selfloops())
        cylcles_g0.append(nx.cycle_basis(G, 'g0'))
        #cylcles_g1.append(nx.cycle_basis(G, 'g1'))
        #cylcles_g2.append(nx.cycle_basis(G, 'g2'))

        avrg_degree = average_degree(graph)
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
                #np.random.seed(args.seed)


                if not utils.is_number(inputs_set):

                    if inputs_set == "None":
                        inputs_arr = None
                    else:
                        inputs_arr = inputs_data[inputs_set].values.reshape(-1, 1)
                    if inputs_set == "Any":
                        inputs_arr *= np.random.randint(0, 2, (size, 1))*2 - 1
                    if args.randomize_inputs:
                        rand_inputs = True
                        inputs_set = inpdata_sourcesuts_set
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
                results_2 = []
                results_3=[]

                while len(results) < num_trials:
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

                    if isinstance(adj_matrix, nx.Graph):
                        adj_matrix = nx.adjacency_matrix(adj_matrix).todense()
                    adj_matrix = np.asarray(adj_matrix)
                    


                                        
                    if rand_inputs:
                        np.random.shuffle(inputs_arr)
                
                    if SR_scale and utils.get_spectral_radius(adj_matrix) == 0:
                        noNilpotent_trials += 1
                        if noNilpotent_trials > 10000:
                            print("NO-NILPOTENT MATRIX NOT FOUND!")
                            raise ValueError
                        continue

                    print("HELLO")
                    prueba = reservoir_performance(data_sources[args.task], adj_matrix,input_weight=inputs_arr,spectral_radius_scale=SR_scale)
                    results.append(prueba[0])
                    results_2.append(prueba[1])

                    rand_results.append(results) 
                    rand_results_2.append(results_2)


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

                fout.write(str(rand_net) + "," + str(np.mean(results)) + "," + str(np.std(results)) + "," + str(graph.order()) + "\n")

i=0
i=i+1
plt.figure()
plt.subplot(2,1,1)
plt.plot(np.linspace(0,len(prueba[3][i][0]),len(prueba[3][i][0])), prueba[3][i][0], label='B')
plt.plot(np.linspace(0,len(prueba[3][i][0]),len(prueba[3][i][0])), prueba[3][i][1], label='D')
plt.legend()
plt.subplot(2,1,2)
plt.plot(np.linspace(0,len(prueba[3][i][0]),len(prueba[3][i][0])), prueba[4][i])
plt.xlabel('Time (min)')
plt.show()

medias = []
desvios = []
for element in rand_results:
    media = np.mean(element)
    desvio=np.std(element)
    medias.append(media)
    desvios.append(desvio)
    
plt.figure()    
plt.errorbar(post_network, medias, desvios, fmt='o', ecolor ='black')
plt.ylim(0,1)
plt.xlabel('Number of genes')
plt.ylabel('Absolute error')
plt.xticks([0,5,10,15,20,25,30])
plt.show()

ff=pd.read_csv("rand_list.csv",sep=",",header=None)
phases=pd.read_csv("phase_list.csv",sep=",",header=None)
phase =[]
error = []
num=[]
for i in range(len(ff)):
    if ff[0][i] ==1:
        phase.append(phases[0][i])
        error.append(prueba[0][i])
        num.append(i)

plt.plot(phase,error,'o')
plt.xlabel('phase')
plt.ylabel('error')



