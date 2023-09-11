#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 18:51:40 2022

@author: leila
"""

# -*- coding: utf-8 -*-
"""Datasets used to test the performance of Reservoir Computing setups."""

import functools
import warnings
import numpy as np
from scipy.integrate import odeint 
import random as rnd
import scipy.stats as st

def stepEuler(vec,dt,gamma,omega0,D):
    x,v = vec
    
    #Making the step
    r = st.norm.rvs(0,1)
    vNew = (-2*gamma*v-omega0**2*x)*dt+r*np.sqrt(D*dt)*1
    xNew = v*dt
    return [xNew,vNew]


def sigmoid(x):
    #return (1/(1+np.exp(-x)))
    return ((x/(np.sqrt((x**2)+1)))+1)/2


def noise():

    t_0 = 0      # define model parameters
    t_end = 1000
    length = 1000
    theta  = 1.1
    mu = 0.1
    sigma  = 0.3

    t = np.linspace(t_0,t_end,length)  # define time axis
    dt = np.mean(np.diff(t))

    y = np.zeros(length)
    y0 = np.random.normal(loc=0.0,scale=1.0)  # initial condition

    drift = lambda y,t: theta*(mu-y) # define drift term, google to learn about lambda
    diffusion = lambda y,t: sigma #define diffusion term
    noise = np.random.normal(loc=0.0,scale=0.1,size=length)*np.sqrt(dt) #define noise process

    #solve SDE

    for i in xrange(1,length):
        y[i] = y[i-1] + drift(y[i-1],i*dt)*dt +diffusion(y[i-1],i*dt)*noise[i]

    #plt.plot(t,y)
    #plt.show()

    return(y)



def keep_bounded(dataset_func, max_trials=100, threshold=1e5):    
    """Wrapper function to regenerate datasets when they get unstable."""
    @functools.wraps(dataset_func)
    def stable_dataset(n_samples=10, sample_len=1000,freq=0.01,amp=0.9,phase=100, noise=0,train="ALL"):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "overflow", RuntimeWarning)
            for i in range(max_trials):
                [x, y] = dataset_func(n_samples=n_samples,
                                      sample_len=sample_len,freq=freq,amp=amp,phase=phase,noise=noise,train=train)
                if np.max((x)) < threshold:
                    return [x, y]
            else:
                errMsg = ("It was not possible to generate dataseries with {} "
                          "bounded by {} in {} trials.".format(
                              dataset_func.__name__, threshold, max_trials))
                raise RuntimeError(errMsg)

    return stable_dataset


@keep_bounded
def narma10(n_samples=10, sample_len=1000):
    """
    Return data for the 10th order NARMA task.

    Generate a dataset with the 10th order Non-linear AutoRegressive Moving
    Average.

    Parameters
    ----------
    n_samples : int, optional (default=10)
        number of example timeseries to be generated.
    sample_len : int, optional (default=1000)
        length of the time-series in timesteps.

    Returns
    -------
    inputs : list (len `n_samples`) of arrays (shape `(sample_len, 1)`)
        Random input used for each sample in the dataset.
    outputs : list (len `n_samples`) of arrays (shape `(sample_len, 1)`)
        Output of the 30th order NARMA dataset for the input used.

    WARNING: this is an unstable dataset. There is a small chance the system
    becomes unstable, leading to an unusable dataset. It is better to use
    NARMA30 which where this problem happens less often.
    """
    system_order = 10
    inputs, outputs = [], []
    for sample in range(n_samples):
        inputs.append(np.random.rand(sample_len, 1) * .5)
        inputs[sample].shape = (-1, 1)
        outputs.append(np.zeros((sample_len, 1)))
        for k in range(system_order-1, sample_len - 1):
            outputs[sample][k + 1] = .3 * outputs[sample][k] +         \
                .05 * outputs[sample][k] *                             \
                np.sum(outputs[sample][k - (system_order-1):k+1]) +    \
                1.5 * inputs[sample][k - system_order-1] * inputs[sample][k] + .1

    return inputs, outputs


@keep_bounded
def narma30(n_samples=10, sample_len=1000):
    """
    Return data for the 30th order NARMA task.

    Generate a dataset with the 30th order Non-linear AutoRegressive Moving
    Average.

    Parameters
    ----------
    n_samples : int, optional (default=10)
        number of example timeseries to be generated.
    sample_len : int, optional (default=1000)
        length of the time-series in timesteps.

    Returns
    -------
    inputs : list (len `n_samples`) of arrays (shape `(sample_len, 1)`)
        Random input used for each sample in the dataset.
    outputs : list (len `n_samples`) of arrays (shape `(sample_len, 1)`)
        Output of the 30th order NARMA dataset for the input used.
    """
    system_order = 30
    inputs, outputs = [], []
    for sample in range(n_samples):
        inputs.append(np.random.rand(sample_len, 1) * .5)
        inputs[sample].shape = (-1, 1)
        outputs.append(np.zeros((sample_len, 1)))
        for k in range(system_order-1, sample_len - 1):
            outputs[sample][k + 1] = .2 * outputs[sample][k] +          \
                .04 * outputs[sample][k] *                              \
                np.sum(outputs[sample][k - (system_order-1):k+1]) +     \
                1.5 * inputs[sample][k - system_order-1] * inputs[sample][k] + .001
    return inputs, outputs


@keep_bounded
def ga3(n_samples=100, sample_len=1000,freq=0.01,amp=0.9):

    inputs, outputs = [], []
    slope_step = (0.001-(-0.001))/n_samples
    a = -0.001
    
    for sample in range(n_samples):

        b = 0
        kd = 0.05

        m_limit = (1/float(sample_len)) * 0.33

        rand = np.random.randint(0,3)

        if rand == 0:
            a = np.random.uniform(1*((1/float(sample_len))),0.8*((1/float(sample_len))))
            k_A = 0.05
            k_B = 0
            k_C = 0
        elif rand == 1:
            a = np.random.uniform(-0.1*((1/float(sample_len))),0.1*((1/float(sample_len))))
            k_A = 0
            k_B = 0.05
            k_C = 0
        else:
            a = np.random.uniform(-0.8*((1/float(sample_len))),-1*((1/float(sample_len))))
            k_A = 0
            k_B = 0
            k_C = 0.05



        def gen_act(x,t):

            z, A, B, C = x 
            dzdt = a
            dAdt = (k_A)-(kd*A)
            dBdt = (k_B)-(kd*B)
            dCdt = (k_C)-(kd*C)



            return(dzdt,dAdt,dBdt,dCdt)

        x0 = [1,0,0,0]

        t = np.linspace(0,sample_len,num=1000)

        x = odeint(gen_act,x0,t)


        x[:,0] += np.random.uniform(-0.1,0.1,len(x[:,0]))

        inputs.append(x[:,0])
        outputs.append(x[:,1:])
        
    return inputs, outputs


@keep_bounded
def ga4(n_samples=10, sample_len=1000,freq=0.01,amp=0.9,phase=100,noise=0,train = "ALL"):

    inputs, outputs, rand_list = [], [], []
    kd = 0.05
    if train == "ALL":

        for sample in range(n_samples):
                  
            rand = np.random.randint(0,3)
            rand_list.append(rand)

            if rand == 0:
                a=0.001
                k_A = 0.05
                k_B = 0
                k_C = 0
    
            elif rand ==1:
                a = 0
                k_A = 0
                k_B = 0.05
                k_C = 0

            elif rand ==2:
                a = -0.001
                k_A = 0
                k_B = 0
                k_C = 0.05
            x0 = [1,0,0,0]
    
            def gen_act(x,t):
    
                z, A,B,C = x
    
                dzdt = a
                dAdt = (k_A)-(kd*A)
                dBdt = (k_B)-(kd*B)
                dCdt = (k_C)-(kd*C)
                return(dzdt,dAdt, dBdt,dCdt)
    
            t = np.linspace(0,sample_len,num=sample_len)
            x = odeint(gen_act,x0,t)
    
            if a ==3:
                if noise == 0:
                    inputs.append(amp*np.sin(freq[sample]*(t+phase[sample]))+10)
                else:    
                    inputs.append(amp*np.sin(freq[sample]*(t+phase[sample]))+10+np.random.normal(0,noise,len(t)))
    
            else:
                if noise==0:
                    inputs.append(x[:,0])
                else:
                    inputs.append(x[:,0]+np.random.normal(0,noise,len(t)))
    
            outputs.append(x[:,1:])
            
    np.savetxt("rand_list.csv",rand_list, delimiter=",",fmt="% s")
    return inputs, outputs




@keep_bounded
def ga3_osc(n_samples=100, sample_len=1000):

    inputs, outputs = [], []
    slope_step = (0.001-(-0.001))/n_samples
    a = -0.001
    
    for sample in range(n_samples):
        
        #inputs.append(np.zeros((sample_len, 1)))
        #outputs.append(np.zeros((sample_len, 3)))

        a = np.random.uniform(-(1/float(sample_len)),(1/float(sample_len)))
        #a += slope_step

        #b = np.random.uniform(0.5,1.5)
        b = 0
        kd = 0.05

        #in_x = np.arange(sample_len).astype(float)
        #inputs[sample] = a * in_x + b  
        #inputs[sample] = (inputs[sample] - (-3000)) / ((3000)-(-3000))
        #print(inputs[sample])


        #m_limit = 0.000165
        m_limit = (1/float(sample_len)) * 0.33

        rand = np.random.randint(0,3)

        if rand == 0:
            a = np.random.uniform(0.1*((1/float(sample_len))),0.20*((1/float(sample_len))))
            amp = 1
            k_A = 0.05
            k_B = 0
            k_C = 0
        elif rand == 1:
            a = np.random.uniform(0.40*((1/float(sample_len))),0.60*((1/float(sample_len))))
            amp = 2
            k_A = 0
            k_B = 0.05
            k_C = 0
        else:
            a = np.random.uniform(0.80*((1/float(sample_len))),1*((1/float(sample_len))))
            amp = 3
            k_A = 0
            k_B = 0
            k_C = 0.05

        readout_target_dyn = open("readout_target_dyn.csv","a")

        def gen_act(x,t):

            z, A, B, C = x 
            dzdt = amp*np.cos(a*t)
            dAdt = 0.05+(k_A)-(kd*A)
            dBdt = 0.05+(k_B)-(kd*B)
            dCdt = 0.05+(k_C)-(kd*C)

            #print("A")
            #print(k_A)
            #print(-kd*A)

            #print("B")
            #print(k_B)
            #print(-kd*B)

            #print("C")
            #print(k_C)
            #print(-kd*C)

            #print(t)

            #if sample == 0:            
                #readout_target_dyn.write(str(t)+",A,"+str(k_A)+","+str(kd*A)+"\n")
                #readout_target_dyn.write(str(t)+",B,"+str(k_B)+","+str(kd*B)+"\n")
                #readout_target_dyn.write(str(t)+",C,"+str(k_C)+","+str(kd*C)+"\n")

            return(dzdt,dAdt,dBdt,dCdt)

        x0 = [1,1,1,1]

        t = np.linspace(0,sample_len,num=1000)
        #print(t)
        dt = t[1]-t[0]

        x = odeint(gen_act,x0,t)
        #x = ode(gen_act).set_integrator("dop853")
        #x.set_initial_value(x0).set_f_params(z[0])

        #print(x)

        inputs.append(x[:,0])
        outputs.append(x[:,1:])

    #outputs = [x+1 for x in outputs] 

        
    return inputs, outputs


@keep_bounded
def ga_abs(n_samples=100, sample_len=1000):

    inputs, outputs = [], []
    slope_step = (0.001-(-0.001))/n_samples
    a = -0.001
    
    for sample in range(n_samples):
        
        inputs.append(np.zeros((sample_len, 1)))
        outputs.append(np.zeros((sample_len, 3)))

        a = np.random.uniform(-0.001,0.001)
        #a += slope_step
        #a = -0.001
        #b = np.random.uniform(0.5,1.5)
        b = 0

        
       

        k_A = 0
        k_B = 0
        k_C = 0

        kd = 0.05

        #in_x = np.arange(sample_len).astype(float)
        #inputs[sample] = a * in_x + b  
        #inputs[sample] = (inputs[sample] - (-3000)) / ((3000)-(-3000))
        #print(inputs[sample])


        m_limit = 0.000333
        #m_limit = 0

        rand = np.random.randint(0,3)

        if rand == 0:
            a = np.random.uniform(-1*((1/float(sample_len))),-0.8*((1/float(sample_len))))
            k_A = 2
            k_B = 1
            k_C = 1
        elif rand == 1:
            a = np.random.uniform(-0.1*((1/float(sample_len))),0.1*((1/float(sample_len))))
            k_A = 1
            k_B = 2
            k_C = 1
        else:
            a = np.random.uniform(0.8*((1/float(sample_len))),1*((1/float(sample_len))))
            k_A = 1
            k_B = 1
            k_C = 2

        for step in range(sample_len):
        
            inputs[sample][step] = a * step + b +1  # LINEAR   
            #inputs[sample][step] = (a*step**2) + b*step   # CUADRATIC
            #inputs[sample][step] = 1/(1+np.exp(-0.0075*step+4))   # SIGMOIDAL
            #inputs[sample][step] = 0.5 * np.sin(a*step) + 0.5 # SINUSOIDAL
            

            outputs[sample][step] = [k_A,k_B,k_C]



        
        
    return inputs, outputs


@keep_bounded
def ga_func(n_samples=100, sample_len=1000):

    inputs, outputs = [], []
    k_A = 0
    k_B = 0
    k_C = 0

    kd = 0.05
    
    for sample in range(n_samples):
        
        inputs.append(np.zeros((sample_len, 1)))
        outputs.append(np.zeros((sample_len, 3)))

        func = np.random.randint(0,3)
        a = np.random.uniform(1,5)
        #a = 1
        
        for step in range(sample_len):
        
            if func == 0:
                inputs[sample][step] = a*0.001*step +1   # LINEAR
                #outputs[sample][step] = [1,0,0]
                k_A = 0.05
                k_B = 0
                k_C = 0
                
            elif func == 1:
                inputs[sample][step] = np.sin(a*0.01*step)+1
                #outputs[sample][step] = [0,1,0]
                k_A = 0
                k_B = 0.05
                k_C = 0

            elif func == 2:
                inputs[sample][step] = inputs[sample][step] = 1/(1+np.exp(-a*0.0075*step+4))
                #outputs[sample][step] = [0,0,1]
                k_A = 0
                k_B = 0
                k_C = 0.05

            outputs[sample][step][0] = outputs[sample][step-1][0] + (k_A - (kd * outputs[sample][step-1][0])) + 1
            outputs[sample][step][1] = outputs[sample][step-1][1] + (k_B - (kd * outputs[sample][step-1][1])) + 1
            outputs[sample][step][2] = outputs[sample][step-1][2] + (k_C - (kd * outputs[sample][step-1][2])) + 1


        
    return inputs, outputs


