# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import numpy as np
#from scipy.integrate import odeint
from scipy.integrate import ode 
from scipy.interpolate import interp2d
import pandas as pd


def sigmoid(x):
    #return (1/(1+np.exp(-x)))
    #return ((x/(np.sqrt((x**2)+1)))+1)/2
    return (((x-0.5)/(np.sqrt(((x-0.5)**2)+0.1)))+1)/2


def _add_constant(x):
    """Add a constant term to the vector 'x'.
    x -> [1 x]
    """
    return np.concatenate((np.ones((x.shape[0], 1), dtype=x.dtype), x), axis=1)
    


class LinearRegression(object):
    
    def __init__(self,  train_x=None, train_y=None, use_bias=True,
                 use_pinv=True, finish_training=False):
        self.use_bias = use_bias
        self.use_pinv = use_pinv

        # Set two temporary variables for the linear regression estimation
        self._xTx = None
        self._xTy = None

        # keep track of how many data points have been sent
        self._tlen = 0

        # final regression coefficients
        # if with_bias=True, beta includes the bias term in the first column


        self.beta = None

        if train_x is not None and train_y is not None:
            self.train(train_x, train_y, finish_training=finish_training)

    def train(self, x, y, finish_training=False):

        if x.shape[0] != y.shape[0]:
            raise ValueError("X and Y do not describe the same number of "
                             "points ({} vs {})".format(x.shape, y.shape))
        # Forget current regression coefficients
        self.beta = None

        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if self.use_bias:
            x = _add_constant(x)


        # initialize internal vars if necessary
        if self._xTx is None:
            x_size = x.shape[1]
            self._xTx = np.zeros((x_size, x_size), x.dtype)
            self._xTy = np.zeros((x_size, y.shape[1]), x.dtype)


        # update internal variables
        self._xTx += (np.dot(x.T,x))
        self._xTy += (np.dot(x.T,y))
        self._tlen += x.shape[0]

        #print(self._xTy)
        invfun = np.linalg.pinv if self.use_pinv else np.linalg.inv
        inv_xTx = invfun(self._xTx)
        W_B = pd.read_csv('W_B.csv',sep=',' ,header=None, skiprows=[1])
        W_D = pd.read_csv('W_D.csv',sep=',' ,header=None, skiprows=[1])
        oned = np.dot(inv_xTx, self._xTy)
        col_B=oned[:,0]
        col_B = col_B.tolist()
        col_D=oned[:,1]
        col_D = col_D.tolist()
        W_B = W_B.append(col_B)
        W_B.to_csv('W_B.csv',index = False)
        W_D = W_D.append(col_D)
        W_D.to_csv('W_D.csv',index = False)



        if finish_training:
            self.finish_training()

    def finish_training(self):
        if self._xTx is None:
            raise RuntimeError("The LinearRegression instance was not "
                               "trained!")
        invfun = np.linalg.pinv if self.use_pinv else np.linalg.inv
        inv_xTx = invfun(self._xTx)
        self.beta = np.dot(inv_xTx, self._xTy)





    def __call__(self, x):

        DeltaT = 1

        #A = np.zeros(x.shape[0])
        #B = np.zeros(x.shape[0])
        #C = np.zeros(x.shape[0])

        #print(type(x))

        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        if self.beta is None:
            self.finish_training()
        
        if self.use_bias:
            x = _add_constant(x)

        #print(np.dot(x[0,:],self.beta[:,0]))

        decay = 0.05

        #print(x)


        # def readout(t,y,res_dyn):
        #     dAdt = np.dot(res_dyn,self.beta[:,0]) -decay*y[0]
        #     print(y[0])
        #     dBdt = np.dot(res_dyn,self.beta[:,0]) -decay*y[1]
        #     dCdt = np.dot(res_dyn,self.beta[:,0]) -decay*y[2]

        #     return (dAdt,dBdt,dCdt)

        # x0 = [0,0,0]
        # t = np.linspace(0,x.shape[0],num=1000)
        # dt = t[1]-t[0]
        # output = np.empty_like(t)

        # res_dyn = x

        # r = ode(readout).set_integrator("dop853")
        # r.set_initial_value([0,0,0],0).set_f_params(res_dyn[0,:])


        # for i in xrange(len(t)):
        #     r.set_f_params(res_dyn[i,:])
        #     r.integrate(r.t+dt)
        #     output[i] = r.y


        # return(output)


        n_outs = 2

        #def readout(t,y,res_dyn,out):
        def readout(t,y,res_dyn):

            #equations = []
            #out = int(param[0])

            #for i in range(0,n_outs):
            equations = ((0.05*sigmoid(np.dot(res_dyn,self.beta[:,out])))-(decay*y))


            return equations




#         def readoutA(t,y,res_dyn):
#             dAdt = (0.05*sigmoid(np.dot(res_dyn,self.beta[:,0]))) -(decay*y)
#             #dAdt = (0.05*(np.dot((res_dyn**n)/((K**n) + (res_dyn**n)),self.beta[:,0]))) -(decay*y)

#             activation = 0
#             alt = 0
#             represion = 1
#             #print(range(x.shape[0]))
#             #print(self.beta)
#             #print(res_dyn)
#             #nothing = raw_input()

#             #dAdt = 0.05*(activation*represion)-decay*y


#             #print(((res_dyn**n)/((K**n) + (res_dyn**n))))
#             #print("A_barra")
#             #print((np.dot(res_dyn,self.beta[:,0]))) 
#             #print(decay*y)
#             #print(t)

#             #readout_real_dyn.write(str(t)+",A,"+str((0.05*np.dot(res_dyn,self.beta[:,0])))+","+str((decay*y)[0])+"\n")
#             return dAdt

#         def readoutB(t,y,res_dyn):
#             dBdt = (0.05*sigmoid(np.dot(res_dyn,self.beta[:,1]))) -(decay*y)
#             #dBdt = (0.05*(np.dot((res_dyn**n)/((K**n) + (res_dyn**n)),self.beta[:,1]))) -(decay*y)

#             activation = 0
#             represion = 1

#             #dBdt = 0.05*(activation*represion)-decay*y
#             #print("B_barra")
#             #print(1*np.dot(res_dyn,self.beta[:,1]))
#             #print(decay*y)
#             #print(t)

#             #readout_real_dyn.write(str(t)+",B,"+str((0.05*np.dot(res_dyn,self.beta[:,0])))+","+str((decay*y)[0])+"\n")
#             return dBdt

#         def readoutC(t,y,res_dyn):
#             dCdt = (0.05*sigmoid(np.dot(res_dyn,self.beta[:,2]))) -(decay*y)
#             #dCdt = (0.05*(np.dot((res_dyn**n)/((K**n) + (res_dyn**n)),self.beta[:,2]))) -(decay*y)

#             activation = 0
#             represion = 1

#             #dCdt = 0.05*(activation*represion)-decay*y
#             #print("C_barra")   
#             #print(1*np.dot(res_dyn,self.beta[:,2]))
#             #print(decay*y)
#             #print(t)
#             #readout_real_dyn.write(str(t)+",C,"+str((0.05*np.dot(res_dyn,self.beta[:,0])))+","+str((decay*y)[0])+"\n")
#             return dCdt

# #### FOURTH GENE

#         def readoutD(t,y,res_dyn):
#             dDdt = (0.05*sigmoid(np.dot(res_dyn,self.beta[:,3]))) -(decay*y)
#             #dCdt = (0.05*(np.dot((res_dyn**n)/((K**n) + (res_dyn**n)),self.beta[:,2]))) -(decay*y)

#             activation = 0
#             represion = 1

#             #dCdt = 0.05*(activation*represion)-decay*y
#             #print("C_barra")   
#             #print(1*np.dot(res_dyn,self.beta[:,2]))
#             #print(decay*y)
#             #print(t)
#             #readout_real_dyn.write(str(t)+",C,"+str((0.05*np.dot(res_dyn,self.beta[:,0])))+","+str((decay*y)[0])+"\n")
#             return dDdt

# ######

        #x0 = [0,0,0]
        #x0 = [0,0,0,0] ### FOURTH GENE
        x0 = np.repeat(0,n_outs)

        t = np.linspace(0,x.shape[0],num=len(x))

        dt = t[1]-t[0]
        
        #output = np.repeat[np.empty_like(t),n_outs]
        output = [np.empty_like(t),np.empty_like(t)]

        outputB = np.empty_like(t)
        outputD = np.empty_like(t)  ### FOURTH GENE

        res_dyn = x

        #interpol = interp2d(t,x,fill_value="extrapolate")  
        #out = odeint(readout,x0,t)

        
        for out in range(0,n_outs):
            r = ode(readout).set_integrator("dop853")
            r.set_initial_value(x0[out],0).set_f_params(res_dyn[0:])
        
        # rA = ode(readoutA).set_integrator("dop853")
        # rA.set_initial_value(x0[0],0).set_f_params(res_dyn[0,:])

        # rB = ode(readoutB).set_integrator("dop853")
        # rB.set_initial_value(x0[1],0).set_f_params(res_dyn[0,:])

        # rC = ode(readoutC).set_integrator("dop853")
        # rC.set_initial_value(x0[2],0).set_f_params(res_dyn[0,:])

        # rD = ode(readoutD).set_integrator("dop853") ### FOURTH GENE
        # rD.set_initial_value(x0[3],0).set_f_params(res_dyn[0,:]) 

            for i in xrange(len(t)):
                
                r.set_f_params(res_dyn[i,:])
                r.integrate(r.t+dt)

                # rA.set_f_params(res_dyn[i,:])
                # rA.integrate(rA.t+dt)

                # rB.set_f_params(res_dyn[i,:])
                # rB.integrate(rB.t+dt)

                # rC.set_f_params(res_dyn[i,:])
                # rC.integrate(rC.t+dt)

                # rD.set_f_params(res_dyn[i,:])  ### FOURTH GENE
                # rD.integrate(rD.t+dt)

                # outputA[i] = rA.y
                # outputB[i] = rB.y
                # outputC[i] = rC.y
                # outputD[i] = rD.y   ### FOURTH GENE
                output[out][i] = r.y
                #print(output)



        #print(np.column_stack((outputA,outputB,outputC))) 
            #print(output)

        #return (np.column_stack((outputA,outputB,outputC)))
        #return (np.column_stack((outputA,outputB,outputC,outputD))) ### FOURTH GENE

        return (output)


        # for t in range(0,int(x.shape[0])):

        #     #print(x[t,:])
        #     #print(self.beta[:,0])

        #     A[t] = (A[t-1] + ((np.dot(x[t,:],self.beta[:,0]))-decay*A[t-1])*DeltaT)
        #     B[t] = (B[t-1] + ((np.dot(x[t,:],self.beta[:,1]))-decay*B[t-1])*DeltaT)
        #     C[t] = (C[t-1] + ((np.dot(x[t,:],self.beta[:,2]))-decay*C[t-1])*DeltaT)

            
        #     print(x[t,:])
        #     print(np.dot(x[t,:],self.beta[:,0]))
        #     print(sigmoid(np.dot(x[t,:],self.beta[:,0])))

        # #print(np.column_stack((A,B,C))) 

        # print("DYNAMICS")
        # #print (x)
        # print("WEIGHTS")
        # #print(self.beta)

        # return (np.column_stack((A,B,C)))
        #return (np.dot(x, self.beta))


class RidgeRegression(LinearRegression):

    def __init__(self,  train_x=None, train_y=None, ridge_param=0,
                 use_bias=True, use_pinv=True, finish_training=False):

        self.ridge_param = ridge_param
        self.ridge_param = 0

        super(RidgeRegression, self).__init__(
            train_x=train_x, train_y=train_y, use_bias=use_bias,
            use_pinv=use_pinv, finish_training=finish_training)

    def finish_training(self):
        if self._xTx is None:
            raise RuntimeError("The LinearRegression instance was not "
                               "trained!")


        invfun = np.linalg.pinv if self.use_pinv else np.linalg.inv

        #print(-np.log((1/(self._xTy))-1))

        inv_xTx = invfun(self._xTx + self.ridge_param*np.eye(*self._xTx.shape))

        #constraint = (np.sinh(self._xTx)**2 + np.cosh((self._xTx)**2))/(np.sinh(self._xTx) * np.cosh((self._xTx)))
        self.beta = (np.dot(inv_xTx,self._xTy))

        #print(self._xTx)
        #print(self._xTy)
        print("LOS PESOS O ALGO")
        print(self.beta)
