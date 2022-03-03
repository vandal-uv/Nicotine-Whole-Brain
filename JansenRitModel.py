# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:57:25 2018

@author: Carlos Coronel

Modified version of the Jansen and Rit Neural Mass Model [1]. 

[1] Jansen, B. H., & Rit, V. G. (1995). Electroencephalogram and visual evoked 
potential generation in a mathematical model of coupled cortical columns. 
Biological cybernetics, 73(4), 357-366.

[2] Deco, G., Cruzat, J., Cabral, J., Knudsen, G. M., Carhart-Harris, R. L., Whybrow, 
P. C., ... & Kringelbach, M. L. (2018). Whole-brain multimodal neuroimaging 
model using serotonin receptor maps explains non-linear functional effects of LSD. 
Current biology, 28(19), 3065-3074.


"""

import numpy as np
from numba import jit,float64, vectorize
from numba.core.errors import NumbaPerformanceWarning
import warnings
import networkx as nx
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


#Simulation parameters
dt = 1E-3 #Integration step
teq = 60 #Simulation time for stabilizing the system
tmax = 1200 + teq * 2 #Simulation time
downsamp = 10 #Downsampling to reduce the number of points        
seed = 0 #Random seed

#Networks parameters

#Structural connectivity
nnodes = 90 #number of nodes
M = nx.to_numpy_array(nx.watts_strogatz_graph(nnodes, 8, 0.05, 1)) #Toy matrix. Replace with the one used in [2]
norm = np.mean(np.sum(M,0)) #Normalization factor

#Node parameters
a = 100 #Inverse of the characteristic time constant for EPSPs (1/sec)
ad = 50 #Inverse of the characteristic time constant for long-range EPSPs (1/sec)
b = 50 #Inverse of the characteristic time constant for IPSPs (1/sec)
p = 4.8 * np.ones(nnodes) #Basal input to pyramidal population
sigma = 0.03162 #Scaling noise factor

C = 135 #Global synaptic connectivity
C1 = C * 1 #Connectivity between pyramidal pop. and excitatory pop.
C2 = C * 0.8 #Connectivity between excitatory pop. and pyramidal pop.
C3 = C * 0.25 #Connectivity between pyramidal pop. and inhibitory pop.
C4 = C * 0.25 * np.ones(nnodes) #Connectivity between inhibitory pop. and pyramidal pop.

A = 3.25 #Amplitude of EPSPs
B = 22 #Amplitude of IPSPs

#Global copuling
alpha = 0  * np.ones(nnodes) #Long-range pyramidal-pyramidal coupling


#Sigmoid function parameters
e0 = 2.5 #Half of the maximum firing rate
v0 = 6 #V1/2
r0, r1, r2 = 0.56, 0.56, 0.56 #Slopes of sigmoid functions
r0 = np.ones(nnodes) * r0

#Initial conditions
ic = np.ones((1, nnodes)) * np.array([0.131,  0.171, 0.343, 0.21,
                                      3.07, 2.96,  25.36, 2.42])[:, None] 

def update():
    f1.recompile()

@vectorize([float64(float64,float64)],nopython=True)
#Sigmoid function
def s(v,r0):
    return (2 * e0) / (1 + np.exp(r0 * (v0 - v)))


@jit(float64[:,:](float64[:,:],float64,float64[:],float64[:],float64[:],float64[:],float64[:,:],float64),nopython=True)
#Jansen & Rit multicolumn model (intra-columnar outputs)
def f1(y,t,alpha,C4,r0,p,M,norm):
    x0, x1, x2, x3, y0, y1, y2, y3 = y

    
    x0_dot = y0
    y0_dot = A * a * (s(C2 * x1 - C4 * x2 + C * alpha * x3, r0)) - \
             2 * a * y0 - a**2 * x0 
    x1_dot = y1
    y1_dot = A * a * (p + s(C1 * x0, r1)) - \
             2 * a * y1 - a**2 * x1
    x2_dot = y2
    y2_dot = B * b * (s(C3 * x0, r2)) - \
             2 * b * y2 - b**2 * x2
    x3_dot = y3
    y3_dot = A * ad * (M / norm @ s(C2 * x1 - C4 * x2 + C * alpha * x3, r0)) - \
             2 * ad * y3 - ad**2 * x3

    return(np.vstack((x0_dot, x1_dot, x2_dot, x3_dot, y0_dot, y1_dot, y2_dot, y3_dot)))


@jit(float64[:,:](float64[:,:],float64),nopython=True)
#Noise
def noise(y,sigma):
    x0, x1, x2, x3, y0, y1, y2, y3 = y

    
    x0_dot = np.zeros(nnodes)
    y0_dot = np.zeros(nnodes)
    x1_dot = np.zeros(nnodes)
    y1_dot = A * a * np.random.normal(0, sigma, nnodes)     
    x2_dot = np.zeros(nnodes)
    y2_dot = np.zeros(nnodes) 
    x3_dot = np.zeros(nnodes)
    y3_dot = np.zeros(nnodes)

    return(np.vstack((x0_dot, x1_dot, x2_dot, x3_dot, y0_dot, y1_dot, y2_dot, y3_dot)))


@jit(float64(float64),nopython=True)
#This function is just for setting the random seed
def set_seed(seed):
    np.random.seed(seed)
    return(seed)


def Sim(verbose = True):
    """
    Run a network simulation with the current parameter values.
    
    Note that the time unit in this model is seconds.

    Parameters
    ----------
    verbose : Boolean, optional
        If True, some intermediate messages are shown.
        The default is False.

    Raises
    ------
    ValueError
        An error raises if the dimensions of M and the number of nodes
        do not match.

    Returns
    -------
    y : ndarray
        Time trajectory for the six variables of each node.
    time_vector : numpy array (vector)
        Values of time.

    """
    global M
         
    if M.shape[0]!=M.shape[1] or M.shape[0]!=nnodes:
        raise ValueError("check M dimensions (",M.shape,") and number of nodes (",nnodes,")")
    
    if M.dtype is not np.dtype('float64'):
        try:
            M=M.astype(np.float64)
        except:
            raise TypeError("M must be of numeric type, preferred float")    

    set_seed(seed); #Set the random seed   
    
    ttotal = teq + tmax #Total simulation time
    Nsim = int(ttotal / dt) #Total simulation time points
    Neq = int(teq / dt / downsamp) #Number of points to discard
    Nmax = int(tmax/dt / downsamp) #Number of points of final simulated recordings
    Ntotal = Neq + Nmax #Total number of points of total simulated recordings
   
    #Time vector
    time_vector = np.linspace(0, ttotal, Ntotal)

    row = 8 #Number of variables of the Jansen & Rit model
    col = nnodes #Number of nodes
    y_temp = np.copy(ic) #Temporal vector to update y values
    y = np.zeros((Ntotal, row, col)) #Matrix to store values
    y[0,:,:] = np.copy(ic) #First set of initial conditions
    
    
    if verbose == True:
        for i in range(1,Nsim):
            y_temp += dt * f1(y_temp, i, alpha, C4, r0, p, M, norm) + np.sqrt(dt) * noise(y_temp, sigma)
            #This line is for store values each dws points
            if (i % downsamp) == 0:
                y[i//downsamp,:,:] = y_temp
            if (i % (10 / dt)) == 0:
                print('Elapsed time: %i seconds'%(i * dt)) #this is for impatient people
    else:
        for i in range(1,Nsim):
            y_temp += dt * f1(y_temp, i, alpha, C4, r0, p, M, norm) + np.sqrt(dt) * noise(y_temp, sigma)
            #This line is for store values each dws points
            if (i % downsamp) == 0:
                y[i//downsamp,:,:] = y_temp 
       
    return(y, time_vector)


def ParamsNode():
    pardict={}
    for var in ('a','b','ad','A','B','r0',
                'r1','r2','e0','v0','C','C1','C2','C3',
                'C4','alpha','p','sigma'):
        pardict[var]=eval(var)
        
    return pardict

def ParamsNet():
    pardict={}
    for var in ('nnodes', 'M'):
        pardict[var]=eval(var)
        
    return pardict

def ParamsSim():
    pardict={}
    for var in ('tmax','teq','dt','downsamp'):
        pardict[var]=eval(var)
        
    return pardict


