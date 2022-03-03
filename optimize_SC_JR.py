# -*- coding: utf-8 -*-
"""
Created on Wen Jan 06 12:23:19 2021

@author: Carlos Coronel

Code for SC optimization based in the method proposed by Deco et al. (2019) [1]. First, the model was fitted to the
empirical Functionl Connectivity matrix using the original Structural Connectivity matrix. Then, Structural
Connectivity was updated iteratively employing the point-to-point difference between empirical and simulated 
Functional Connectivity matrices. Negative values within the optimized Structural Connectivity matrix were discarded.

The code started with an alpha = 0.651 employing the human Structural Connectivity matrix used in [2]

[1] Deco, G., Cruzat, J., Cabral, J., Tagliazucchi, E., Laufs, H., Logothetis, N. K., 
& Kringelbach, M. L. (2019). Awakening: Predicting external stimulation to 
force transitions between different brain states. Proceedings of the National 
Academy of Sciences, 116(36), 18088-18097.

[2] Deco, G., Cruzat, J., Cabral, J., Knudsen, G. M., Carhart-Harris, R. L., Whybrow, 
P. C., ... & Kringelbach, M. L. (2018). Whole-brain multimodal neuroimaging 
model using serotonin receptor maps explains non-linear functional effects of LSD. 
Current biology, 28(19), 3065-3074.

"""

import numpy as np
from scipy import signal
import BOLDModel as BD
import JansenRitModel as JR
from scipy import stats
import graph_utils
import networkx as nx

#Uppter triangular of the empirical Functional Connectivity matrix
dist_placebo_rest = graph_utils.get_uptri(np.load('FC_placebo_rest.npy'))

#Simulation parameters
JR.dt = 1E-3 #Integration step
JR.teq = 60 #Simulation time for stabilizing the system
JR.tmax = 1200 + JR.teq * 2 #Simulation time
JR.downsamp = 10 #Downsampling to reduce the number of points        
Neq = int(JR.teq / JR.dt / JR.downsamp) 
Nmax = int(JR.tmax / JR.dt / JR.downsamp)
Ntotal = Neq + Nmax 
ttotal = JR.teq + JR.tmax #Total simulation time

#Structural connectivity
nnodes = 90 #number of nodes
JR.nnodes = 90
JR.M = nx.to_numpy_array(nx.watts_strogatz_graph(nnodes, 8, 0.05, 1)) #Toy matrix. Replace with the one used in [2]
JR.norm = np.mean(np.sum(JR.M,0)) #Normalization factor

#Node parameters
JR.alpha = 0.651 * np.ones(nnodes) #Long-range pyramidal-pyramidal coupling
JR.C4 = (0.3 + JR.alpha * 0.3 / 0.5) * 135 #Connectivity between inhibitory pop. and pyramidal pop.

#Current copy of the SC matrix
C = np.copy(JR.M)
JR.norm = np.mean(np.sum(JR.M,0))
    
seeds = 20 #Number of FCs computed in each iterations
iters = 80 #Number of iterations
fitting = np.zeros((4,iters)) #Matrix to save the results: fitting-related metrics across iterations
all_SCs = np.zeros((90,90,iters)) #Array to save all SCs across iterations
epsilon = 0.01 #Convergence rate

#JR update
JR.update()

for i in range(0,iters):
    FC_BOLD = np.zeros((90,90))
    for ss in range(0,seeds):
        all_SCs[:,:,i] = np.copy(C)
        
        #np.random.seed(seed)
        JR.seed = ss
        y, time_vector = JR.Sim(verbose = False)
        pyrm = JR.C2 * y[:,1] - JR.C4 * y[:,2] + JR.C * JR.alpha * y[:,3] #EEG-like output of the model
        
        #Firing rates
        rE = JR.s(pyrm,JR.r0)
        
        #Simulating BOLD
        BOLD_signals = BD.Sim(rE, nnodes, JR.dt * JR.downsamp)
        BOLD_signals = BOLD_signals[Neq:,:]
        
        #Same dt as empirical signals (RT = 1.5 seconds)
        BOLD_dt = 1.5        
        BOLD_signals = signal.decimate(BOLD_signals, n = 3,
                                       q = int(BOLD_dt / JR.dt / JR.downsamp), axis = 0)
        
        
        #Filter the BOLD-like signal between 0.01 and 0.1 Hz
        Fmin, Fmax = 0.01, 0.1
        a0, b0 = signal.bessel(3, [2 * BOLD_dt * Fmin, 2 * BOLD_dt * Fmax], btype = 'bandpass')
        BOLD_filt = signal.filtfilt(a0, b0, BOLD_signals[:,:], axis = 0)        
        cut0, cut1 = int(Neq / (BOLD_dt / JR.dt / JR.downsamp)), int((Nmax - Neq) / (BOLD_dt / JR.dt / JR.downsamp))
        BOLD_filt = BOLD_filt[cut0:cut1,:]  
        
        FC_BOLD += np.corrcoef((BOLD_filt.T))
        print([ss,i])
        
    FC_BOLD /= seeds #Averaged FC matrix across random seeds
    dist_sim = graph_utils.get_uptri((FC_BOLD)) #Upper triangular of the simulated FC matrix
    
    #Fitting metrics
    fitting[0,i] = np.mean(dist_placebo_rest) - np.mean(dist_sim) #Difference in mean Overall FC (global correlations)
    fitting[1,i] = stats.ks_2samp(dist_placebo_rest, dist_sim)[0] #Difference in weights' distributions
    fitting[2,i] = np.linalg.norm(dist_placebo_rest - dist_sim) #Euclidean distance
    fitting[3,i] = stats.pearsonr(dist_placebo_rest,dist_sim)[0] #Pearson Correlation
    
    #For following up the optimization
    print(ss,fitting[:,i],np.sum(C), np.sum(C>0))
    
    
    #Updating the C matrix
    C += graph_utils.matrix_recon(epsilon*(dist_placebo_rest - dist_sim))
    C = graph_utils.thresholding(C,0.3) #Fixing density
    C[C < 0] = 0 #Discarding negative values
    C = C * 138 / np.sum(C) #Fixing the sum of weights
    JR.M = C #Update the SC within the model
    JR.norm = np.mean(np.sum(JR.M,0)) #Update the normalization factor



