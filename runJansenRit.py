# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:57:25 2018

@author: Carlos Coronel

Whole brain Jansen & Rit model, with some modifications:
    
[1] Jansen, B. H., & Rit, V. G. (1995). Electroencephalogram and visual evoked 
potential generation in a mathematical model of coupled cortical columns. 
Biological cybernetics, 73(4), 357-366.

Hemodynamic model:

[2] Stephan, K. E., Weiskopf, N., Drysdale, P. M., Robinson, P. A., & Friston, K. J. 
(2007). Comparing hemodynamic models with DCM. Neuroimage, 38(3), 387-401.

[3] Deco, Gustavo, et al. "Whole-brain multimodal neuroimaging model using serotonin 
receptor maps explains non-linear functional effects of LSD." Current Biology 
28.19 (2018): 3065-3074.
 
PET maps obtained in:  https://github.com/netneurolab/hansen_receptors :
    
[4] Hansen, J., Shafiei, G., Markello, R., Smart, K., Cox, S., Wu, Y., ... & Misic, B. (2022).
 Mapping neurotransmitter systems to the structural and functional organization of the human neocortex.

Graph metrics computed using Brain Connectivity Toolbox for Python: https://github.com/fiuneuro/brainconn :
    
[5] Rubinov, M., & Sporns, O. (2010). Complex network measures of brain connectivity: 
uses and interpretations. Neuroimage, 52(3), 1059-1069.

Empirical data from:
    
[6] Gießing, C., Thiel, C. M., Alexander-Bloch, A. F., Patel, A. X., & Bullmore, E. T. (2013).
Human brain functional network changes associated with enhanced and impaired attentional 
task performance. Journal of Neuroscience, 33(14), 5903-5914.

Original SC matrix from:
(not the same as the optimized connectivity)    
    
[7] Deco, G., Cruzat, J., Cabral, J., Knudsen, G. M., Carhart-Harris, R. L., Whybrow,
P. C., ... & Kringelbach, M. L. (2018). Whole-brain multimodal neuroimaging model using 
serotonin receptor maps explains non-linear functional effects of LSD. Current biology, 
28(19), 3065-3074.

"""

import numpy as np
from scipy import signal
import time
import BOLDModel as BD
import JansenRitModel as JR
import graph_utils
import warnings
warnings.filterwarnings("ignore")

#Empirical Data
diff_vec = np.load('diff_vec.npy') #Change in nodal strength (task minus rest)
diff_vec /= 89 #Normalizing by the number of possible connections between nodes


#Simulation parameters
JR.dt = 1E-3 #Integration step

JR.teq = 60 #Simulation time for stabilizing the system
JR.tmax = 1200 + JR.teq * 2 #Simulation time
JR.downsamp = 10 #Downsampling to reduce the number of points        
Neq = int(JR.teq / JR.dt / JR.downsamp) 
Nmax = int(JR.tmax / JR.dt / JR.downsamp)
Ntotal = Neq + Nmax #Total number of points
ttotal = JR.teq + JR.tmax #Total simulation time

JR.nnodes = 90 #number of nodes
nnodes = JR.nnodes     

#Optimized connectivity
JR.M = np.load('SC_optimized_Nicotine_JR.npy')        
JR.norm = np.mean(np.sum(JR.M,0))                      

#Parameters for each condition
#Placebo Rest: alpha = 0.648, beta = 0
#Placebo Task: alpha = 0.648, beta = 0.58
#Nicotine Rest: alpha = 0.644, beta = 0
#Nicotine Task: alpha = 0.641, beta = 0.58 

#Node parameters
JR.alpha = 0.648 * np.ones(nnodes) #Long-range pyramidal-pyramidal coupling   
JR.C4 = (0.3 + 0.3 / 0.5 * JR.alpha) * JR.C #Connectivity between inhibitory pop. and pyramidal pop.
beta = 0 #Input slope 
JR.p = 4.8 - diff_vec * beta #Heterogeneous external inputs

#Random seed
JR.seed = 0

#JR update
JR.update()

init = time.time()

#Simulate the EEG-like signals
y, time_vector = JR.Sim(verbose = False)
pyrm = JR.C2 * y[:,1] - JR.C4 * y[:,2] + JR.C * JR.alpha * y[:,3] #EEG-like output of the model

#Firing rates
rE = JR.s(pyrm, JR.r0)

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
  
#Funcional Connectivity Matrix
FC_BOLD = np.corrcoef(BOLD_filt.T)

#Overall FC (global correlations)
mean_FC = np.mean(graph_utils.get_uptri(FC_BOLD))

#####Proportional Thresholding
trheshold = 0.1
FC_BOLD_th = graph_utils.thresholding(np.copy(FC_BOLD), trheshold)
#Binarize
FC_BOLD_th[FC_BOLD_th > 0] = 1
#Binary Global Efficiency
GE_bin = graph_utils.efficiency_bin(FC_BOLD_th)
#Binary Transitivity
TT_bin = graph_utils.transitivity_bu(FC_BOLD_th)


end = time.time()
print(end-init)



