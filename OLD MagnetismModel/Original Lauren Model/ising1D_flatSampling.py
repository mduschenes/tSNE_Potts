###################### PHYS 602: PSI STATISTICAL MECHANICS CORE #####################
### Code by Lauren Hayward Sierens
###
### This code estimates the average energy for the classical one-dimensional Ising
### model using flat sampling.
### This code assumes open boundary conditions and no external field (h=0).
#####################################################################################

import matplotlib.pyplot as plt
import numpy as np
import os
import random

### Input parameters: ###
T_list = np.linspace(5.0,0.5,10) #temperature list
L = 10                           #linear size of the lattice
d = 1
N_spins = L^d                    #total number of spins
J = 1                            #coupling parameter

### Sampling parameters: ###
M_samples = 10000

### Create arrays to store the spin states and the energy estimator results: ###
spins       = np.zeros(N_spins,dtype=np.int)
E_estimator = np.zeros(len(T_list))

### Function to calculate the total energy (with open boundary conditions) ###
def getEnergy():
  currEnergy = 0
  for i in range(N_spins-1):
    currEnergy += -J*( spins[i]*spins[(i+1)] )
  return currEnergy
#end of getEnergy() function

### For each temperatures, estimate the energy from M random state configurations: ###
for (iT,T) in enumerate(T_list):
  print('\nT = %f' %T)

  E_samples = np.zeros(M_samples)
  for m in range(M_samples):
    
    #Choose a state at random:
    for i in range(N_spins):
      spins[i] = 2*random.randint(0,1) - 1 #either +1 or -1

    E_samples[m] = getEnergy()
  #end loop over m

  E_estimator[iT] = np.sum(E_samples*np.exp(-E_samples*1.0/T)) / (1.0*np.sum(np.exp(-E_samples*1.0/T))*N_spins)
#end loop over temperature

### Calculate the known solution (see Tutorial 2): ###
E_solution = -(N_spins-1)*J*np.tanh(J*1.0/T_list)/(1.0*N_spins)

### Plot the results: ###
plt.plot(T_list, E_estimator, 'o-', label='M=%d Estimator' %M_samples)
plt.plot(T_list, E_solution, '-', label = 'Solution')
plt.xlabel('$T$')
plt.ylabel('$<E>/N$')
plt.legend(loc='best')
plt.title('L=%d Ising model' %(L))
plt.show()
