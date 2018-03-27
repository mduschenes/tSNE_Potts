# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 14:18:39 2018

@author: Matt
"""

import numpy as np
import time

#from ModelFunctions import caps,flatten,signed_val,list_f,delta_f


from Plot_Sites import Plot_Sites
from ModelFunctions import flatten


class MonteCarloUpdate(object):
    
    def __init__(self,sites,neighbour_sites,observables_f,T,modelprops,
                      update,transition,animate):
        # Perform Monte Carlo Updates for nclusters of sites and Plot Data
        
        #  Monte Carlo Update Parameters: Perform Monte Carlo Update Boolean
        #                                 Number of initial updates Neqb,
        #                                 Number of Measurement Updates Nmeas
        #                                 Number of Clusters Ncluster
        #                                 Monte Carlo Update Algorithm
        # State Transition Parameters:
        #                                 Transition Probability prob_transiton
        #                                 Transition Values site_states
        #
        # Animate: [Boolean for Sites, Cluster, Edges Animation,
        #           plot_range,Save File]
        
        
        # Define System Configuration
        self.sites = sites
        self.neighbour_sites = neighbour_sites
        
        self.modelprops = modelprops
        self.Nspins = np.size(self.sites)
        self.d = modelprops['d']
        
        self.T = np.atleast_1d(T).tolist()
        self.observables_f = observables_f
        self.observables = []
        
        # Define Transition Probability and Possible Site Values
        self. prob_updates = transition[0]
        self.state_sites = transition[1]
        
        
        # Define Monte Carlo Update Parameters:
        # Number of updates to reach "equilibrium" before measurement, 
        # Number of measurement updates.
        # Monte Carlo update alogrithms
        self.mcupdate = update[0]
        self.Neqb = int((1)*self.Nspins) if update[1]==None else update[1]
        self.Nmeas = int(10*self.Nspins) if update[2]==None else update[2]
        self.Ncluster = update[3]


        self.update_algs = {'metropolis': self.metropolis, 'wolff': self.wolff}

        
        # Initialize Plotting
        self.animate = animate[0:-1]
        plot_range = animate[-1]
        plot_titles = [['Spin Configurations','Cluster','Edge'] ,
                                          lambda i: r'$t_{MC}$: %d'%i,
                                          lambda i: r'T = %0.1f'%self.T[i]]
        data_process = lambda data: self.sites_region(data).reshape(
                                  [int(np.power(self.Nspins,1/self.d))]*self.d)
        
        self.plotf = Plot_Sites(self.animate,np.size(self.T),sum(self.animate),
                                     plot_titles,data_process,plot_range)

        return
                    
        
            
            
            
            
    def MCalg(self,algorithm):
        # Perform Monte Carlo Update Algorithm and Plot Spin Sites
        
        self.MCUpdate_alg = self.update_algs[algorithm]
        
        self.prob_update = self.prob_updates[algorithm]
        
        t0 = time.clock()
        
        # Create Observables, Clusters, and Edges Arrays
        observable = []
        self.clusters = []
        self.cluster_edges = []
        
        for i_t,t in enumerate(self.T):
            
            self.t = t
            
            #t1 = time.clock()
        
            for i_iter in range(self.Neqb):
                self.MCUpdate_alg()
                
            for i_iter in range(self.Nmeas):
    
    
                self.MCUpdate_alg()
    
                
                if self.animate[0] and ((False or np.size(self.T)==1) or ( 
                                                     i_iter == self.Nspins-1)):
                        
                    site_data = [self.sites,self.cluster_sites,
                                       self.cluster_sites]
                    
                    for i_a,a in reversed(list(enumerate(self.animate))): 
                        if a:
                            self.plotf.plot_sites(site_data[i_a],i_t,i_a,i_iter)
                        
                # Update Observables Array
                if 0 % self.Nmeas/int(np.sqrt(self.Nspins)) == 0 or True:
                    observable.append(flatten(self.observables_f(
                            self.sites,self.neighbour_sites,self.t)))
                                
    
            #print('T = %0.1f   %0.5f' %(t,time.clock()-t1))
        self.observables.append(observable)


      
        # Calculate Total Runtime for system class
#        print('final order: '+str(
#               (self.observables_f(self.sites,self.neighbour_sites,self.t)[2]/
#                                                                 self.Nspins)))
        print('%s runtime: %0.5f'%(algorithm,time.clock()-t0)) 
#        
        # Pause after last Iteration
        if i_t == len(self.T)-1:
                    # Save Data Plots
            if self.animate[0]: 
                self.plotf.plot_save(''.join(
                       [self.modelprops[t] for t in ['data_dir','data_file']]))  
                self.plotf.plot_show(1)
        
        return
     
        
        
    def metropolis(self):
        # Randomly alter random spin sites and accept spin alterations
        # if energetically favourable or probabilistically likely
        for i in range(self.Nspins):
            E0 = self.prob_update(self.sites,self.neighbour_sites,self.t)
            
            isites = np.random.randint(0,self.Nspins,self.Ncluster)
            sites0 = np.copy(self.sites[isites])
            self.sites[isites] = self.state_sites(self.Ncluster,sites0)

            dE = self.prob_update(self.sites,self.neighbour_sites,self.t)- E0

            if dE > 0:
                if np.exp(-dE/self.t) < np.random.random():
                    self.sites[isites] = np.copy(sites0)
                else:
                    self.cluster_sites = isites
                    self.clusters.append(isites)
            else:
                self.cluster_sites = isites
                self.clusters.append(isites)

            return
    
    
    
    def wolff(self):
        
        # Create Cluster Array and Choose Random Site
        isite = np.random.randint(self.Nspins)
        self.cluster_sites = []
        self.cluster_rejections = []
        self.cluster_value = self.state_sites(self.Ncluster,
                                                self.sites[isite])
        self.cluster_value0 = self.sites[isite]
    
        # Perform cluster algorithm to find indices in cluster
        self.cluster(isite)
        
        # Flip spins in cluster to new value
        self.sites[self.cluster_sites] = self.cluster_value        
        
        self.clusters.append(self.cluster_sites)
        
        
        # Find Edges of Cluster
        self.edges(self.cluster_sites)
        #print(np.size(self.edges))
        # Plot Sites, Updated Cluster, and Updated Cluster Edges
                
               
    
    
    def edges(self,cluster):
        for c in np.atleast_1d(cluster):
             self.cluster_edges.append([i for i in np.atleast_1d(c) if len(
                                             [j for j in np.atleast_1d(c) if j
                                    in self.neighbour_sites[0][i]]) 
                                          < len(self.neighbour_sites[0][i])])
        return
    
    def cluster(self,i):
        self.cluster_sites.append(i)
        if len(self.cluster_sites) < int(0.95*self.Nspins):
            J = (j for j in self.neighbour_sites[0][i] if (
                                    j not in self.cluster_sites) and 
                                    (self.sites[j] == self.cluster_value0) and 
                                    (j not in self.cluster_rejections))
            for j in J:
                if self.prob_update(self.t) > np.random.rand():
                        self.cluster(j)
                else:
                    self.cluster_rejections.append(j)
        return

    def sites_region(self,sites0):
        if  np.array_equiv(sites0,self.sites):
            return sites0
        else:
            region = np.zeros(np.shape(self.sites))
            region[:] = np.nan
            region[sites0] = self.sites[sites0]
            return region
    
    
    
    
