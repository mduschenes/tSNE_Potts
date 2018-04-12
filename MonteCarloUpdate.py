# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 14:18:39 2018
@author: Matt
"""

import numpy as np

from Plot_Data import Plot_Data
from ModelFunctions import flatten, caps


class MonteCarloUpdate(object):
    
    def __init__(self,sites,neighbour_sites,model_props,
                      update,animate):
        # Perform Monte Carlo Updates for nclusters of sites and Plot Data
        
        #  Monte Carlo Update Parameters: Perform Monte Carlo Update Boolean
        #                                 Number of initial updates Neqb,
        #                                 Number of Measurement Updates Nmeas
        #                                 Measurement Update Frequency Nmeas_f
        #                                 Number of Clusters Ncluster
        # State Transition Parameters:
        #                                 Transition Probability prob_transiton
        #                                 Transition Values site_states
        #
        # Animate: [Boolean for Sites, Cluster, Edges Animation,
        #           plot_range,Save File]
        
        
        # Define System Configuration
        self.sites = sites
        self.neighbour_sites = neighbour_sites
        
        self.model_props = model_props
        self.Nspins = np.size(self.sites)
        self.d = model_props['d']
        
        self.T = np.atleast_1d(self.model_props['T']).tolist()
        self.t = self.T[0]
        
        
        
        # Define Monte Carlo Update Parameters:
        # Number of updates to reach "equilibrium" before measurement, 
        # Number of measurement updates.
        # Monte Carlo update alogrithms
        self.mcupdate = update[0]
        self.Neqb = int(((1) if update[1]==None else update[1])*self.Nspins)
        self.Nmeas = int(((1) if update[2]==None else update[2])*self.Nspins)
        self.Nmeas_f = int(((1) if update[3]==None else update[3])*self.Nspins)
        self.Ncluster = update[4]

        # Define Update Algorithms and State Generator
        self.update_algs = {'metropolis': self.metropolis, 'wolff': self.wolff}
        self.state_gen = self.model_props['state_gen']


        
        # Initialize Plotting and plot_sites class
        self.animate = animate
        plot_range = self.model_props['state_range']

        
        site_titles = [lambda i:['Spin Configurations'+' - '+
                                 caps(self.model_props['model'])+' - '+
                                 caps(self.model_props['algorithm']),
                                 'Cluster','Edge'][i],
                       lambda i: r'$t_{MC}$: %d'%i,
                       lambda i: r'T = %0.1f'%self.T[i],
                       lambda i: 'Spin Values']
#        
#        # titles object is 4 functions, which will act based on 
#        plot_labels = []
#        i_title = lambda col,row,num,cols,rows: [[col,row,0],
#               [num,row,rows-1],
#               [row,col,0],[0,col,cols-1]]
#    plot_labels.append(choose_f(t,i_title[i_t]
#    
#    
#    
#        lambda col,row,num,cols,rows: list(map(lambda i_t,t: i_title(col,row,num,cols,rows),'')),enumerate(site_titles)))
            
    
    
    
    
        data_plot_shape = [int(np.power(self.Nspins,1/self.model_props['d']))]*self.model_props['d']
        data_process = lambda data: self.sites_region(
                                                data).reshape(data_plot_shape)
                                  
        
        self.plot_sites = Plot_Data(animate = self.animate,
                                    plot_rows = np.size(self.T),
                                    plot_cols = sum(self.animate),
                                    plot_titles = site_titles, 
                                    data_process = data_process,
                                    plot_range = plot_range,
                                    plot_type = 'sites')

        return
                    
        
    
    
    
            
            
            
            
    def MCAlg(self,algorithm='wolff',new_figure=False):
        # Perform Monte Carlo Update Algorithm and Plot Spin Sites
        
        
        self.model_props['alogorithm']=algorithm
        self.MCUpdate_alg = self.update_algs[algorithm]
        
        self.prob_update = lambda : self.model_props['prob_trans'][algorithm](
                                        self.sites,self.neighbour_sites,self.t)
        
           
        self.observables = []

        
        
        # Perform Monte Carlo at temperatures t = T
        for i_t,t in enumerate(self.T):
            
            
        # Create Observable, Cluster, and Edges Array
            observable = []
            self.clusters = []
            self.cluster_values = []
            self.cluster_edges = []
            
            
            self.t = self.T[i_t]
            
       
            # Perform Equilibration Monte Carlo steps initially
            for i_mc in range(self.Neqb):
                self.MCUpdate_alg()


            
            # Perform Measurement Monte Carlo Steps
            for i_mc in range(self.Nmeas):
    
                
                self.MCUpdate_alg()
                
                # Plot Sites Data
                if self.animate[0] and i_mc == self.Nmeas-1: # (np.size(self.T)==1)
                        
                    site_data = [self.sites,self.cluster_sites,
                                       self.cluster_sites]
                    
                    for i_a,a in reversed(list(enumerate(self.animate))): 
                        if a:
                            self.plot_sites.plot_functions['sites'](
                            site_data[i_a],[i_t, i_a, i_mc],new_figure,-1,'sites',None)
                        
                        
                # Update Observables Array every Monte Carlo Step: i_mc=nNspins
                if i_mc % self.Nmeas_f == 0:
                    observable.append(flatten(self.model_props['observables'](
                            self.sites,self.neighbour_sites,self.t)))
                    # Print a progress ticker              
                    #sys.stdout.write('.'); sys.stdout.flush();  
            
            
            
            self.observables.append(observable)
                
        
        return
     
        
    # Update Algorithms
    def metropolis(self):
        # Randomly alter random spin sites and accept spin alterations
        # if energetically favourable or probabilistically likely

        # Calculate Energy and Generate Random Spin Site
        E0 = self.prob_update()
        sites0 = np.copy(self.sites)

        isite = [np.random.randint(self.Nspins) for j in 
                  range(self.Ncluster)]
        
        self.sites[isite] = self.state_gen(self.Ncluster,sites0[isite])
        
        self.cluster_sites = isite
        self.clusters.append(isite)
        self.cluster_value = np.copy(self.sites[isite])

        # Calculate Change in Energy and decide to Accept/Reject Spin Flip
        dE = self.prob_update()-E0
        if dE > 0:
            if np.exp(-dE/self.t) < np.random.random():
                self.sites = sites0
                self.cluster_value = [np.nan]
                
        
        
        self.cluster_values.append(self.cluster_value)
        
        return
    
    
    def wolff(self):      
        # Create Cluster Array and Choose Random Site
 
        isite = np.random.randint(self.Nspins)
        self.cluster_sites = []
        self.cluster_rejections = []
        self.cluster_value = self.state_gen(self.Ncluster,
                                                self.sites[isite])
        self.cluster_value0 = np.copy(self.sites[isite])
    
        # Perform cluster algorithm to find indices in cluster
        self.cluster(isite)
        
        # Flip spins in cluster to new value
        self.sites[self.cluster_sites] = np.copy(self.cluster_value)
        
        self.clusters.append(self.cluster_sites)
        self.cluster_values.append(self.cluster_value)

        # Find Edges of Cluster
        #self.edges(self.cluster_sites)
        
        return
        
               
    
    # Cluster Functions
    def edges(self,cluster):
        for c in np.atleast_1d(cluster):
             self.cluster_edges.append([i for i in np.atleast_1d(c) if len(
                                             [j for j in np.atleast_1d(c) if j
                                    in self.neighbour_sites[0][i]]) 
                                          < len(self.neighbour_sites[0][i])])
        return
    
    def cluster(self,i):
        self.cluster_sites.append(i)
        #if len(self.cluster_sites) < int(0.95*self.Nspins):
        J = (j for j in self.neighbour_sites[0][i] if (
                                j not in self.cluster_sites) and 
                                (self.sites[j] in self.cluster_value0) and 
                                (j not in self.cluster_rejections))
        for j in J:
                if self.prob_update() > np.random.rand():
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
            return np.array(region)
    
    
    
    