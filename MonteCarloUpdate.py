# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 14:18:39 2018
@author: Matt
"""

import numpy as np

from Data_Plot import Data_Plot
from ModelFunctions import flatten, caps, display


class MonteCarloUpdate(object):
    
    def __init__(self,sites,neighbour_sites,model_props,update):
        # Perform Monte Carlo Updates for nclusters of sites and Plot Data
        
        #  Monte Carlo Update Parameters: Perform Monte Carlo Update Boolean
        #                                 Number of initial updates Neqb,
        #                                 Number of Measurement Updates Nmeas
        #                                 Measurement Update Frequency Nmeas_f
        #                                 Number of Clusters Ncluster
        # State Transition Parameters:
        #                                 Transition Probability prob_transiton
        #                                 Transition Values site_states
        
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
        
        # Define Animation
        Data_Plot().plot_close()

        return

        
            
            
            
    def MCAlg(self,algorithm='wolff',data_keys={}):
        # Perform Monte Carlo Update Algorithm and Plot Spin Sites
        # Observe: [Boolean to Animate [Sites, Cluster, Edges] [Observables]]

        self.model_props['algorithm']=algorithm
        self.MCUpdate_alg = self.update_algs[algorithm]
        
        self.prob_update = lambda : self.model_props['prob_trans'][algorithm](
                                        self.sites,self.neighbour_sites,self.t)
        
        
        display(True,True,'Monte Carlo Simulation... T = %s'%str(self.T))



        # Create Sites and Observables Dictionaries
        observables = {t:{k:[] for k in flatten(data_keys['observables'])}
                            for t in self.T}
        observables_func = {k: self.model_props['observables'][k[1]] 
                                for k in flatten(data_keys['observables'])}
        
        configurations_func = {k: lambda: f for k,f in zip(set([k[0] for k in flatten(data_keys['obs'])]),
                                                                  [self.sites,self.cluster_sites])} 
                                 

        configurations = {t:{k:[] for k in flatten(data_keys['configurations'])}
                            for t in self.T}
       
        # Initialize Plotting
        self.plot_sites = Data_Plot(self.keys_obs,plot=self.observe[1][0])


        # Perform Monte Carlo at temperatures t = T
        for i_t,t in enumerate(self.T):
            
            self.clusters = []
            self.cluster_values = []
            self.cluster_edges = []
                   
            
            self.t = t
            
       
            # Perform Equilibration Monte Carlo steps initially
            for i_mc in range(self.Neqb):
                self.MCUpdate_alg()


            
            # Perform Measurement Monte Carlo Steps
            for i_mc in range(self.Nmeas):
    
                
                self.MCUpdate_alg()
                
                # Plot Sites Data
                if i_mc == self.Nmeas-1: # (np.size(self.T)==1)
                        
                    
                    for k in flatten(data_keys['configurations']:
                        self.configurations[t][k].append()
                    data_sites = {k:d for k,d in 
                                 zip([k for k in flatten(data_keys['configurations']) 
                                      if k[0]==t],
                                 [self.sites,self.cluster_sites])
                                 }                                 
                    
                    
                    
                    
                    # Update plot_props_obs and Plot data_obs

                    self.plot_sites.plotter(data = data_sites,
                                            plot_props = self.MCPlot_sites(
                                                     data_keys['configurations'],i_mc))
                        
                    
                    
                    
                # Update Observables Array every Monte Carlo Step: i_mc=nNspins
                if i_mc % self.Nmeas_f == 0:
                    
                    
                    
                    for k in flatten(data_keys['obs']):
                        self.observables[k][t].append(
                              (self.sites,
                                                          self.neighbour_sites,
                                                                       self.t))
                    
                    # Print a progress ticker              
                    #sys.stdout.write('.'); sys.stdout.flush();  
            
            
    
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
    








    def MCPlot_sites(self,data_keys,*args):
        
        
        def Plot_Props(keys):
        
            return {
                 k: {
                
                  'set':   {'title' : '', 
                            'xlabel': '', 
                            'ylabel': ''},
                  
                  'plot':  {},
                  
                  'data':  {'plot_type':'image',
                            'plot_range': '',
                            'data_process':lambda data: np.real(data)},
                            
                  'other': {'cbar_plot':False, 'cbar_title':'Spin Values',
                           'cbar_color':'bone','cbar_color_bad':'magenta',
                            'label': '','pause':0.02}
                 }
                for k in keys}
                  
                  
        # Set Varying Properties                  
        def set_prop(props,key,func,*args):
              for k in props.keys():
                  props[k][key[0]][key[1]] = func(k,*args)
              return
         
            
        # Properties Specific to Plot
        plot_props_sites = {'title': '', 'xlabel': '', 'ylabel': ''}
            
            
        def plot_title(k,*args):
            if k[0] != data_keys[0][0][0]:
                return plot_props_sites['title']
            
            elif k[1] == data_keys[0][0][1]:
                return (k[1] +' - '+
                       caps(self.model_props['model'])+' - '+
                       caps(self.model_props['algorithm']))
            else:
                return  k[1] # Clusters or Edges
            
        def plot_ylabel(k,*args):
            if k[1] != data_keys[0][0][1]:
                return plot_props_sites['ylabel']
            else:
                return r'T = %0.1f'%k[0] # Temperature
            
        def plot_xlabel(k,*args):
            if k[0] != data_keys[-1][-1][0]:
                return plot_props_sites['xlabel']
            else:
                return r'$t_{MC}$: %d'%args[0] 
        
        def cbar_plot(k,*args):
            if k[1] == data_keys[0][-1][1]:
                return True
            else:
                return False 

        def data_process(k,*args):
            
            data_plot_shape = [int(np.power(self.Nspins,
                          1/self.model_props['d']))]*self.model_props['d']
            
            return lambda d: self.sites_region(d).reshape(data_plot_shape)
        
        def plot_range(k,*args):
            return np.append(self.model_props['state_range'],
                             self.model_props['state_range'][-1]+1)    
        
        
        
    
        plot_props = Plot_Props(flatten(data_keys))
        
        set_prop(plot_props,['set','title'],plot_title)
        set_prop(plot_props,['set','xlabel'],plot_xlabel,*args)
        set_prop(plot_props,['set','ylabel'],plot_ylabel)
        set_prop(plot_props,['data','data_process'],data_process)
        set_prop(plot_props,['data','plot_range'],plot_range)
        set_prop(plot_props,['other','cbar_plot'],cbar_plot)
            
                
        return plot_props
    



    def MCPlot_obs(self,data_keys,*args):
        
        
        def Plot_Props(keys):
        
            return {
                 k: {
                
                  'set':   {'title' : '', 
                            'xlabel': '', 
                            'ylabel': ''},
                  
                  'plot':  {'stacked':True, 'fill':True, 'alpha': 0.35, 
                            'histtype':'bar'},
                  
                  'data':  {'plot_type':'histogram',
                            'plot_range': '',
                            'data_process':lambda data: np.real(data)},
                            
                  'other': {'cbar_plot':True,  'cbar_title':'Spin Values',
                           'cbar_color':'bone','cbar_color_bad':'magenta',
                           'label': caps(k[0]) + ', T = %0.1f'%self.t,
                            'pause':0.02}
                 }
                for k in keys}
         
        
        plot_props_obs = {'title': 'Observables',
                           'xlabel': '',
                            'ylabel': 'counts'}
        
       
        
 
                  
        # Set Varying Properties                  
        def set_prop(props,key,func,*args):
              for k in props.keys():
                  props[k][key[0]][key[1]] = func(k,*args)
              return
        
        
        
        def plot_title(k,*args):
            if k[0] != data_keys[0][0][0]:
                return ''
            
            elif k[1] == data_keys[0][0][1]:
                return (k[1] +' - '+
                       caps(self.model_props['model'])+' - '+
                       caps(k[0]))
            else:
                return  k[1] # Clusters or Edges
            
        def plot_ylabel(k,*args):
            if k[1] != data_keys[0][0][1]:
                return ''
            else:
                return plot_props_obs['ylabel']
            
        def plot_xlabel(k,*args):
            if k[0] != data_keys[-1][-1][0]:
                return ''
            else:
                return caps(k[1])

        
        
        
        plot_props = Plot_Props(flatten(data_keys))
        
        set_prop(plot_props,['set','title'],plot_title)
        set_prop(plot_props,['set','xlabel'],plot_xlabel,*args)
        set_prop(plot_props,['set','ylabel'],plot_ylabel)
        
    
        return plot_props
                     