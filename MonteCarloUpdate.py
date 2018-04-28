# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 14:18:39 2018
@author: Matt
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from Data_Process import Data_Process
from ModelFunctions import flatten, caps, display


class MonteCarloUpdate(object):
    
    def __init__(self,sites,neighbour_sites,model_props,
                 update = [True,20,10,1,1/5],
                 observe = {'configurations': [True,'sites','cluster'],
                            'observables': [True,'temperature','energy','order'
                                            ]}):

        # Perform Monte Carlo Updates for nclusters of sites and Plot Data
        
        #  Monte Carlo Update Parameters: Perform Monte Carlo Update Boolean
        #                                 Number of initial updates Neqb,
        #                                 Number of Measurement Updates Nmeas
        #                                 Measurement Update Frequency Nmeas_f
        #                                 Number of Clusters Ncluster
        # State Transition Parameters:
        #                                 Transition Probability prob_transiton
        #                                 Transition Values site_states
        # Observe: [Boolean to Animate [Sites, Cluster, Edges] [Observables]]

        # Define System Configuration
        self.sites = sites
        self.cluster = []
        self.neighbour_sites = neighbour_sites
        
        self.model_props = model_props
        self.Nspins = np.size(self.sites)
        
        self.T = np.atleast_1d(self.model_props['T']).tolist()
        
        
        # Define Monte Carlo Update Parameters:
        # Number of updates to reach "equilibrium" before measurement, 
        # Number of measurement updates.
        self.mcupdate = update[0]
        self.Neqb = int(((1) if update[1]==None else update[1])*self.Nspins)
        self.Nmeas = int(((1) if update[2]==None else update[2])*self.Nspins)
        self.Nmeas_f = int(((1) if update[3]==None else update[3])*self.Nspins)
        self.Ncluster = update[4]


        self.prob_update = lambda t: self.model_props['prob_trans'][
                                       self.model_props['algorithm']](
                                        self.sites,self.neighbour_sites,t)


        self.update_algs = {k: getattr(self,k,lambda *x: None) 
                            for k in model_props['algorithms']}
         
        
        
        # Define Configurations and Observations Keys
        
        data_keys = {'configurations':[[(k,t) for t in self.T]
                                       for k in observe['configurations'][1:]],
                       
                       'observables' :[[k for k in observe['observables'][1:]]]
                    }


        # Define Configurations and Observables Dictionaries

        


        self.data = {'configurations': {kt:[] 
                               for kt in flatten(data_keys['configurations'])},
        
                    'observables': {k:{(a,t): [] 
                                    for t in self.T
                                    for a in self.model_props['algorithms']}
                                    for k in flatten(data_keys['observables'])}
                    }
    
        self.data_mean = lambda: {'observables': {k:{a: [np.mean(self.data[
                                                      'observables'][k][(a,t)]) 
                                                for t in self.T]
                                    for a in self.model_props['algorithms']}
                                    for k in flatten(data_keys['observables'])}
                         }   
        
        self.data_func = {'configurations': lambda t: [
                                         self.data['configurations'][k].append(
                                         getattr(self,k[0]))
                                  for k in flatten(data_keys['configurations']) 
                                  if k[1]==t],
        
                         'observables': lambda sites,t: [
                                     self.data['observables'][k][
                                     (self.model_props['algorithm'],t)].append(
                                     self.model_props['observables'][k](
                                     sites,self.neighbour_sites,t)) 
                                    for k in flatten(data_keys['observables'])]
                         }



        # Define Animation
        Data_Process().plot_close()
        
        self.plot_init = lambda k,p=None: Data_Process(data_keys[k],
                                                       plot = observe[k][0] 
                                                       if p == None else p) 
                            

        self.plot_obj = {k: None for k in data_keys.keys()}

        self.plotter = {'configurations': lambda T,*args: self.plot_obj[
                                                'configurations'].plotter(
                             data = {kt: self.data['configurations'][kt][-1]
                                 for kt in flatten(data_keys['configurations']) 
                                 if kt[1] in T},       
                             plot_props = self.MCPlot_props('configurations',
                                                   data_keys['configurations'],
                                                   *args)),
        
        
        
                    'observables': lambda T,*args: self.plot_obj[
                                                    'observables'].plotter(
                                              data = {k: {(self.model_props[
                                                    'algorithm'],t): self.data[
                                                    'observables'][k][
                                                    (self.model_props[
                                                            'algorithm'],t)] 
                                                    for t in T} 
                                                    for k in flatten(data_keys[
                                                            'observables'])},       
                                              plot_props = self.MCPlot_props(
                                                      'observables',
                                                      data_keys['observables'],
                                                      *args))
                    }


        return

        
            
            
    # Perform Monte Carlo Update Algorithm and Plot Sites and Observables            
    def MCAlg(self,algorithm='wolff',n_iter=1):
        algorithm = np.atleast_1d(algorithm)
        
        n_alg = np.size(algorithm)
        if n_iter == 1 and n_alg > 1:
            n_iter = n_alg
    
        self.plot_obj['observables'] = self.plot_init('observables')
        
        # Perform Monte Carlo Algorithm for n_iter configurations
        display(True,False,'Monte Carlo Simulation... %s:\nT = %s\nalg = %s'%(
                      (self.model_props['model'],str(self.T),str(algorithm))))
        
        for i in range(n_iter):
                
            
            # Initialize sites with random spin at each site for each Iteration
            self.sites = self.model_props['state_gen'](self.Nspins)
            self.cluster = []
            self.plot_obj['configurations'] = self.plot_init('configurations')
            
            alg = algorithm[i%n_alg]
            self.model_props['algorithm'] = alg
    
            display(True,False,caps(alg)+' Algorithm')
    
            # Perform Monte Carlo at temperatures t = T
            for i_t,t in enumerate(self.T):
           
                self.t = t
                display(True,True,'Updates: T = %0.1f'%t)
                
                # Perform Equilibration Monte Carlo steps initially
                for i_mc in range(self.Neqb):
                    self.update_algs[alg](t)
                                
                # Perform Measurement Monte Carlo Steps
                for i_mc in range(self.Nmeas):

                    self.update_algs[self.model_props['algorithm']](t)                        
                     
                    # Update Configurations and Observables
                    if i_mc % self.Nmeas_f == 0:
                        self.data_func['configurations'](t)
                        self.plotter['configurations']([t],i_mc/self.Nspins)
                
                self.data_func['observables'](self.sites,t)
                self.plotter['observables']([t])
                
                # Save Observables Data 
                for k,obj in self.plot_obj.items():
                    obj.plot_save(self.model_props,k)
            
                Data_Process().exporter(self.data,self.model_props)  
                
                
            display(True,True,'%s runtime: '%(self.model_props['algorithm']))                

            
            
        self.data_mean = lambda: {'observables': {k:{a: [np.mean(self.data[k][(t,a)]) 
                                               for t in self.T]
                                   for a in self.model_props['algorithms']}
                                   for k in ['temperature','energy','order']}
                                }
                                                
        
        return
     
        
    # Update Algorithms
    def metropolis(self,T=1):
        # Randomly alter random spin sites and accept spin alterations
        # if energetically favourable or probabilistically likely

        # Calculate Energy and Generate Random Spin Site
        E0 = self.prob_update(T)

        isite = [np.random.randint(self.Nspins) for j in 
                  range(self.Ncluster)]
        
        sites0 = self.sites[isite]
        
        self.sites[isite] = self.model_props['state_gen'](self.Ncluster,
                                                          sites0)
        
        self.cluster = isite

        # Calculate Change in Energy and decide to Accept/Reject Spin Flip
        dE = self.prob_update(T)-E0
                
        if dE > 0:
            if np.exp(-dE/T) < np.random.random():
                self.sites[isite] = sites0                
        return
    
    
    def wolff(self,T=1):      
        # Create Cluster Array and Choose Random Site
 
        isite = np.random.randint(self.Nspins)
        self.cluster = []
        self.cluster_rejections = []
        self.cluster_value = self.model_props['state_gen'](self.Ncluster,
                                                           self.sites[isite])
        self.cluster_value0 = np.copy(self.sites[isite])
    
        # Perform cluster algorithm to find indices in cluster
        self.cluster_update(isite,T)
        
        
        #print(self.cluster)
        
        # Flip spins in cluster to new value
        self.sites[self.cluster] = np.copy(self.cluster_value)
        return
        
               
    
    # Cluster Functions
    def edges(self,cluster):
        for c in np.atleast_1d(cluster):
             self.cluster_edges.append([i for i in np.atleast_1d(c) if len(
                                             [j for j in np.atleast_1d(c) if j
                                    in self.neighbour_sites[0][i]]) 
                                          < len(self.neighbour_sites[0][i])])
        return
    
    def cluster_update(self,i,T):
        self.cluster.append(i)
        #if len(self.cluster_sites) < int(0.95*self.Nspins):
        J = (j for j in self.neighbour_sites[0][i] if (
                                j not in self.cluster) and 
                                (self.sites[j] in self.cluster_value0) and 
                                (j not in self.cluster_rejections))
        for j in J:
                if self.prob_update(T) > np.random.rand():
                        self.cluster_update(j,T)
                else:
                    self.cluster_rejections.append(j)
        return
    
    

    def sites_region(self,sites0):
        if  np.array_equiv(sites0,self.sites):
            return self.sites
        else:
            region = np.zeros(np.shape(self.sites))
            region[:] = np.nan
            region[sites0] = np.copy(self.sites[sites0])
            return region
    




    def MCPlot_props(self,data_type,data_keys,*args):
        
        if data_type == 'configurations':
            
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
                                'label': '','pause':2,
                                'sup_title': 'Monte Carlo Updates' + ' - '+ 
                                            caps(self.model_props['model'])+
                                            ' - '+
                                            caps(self.model_props['algorithm'])
                                }
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
                
                else:
                   return 'T = %0.1f'%k[1]
                
            def plot_ylabel(k,*args):
                if k[1] != data_keys[0][0][1]:
                    return plot_props_sites['ylabel']
                else:
                    return k[0]
                           
                
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
                
                return lambda d: np.reshape(self.sites_region(d),
                                            data_plot_shape)
            
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
        

        elif data_type == 'observables':

        
            
            
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
                               'label':lambda *args: 'T = %0.1f   %s'%(self.t,
                                                self.model_props['algorithm']),
                               'sup_title': 'Observables Histogram - '+
                                                self.model_props['model'],
                                'pause':2}
                     }
                    for k in keys}
             
            
            
           
            
     
                      
            # Set Varying Properties                  
            def set_prop(props,key,func,*args):
                  for k in props.keys():
                      props[k][key[0]][key[1]] = func(k,*args)
                  return
            
            
            def plot_title(k,*args):
                return ''
                
            def plot_ylabel(k,*args):
                if k != data_keys[0][0]:
                    return ''
                else:
                    return 'Counts'
                
            def plot_xlabel(k,*args):
                return caps(k)
    
            
            
            
            plot_props = Plot_Props(flatten(data_keys))
            
            set_prop(plot_props,['set','title'],plot_title)
            set_prop(plot_props,['set','xlabel'],plot_xlabel,*args)
            set_prop(plot_props,['set','ylabel'],plot_ylabel)
            
        
            return plot_props
                     