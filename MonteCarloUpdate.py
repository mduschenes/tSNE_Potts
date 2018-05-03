# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 14:18:39 2018
@author: Matt
"""

import numpy as np
import warnings,copy
warnings.filterwarnings("ignore")


from Data_Process import Data_Process
from ModelFunctions import flatten,array_dict, caps, display


class MonteCarloUpdate(object):
    
    def __init__(self,sites,neighbour_sites,model_props,
                 update = [True,50,100,1,1],
                observe = {'configurations': [False,'sites','cluster'],
                           'observables': [True,'energy','order']
                           }):

        # Perform Monte Carlo Updates for nclusters of sites and Plot Data
        
        #  Monte Carlo Update Parameters: Perform Monte Carlo Update Boolean
        #                                 Number of initial updates Neqb,
        #                                 Number of Measurement Updates Nmeas
        #                                 Measurement Update Frequency Nmeas_f
        #                                 Number of Clusters Ncluster
        # State Transition Parameters:
        #                                 Transition Probability prob_transiton
        #                                 Transition Values site_states
        # Observe: [Boolean to Animate [Sites, Observables]]


#        model_props = {'L': L, 'd': d, 'q': m.q, 'T': T,
#                            'state_range': m.state_range,
#                            'state_gen': m.state_gen,
#                            'prob_trans': {'metropolis': m.energy,
#                                           'wolff':m.model_params[
#                                                   m.model.__name__][
#                                                   'bond_prob']},
#                            'model': m.model.__name__,
#                            'algorithm': 'wolff',
#                            'algorithms': ['metropolis','wolff'],
#                            'observables': m.observables_functions,
#                            'observables_props': m.observables_props,
#                            'observables_mean': m.obs_mean,
#                            'data_dir': '%s_Data/'%(caps(m.model.__name__)),
#                            'data_file': '%s_d%d_L%d__%s' %(
#                                          caps(m.model.__name__),d,L,
#                                          datetime.datetime.now().strftime(
#                                                           '%Y-%m-%d-%H-%M'))}





        # Define System Configuration
        self.sites = sites
        self.cluster = []
        self.neighbour_sites = neighbour_sites
        
        self.model_props = model_props       
        
        self.Nspins = np.size(self.sites)
        self.T = np.atleast_1d(self.model_props['T'])
        
        
        # Define Monte Carlo Update Parameters:
        # Number of updates to reach "equilibrium" before measurement, 
        # Number of measurement updates.
        self.mcupdate = update[0]
        self.Neqb = int(((1) if update[1]==None else update[1])*self.Nspins)
        self.Nmeas = int(((1) if update[2]==None else update[2])*self.Nspins)
        self.Nmeas_f = int(((1) if update[3]==None else update[3])*self.Nspins)
        self.Ncluster = update[4]

        self.update_algs = {k: getattr(self,k,lambda *x: None) 
                            for k in model_props['algorithms']}
        self.state_int = self.model_props['state_int']
        self.state_gen = self.model_props['state_gen']
        
        




        # Define Configurations and Observables Data Dictionaries
        self.data = {'observables': {k: np.array([[] for t in self.T])
                                    for k in observe['observables'][2:]}}
        
        self.data['sites'] = np.zeros((len(self.T),self.Nmeas//self.Nmeas_f,
                                          self.Nspins))   





        # Initialize Plotting
        Data_Process().plot_close()        
        
        # Define Configurations and Observations Plot Shape Keys
        # Subplots are plotted based on the key passed associated with each
        # data set.
        plot_keys = {'configurations': [[(k,t) for t in self.T]
                                        for k 
                                        in observe['configurations'][1:]],
                       
                       'observables': [[k 
                                       for k 
                                       in self.data['observables'].keys()]],
                        
                        'observables_mean': [[k 
                                       for k 
                                       in self.data['observables'].keys()]]
                    }
        
        # Define function to initialize plotting instances
        self.plot_init = lambda k,p=None: Data_Process({k:plot_keys[k]},
                                                       plot = observe[k][0] 
                                                       if p == None else p) 
                            
        # Define plotting instances and plotting functions
        self.plot_obj = {k: None for k in plot_keys.keys()}

        self.plotter = {'configurations': lambda T,A=[], *args:
                           self.plot_obj['configurations'].plotter( 
                             data = {kt: getattr(self,kt[0])
                                 for kt in flatten(plot_keys['configurations']) 
                                 if kt[1] in T},
                             plot_props = self.MCPlot_props('configurations',
                                      plot_keys['configurations'],*args),
                             data_key = 'configurations'),
        
         
                       'observables': lambda T,A=[],*args: 
                          self.plot_obj['observables'].plotter(
                            data = {k: {(a,t): 
                                self.data['observables'][ia][k][it] 
                                for it,t in enumerate(T)
                                for ia,a in enumerate(A)}
                                for k in flatten(plot_keys['observables'])},       
                            plot_props = self.MCPlot_props('observables',                                                      
                                plot_keys['observables'],*args),
                            data_key = 'observables'),
                                                           
                      'observables_mean': lambda T,A,*args: 
                          self.plot_obj['observables_mean'].plotter(
                            data = {k: {a: self.data['observables'][ia][k]
                                for ia,a in enumerate(A)}
                                for k in flatten(plot_keys['observables'])},
                            domain = {k: {a: T 
                                for ia,a in enumerate(A)}
                                for k in flatten(plot_keys['observables'])},
                            plot_props = self.MCPlot_props('observables_mean',                                                      
                                plot_keys['observables_mean'],*args),
                            data_key = 'observables_mean')
                       }


        return

        
            
            
    # Perform Monte Carlo Update Algorithm and Plot Sites and Observables            
    def MCUpdate(self,props_iter={'algorithm':'wolff'}):
        
        # Initialize props_iter as array of dictionaries
        props_iter,n_iter = array_dict(props_iter)
        
        # Initialize Plotting and Data for n_iterations
        self.plot_obj['observables'] = self.plot_init('observables')
        self.plot_obj['observables_mean'] = self.plot_init('observables_mean')

        
        for k in self.data.keys():
            self.data[k] = np.array([copy.deepcopy(self.data[k]) 
                                    for _ in range(n_iter)])
        
        
        display(True,False,'Monte Carlo Simulation... \n%s: q = %d \nT = %s'%(
                      (self.model_props['model'],self.model_props['q'],
                       str(self.T))) + '\nNeqb = %d, Nmeas = %d'%(
                                 self.Neqb/self.Nspins,self.Nmeas/self.Nspins),
                       line_break=True)
                      
                      
        
        
        # Perform Monte Carlo Algorithm for n_iter configurations        
        for i_iter in range(n_iter):
                
            # Update dictionary for each Iteration
            self.model_props.update(props_iter[i_iter])
            
            
            # Initialize sites with random spin at each site for each Iteration
            self.sites = self.state_gen(self.Nspins)
            self.cluster = []
            
            self.prob_update = self.model_props['prob_update'][
                                       self.model_props['algorithm']]
            #self.plot_obj['configurations'] = self.plot_init('configurations')
    
            display(1,0,'Iter %d: %s Algorithm'%(
                                   i_iter,caps(self.model_props['algorithm'])))
    
    
            # Perform Monte Carlo at temperatures t = T
            for i_t,t in enumerate(self.T):
           
                self.t = t
                
                # Perform Equilibration Monte Carlo steps initially
                for i_mc in range(self.Neqb):
                    self.update_algs[self.model_props['algorithm']](t)
                          
                    
                    
                # Perform Measurement Monte Carlo Steps
                for i_mc in range(self.Nmeas):

                    self.update_algs[self.model_props['algorithm']](t)                        
                     
                    # Update Configurations and Observables
                    if i_mc % self.Nmeas_f == 0:
                        self.data['sites'][i_iter,i_t,i_mc//self.Nmeas_f,:] = (
                                                           np.copy(self.sites))
                        #self.plotter['configurations']([t],[],i_mc/self.Nspins)                
              
                display(m='Updates: T = %0.2f'%t)
            
            
            
            display(m='Runtime: ',t0=-(len(self.T)+2),line_break=True)
                
                
        # Compute, Plot and Save Observables Data and Figures
        for i_iter in range(n_iter):        
            for k in self.data['observables'][i_iter].keys():
                self.data['observables'][i_iter][k] =  self.model_props[
                                                'observables'][k](
                                                self.data['sites'][i_iter],
                                                self.neighbour_sites,
                                                self.T)                                                    
            
            
            

        self.plotter['observables'](self.T,[p['algorithm']
                                            for p in props_iter])
        self.plotter['observables_mean'](self.T,[p['algorithm']
                                                  for p in props_iter])
        
        for k,obj in self.plot_obj.items():
            if obj:
                obj.plot_save(self.model_props,k)
    
        Data_Process().exporter(self.data,self.model_props)  
                                               
        
        return
     
        
    # Update Algorithms
    def metropolis(self,T=1):
        # Randomly alter random spin sites and accept spin alterations
        # if energetically favourable or probabilistically likely

        # Generate Random Spin Site and store previous Spin Value

        isite = [np.random.randint(self.Nspins) for j in 
                  range(self.Ncluster)]
        self.cluster = isite
        sites0 = self.sites[isite]
        
        # Update Spin Value
        self.sites[isite] = self.state_gen(self.Ncluster,sites0)
        
        # Calculate Change in Energy and decide to Accept/Reject Spin Flip
        nn = self.sites[self.neighbour_sites[0][isite]]
        dE = -np.sum(self.state_int(self.sites[isite],nn)) + (
              np.sum(self.state_int(sites0,nn)))
        
        if dE > 0:
            if self.prob_update[dE,T] < np.random.random():
                self.sites[isite] = sites0                
        return
    
    
    def wolff(self,T=1):      
        # Create Cluster Array and Choose Random Site
 
        isite = np.random.randint(self.Nspins)
        
        self.cluster = np.zeros(self.Nspins,dtype=bool)
        self.cluster_value0 = self.sites[isite]
    
        # Perform cluster algorithm to find indices in cluster
        self.cluster_update(isite,T)
        
        # Flip spins in cluster to new value
        self.sites[self.cluster] = self.state_gen(1,self.cluster_value0)
        return
        
               
    
    # Cluster Function
    def cluster_update(self,i,T):

        # Add indices to cluster
        self.cluster[i] = True
        J = (j for j in self.neighbour_sites[0][i] if (not self.cluster[j]) and 
                                        (self.sites[j] == self.cluster_value0)) 

        for j in J:
            if self.prob_update[T] > np.random.random():
                self.cluster_update(j,T)

        return
    

    # Function plot sites or clusters of sites
    def sites_region(self,sites0):
        if  np.array_equiv(sites0,self.sites):
            return self.sites
        else:
            region = np.zeros(np.shape(self.sites))
            region[:] = np.nan
            region[sites0] = np.copy(self.sites[sites0])
            return region
    



    # Data type dependent plot properties keys
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
                   return 'T = %0.2f'%k[1]
                
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
                               'label':'',
                               'sup_title': 'Observables Histogram - %s'%(
                                            caps(self.model_props['model'])) + 
                                            ' - q = %d'%(
                                            self.model_props['q'] + (1 if 
                                             self.model_props['model']=='ising' 
                                             else 0)),
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
            
            def plot_label(k,*args):
                return lambda k: 'T = %0.2f   %s'%(k[1],k[0])                                             
    
            
            
            
            plot_props = Plot_Props(flatten(data_keys))
            
            set_prop(plot_props,['set','title'],plot_title)
            set_prop(plot_props,['set','xlabel'],plot_xlabel,*args)
            set_prop(plot_props,['set','ylabel'],plot_ylabel)
            set_prop(plot_props,['other','label'],plot_label)
            
        
            return plot_props
        
        
        elif data_type == 'observables_mean':

        
            
            
            def Plot_Props(keys):
            
                return {
                     k: {
                    
                      'set':   {'title' : '', 
                                'xlabel': '', 
                                'ylabel': ''},
                      
                      'plot':  {},
                      
                      'data':  {'plot_type':'plot',
                                'plot_range': '',
                                'data_process':''
                               },
                                
                      'other': {'cbar_plot':True,  'cbar_title':'Spin Values',
                               'cbar_color':'bone','cbar_color_bad':'magenta',
                               'label':'',
                               'sup_title': 'Observables Plots - %s'%(
                                            caps(self.model_props['model'])) + 
                                            ' - q = %d'%(
                                            self.model_props['q'] + (1 if 
                                             self.model_props['model']=='ising' 
                                             else 0)),
                                'pause':2}
                     }
                    for k in keys}
             
            
            
           
            
     
                      
            # Set Varying Properties                  
            def set_prop(props,key,func,*args):
                  for k in props.keys():
                      props[k][key[0]][key[1]] = func(k,*args)
                  return
            
            
            def plot_title(k,*args):
                return k
                
            def plot_ylabel(k,*args):
                return ''
                
            def plot_xlabel(k,*args):
                return 'Temperature'
            
            def plot_label(k,*args):
                return lambda k: k                                           
    
            
            def data_process(k,*args):
                if k == 'order':
                    return lambda x:  np.mean(np.abs(x),axis=-1)
                else:
                    return lambda x:  np.mean(x,axis=-1)
            
            
            
            plot_props = Plot_Props(flatten(data_keys))
            
            set_prop(plot_props,['set','title'],plot_title)
            set_prop(plot_props,['set','xlabel'],plot_xlabel,*args)
            set_prop(plot_props,['set','ylabel'],plot_ylabel)
            set_prop(plot_props,['other','label'],plot_label)
            set_prop(plot_props,['data','data_process'],data_process)
            
        
            return plot_props
                     