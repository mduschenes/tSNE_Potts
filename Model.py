# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 20:23:39 2018

@author: Matt
"""

import numpy as np

from ModelFunctions import delta_f,flatten




class Model(object):
    
    # Define a Model class for spin model with: 
    # Lattice Model Type: Model Name and Max Spin Value q
    
    def __init__(self,model=['ising',1],orderparam = [0,1],d=2):
            # Define Models dictionary for various spin models, with
            # [ Individual spin calculation function, 
            #   Spin Upper, Lower, and Excluded Values]
            
            # Define spin value model and q (~max spin value) parameters
            # (i.e: Ising(s)-> s, Potts(s) -> exp(i2pi s/q)  )
            self.q = model[1]
            self.orderparam = orderparam
            self.d = d
            
            self.model_params = {'ising': {'value': self.ising,
                                           'energy': self.ising_energy,
                                           'order': self.ising_order,
                                           'bond_prob': 
                                               lambda T: 
                                            1- np.exp(-2/T*self.orderparam[1]),
                                           'value_range': [-self.q,self.q,0]},
                                 'potts': {'value': self.potts,
                                           'energy': self.potts_energy,
                                           'order': self.potts_order,
                                           'bond_prob': 
                                               lambda T: 
                                            1- np.exp(-1/T*self.orderparam[1]),
                                           'value_range': [1,self.q,None]}} 
                                     

            # Define Model Name        
            self.model = self.model_params[model[0].lower()]['value']
            
            # Define Range of Possible Spin Values and Lookup Table of Values                        
            self.state_range = self.state_ranges()
            self.state_values = dict(zip(self.state_range, self.model(self.state_range)))
            
            
            # Define Model Energy and Order Parameter
            self.site_energy = self.model_params[self.model.__name__]['energy']
            self.order = self.model_params[self.model.__name__]['order']
            
            
            
            # Define Observables
            
            self.observables_functions = [self.temperature,self.energy,
                          self.order_param]
            #self.observables_data = lambda:list(map(lambda f,g= lambda x:x : g(f())
            #                                        ,self.observables_functions))
            self.observables_sizes = lambda s,n,T: [np.size(f(s,n,T)) 
                            for f in self.observables_functions]
            
            
            self.observables_data = lambda s,n,T: [f(s,n,T) 
                                          for f in self.observables_functions]
            
            
            return
            
    def state_ranges(self,xNone=None,xmin=float('inf'),xmax=-float('inf')):
        if xNone == None:
            xNone = np.atleast_1d([self.model_params[self.model.__name__][
                                                            'value_range'][2]])
        # List of range of possible spin values, depending on model
        return [x for x in range(min([xmin,self.model_params[
                                      self.model.__name__]['value_range'][0]]),
                                 max([xmax,self.model_params[
                                    self.model.__name__]['value_range'][1]+1])) 
                               if x not in xNone]

    def state_gen(self,n0=None):
        # Model dependent generator of spin values
        if n0 is None:
            return np.random.choice(self.state_range[:])
        else:
             return np.random.choice(self.state_range[:].remove(n0))
             
    def state_sites(self,N=1,n0=None):
        # Return array of N random spins, per possible state_range spin values
        # excluding the possible n0 spin
        # Model dependent generator of spin values      
        
        if np.all(n0 is None):
            return np.array([self.state_values[i] for i in np.random.choice(self.state_range,N)])
        else:
            n0 = np.array(n0)
            return np.array([self.state_values[j] for j in np.array(
                                              [np.random.choice(
                                              [x for x in self.state_range 
                                               if x != np.atleast_1d(n0)[i]]) 
                                               for i in range(N)])])        
        
#        if np.all(n0 is None):
#            return self.model(np.random.choice(self.state_range,N))
#        else:
#            n0 = np.array(n0)
#            return self.model(np.array([np.random.choice(
#                                              [x for x in self.state_range 
#                                               if x != np.atleast_1d(n0)[i]]) 
#                                               for i in range(N)]))                               
    
   
    # Site Values                                 
    def ising(self,s):
        return s
    
    def potts(self,s):
        return s #(np.exp(2j*np.divide(np.multiply(np.pi,s),self.q)))

    # Model Energy
    def ising_energy(self,*args):
        try:
            return args[0]*args[1]
        except IndexError:
            return args[0]
    
    def potts_energy(self,*args):
        try:
            return delta_f(args[0],args[1])
        except IndexError:
            return delta_f(args[0],np.zeros(np.shape(args[0])))
        

    # Model Order Parameter
    def ising_order(self,s):
        return np.sum(s)
    
    def potts_order(self,s):
        return np.abs(np.sum(s))

    
    
    
    # Model Observables
    def temperature(self,sites,neighbours,T):
        return T
    

    def energy(self,sites,neighbours,T):
        # Calculate energy of spins as sum of spins 
        # + sum of r-distance neighbour interactions
        # i.e) -self.orderp[0]*sum(self.sites) + (
#                        sum(-self.orderp[r]/2 
#                            *sum(sum(
#                                     n*self.sites[j] 
#                            for n in self.neighbours(r)[j]) 
#                            for j in range(self.Nspins)) 
#                            for r in range(1,len(self.orderp))))
        # Calculate spin energy function S(sites):
#        e_sites = self.m.site_energy(np.copy(self.sites))
        
        return (-self.orderparam[0]*np.sum(self.site_energy(sites)))+(
                    -(1/2)*np.sum([
                            self.orderparam[i]*
                            self.site_energy(sites[:,np.newaxis],
                sites[neighbours[i-1]]) 
                for i in range(1,len(self.orderparam))]))
    
    def order_param(self,sites,neighbours,T):
        return self.order(sites)
    
    def correlation(self,sites,neighbours,T,r = None):
        # Calculate correlation function c(r) = <s_i*s_j> for all spin pairs
        # where the ri-ir neighbour distance is r = {1:L/2}
        # i.e) [(1/(self.Nspins))*
#                sum(sum(n*self.sites[j] for n in self.neighbours(rr)[j])
#                - ((sum(self.sites)/self.Nspins)**2) 
#                    for j in range(self.Nspins)) for rr in r]
        
        Nspins = np.size(sites)
        
        
        if r is None:
            r = np.arange(np.arange(1,np.ceil(np.power(Nspins,1/self.d)/2),
                                                               dtype=np.int64))

        return list((1/2)*((1/Nspins)**2)*np.sum(sites[:,np.newaxis]
                *sites[neighbours[r-1]],(1,2)) - (
                (1/Nspins**2)*np.sum(sites))**2)
#        
            
    def Tcrit(self,d):
        # Declare the critical Ising Model Temperature in d-dimensions
        if d >= 4:
            Tc = self.orderparam[1]
        elif d == 1:
            Tc = 0
        elif d == 2:
            Tc = 2.0/np.log(1.0 + np.sqrt(2))*self.orderparam[1]
        else: # self.d == 3:
            Tc = None
        return Tc



#            
#            sr = np.copy(self.state_range) 
#            n = empty
#            for n00 in n0:
#                np.remove(sr,n00)
#            np.remove(sr,)
#            n0 = np.array(n0)
#            n_n0 = n==n0
#            print(n_n0)
#            while np.all(n_n0):
#                print(n_n0)
#                n[n_n0] = np.random.choice(self.state_range,n[n_n0].size)
#                n_n0 = n==n0
#            return n
##        n0 = n0 if isinstance(n0,list) or isinstance(n0,np.ndarray) else [n0]
#        try:
#             return np.array([self.model(
#                          np.random.choice(self.state_range[:].remove(n0[i])))
#                             for i in range(N)])
#        except ValueError:
#             return np.array([self.model(np.random.choice(self.state_range))
#                              for i in range(N)])
        #return np.array([self.model(self.state_gen(n0)) for i in range(N)])