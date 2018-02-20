# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 20:23:39 2018

@author: Matt
"""

import numpy as np


def delta_f(x,y,f=np.multiply):
    return f(x,y)[x==y]


class Model(object):
    
    # Define a Model class for spin model with: 
    # Lattice Model Type: Model Name and Max Spin Value q
    
    def __init__(self,model=['ising',1],orderparam = [0,1]):
            # Define Models dictionary for various spin models, with
            # [ Individual spin calculation function, 
            #   Spin Upper, Lower, and Excluded Values]
            
            # Define spin value model and q (~max spin value) parameters
            # (i.e: Ising(s)-> s, Potts(s) -> exp(i2pi s/q)  )
            self.q = model[1]
            self.orderparam = orderparam
            
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
                                     

            self.model = self.model_params[model[0].lower()]['value']
                        
            
            # List of range of possible spin values, depending on model
            self.state_range = [x for x in 
                               range(self.model_params[self.model.__name__]['value_range'][0],
                                   self.model_params[self.model.__name__]['value_range'][1]+1) 
                               if x not in 
                               np.atleast_1d(
                                  [self.model_params[self.model.__name__]['value_range'][2]])]
            
            # Define Model Energy and Order Parameter
            self.site_energy = self.model_params[self.model.__name__]['energy']
            self.order = self.model_params[self.model.__name__]['order']

                    
        

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
            return self.model(np.random.choice(self.state_range,N))
        else:
            n0 = np.array(n0)
            return self.model(np.array([np.random.choice(
                                              [x for x in self.state_range 
                                               if x != np.atleast_1d(n0)[i]]) 
                                               for i in range(N)]))                               
    # Site Values                                 
    def ising(self,s):
        return s
    
    def potts(self,s):
        return (np.exp(2j*np.divide(np.multiply(np.pi,s),self.q)))

    # Model Energy
    def ising_energy(self,*args):
        return args[0]*args[1]
    
    def potts_energy(self,*args):
        return delta_f(args[0],args[1])

    # Model Order Parameter
    def ising_order(self,s):
        return np.sum(s)
    
    def potts_order(self,s):
        return np.abs(np.sum(s))



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