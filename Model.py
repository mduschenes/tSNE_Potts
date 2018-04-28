# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 20:23:39 2018
@author: Matt
"""

import numpy as np

from ModelFunctions import get_attr


class Model(object):
    
    # Define a Model class for spin model with: 
    # Lattice Model Type: Model Name and Max Spin Value q
    
    def __init__(self,model=['ising',1,[0,1]],d=2,
                 observe=['temperature','energy','order']):
        # Define Models dictionary for various spin models, with
        # [ Individual spin calculation function, 
        #   Spin Upper, Lower, and Excluded Values]
        
        # Define spin value model and q (~max spin value) parameters
        # (i.e: Ising(s)-> s, Potts(s) -> exp(i2pi s/q)  )
        self.q = model[1]
        self.orderparam = model[2]
        self.d = d
        
        self.model_params = {'ising': {'value': self.ising,
                                       'energy': self.ising_energy,
                                       'order': self.ising_order,
                                       'bond_prob': 
                                           lambda s,n,T: 
                                        1- np.exp(-2*self.orderparam[1]/T),
                                       'value_range': [-self.q,self.q,0]},
                             'potts': {'value': self.potts,
                                       'energy': self.potts_energy,
                                       'order': self.potts_order,
                                       'bond_prob': 
                                           lambda s,n,T: 
                                        1- np.exp(-1*self.orderparam[1]/T),
                                       'value_range': [1,self.q,None]}} 
                                 

        # Define Model Name        
        self.model = self.model_params[model[0].lower()]['value']
        
        # Define Range of Possible Spin Values and Lookup Table of Values                        
        self.state_range = self.state_ranges()
#            self.state_values = dict(zip(self.state_range,
#                                         self.model(self.state_range)))
        
        # Define Model Energy and Order Parameter
        self.site_energy = self.model_params[self.model.__name__]['energy']
        self.site_order = self.model_params[self.model.__name__]['order']
        

        
        # Define Observables
        self.observables_functions = {k: getattr(self,k,lambda *args:[]) 
                                      for k in observe}
        
        
        self.observables_props = lambda prop,*arg: {k: get_attr(v,prop,v,*arg) 
                                 for k,v in self.observables_functions.items()}
        
        
        
        
        return
        
           
     
        
    # Generate Range of q values
    def state_ranges(self,xNone=None,xmin=float('inf'),xmax=-float('inf')):
        
        vals = self.model_params[self.model.__name__]['value_range']
        
        # Exclude xNone Values
        if xNone == None:
            xNone = np.atleast_1d([vals[2]])
            
        # List of range of possible spin values, depending on model
        return [x for x in range(min([xmin,vals[0]]),max([xmax,vals[1]+1])) 
                               if x not in xNone]




    # Generate N Random q values, exluding n0
    def state_gen(self,N=1,n0=None):
        # Return array of N random spins, per possible state_range spin values
        # excluding the possible n0 spin
        # Model dependent generator of spin values 
        
        return self.model(np.random.choice(
                                np.setdiff1d(self.state_range,n0),N))
        
        
        
#        return np.array([self.state_values[j] for j in np.random.choice(
#                                              [x for x in self.state_range 
#                                               if not (x in np.atleast_1d(n0)) 
#                                               ],N)])        
    
    
    
    
##### Model Observables #########
 
    def temperature(self,sites,neighbours,T):
        return T
    
    def sites(self,sites,neighbours,T):
        return sites

    def energy(self,sites,neighbours,T):
        # Calculate energy of spins as sum of spins 
        # + sum of r-distance neighbour interactions
        # i.e) -self.orderp[0]*sum(self.sites) + (
        #                        sum(-self.orderp[r]/2 
        #                            *sum(sum(
        #                                     n*self.sites[j] 
        #                          for n in self.neighbours(r)[j]) 
        #                          for j in range(self.Nspins)) 
        #                          for r in range(1,len(self.orderp))))
        # Calculate spin energy function S(sites):
        #        e_sites = self.m.site_energy(np.copy(self.sites))
        
        
        return (-self.orderparam[0]*np.sum(self.site_energy(sites)))+(
                    -(1/2)*np.sum([
                            self.orderparam[i]*
                            self.site_energy(sites[:,np.newaxis],
                sites[neighbours[i-1]]) 
                for i in range(1,len(self.orderparam))])) 
#        if E < -self.orderparam[1]*np.size(sites)*self.d:
#            print('Error - E < -dNJ')
#        elif E > 0:
#            print('Error - E > 0')
#        return E
    
    def order(self,sites,neighbours,T):        
        return self.site_order(sites)/np.size(sites)
    
    def specific_heat(self,sites,neighbours,T):        
      return list(map(lambda t:np.mean(list(map(lambda s: np.power(self.energy(
                                                  s,neighbours,t),2),sites))) -
                 np.mean(list(map(lambda s: self.energy(
                                                 s,neighbours,t),sites))),T))/(
                                                                np.power(T,2))
        
    
    def correlation(self,sites,neighbours,T,r = None):
        # Calculate correlation function c(r) = <s_i*s_j> 
        # for all spin pairs, where the ri-ir neighbour distance is
        #  r = {1:L/2}
        # i.e) [(1/(self.Nspins))*
        #      sum(sum(n*self.sites[j] for n in self.neighbours(rr)[j])
        #       - ((sum(self.sites)/self.Nspins)**2) 
        #       for j in range(self.Nspins)) for rr in r]
        
        Nspins = np.size(sites)
        
        
        if r is None:
            r = np.arange(np.arange(1,np.ceil(
                                np.power(Nspins,1/self.d)/2),
                                dtype=np.int64))

        return list((1/2)*((1/Nspins)**2)*np.sum(sites[:,np.newaxis]
                *sites[neighbours[r-1]],(1,2)) - (
                (1/Nspins**2)*np.sum(sites))**2)
   
            
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
    
    
    
    
    
    
    
    
   
    # Site Values                                 
    def ising(self,s):
        return s
    
    def potts(self,s):
        return s 

    # Model Energy
    def ising_energy(self,*args):
        try:
            return args[0]*args[1]
        except IndexError:
            return args[0]
    
    def potts_energy(self,*args):

        try:
            return args[0]==args[1]
        except IndexError:
            return args[0]
        

    # Model Order Parameter
    def ising_order(self,s):
        return np.sum(s)
    
    def potts_order(self,s):
        return np.real(np.sum(np.exp(2j*np.pi*s/self.q)))

    



    def dEN(self,J=1):
        dN = np.array([[0,0],
              [0,1],
              [0,2],
              [0,3],
              [0,4], 
              [1,0],       
              [1,1],       
              [1,2],       
              [1,3],       
              [2,0],
              [2,1],       
              [2,2],       
              [3,0],       
              [3,1],       
              [4,0]
             ])
        dE = -J*(dN[:,1] - dN[:,0])
        
        print(dE)
        
        
        PdE = lambda q: [(5*q-17)/(q-1),7/(q-1),3/(q-1),1/(q-1),1/(q-1),
               (3*q-3)/(q-1),4/(q-1),1/(q-1),1/(q-1),
               (2*q-5)/(q-1),2/(q-1),1/(q-1),
               (q-2)/(q-1),1/(q-1),
               1]
        
        
    
#        plt.figure()
        for q  in [2,3,4,5,6]:
            print('Ptot = %0.4f, %d'%(sum(PdE(q)),q))
#            plt.plot(dE,PdE(q),'*',label=str(q))
#        plt.legend()
        return
    
    
    
    
    