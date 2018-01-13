# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 22:56:21 2017

@author: Matt
"""
import numpy as np
import sympy as sp
import scipy as sc
from inspect import signature
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import time
import datetime
import os

def listmultichange(x,i,a=lambda b:b):
    # Return a list x after changing all jth elements by a function a[n](x[j])
    # for j indicies in list ielements 
    # CURRENTLY NOT WORKING PROPERLY
            
            if isinstance(a,list):
                for ia,aa in enumerate(a):
                    if not callable(aa):
                        a[ia] = lambda b: aa
            else:
                if not callable(a):
                    a = lambda b: a
                a = [a]
            
            if not isinstance(i,list):
                i = [i]
            print(a,i)
            
            print(type(a))
            
            for n,j in enumerate(i):
                print(n,j)
                print(a[0])
                print(type(a[n]))
                x[j] = a[0](x[j])
            return x

def listchange(x,i,a=lambda b:b):
    # Return a list x after changing its ith element by a function a(x[i])
    # CURRENTLY only accepts function arguements for a 
    # (desire constant arguments that are converted to lamba x: a functions)
    
#    if not callable(a):
#        print('a is not a function')
#        a = lambda c: a
#    
#    print(a(1))
##    
##    f = lambda y:a    
##    print(a(0))
##    print(x[i])
##    print(f(x[i]))
    x[i] = a(x[i])
    return x

def flatten(x):
    # Return a 1d list of all elements in inner lists of x of arbirtrary shape
    if  not isinstance(x,list):
        return x
    elif len(x) == 1:
        return x[0]
    xlist = []
    for y in x:
        if isinstance(y,type([])):
            xlist.extend(flatten(y))
        else: xlist.append(y)
    return xlist
    
def fappend(x,F):
   for f in F:
       if not callable(f):
           f = lambda : f
       #print(f())
       ft = f()
       if not((isinstance(ft,np.array)) or (isinstance(ft,list))):
           x = np.append(x,ft)
       else:
           for j in ft:
               x = np.append(x,j)
   return x

def realimag(x,tol=1e-14):
    # Return entirely real or imaginary components of complex x, 
    # if the conjugate component is zero, or below a tolerance
    r = x.real
    i = x.imag
    
    if i == 0:
        return x
    
    if abs(r) <= tol:
        return i
    elif abs(i)<=tol:
        return r
    else:
        return x
        
def index(x,i):
    # Return array of elements in list x, specficied in indicices in list i
    return [x[j] for j in i] if isinstance(i,list) else x[i]
    
def listindex(x,i):
    #print(x)
    if not isinstance(x,list):
        return x
    if not isinstance(i,list):
        return x[i]
    elif len(i) == 1:
        try:
            return [ y if not isinstance(y,list) 
                else (y[i[0]]) if len(y)>= i[0] 
                else y[-1] for y in x]
        except IndexError: 
            return [y for y in x]
        
    else:    
        return listindex([y if not isinstance(y,list) 
                else (y[i[0]]) if len(y)>= i[0]+1 
                else y[-1]  for y in x]
                ,i[1:])
    
class system(object):
    # Define system class for general model of lattice sites with
    #  Lattice Parameters: Lattice Length L, Lattice Dimension d, Temperature T
    #  Lattice Model Type: Model Name and Max Spin Value q
    #  Order Parameters: Hamiltonian Coupling Constants, 
    #                    Order Parameter Function
    #  Monte Carlo Update Parameters: Number of initial updates Neqb,
    #                                 Number of Measurement Updtes Nmeas
    #                                 Monte Carlo Update Algorithm
    #                                 Monte Carlo Plot Results
    # DataSave: Save Boolean
    def __init__(self,L=6,d=2,T=3,
                 model=['ising',1],
                 orderparam=[[0,1],lambda x: sum(x)],
                 update = [None,None,'metropolis',False],
                 datasave = True):
        # Time runtime of system class
        t0 = time.clock() 
        
        # Initialize model class, lattice class
        self.m = Model(model)
        t1 = time.clock()
        self.l = Lattice0(L,d)
        t2 = time.clock()
        # Define system parameters of:
        # Temperature, Size, Dimension, Number of Spins, 
        # Maximum spin q value and specific spin model
        self.T = T
        self.Tlist = [5,4.5,4,3.5,3,2.5,2,1.5,1,0.5]
        self.L = L
        self.d = d
        self.Nspins = self.l.Nspins
        self.q = self.m.q
        self.model = self.m.model

        
        # Define order hamiltonian parameters 
        # and function to calculate order parameter
        self.orderp = orderparam[0]
        if self.model[0] == 'ising':
            self.h = self.orderp[0]
            self.J = self.orderp[1]
            self.Tcrit()
        self.orderf = orderparam[1]
        
        
        # Initialize random spin at each site
        self.sites = self.m.sigma_sites(self.Nspins)
        #self.nn1 = self.neighbours(1)
        
        # Initilize list of observations values
        self.observables = []
        self.observables_functions = [self.temperature,self.energy,
                                      self.order,self.correlation]
        
        # Save observations data
        self.save = datasave
        self.datasave()
        
        # Define Monte Carlo Update:
        # Number of updates to reach "equilibrium" before measurement, 
        # Number of measurement updates.
        # Monte Carlo update alogrithms
        self.Neqb = 0*L**(d-1) if update[0]==None else update[0]
        self.Nmeas = L**(d) if update[1]==None else update[1]

        self.updatealgs = {'metropolis': self.metropolis, 'wolff': self.wolff}
        self.algorithm = update[2]
        self.animate = update[3]
        self.MCUpdatealg = self.updatealgs[self.algorithm]
        
        # Perform Monte Carlo Updates for various Temperatures
        t3 = time.clock()
        for t in self.Tlist:
            self.T = t
            self.MCUpdate()
            print('T = ',str(t))
            print(time.clock()-t3)
            t3 = time.clock()
            print('')
        self.datasave()
        # print(self.correlation())
        # Calculate total runtime for system class
        print('system class runtime: '+str(time.clock()-t0)) # -self.Nmeas*1.5
        print('model class runtime: '+str(t1-t0)) # -self.Nmeas*1.5
        print('lattice class runtime: '+str(t2-t1)) # -self.Nmeas*1.5
        plt.pause(3)
        
    def MCUpdate(self,ncluster = 1):
        # Perform Monte Carlo Updates for nclusters of sites and Plot Data
        bw_cmap = colors.ListedColormap(['black', 'white'])
        
        for i in range(self.Neqb):
            self.MCUpdatealg(ncluster)
            
        for i in range(self.Nmeas):
            
            # Perform Monte Carlo Update Algorithm and Plot Spin Sites
            #tmc = time.clock()
            self.MCUpdatealg(ncluster)
            #print(time.clock()-tmc)
            if i % self.Nmeas/self.L == 0 or True:
                self.observables.append([
                                      f() for f in self.observables_functions])
                        
            if self.animate and i == self.Nmeas-1:
                plt.clf()
                plt.imshow(np.real(self.sites.reshape((self.L,self.L))), 
                           cmap=bw_cmap, norm=colors.BoundaryNorm([-1,0,1], 
                           bw_cmap.N),interpolation='nearest')
                plt.xticks([])
                plt.yticks([])
                plt.title('%d^%d %s model, m = %f ---- q = %d, T = %.3f' 
                          %(self.L,self.d,
                            self.model[0][0].upper()+self.model[0][1:].lower(),
                            self.order()/self.Nspins,
                            #self.correlations[-1][0],
                            self.q,
                            self.T))
                plt.pause(0.5)
        # Plot Corellation length
        #plt.plot()
            
            
    def metropolis(self,ncluster=1):
        # Randomly alter random spin sites and accept spin alterations
        # if energetically favourable or probabilistically likely
        for i in range(self.Nspins):
            E0 = self.energy()
            sites0 = np.copy(self.sites)
            
            isites = [np.random.randint(self.Nspins) for j in range(ncluster)]
            
            
            for isite in isites:
                #print(self.sites[isite])
                self.sites[isite] = self.m.sigma_sites(1,sites0[isite])
                #print(self.sites[isite])
                #print(sites0[isite])
            dE = self.energy()-E0
            ranp = np.random.random()
            if dE > 0:
                if np.exp(-dE/self.T) < ranp:
                    #print('change sites back')
                    for isite in isites:
                        #print(self.sites[isite])
                        self.sites[isite] = sites0[isite]
                        #print(self.sites[isite])
            #print(self.sites[isite])
            #print(dE)
            #for isite in isites:
                #print(self.sites[isite]==sites0[isite])
            #print('')
            return
            
        
    def wolff(self,ncluster=1):
        return
                
    def neighbours(self,r=1):
        # Return spins of r-distance neighbours for all spin sites
        return np.array([index(self.sites,self.l.neighbour_sites[r-1][i]) 
                                            for i in range(self.Nspins)])
        
    def temperature(self):
        return self.T
    
    def order(self):
        # Calculate order parameter with orderf function
        return realimag(self.orderf(self.sites))
    
    def energy(self):
        # Calculate energy of spins as sum of spins 
        # + sum of r-distance neighbour interactions
        return realimag(-self.orderp[0]*sum(self.sites) +
                        sum(-self.orderp[r]/2 
                            *sum(sum(
                                     n*self.sites[j] 
                            for n in self.neighbours(r)[j]) 
                            for j in range(self.Nspins)) 
                            for r in range(1,len(self.orderp))))
    
    def correlation(self, r = None):
        # Calculate correlation function c(r) = <s_i*s_j> for all spin pairs
        # where the ri-ir neighbour distance is r = {1:L/2}
        if r == None:
            r = [x for x in range(1,int(np.ceil(self.L/2)))]    
        elif not isinstance(r) != list:
            r = [r]
        return [(1/(self.Nspins))*
                sum(sum(n*self.sites[j] for n in self.neighbours(rr)[j])
                - ((sum(self.sites)/self.Nspins)**2) 
                    for j in range(self.Nspins)) for rr in r]
            
    def Tcrit(self):
        # Declare the critical Ising Model Temperature in d-dimensions
        if self.d >= 4:
            self.Tc = self.J
        elif self.d == 1:
            self.Tc = 0
        elif self.d == 2:
            self.Tc = 2.0/np.log(1.0 + np.sqrt(2))*self.J
        else: # self.d == 3:
            self.Tc = None
    
    def datasave(self):
        if self.save == True:
            # Open and write observables to a file in a specific directory
            dataDir = '%s_Data' %(
                            self.model[0][0].upper()+self.model[0][1:].lower())
            if self.observables == []:
                if not(os.path.isdir(dataDir)):
                        os.mkdir(dataDir)
                return
            
            dataName         = '%s/%s_d%d_L%d__%s.txt' %(
                                    dataDir,
                                    self.model[0][0].upper()+
                                    self.model[0][1:].lower(),
                                    self.d,self.L,
                                    datetime.datetime.now().strftime(
                                                            '%Y-%m-%d-%H-%M'))
            # Write headers to file, based on observable functions
            file = open(dataName, 'w')
            headers = []
            for i,f in enumerate(self.observables_functions):
                headernum = max([len(obs if isinstance(obs,list) else [obs])
                                 for obs in listindex(self.observables,[i])])
                for j in range(headernum):
                    headers.append(f.__name__+'_'+str(j+1) if headernum > 1
                                   else f.__name__)
            file.write('\t'.join(headers) + '\n')
            
            
            
            # Write data to file, one line of observables per temperature
            for data in self.observables:
                dataline = ''
                for d in flatten(data):            
                        dataline += '%0.8f \t' %(float(d))
                dataline += '\n'
                file.write(dataline)
            
            file.close()
        return
    
class Model(object):
    
    # Define a Model class for spin model with: 
    # Lattice Model Type: Model Name and Max Spin Value q
    
    def __init__(self,model=['ising',1]):
        # Define Models dictionary for various spin models, with
        # [ Individual spin calculation function, 
        #   Random generator function of possible spin values ]
        self.models = {'ising': [self.ising,lambda n: np.random.choice(
                                        [x for x in range(-n,n+1) if x != 0])],
                       'potts': [self.potts,lambda n: np.random.randint(1,n+1)]
                      }
        # Define model and q (~max spin value) parameters
        self.model = model
        self.model[0] = self.model[0].lower()
                      
        self.q = model[1]
        
        # Define spin value calculation function 
        # (i.e: Ising(s)-> s, Potts(s) -> exp(i2pi s/q)  )
        self.sigma = self.models[self.model[0]][0]
        self.state_gen = self.models[self.model[0]][1]
        
                  
    def ising(self,s):
        return s 
    
    def potts(self,s):
        return (np.exp(2j*np.divide(np.multiply(np.pi,s),self.q)))
        
#    def sigma(self,s):
#        return self.site_state(s)
    
    def sigma_sites(self,N,s0=None):
        # Return array of N random spins, as per possible state_gen spin values
        # excluding the possible s0 spin
        def rands0():
            rands = self.sigma(self.state_gen(self.q))
            while rands == s0:
                rands = self.sigma(self.state_gen(self.q))
            return rands
        
        if N == 1:
            return rands0()
        else:
            return np.array([rands0() for i in range(N)])
            
class Lattice0(object):
    
    # Define a Lattice class for lattice sites configurations with:
    # Lattice Length L, Lattice Dimension d
    
    def __init__(self,L=6,d=2):
        # Define parameters of system        
        self.L = L
        self.d = d
        self.Nspins = L**d
        
        # Calculate array of lists of r-distance neighbours,
        # for each site, for r = 1: L/2 (Time-intensive for large L)
        self.neighbour_sites = np.array([[self.neighboursites(i,r) 
                                         for i in range(self.Nspins)]
                                         for r in range(1,int(np.ceil(self.L/2)))])
        #print(self.neighbour_sites)
        
    def position(self,site):
        # Return position coordinates in d-dimensional L^d lattice 
        # from given linear site position in 1d Nspins^2 length array
        return [int(site/(self.L**(i))) % self.L for i in range(self.d)]
    
    def site(self,position):
        # Return linear site position in 1d Nspins^2 length array 
        # from given position coordinates in d-dimensional L^d lattice 
        return sum(position[i]*self.L**i for i in range(self.d))
    
    def neighboursites(self,site,r=1):
        # Return array of neighbour spins 
        # for a given site and r-distance neighbours
        return np.array([self.site(
                         listchange(self.position(site),i,
                         lambda x: np.mod(x + p*r,self.L))) 
                         for i in range(self.d)for p in [1,-1]])
    

if __name__ == "__main__":
    #pass
    s = system()
#   print(type(self.sites))
#    print(s.sites)
##   print('')
##   print(type(self.neighboursites_1[0]))
#    print(s.nn1)
##   print('')
#    print(s.order())    
#    print(s.energy())