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
    # CURRENTLY only accepts function arguments for a 
    # (desire constant arguments that are converted to functions lamba x: a )
    
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

def flatten(x,flattenint=True):
    # Return a 1d list of all elements in inner lists of x of arbirtrary shape
    if  not isinstance(x,list):
        return x
    elif len(x) == 1 and flattenint:
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

def signed_val(x):
    xs = np.shape(x)
    xf = np.array(x).flatten()
    ri = np.stack((np.real(xf),np.imag(xf)),1)
    xms = np.sign(ri[np.arange(np.size(ri,0)), np.argmax(np.abs(ri),1)])*np.abs(xf)
    return np.reshape(xms,xs)
    
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
        
def caps(word):
    return word[0].upper()+word[1:].lower()

def delta_f(x,y,f=np.multiply):
    return f(x,y)[x==y]

class Observables(object):
    def __init__(self,f):
        self.f = f
        self.size = np.size(f())
        self.name = f.__name__

    def F(self):
        return self.f()
        
        
class system(object):
    # Define system class for general model of lattice sites with
    #  Lattice Parameters: Lattice Length L, Lattice Dimension d, Temperature T
    #  Lattice Model Type: Model Name and Max Spin Value q
    #  Order Parameters: Hamiltonian Coupling Constants, 
    #                    Order Parameter Function
    #  Monte Carlo Update Parameters: Perform Monte Carlo Update
    #                                 Number of initial updates Neqb,
    #                                 Number of Measurement Updtes Nmeas
    #                                 Monte Carlo Update Algorithm
    #                                 Monte Carlo Plot Results
    # DataSave Boolean
    
    def __init__(self,L=6,d=2,T=3, model=['ising',1],
                 orderparam=[0,1],
                 update = [True,None,None,'wolff',True],
                 datasave = False):
        # Time runtime of system class
        t0 = time.clock() 
        
        # Initialize model class, lattice class
        self.m = Model(model,orderparam)
        t1 = time.clock()
        self.l = Lattice(L,d)
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

     
        
        # Initialize numpy array with random spin at each site
        self.sites = self.m.state_sites(self.Nspins)
#        # Initialize nearest neighbour sites array
#        self.nn = self.l.neighbour_sites[0]
        
        
        # Define Monte Carlo Update:
        # Number of updates to reach "equilibrium" before measurement, 
        # Number of measurement updates.
        # Monte Carlo update alogrithms
        self.mcupdate = update[0]
        self.Neqb = int((1/3)*L**(d)) if update[1]==None else update[1]
        self.Nmeas = int(5*L**(d)) if update[2]==None else update[2]

        self.update_algs = {'metropolis': self.metropolis, 'wolff': self.wolff}
        self.algorithm = update[3]
        self.animate = update[4]
        self.MCUpdatealg = self.update_algs[self.algorithm]

        # Initilize list of observations values
        # obsvmax = len(self.Tlist)*(self.Neqb + self.Nmeas)
        self.observables = []
        
        self.observables_functions = [self.temperature,self.energy,
                                      self.order]
        self.observables_data = lambda:list(map(lambda f,g= lambda x:x : g(f())
                                                ,self.observables_functions))
        self.observables_sizes = [np.size(f()) 
                                        for f in self.observables_functions]
        # Save observations data
        self.save = datasave
        self.datasave()



        
        # Perform Monte Carlo Updates for various Temperatures
        if self.mcupdate:
            t3 = time.clock()
            for t in self.Tlist:
                self.T = t
                self.MCUpdate()
                print('T = ',str(t))
                print(time.clock()-t3)
                t3 = time.clock()
                print('')
            self.datasave(lambda : 'system class runtime: '+
                          str(time.clock()-t0))
            # print(self.correlation())
            # Calculate total runtime for system class
            print('final order: '+str(self.order()/self.Nspins))
            print('system class runtime: '+str(time.clock()-t0)) 
            print('model class runtime: '+str(t1-t0)) 
            print('lattice class runtime: '+str(t2-t1)) 
            plt.pause(5)
            
    def MCUpdate(self,ncluster = 1):
        # Perform Monte Carlo Updates for nclusters of sites and Plot Data
        # bw_cmap = colors.ListedColormap(['black', 'white'])
        
        self.bond_prob = self.m.model_params[
                self.m.model.__name__]['bond_prob'](self.T)
        
        
        for i in range(self.Neqb):
            self.MCUpdatealg(ncluster)
            
        for i in range(self.Nmeas):
            
            # Perform Monte Carlo Update Algorithm and Plot Spin Sites
            #tmc = time.clock()
            self.MCUpdatealg(ncluster)
            #print(time.clock()-tmc)
            if i % self.Nmeas/self.L == 0 or True:
                self.observables.append(signed_val(flatten([
                                    f() for f in self.observables_functions])))
            #print(self.observables)
                        
            if self.animate and i == self.Nmeas-1:
                plot_title = '%d^%d %s model, m = %f ---- q = %d, T = %.3f'%(self.L,self.d,caps(self.m.model.__name__),
                            signed_val(self.order())/self.Nspins,
                            #self.correlations[-1][0],
                            self.q,
                            self.T)
                data = signed_val(np.copy(self.sites)).reshape(self.L,self.L)
                self.plot_sites(data,plot_title)
        # Plot Corellation length
        #plt.plot()
            
    def metropolis(self,ncluster=1):
        # Randomly alter random spin sites and accept spin alterations
        # if energetically favourable or probabilistically likely
        for i in range(self.Nspins):
            E0 = self.energy()
            sites0 = np.copy(self.sites)
            
#            print(self.sites)
#            print(self.sites0)
#            print('')
            isites = [np.random.randint(self.Nspins) for j in range(ncluster)]
            self.sites[isites] = self.m.state_sites(ncluster,sites0[isites])
#            print(self.sites)
#            print(sites0)
#            print('')
            
#            for isite in isites:
#                print(self.sites[isite])
#                self.sites[isite] = self.m.state_sites(1,sites0[isite])
#                print(self.sites[isite])
                #print(sites0[isite])
            dE = self.energy()-E0
            dE = np.sign(np.real(dE))*np.abs(dE)
            if dE > 0:
                if np.exp(-dE/self.T) < np.random.random():
#                    print('change sites back')
                    #for isite in isites:
                        #print(self.sites[isite])
                    self.sites[isites] = np.copy(sites0[isites])
#                else:
                    
                    #print('dE>0 but no change')
#            else:
#                pass
                #print('dE < 0')
                        #print(self.sites[isite])
            #print(self.sites[isite])
            #print(dE)
            #print(np.all(self.sites[isites]==sites0[isites]))
            #print('')
#            print(self.sites)
#            print(self.sites0)
#            print('')
            return
            
        
    def wolff(self,ncluster=1):
        
#        # Create list of unique clusters and their values
#        self.clusters = []
#        self.cluster_values = []
#        
#        
#        # Create Cluster Array and Choose Random Site
#        isite = np.random.randint(self.Nspins)
#        self.cluster_value = self.m.state_sites(ncluster,self.sites[isite])
#        self.cluster_sites.append(isite)
#        
#        # Create list of indices in Cluster and original value of Random Site
#        self.cluster_sites = []
#        self.cluster_value0 = self.sites[isite]
#    
#        # Perform cluster algorithm to find indices in cluster
#        self.cluster(isite)
#        
#        # Flip spins in cluster to new value
#        self.sites[self.cluster_sites] = self.cluster_value        
#        
#        self.clusters.append(self.cluster_sites)
        
        
        for i in range(self.Nspins):    
            # Create Cluster Array and Choose Random Site
            isite = np.random.randint(self.Nspins)
            self.cluster_value = self.m.state_sites(ncluster,self.sites[isite])
            
            self.cluster_sites = []
            for c in self.clusters:
                if isite in c:
                    self.sites[c] = self.cluster_value
                    if self.cluster_values in self.cluster_values:
                        
                    break
                else:
                    self.cluster_sites.append(isite)
            
                    self.cluster_sites = []
                    
            self.cluster_value0 = self.sites[isite]
        
            
            self.cluster(isite)
            
            self.sites[self.cluster_sites] = self.cluster_value        
            
            self.clusters.append
            
        return


    def cluster_edge(self):
        self.edges = []
        for c in self.clusters:
             self.edges.append([i for i in c if len([j for j in c if j in self.l.neighbour_sites[0][i]]) < len(self.l.neighbour_sites[0][i])])
                

    def cluster(self,i):
        self.cluster_sites.append(i)
        if len(self.cluster_sites) < int(0.8*self.Nspins):
            J = (j for j in self.l.neighbour_sites[0][i] if (j not in self.cluster_sites) and (self.sites[j] == self.cluster_value0) )
            for j in J:
                if self.bond_prob > np.random.rand():
                        self.cluster(j)
        return


    def neighbours(self,r=1):
        # Return spins of r-distance neighbours for all spin sites
        return np.array([index(self.sites,self.l.neighbour_sites[r-1][i]) 
                                            for i in range(self.Nspins)])
        
    def temperature(self):
        return self.T
    

    def energy(self):
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
        
        return (-self.m.orderparam[0]*np.sum(self.sites)) + (-(1/2)*np.sum([
                self.m.orderparam[i]*self.m.site_energy(self.sites[:,np.newaxis],
                self.sites[self.l.neighbour_sites[i-1]]) 
                for i in range(1,len(self.m.orderparam))]))
    
    def order(self):
        return self.m.order(self.sites)
    
    def correlation(self, r = None):
        # Calculate correlation function c(r) = <s_i*s_j> for all spin pairs
        # where the ri-ir neighbour distance is r = {1:L/2}
        # i.e) [(1/(self.Nspins))*
#                sum(sum(n*self.sites[j] for n in self.neighbours(rr)[j])
#                - ((sum(self.sites)/self.Nspins)**2) 
#                    for j in range(self.Nspins)) for rr in r]
        if r is None:
            r = self.l.R
        
        return list((1/2)*((1/self.Nspins)**2)*np.sum(self.sites[:,np.newaxis]
                *self.sites[self.l.neighbour_sites[r-1]],(1,2)) - (
                (1/self.Nspins**2)*np.sum(self.sites))**2)
#        
            
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
        
    
    def plot_sites(self,data,plot_title):
                plt.clf()
                plt.imshow(data,interpolation='nearest')
                #plt.colorbar()
                plt.xticks([])
                plt.yticks([])
                plt.title()
                plt.pause(0.5)
    
    
    def datasave(self,*args):
        
        if self.save == True:
            # Open and write observables to a file in a specific directory
            dataDir = '%s_Data' %(caps(self.m.model.__name__))
            if not self.observables :
                if not(os.path.isdir(dataDir)):
                        os.mkdir(dataDir)
                return
            
            dataName         = '%s/%s_d%d_L%d__%s.txt' %(
                                    dataDir,caps(self.m.model.__name__),
                                    self.d,self.L,
                                    datetime.datetime.now().strftime(
                                                            '%Y-%m-%d-%H-%M'))
            # Make observables headers for file
            file = open(dataName, 'w')
            headers = []
            for i,f in enumerate(self.observables_functions):
                for j in range(self.observables_sizes[i]):
                    headers.append(f.__name__+'_'+str(j+1) 
                                    if self.observables_sizes[i] > 1
                                    else f.__name__)
            file.write('\t'.join(headers) + '\n')
            
            # Convert lists of lists of observables to array
            self.observables = np.array([flatten(x) for x in self.observables])
            for data in self.observables:
                dataline = ''
                for d in data:
                        dataline += '%0.8f \t' %(float(d))
                dataline += '\n'
                file.write(dataline)
            if args:
                for a in args:
                    file.write(str(a())+'\n')
            
            file.close()
        return
    
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
            
class Lattice(object):
    
    # Define a Lattice class for lattice sites configurations with:
    # Lattice Length L, Lattice Dimension d
    
    def __init__(self,L=6,d=2):
        # Define parameters of system        
        self.L = L
        self.d = d
        self.Nspins = L**d
        
        if self.Nspins > 2**32:
            self.dtype = np.int64
        else:
            self.dtype=np.int32

        # Prepare arrays for Lattice functions

        # Define array of sites
        self.sites = np.arange(self.Nspins)
        
        # L^i for i = 1:d array
        self.L_i = np.power(self.L,np.arange(self.d,dtype=self.dtype))
        
        # r = [0....ri...0] for i = 1:d array
        self.I = np.identity(self.d)
        self.R = np.arange(1,np.ceil(self.L/2),dtype=self.dtype)
        #self.Rn = np.concatenate((self.R,-self.R))
#        self.Rn = np.array([(x,-x) for x in self.R ])
#        print(self.Rn)
#        self.Rnp = np.kron(self.Rn,self.I).reshape((-1,self.d))
        #print(self.Rnp)
        
        # Calculate array of arrays of r-distance neighbour sites,
        # for each site, for r = 1 : L/2 
        # i.e) self.neighbour_sites = np.array([[self.neighboursites(i,r) 
        #                                 for i in range(self.Nspins)]
        #                                 for r in range(1,
        #                                             int(np.ceil(self.L/2)))])
        self.neighbour_sites = self.neighboursites(None,None)
        #print(self.neighbour_sites.reshape((self.Nspins,-1,2*self.d)),'F')
        #print(self.neighbour_sites)
        #.reshape((len(site),-1,4))

        
    def position(self,site):
        # Return position coordinates in d-dimensional L^d lattice 
        # from given linear site position in 1d Nspins^2 length array
        # i.e) [int(site/(self.L**(i))) % self.L for i in range(self.d)]
        return np.mod(((np.atleast_1d(site)[:,np.newaxis]/self.L_i)).
                        astype(self.dtype),self.L)
    
    def site(self,position):
        # Return linear site position in 1d Nspins^2 length array 
        # from given position coordinates in d-dimensional L^d lattice
        # i.e) sum(position[i]*self.L**i for i in range(self.d))
        return (np.dot(np.atleast_2d(position),self.L_i)).astype(self.dtype)
    
    def neighboursites(self,site=None,r=1):
        # Return array of neighbour spin sites 
        # for a given site and r-distance neighbours
        # i.e) np.array([self.site(np.put(self.position(site),i,
        #                 lambda x: np.mod(x + p*r,self.L))) 
        #                 for i in range(self.d)for p in [1,-1]]) 
        #                 ( previous method Time-intensive for large L)
        
        if site==None:
            site = self.sites
        
        sitepos = self.position(site)[:,np.newaxis]
        
        if r==None:
            Rrange = self.R
        elif isinstance(r,list):
            Rrange = r
        else:
            Rrange = [r]
                            
        return np.stack((np.concatenate(
                            (self.site(np.mod(sitepos+R*self.I,self.L)),
                             self.site(np.mod(sitepos-R*self.I,self.L))),1)
                                for R in Rrange))          
#        return 
#        np.stack((self.site(np.mod(sitepos+R*self.I,self.L))) for R in Rrange) 

            
        #  rvals = np.diag(np.delete(np.arange(-self.L,self.L+1),0))
        #  self.rsites = np.kron(rvals,np.identity(self.d))
        #  print(self.rsites)

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