# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 20:00:49 2018

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

    def plot_sites(self,data,plot_title):
                plt.clf()
                plt.imshow(data),interpolation='nearest')
                #plt.colorbar()
                plt.xticks([])
                plt.yticks([])
                plt.title()
                plt.pause(0.5)
                
                
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