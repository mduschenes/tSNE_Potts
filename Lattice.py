# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 20:22:11 2018

@author: Matt
"""

import numpy as np

class Lattice(object):
    
    # Define a (Square) Lattice class for lattice sites configurations with:
    # Lattice Length L, Lattice Dimension d
    
    def __init__(self,L=10,d=3,**kwargs):
        # Define parameters of system        
        self.L = L
        self.d = d
        self.N = L**d
        self.z = 2*d
        
        if self.N > 2**32-1:
            self.dtype = np.int64
        else:
            self.dtype=np.int32

        # Prepare arrays for Lattice functions

        # Define array of sites
        self.sites = np.arange(self.N)
        
        # L^i for i = 1:d array
        self.L_i = np.power(self.L,np.arange(self.d,dtype=self.dtype))
        
        # Arrays for finding coordinate and linear position in d dimensions
        self.I = np.identity(self.d)
        self.R = np.arange(1,np.ceil(self.L/2),dtype=self.dtype)

        
        # Calculate array of arrays of r-distance neighbour sites,
        # for each site, for r = 1 : L/2 
        # i.e) self.neighbour_sites = np.array([[self.neighboursites(i,r) 
        #                                 for i in range(self.N)]
        #                                 for r in range(1,
        #                                             int(np.ceil(self.L/2)))])
        self.neighbours = self.neighbour_sites()


        
    def position(self,site):
        # Return position coordinates in d-dimensional L^d lattice 
        # from given linear site position in 1d N^2 length array
        # i.e) [int(site/(self.L**(i))) % self.L for i in range(self.d)]
        return np.mod(((np.atleast_1d(site)[:,np.newaxis]/self.L_i)).
                        astype(self.dtype),self.L)
    
    def site(self,position):
        # Return linear site position in 1d N^2 length array 
        # from given position coordinates in d-dimensional L^d lattice
        # i.e) sum(position[i]*self.L**i for i in range(self.d))
        return (np.dot(np.atleast_2d(position),self.L_i)).astype(self.dtype)
    
    def neighbour_sites(self,r=None,sites=None):
        # Return array of neighbour spin sites 
        # for a given site and r-distance neighbours
        # i.e) np.array([self.site(np.put(self.position(site),i,
        #                 lambda x: np.mod(x + p*r,self.L))) 
        #                 for i in range(self.d)for p in [1,-1]]) 
        #                 ( previous method Time-intensive for large L)
        
        if sites is None:
            sites = self.sites
        
        sitepos = self.position(sites)[:,np.newaxis]
        
        if r is None:
            Rrange = self.R
        elif isinstance(r,list):
            Rrange = r
        else:
            Rrange = [r]
        return np.array([np.concatenate(
                            (self.site(np.mod(sitepos+R*self.I,self.L)),
                             self.site(np.mod(sitepos-R*self.I,self.L))),1)
                                for R in Rrange])                     

        
    def neighbour_states(self,r=1):
        # Return spins of r-distance neighbours for all spin sites
        return np.array([np.index(self.sites,self.neighbour_sites[r-1][i]) 
                                    for i in range(len(self.sites))])

        
if __name__ == '__main__':
    l = Lattice