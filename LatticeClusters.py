# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 20:00:49 2018

@author: Matt
"""

import numpy as np
import datetime

import matplotlib.pyplot as plt

from Lattice import Lattice
from Model import Model
from Plot_Sites import Plot_Sites
from ModelFunctions import caps

class system(object):

    def __init__(self,L=10,d=2,T=2.5,model=['potts',4]): #2.0/np.log(1.0 + np.sqrt(2))
        self.m = Model(model = model)
        self.l = Lattice(L,d)
        self.sites = self.m.state_sites(self.l.Nspins)
        
        self.L = L
        self.d = d
        self.T = np.atleast_1d(T)
        self.bond_prob_f = lambda T: 1-np.exp(-1/T)
        
        self.wolff()
        return 
            
             
                
    def wolff(self,ncluster=1,animate=[True,True,True]):
    
        # Create list of unique clusters and their values
        self.clusters = []
        self.cluster_edges = []
        
        
        
        # Create Cluster Array and Choose Random Site

        # Initialize Plot Class
        
        self.plotf = Plot_Sites(animate,np.size(self.T),sum(animate),
                                        self.m.state_ranges(
                                                    xmax=self.m.q+2),
                                         [['Spin Configurations',
                                                     'Cluster','Edge'] ,
                                          lambda i: r'$t_{MC}$: %d'%i,
                                          lambda i: r'T = %0.1f'%self.T[i]],
                                         lambda d: self.sites_region(
                                                  d).reshape(self.L,self.L))
                                        
            


        for i_t,t in enumerate(self.T):
            print(t)
            self.bond_prob = self.bond_prob_f(t) 

            for i_update in range(self.l.Nspins):
                isite = np.random.randint(self.l.Nspins)
                self.cluster_sites = []
                self.cluster_rejections = []
                self.cluster_value = self.m.state_sites(ncluster,
                                                        self.sites[isite])
                self.cluster_value0 = self.sites[isite]
            
                # Perform cluster algorithm to find indices in cluster
                self.cluster(isite)
                
                # Flip spins in cluster to new value
                self.sites[self.cluster_sites] = self.cluster_value        
                
                self.clusters.append(self.cluster_sites)
                
                
                # Find Edges of Cluster
                self.edges(self.cluster_sites)
                #print(np.size(self.edges))
                # Plot Sites, Updated Cluster, and Updated Cluster Edges
                
                if animate[0] and ((False or np.size(self.T)==1) or ( 
                                                 i_update == self.l.Nspins-1)):
                    
                    self.plotf.data = [
                              self.sites,self.cluster_sites,self.cluster_sites]
                    
                    for i_a,a in reversed(list(enumerate(animate))): #enumerate(animate):
                        if a:
                            self.plotf.plot_sites(i_t,i_a,i_update)
                    
                                    
        
        # Save Data Plots
        if animate[0]: 
            dataDir = '%s_Data' %(caps(self.m.model.__name__))
            dataName = '%s/%s_d%d_L%d__%s.pdf' %(
                        dataDir,caps(self.m.model.__name__),self.d,self.L,
                        datetime.datetime.now().strftime(
                                                '%Y-%m-%d-%H-%M'))
            self.plotf.plot_save(dataName)
                    
                    
        return
    
    
    def edges(self,cluster):
        for c in np.atleast_1d(cluster):
             self.cluster_edges.append([i for i in np.atleast_1d(c) if len(
                                             [j for j in np.atleast_1d(c) if j
                                    in self.l.neighbour_sites[0][i]]) 
                                          < len(self.l.neighbour_sites[0][i])])
        return
    
    def cluster(self,i):
        self.cluster_sites.append(i)
        if len(self.cluster_sites) < int(0.95*self.l.Nspins):
            J = (j for j in self.l.neighbour_sites[0][i] if (
                                    j not in self.cluster_sites) and 
                                    (self.sites[j] == self.cluster_value0) and 
                                    (j not in self.cluster_rejections))
            for j in J:
                if self.bond_prob > np.random.rand():
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
            return region
        


    
                
                

if __name__ == "__main__":
    T = [5,2.5,2,1.5,1,0.5]
    T0 = 0.5
    s = system(T=T0)
    plt.pause(1)
    