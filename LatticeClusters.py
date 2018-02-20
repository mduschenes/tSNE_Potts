# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 20:00:49 2018

@author: Matt
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from Lattice import Lattice
from Model import Model

bw_cmap = colors.ListedColormap(['black', 'white'])

class system(object):

    def __init__(self,L=10,d=2,T=2.0/np.log(1.0 + np.sqrt(2)),model=['potts',3]):
        self.m = Model(model = model)
        self.l = Lattice(L,d)
        self.sites = self.m.state_sites(self.l.Nspins)
        
        self.L = L
        self.d = d
        self.T = T
        self.bond_prob = 1-np.exp(-2/self.T)
        
        self.wolff()
        return 
            
             
                
    def wolff(self,ncluster=1):
    
        # Create list of unique clusters and their values
        self.clusters = []
        self.cluster_values = []
        
        
        # Create Cluster Array and Choose Random Site
        f, (ax_sites, ax_cluster) = plt.subplots(1, 2)
        for i in range(self.l.Nspins):
            
            isite = np.random.randint(self.l.Nspins)
            self.cluster_sites = []
            self.cluster_rejections = []
            self.cluster_value = self.m.state_sites(ncluster,self.sites[isite])
            self.cluster_value0 = self.sites[isite]
        
            # Perform cluster algorithm to find indices in cluster
            self.cluster(isite)
            
            # Flip spins in cluster to new value
            self.sites[self.cluster_sites] = self.cluster_value        
            
            self.clusters.append(self.cluster_sites)
            
            # Plot Created Cluster
            self.sites_clusters = np.zeros(np.shape(self.sites))
#            for c in self.clusters:
            self.sites_clusters[self.cluster_sites] = self.sites[self.cluster_sites]
            self.plot_sites(self.sites_clusters.reshape(self.L,self.L),'Cluster',ax_cluster)
            
            # Plot Sites
            self.plot_sites(self.sites.reshape(self.L,self.L),'Spin Configurations',ax_sites)
            
        return
    
    
    def cluster_edge(self,clusters):
        self.edges = []
        for c in clusters:
             self.edges.append([i for i in c if len([j for j in c if j in self.l.neighbour_sites[0][i]]) < len(self.l.neighbour_sites[0][i])])
        return
    
    def cluster(self,i):
        self.cluster_sites.append(i)
        if len(self.cluster_sites) < int(0.8*self.l.Nspins):
            J = (j for j in self.l.neighbour_sites[0][i] if (j not in self.cluster_sites) and (self.sites[j] == self.cluster_value0) and (j not in self.cluster_rejections))
            for j in J:
                if self.bond_prob > np.random.rand():
                        self.cluster(j)
                else:
                    self.cluster_rejections.append(j)
        return





    def plot_sites(self,data,plot_title='Spin Configurations',ax=None):
                
                # Create color map of fixed colors
#                cmap = colors.ListedColormap(['red', 'white','black']) , norm=norm
#                bounds=[-1,0,1,2]
#                norm = colors.BoundaryNorm(bounds,cmap.N)
                cmap=plt.cm.get_cmap('RdBu',len(self.m.state_range)+1)
                if ax == None:
                    f,ax = plt.subplots()
                plt.sca(ax)
                ax.clear()
                plot = plt.imshow(np.real(data),cmap=cmap,interpolation='nearest')
                plt.colorbar(plot,cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05),ticks = self.m.state_range)
                #cbar.set_ticklabels(self.m.state_range)
                plt.sca(ax)
                plt.xticks([])
                plt.yticks([])
                plt.title(plot_title)
                plt.pause(0.5)

if __name__ == "__main__":
    s = system()
    