# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 20:00:49 2018

@author: Matt
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

from Lattice import Lattice
from Model import Model

bw_cmap = colors.ListedColormap(['black', 'white'])

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
            
             
                
    def wolff(self,ncluster=1,animate=True,animate_clusters=True):
    
        # Create list of unique clusters and their values
        self.clusters = []
        self.cluster_values = []
        
        
        # Create Cluster Array and Choose Random Site
        
        if animate:
            _, ax = plt.subplots(np.size(self.T),animate_clusters+1)
            ax = np.atleast_1d(ax)
            if not animate_clusters and np.size(self.T)>1:
                ax = ax[:,np.newaxis] 
            elif np.size(self.T)==1:
                ax = ax[np.newaxis,:] 

        for j ,t in enumerate(self.T):

            self.bond_prob = self.bond_prob_f(t) 

            for i in range(10*self.l.Nspins):
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
                
                # Plot Sites and Created Cluster
                self.sites_clusters = np.zeros(np.shape(self.sites))
                self.sites_clusters[:] = np.nan
    #            for c in self.clusters:
                self.sites_clusters[self.cluster_sites] = self.sites[
                                                            self.cluster_sites]

                if (animate and np.size(self.T)==1) or i == self.l.Nspins-1:
                    if animate_clusters:
                        title_cluster = 'Cluster' if j == 0 else ''
                        self.plot_sites(
                                self.sites_clusters.reshape(self.L,self.L),
                                        title_cluster,
                                        r'$t_{MC}$: %d'%i,ax[j,1],
                                        sitevalues = self.m.state_ranges(
                                                xmax=self.m.q+2))
                
                    title_site = 'Spin Configurations' if j == 0 else ''
                    self.plot_sites(self.sites.reshape(self.L,self.L),
                                    title_site,'T = %0.1f'%t,
                                     ax[j,0], sitevalues = self.m.state_ranges(
                                            xmax=self.m.q+2))
                
        return
    
    
    def cluster_edge(self,clusters):
        self.edges = []
        for c in clusters:
             self.edges.append([i for i in c if len([j for j in c if j
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





    def plot_sites(self,data,plot_title='Spin Configurations',plot_label='',
                   ax=None,sitevalues=None):              
                # Create color map of fixed colors
#                cmap = colors.ListedColormap(['red', 'white','black']) , norm=norm
#                bounds=[-1,0,1,2]
#                norm = colors.BoundaryNorm(bounds,cmap.N)
#                sitevalues = self.m.state_ranges(xNone=[],xmin=0,xmax=self.m.q+2)
                if sitevalues == None:
                    sitevalues = range(np.min(data),np.max(data)+1)

                n_sitevalues = len(sitevalues)
                norm = colors.BoundaryNorm(sitevalues, ncolors=n_sitevalues)
                cax = make_axes_locatable(
                        ax).append_axes('right', size='5%', pad=0.05)
            
#                if ax == None:
#                    _,ax = plt.subplots()
                
                ax.clear()
                cax.clear()
                plt.sca(ax)
                cmap=plt.cm.get_cmap('bone',n_sitevalues)
                plot = plt.imshow(np.real(data),cmap=cmap,norm=norm,
                                  interpolation='nearest')
                
                
                cmap.set_bad(color='magenta')             

               
                
                
#                plt.clim(min(sitevalues)-1, min(sitevalues)+1)
                #cbar.set_ticklabels(self.m.state_range)
                plt.sca(ax)
                plt.xticks([])
                plt.yticks([])
                plt.title(plot_title)
                plt.xlabel(plot_label)
                
                cbar = plt.colorbar(plot,cax= cax,label='Spin value')
                cbar.set_ticks(np.array(sitevalues)+0.5)
                cbar.set_ticklabels(np.array(sitevalues))
                
                plt.pause(0.2)

if __name__ == "__main__":
    T = [5,2.5,2,1.5,1,0.5]
    T0 = 0.5
    s = system(T=T)
    plt.pause(100)
    