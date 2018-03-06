# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 14:18:39 2018

@author: Matt
"""

import numpy as np


class MCUpdate(object):
    
    def __init__(self,sites,T,update_props = [Nmeas,Neqb,bond_prob,ncluster = 1]):
        # Perform Monte Carlo Updates for nclusters of sites and Plot Data
        
        # Define System Parameters
        self.Nspins = np.size(sites)
        
        
        
        
        self.bond_prob = bond_prob
        
        
        
        self.m.model_params[
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
                plot_title = '%d^%d %s model, m = %f \n q = %d, T = %.3f'%(self.L,self.d,caps(self.m.model.__name__),
                            signed_val(self.order())/self.Nspins,
                            #self.correlations[-1][0],
                            self.q,
                            self.T)
                # Plot Sites 
                self.plot_sites(self.sites_clusters.reshape(self.L,self.L),self.m.state_ranges(xmax=self.m.q+2),'Cluster',self.axis[1])
                self.plot_sites(self.sites.reshape(self.L,self.L),self.m.state_ranges(xmax=self.m.q+2),plot_title,self.axis[0])
            
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
    
        # Create list of unique clusters and their values
        self.clusters = []
        self.cluster_values = []
        
        
        # Create Cluster Array and Choose Random Site
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
            self.sites_clusters[:] = np.nan
#            for c in self.clusters:
            self.sites_clusters[self.cluster_sites] = self.sites[self.cluster_sites]
            
            
        return        
        
    def wolff1(self,ncluster=1):
        
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


    def cluster_edge(self,clusters):
        self.edges = []
        for c in clusters:
             self.edges.append([i for i in c if len([j for j in c if j in self.l.neighbour_sites[0][i]]) < len(self.l.neighbour_sites[0][i])])
        return
    
    def cluster(self,i):
        self.cluster_sites.append(i)
        if len(self.cluster_sites) < int(0.95*self.l.Nspins):
            J = (j for j in self.l.neighbour_sites[0][i] if (j not in self.cluster_sites) and (self.sites[j] == self.cluster_value0) and (j not in self.cluster_rejections))
            for j in J:
                if self.bond_prob > np.random.rand():
                        self.cluster(j)
                else:
                    self.cluster_rejections.append(j)
        return


    def neighbours(self,r=1,sites,neighbour_sites):
        # Return spins of r-distance neighbours for all spin sites
        return np.array([index(sites,neighbour_sites[r-1][i]) 
                                            for i in range(len(sites))])