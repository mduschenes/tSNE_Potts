# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 18:29:03 2018

@author: Matt
"""

# Calculate Mutual Information for Neural Network Input X, Output Y, Layers T

import numpy as np

# Calculate Mutual Information for Neural Network Input X, Output Y, Layers T

#######  datafolder+ dataset_file[2] = /path_to/"labels64.npy"  ###############

class MUT_INFO(object):
    def __init__(self, n_bins=1000, dataset_file=None):
        
        # Initialize Mutual Information Class for given dataset
        # mut_info method will be called with layer output array:
        #     t = (Nx by n_t) array, where Nx is the number of datasets,
        #                             n_t is the number of neurons in the layer
        
        # Define bin size
        self.n_bins = n_bins
        
        # Calculate global p(x,y), p(x), |X|
        self.dataset_file = dataset_file
        self.prob_joint_X_binaryY()
        
        return
    
    
    def mut_info(self,t):
        # Estimate Mutual Information of between 
        # Random Variables (X,Y,T):   I(X,T) and I(Y,T)
        
        # Probability of p(t(x)) and delta(t(x),t(x'))
        p_tx,delta_tx = self.prob_t_x(t,n_bins=self.n_bins)      
        
        # Calculate Mutual Information of I(T,Y)
        # p_xy: (Ny(=2 for binary) by Nx) array,    p_tx = (Nx by 1) array,   
        # delta_tx = (Nx by Nx) array,  p_x = (Nx by 1) array
   
        I_TY = np.nansum(self.p_xy*
                         np.log2(np.dot(self.p_xy,delta_tx)/
                         np.sum(self.p_xy,1)[:,np.newaxis]/p_tx))
    
        I_TX = -np.dot(self.p_x,np.log2(p_tx))
        
        return [I_TX, I_TY]
    
        
    def prob_joint_X_binaryY(self):
    
        def py_x(u,gamma=30.5,theta=34):
            return 1.0/(1.0 + np.exp(-gamma*(u-theta)))

        # Import Original X Data and calculate size of 
        X = np.load(self.dataset_file).astype(np.float)
        
        # Define Probability Space NX
        self.NX = np.size(X)
        
        # Calculate p(x) and p(x,y)
        self.p_x = np.ones(self.NX)*1/self.NX
        
        self.p_xy = np.array([(1-py_x(X))*self.p_x,
                                  py_x(X)*self.p_x])
        
        return 
    

    def prob_t_x(self,t, n_bins): # Thanks Lauren!
        # Takes the layer's output t(x) and a number of bins
        # Returns a probability p(t(x)) 
        # as a vector and a matrix for KroneckerDelta(t(x), t(x'))

        # Define bins
        bins = np.linspace(-1, 1, n_bins)
        
        # Count number of appearance of each vector
        _, indices, counts= np.unique(np.digitize(t, bins), 
                                return_inverse=True, 
                                return_counts=True, axis=0)
        # Create delta matrix from indices
        delta = (np.array([indices,] * 
                          len(indices)).T == indices).astype(np.int)
        
        # Return p(t_x), delta
        return counts[indices]/self.NX, delta

















#class MUT_INFO(object):
#    def __init__(self, n_bins=1000, dataset_file=data_folder+dataset_file[2]):
#        
#        # Initialize Mutual Information Class for given dataset
#        # mut_info method will be called with layer output array:
#        #     t = (Nx by n_t) array, where Nx is the number of datasets, 
#        #     and n_t is the number of neurons in the layer
#        
#        
#        # Define bin size
#        self.n_bins = n_bins
#        
#        
#        # Obtain constant p_xy distribution from dataset and calculate: 
#        # p(x) and size of probability space: NX
#        self.p_xy = self.prob_joint_X_binaryY(dataset_file)
#        
#        return
#        
#    def mut_info(self,t):
#        # Estimate Mutual Information of between 
#        # Random Variables (X,Y,T):    I(X,T) and I(Y,T)
#        
#        # Probability of p(t(x)) and delta(t(x),t(x'))
#        p_tx,delta_tx = self.prob_t_x(t,n_bins=self.n_bins)      
#        
#        # Calculate Mutual Information of I(T,Y) and I(T,X)
#        
#        I_TY = np.nansum(self.p_xy*(np.log2(np.dot(self.p_xy,delta_tx)/np.sum(self.p_xy,1)[:,np.newaxis]/p_tx)))
#    
#        I_TX = -np.dot(self.p_x,np.log2(p_tx))
#
#
#        return [I_TX, I_TY]
#        
#        
#    def prob_joint_X_binaryY(self,dataset_file):
#    
#        def py_x(u,gamma=30.5,theta=34):
#            return 1.0/(1.0 + np.exp(-gamma*(u-theta)))
#
#        # Import Original X Data and calculate size of Probability Space NX
#        X = np.load(dataset_file).astype(np.float)
#        self.NX = np.size(X)
#        
#        # Calculate p(x)
#        self.p_x = np.ones(self.NX)*1/self.NX
#        
#        
#        py_x = py_x(X) 
#        #print('pyx sig: ',pyx)
#        
#        return np.array([(1-py_x(X))*self.p_x, py_x(X)*self.p_x])
#    
#    
#
#    def prob_t_x(self,t, n_bins): # Thanks Lauren!
#        # Takes the layer's output t(x) and a number of bins
#        # Returns a probability p(t(x)) as a vector and a matrix for KroneckerDelta(t(x), t(x'))
#
#        # Define bins
#        bins = np.linspace(-1, 1, n_bins)
#        
#        # Count number of appearance of each vector
#        _, indices, counts= np.unique(np.digitize(t, bins), 
#                                return_inverse=True, return_counts=True, axis=0)
#        
#        # Create delta matrix from indices
#        delta = (np.array([indices,] * len(indices)).T == indices).astype(np.int)
#        
#        # Return p(t_x), delta
#        return counts[indices]/self.NX, delta

    
    