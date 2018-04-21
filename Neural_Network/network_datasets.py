# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 01:08:44 2018

@author: Matt
"""
import numpy as np
import matplotlib.pyplot as plt



def spiral_dataset():    
    
    
    
    #    from network_datasets import spiral_dataset 
#    x_train,y_train,plot_data = spiral_dataset()
#    x_test = None
#    y_test = None
#    data_format = 'values'
#    one_hot = True
#    kwargs = {'y_estimate': plot_data}
    
    
    
    
    
    
    # K Branch Data Set
    N = 50 # number of points per branch
    K = 3  # number of branches
    
    N_train = N*K # total number of points in the training set
    x_train = np.zeros((N_train,2)) # 2-dimensional datapoints
    y_train = np.zeros((N_train,1), dtype='uint8') # (not one-hot labels)
    
    mag_noise = 0.2 # controls how much noise gets added to the data
    dTheta    = 4    # difference in theta in each branch
    
    ### Data generation: ###
    for j in range(K):
      ix = range(N*j,N*(j+1))
      r = np.linspace(0.01,1,N) # radius
      t = np.linspace(j*(2*np.pi)/K,j*(2*np.pi)/K + dTheta,N) + (
                                         np.random.randn(N)*mag_noise) # theta
      x_train[ix] = np.c_[r*np.cos(t), r*np.sin(t)]
      y_train[ix] = j
            
    
    ### Generate coordinates covering the whole plane: ###
    padding = 0.1
    spacing = 0.02
    x1_min, x1_max = x_train[:, 0].min()-padding, x_train[:, 0].max()+padding
    x2_min, x2_max = x_train[:, 1].min()-padding, x_train[:, 1].max()+padding
    x1_grid, x2_grid = np.meshgrid(np.arange(x1_min, x1_max, spacing),
                         np.arange(x2_min, x2_max, spacing))
    

    def plot_data(y_est,fig,ax):
        
        
        y_estimate =  lambda f,y: np.argmax(f(y,
                                    [np.c_[x1_grid.ravel(),x2_grid.ravel()]]),
                                    axis=1).reshape(x1_grid.shape)

            
        plt.contourf(x1_grid, x2_grid, y_estimate(y_est[0],y_est[1]),
                     K, alpha=0.8)
        plt.scatter(x_train[:,0],x_train[:,1],
                    c=y_train, s=40)
        plt.xlim(x1_grid.min(), x1_grid.max())
        plt.ylim(x2_grid.min(), x2_grid.max())
        plt.xlabel('x1')
        plt.ylabel('x2')
        
        plt.pause(0.05)
    


    return x_train,y_train,plot_data