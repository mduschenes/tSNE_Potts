# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 11:04:42 2018

@author: Matt
"""

#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

seed=1234
np.random.seed(seed)
tf.set_random_seed(seed)

#from mutual_information import MUT_INFO
from data_transfer import data_transfer


#network_params = {'n_neuron': [None,10,7,5,4,3,None],'alpha_learn': 0.5, 
#                  
#                  'neuron_func':{
#                       'layer': tf.tanh,
#                       'output': tf.sigmoid,
#                       'cost': {
#                         'cross_entropy': lambda y_label,y_est, eps= 10**(-6): 
#                                         tf.reduce_mean(-tf.reduce_sum(
#                                             y_label*tf.log0(y_est+eps) +
#                                          (1.0-y_label)*tf.log(1.0-y_est +eps),
#                                          axis=1)),
#                          'mse': lambda y_label,y_est: (1/2)*tf.reduce_mean(
#                                  tf.reduce_sum(tf.square(y_est-y_label),
#                                  axis=1)),
#                          'entropy_logits': lambda y_label,y_est: 
#                              tf.nn.sigmoid_cross_entropy_with_logits(
#                                      labels=y_label, logits=y_est)},
#                    'optimize': lambda a,c: tf.train.GradientDescentOptimizer(s).minimize(c)
#                             },
#                           
#                  'n_epochs': 100,
#                  'n_batch_train': 1/20, 'n_epochs_meas': 1/20,
#                  }
#                           
#data_params  =       {'data_files': ['x_train','y_train','x_test','y_test'],
#                      'data_types': ['x_train','y_train','x_test','y_test'],
#                      'data_format': 'npz',
#                      'data_dir': 'dataset/',
#                      'one_hot': False
#                     }


class neural_net(object):
    
    def __init__(self,nn_params):
                
        # Define Neural Network Properties Dictionary
        self.nn_params = nn_params  
       
        return
        

    def training(self,data_params = 
                     {'data_files': ['x_train','y_train','x_test','y_test'],
                      'data_types_train': ['x_train','y_train'],
                      'data_types_test': ['x_test','y_test'],
                      'data_format': 'npz',
                      'data_dir': 'dataset/',
                      'one_hot': False
                     },
                      train = False,
                      test = False,
                      cost_f = 'cross_entropy'):
    
        
        # Import Data by Data Type
        if not(data_params['data_types_train'] and 
               data_params['data_types_test']):
            
            data_params['data_types'] = ['x_train','y_train','x_test','y_test']
        
        else:
            
            data_params['data_types'] = (data_params['data_types_train'] +
                                         data_params['data_types_test'])
        

        data, data_size = data_transfer().importer(data_params)
        
        
        if not any('test' in key for key in data.keys()):
            test = False
        
#        print(data)
#        print(data_size)
        
        # Define Number of Neurons at Input and Output
        self.nn_params['n_dataset_train'],self.nn_params['n_neuron'][0] = (
                                                          data_size['x_train'])
        self.nn_params['n_neuron'][-1] = data_size['y_train'][1]
        self.nn_params['n_dataset_test'] = data_size.get('x_test',[None])[0]
        
        # Define Training Parameters
        self.nn_params['n_batch_train'] = int(self.nn_params['n_batch_train']*
                                             self.nn_params['n_dataset_train'])
        self.nn_params['n_epochs_meas'] = int(self.nn_params['n_epochs']*
                                              self.nn_params['n_epochs_meas'])
    
        
    
        # Initialize Layers
        y_est,x_,y_ = self.layers()
        
        # Initialize arrays to collect accuracy and error data during training
#        y_cost  = np.empty(self.nn_params['n_epochs'])
#        y_train_accuracy = np.zeros(self.nn_params['n_epochs'])
#        y_test_accuracy = np.zeros(int(self.nn_params['n_epochs']/
#                                        self.nn_params['n_epochs_meas']))
        
        # Initalize Tensorflow session
        sess = tf.Session()
        
        # Session Run Function
        sess_run = lambda f,x=[data['x_train'],data['y_train']]: sess.run(
                f,feed_dict = {xi:x for xi,x in zip([x_,y_],x)})
               
        # Initialize Accuracy Functions
        y_predictions = tf.equal(tf.argmax(y_est,axis=1),tf.argmax(y_,axis=1))
        
        train_acc = tf.reduce_mean(tf.cast(y_predictions, tf.float32))
        test_acc = train_acc         
        
        # Define Cost Function      
        cost = tf.reduce_mean(
                self.nn_params['neuron_func']['cost'][cost_f](y_,y_est))
        
                       
        # Define Learning Rate Corrections
        #global_step = tf.Variable(0, trainable=False)
        alpha_learn = self.nn_params['alpha_learn']
#        tf.train.exponential_decay(self.nn_params['apha_learn'],
#                              global_step, self.nn_params['n_epochs'],
#                              0.96, staircase=True)
#        
        # Training Output with Learning Rate Alpha
        train_step = self.nn_params[
                                   'neuron_func']['optimize'](alpha_learn,cost)

        
        # Initialize all tensorflow variables
        sess.run(tf.global_variables_initializer())
        
        
        # Initialize Results Dictionaries
        results_keys_train = ['cost','train_acc']
        results_keys_test =  ['test_acc', 'y_est']
        results_keys = results_keys_train + results_keys_test
        
        loc = locals()
        loc= {key:loc[key] for key in results_keys}

        results_func = {key: lambda *args: sess_run(loc[key],*args) 
                                            for key in results_keys}

        results = {key: [] for key in results_keys}
                        
        
        # Ability to save and restore all the variables
        #saver = tf.train.Saver(max_to_keep=self.nn_params['n_epochs'])  
        

    

        if train:
            
            print('Training...')
            epoch_range = np.arange(self.nn_params['n_epochs'])
            dataset_range = np.arange(self.nn_params['n_dataset_train'])
            batch = {}
            # Train Model over n_epochs with Stochastic Gradient Descent 
            for epoch in epoch_range:
                           
                # Divide training data into batches for training 
                # with Stochastic Gradient Descent
                for _ in dataset_range:
                    
                    # Choose Random Batch of data
                    np.random.shuffle(dataset_range)
                    
                    for key in data_params['data_types_train']:
                        batch[key] = data[key][dataset_range,:][
                                           0:self.nn_params['n_batch_train'],:]
                    
                    sess_run(train_step,[data[key] 
                                  for key in data_params['data_types_train']])
                    
                
                
                    
                if epoch+1 % self.nn_params['n_epochs_meas']:
                    
                    # Record Training Cross Entropy and Training Accuracy
    #                
    #                results['cost'].append(sess_run(cost,[batch_x,batch_y]))
    #                results['train_acc'].append(sess_run(y_acc,[batch_x,batch_y]))
                    for key,val in results.items():
                        if key in results_keys_train:
                            val.append(results_func[key]([data[k] 
                            for k in data_params['data_types_train']]))
                    
                    
                    print('Epoch: %d'% epoch +
                          'Testing Accuracy: '+str(results['train_acc'][-1])+
                          '\n') 
                    
#                    self.data_process(results,save=False,plot=True,
#                                      **{'x1': x1_grid, 'x2': x2_grid,
#                                       'x_train': data['x_train'],
#                                       'y_est':y_est,'f': sess_run,'K': K})
                                    
                    if test:
                        #t_acc = sess_run(y_acc,[data['x_test'],data['y_test']])
                        t_acc = results_func['test_acc'](
                                [data['x_test'],data['y_test']])
                        results['test_acc'].append(t_acc)
                        print('Epoch: %d'% epoch +
                              'Testing Accuracy: %0.6f'%(t_acc) + '\n')    

        
            # Save and Plot Accuracy and Error Data
            #self.data_process(results,save=False,plot=False)
        
        if test:
            self.testing('y_est',
                         [data[key] for key in data_params['data_types_test']])


        data_params['results'] = results
        data_params['results_func'] = results_func

        self.data_params = data_params
        
        # Calculate Mutual Information from Loaded Data
        #I,I_file = self.info_plane(plot=False)

        
    def testing(self,data_params,keys=None,*data):
        
        if data_params is None:
            data_params = self.data_params
        
        if keys is None:
            keys = data_params['results'].keys()
        
        items = zip(keys,data_params['results'].values())
        
        for key,val in items:
            val.append(data_params['results_func'][key](*data))
        
    
    def layers(self,sigma=0.1):
        
        def neuron_var(shape,sigma=0.1):
            initial = tf.truncated_normal(shape,stddev=sigma)
            return tf.Variable(initial)        
        
        
        # Define Numbers In + Out + Hidden Layers
        self.nn_params['n_layers'] = np.size(self.nn_params['n_neuron'])    
        
        # Initialize Weights, Biases and Input/Output Placeholders
        x_ = tf.placeholder(tf.float32, [None,self.nn_params['n_neuron'][0] ])
        y_ = tf.placeholder(tf.float32, [None,self.nn_params['n_neuron'][-1]])
        
        W = [None]*(self.nn_params['n_layers']-1)
        b = [None]*(self.nn_params['n_layers']-1)
        T = [None]*(self.nn_params['n_layers'])
        
        # Define Input
        T[0] = x_
        
        # Create Neuron Parameters in Layers
        for i in range(self.nn_params['n_layers']-1):
            Wshape = [self.nn_params['n_neuron'][i],
                      self.nn_params['n_neuron'][i+1]]
            bshape = [self.nn_params['n_neuron'][i+1]]
        
            W[i] = neuron_var(Wshape,sigma)
            b[i] = neuron_var(bshape,sigma)
            
            # Calculate Activation function for ith layer
            if i != self.nn_params['n_layers']:
                T[i+1] = self.nn_params['neuron_func']['layer'](
                                              tf.matmul(T[i],W[i]) + b[i])
            else:
                # Cross Entropy performs activation on output
                T[i+1] = self.nn_params['neuron_func']['output'](
                                              tf.matmul(T[i],W[i]) + b[i])
        
        
        # Define Ouput
        y_est = T[-1]
        
        return y_est,x_,y_
    
    
    
    def data_process(self,data,save=False,plot=True,**kwargs):
        
        
        n_data = np.size(data)
        
        # Save or Load Data and Plot Data
        file_names = lambda value: '{}epochs_'.format(n_data) + value
        
        
        if plot:
            fig, ax = plt.subplots(n_data+1)
            i = 0
        
        
        for key,val in data.items():
            
            if save:    
                np.savez_compressed(file_names(key),a=val)
            elif not val:
                val = np.load(file_names(key))['a']
        
            if plot:
                
                plt.sca(ax[i])
                plt.plot(np.arange(np.size(val)),val, color='r')
                plt.title(key)
                plt.ylabel(key)
                plt.xlabel('Epoch')
                
                i += 1
        
        if plot and kwargs:
            
            y_estimate = np.argmax(kwargs['f'](kwargs['y_est'],np.c_[kwargs['x1'].ravel(), kwargs['x2'].ravel()]), axis=1).reshape(kwargs['x1'].shape)
            
            plt.sca(ax[i])
            plt.contourf(kwargs['x1'], kwargs['x2'], y_estimate, kwargs['K'], alpha=0.8)
            plt.scatter(kwargs['x_train'][:, 0], kwargs['x_train'][:, 1], c=kwargs['y_train'], s=40)
            plt.xlim(x1_grid.min(), x1_grid.max())
            plt.ylim(x2_grid.min(), x2_grid.max())
            plt.xlabel('x1')
            plt.ylabel('x2')
        



    
#    def updatePlot(self,x_train,y_train,y_est,f,results):
#    
#          
#        y_estimate = np.argmax(f(y_est,np.c_[x1_grid.ravel(), x2_grid.ravel()]), axis=1)
#    
#        plt.clf()
#    
#    
#        ### Plot the classifier: ###
#        plt.subplot(121)
#        plt.contourf(x1_grid, x2_grid, y_estimate.reshape(x1_grid.shape), K, alpha=0.8)
#        plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, s=40)
#        plt.xlim(x1_grid.min(), x1_grid.max())
#        plt.ylim(x2_grid.min(), x2_grid.max())
#        plt.xlabel('x1')
#        plt.ylabel('x2')
#    
#        ### Plot the cost function during training: ###
#        plt.subplot((2,2,i))
#        plt.plot(np.arange(np.size(cost)),cost,'o-')
#        plt.xlabel('Epoch')
#        plt.ylabel('Training cost')
#    
#        ### Plot the training accuracy: ###
#        plt.subplot(224)
#        plt.plot(np.arange(np.size(acc)),acc,'o-')
#        plt.xlabel('Epoch')
#        plt.ylabel('Training accuracy')
#        
#        plt.pause(0.1)
#        plt.show()
        
    

if __name__ == '__main__':
    
    # Specify Plotting Parameters
    plt.ion() # turn on interactive mode (for plotting)

    #Specify font sizes for plots:
    plt.rcParams['axes.labelsize']  = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    
    
    # K Branch Data Set
    N = 50 # number of points per branch
    K = 3  # number of branches
    
    N_train = N*K # total number of points in the training set
    x_train = np.zeros((N_train,2)) # matrix containing the 2-dimensional datapoints
    y_train = np.zeros((N_train,1), dtype='uint8') # labels (not in one-hot representation)
    
    mag_noise = 0.05  # controls how much noise gets added to the data
    dTheta    = 4    # difference in theta in each branch
    
    ### Data generation: ###
    for j in range(K):
      ix = range(N*j,N*(j+1))
      r = np.linspace(0.01,1,N) # radius
      t = np.linspace(j*(2*np.pi)/K,j*(2*np.pi)/K + dTheta,N) + np.random.randn(N)*mag_noise # theta
      x_train[ix] = np.c_[r*np.cos(t), r*np.sin(t)]
      y_train[ix] = j
            
    
    ### Generate coordinates covering the whole plane: ###
    padding = 0.1
    spacing = 0.02
    x1_min, x1_max = x_train[:, 0].min() - padding, x_train[:, 0].max() + padding
    x2_min, x2_max = x_train[:, 1].min() - padding, x_train[:, 1].max() + padding
    x1_grid, x2_grid = np.meshgrid(np.arange(x1_min, x1_max, spacing),
                         np.arange(x2_min, x2_max, spacing))
    
    
    
    network_params = {'n_neuron': [None,4,None],'alpha_learn': 0.5, 
                  
                  'neuron_func':{
                       'layer':  tf.nn.sigmoid,
                       'output': tf.nn.sigmoid,
                       'cost': {
                         'cross_entropy': lambda y_label,y_est, eps= 10**(-6): 
                                         tf.reduce_mean(-tf.reduce_sum(
                                             y_label*tf.log(y_est+eps) +
                                          (1.0-y_label)*tf.log(1.0-y_est +eps),
                                          axis=1)),
                          'mse': lambda y_label,y_est: (1/2)*tf.reduce_mean(
                                  tf.reduce_sum(tf.square(y_est-y_label),
                                  axis=1)),
                          'entropy_logits': lambda y_label,y_est: 
                              tf.nn.sigmoid_cross_entropy_with_logits(
                                      labels=y_label, logits=y_est)},
                    'optimize': lambda a,c: tf.train.GradientDescentOptimizer(a).minimize(c)
                             },
                           
                  'n_epochs': 100,
                  'n_batch_train': 1/20, 'n_epochs_meas': 1/20,
                  }
    
    data_params =    {'data_files': [x_train,y_train],
                      'data_types_train': ['x_train','y_train'],
                      'data_types_test': ['x_test','y_test'],
                      'data_format': 'values',
                      'data_dir': 'dataset/',
                      'one_hot': True
                     }

    # Run Neural Network
    nn = neural_net(network_params).training(data_params,train=True)