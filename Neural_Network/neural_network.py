# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 11:04:42 2018

@author: Matt
"""

#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

from data_process import Data_Process

import numpy as np
import tensorflow as tf
import time

tf.reset_default_graph()

seed=1234
np.random.seed(seed)
tf.set_random_seed(seed)


times = [time.clock()]
def display(printit=False,timeit=False,m=''):
    if timeit:
        times.append(time.clock())
        if printit:
            print(m,times[-1]-times[-2])
    elif printit:
        print(m)

def array_sort(a,b,axis=0,dtype='list'):
    # Sort 2-dimensional a by elements in 1-d array b
    
    b = np.reshape(b,(-1,))
    
    if dtype == 'dict':
        return {i: np.take(a,np.where(b==i),axis) for i in set(b)}
    elif dtype == 'list':
        return [np.take(a,np.where(b==i),axis) for i in set(b)]
    elif dtype == 'sorted':
        return np.concatenate(list(np.take(a,np.where(b==i),axis)
                                   for i in set(b)),1)
    else:
        return a



#from mutual_information import MUT_INFO


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
                      'data_types': ['x_train','y_train','x_test','y_test'],
                      'data_format': 'npz',
                      'data_dir': 'dataset/',
                      'one_hot': False
                     },
                     
                     alg_params = {'n_epochs':10, 
                                   'n_batch_train': 1/20,'n_epochs_meas': 1/20,
                                   'alpha_learn': 0.5, 'eta_reg': 0.005,                  
                                   'cost_func': 'cross_entropy', 
                                   'optimize_func':'grad',
                                   'regularize': None
                                  }, 
                     train = True, test = False, other = True,
                     timeit = True, printit=True,
                     save = False, plot = False,**plot_args):
    
        # Initialize Network Data and Parameters
        
        display(printit,timeit,'Neural Network Starting...')

        # Import Data
        Data_Proc = Data_Process()
        Data_Proc.plot_close()
        
        data, data_size = Data_Proc.importer(data_params)

        print(data_size)
        
        data_typed = {t: {k[0]: data[k] 
                          for k in data_params['data_types'] if t in k} 
                          for t in ['train','test','other']}
        
        
        display(printit,timeit,'Data Imported...')
                
        if not any('test' in key for key in data.keys()):
            test = False
        
        
        # Define Number of Neurons at Input and Output
        alg_params['n_dataset_train'],self.nn_params['n_neuron'][0] = (
                                                          data_size['x_train'])
        self.nn_params['n_neuron'][-1] = data_size['y_train'][1]
        alg_params['n_dataset_test'] = data_size.get('x_test',[None])[0]
        
        # Define Training Parameters
        alg_params['n_batch_train'] = max(1,int(alg_params['n_batch_train']*
                                                alg_params['n_dataset_train']))
        alg_params['n_epochs_meas'] = max(1,int(alg_params['n_epochs']*
                                                alg_params['n_epochs_meas']))
        
    
        # Initialize Layers
        y_est,x_,y_ = self.layers()
        
        display(printit,timeit,'Layers Initialized...')
        
               
        # Initalize Tensorflow session
        sess = tf.Session()
        
        # Session Run Function
        sess_run = lambda var,data_: sess.run(var,feed_dict={k:data_[v] 
                                            for k,v in zip([x_,y_],['x','y'])})
                       
        # Initialize Lable, Accuracy and Cost Functions
        y_out = tf.reduce_mean(y_est,axis=0)
        
        y_equivalence = tf.equal(tf.argmax(y_est,axis=1),tf.argmax(y_,axis=1))
        
        train_acc = tf.reduce_mean(tf.cast(y_equivalence, tf.float32))
        test_acc  = tf.reduce_mean(tf.cast(y_equivalence, tf.float32))
        other_acc = tf.reduce_mean(tf.cast(y_equivalence, tf.float32))
                       
        # Define Learning Rate Corrections
        #global_step = tf.Variable(0, trainable=False)
        alpha_learn = alg_params['alpha_learn']
#        tf.train.exponential_decay(self.nn_params['apha_learn'],
#                              global_step, self.nn_params['n_epochs'],
#                              0.96, staircase=True)
       
        
        # Define Cost Function (with possible Regularization)
        cost = tf.reduce_mean(self.nn_params['cost_func'][
                                            alg_params['cost_func']](y_,y_est))
        
        if alg_params['regularize']:
            cost += self.nn_params['cost_func'][
                 alg_params['regularize']]([v for v in tf.trainable_variables() 
                                                       if 'weights' in v.name],
                                           alg_params['eta_reg'])
        
        
        
        # Training Output with Learning Rate Alpha and regularization Eta
        train_step = self.nn_params['optimize_func'][
                                 alg_params['optimize_func']](alpha_learn,cost)

         
       
        
        # Initialize Results Dictionaries
        results_keys_train = ['train_acc','cost']
        results_keys_test =  ['test_acc']
        results_keys_other = ['other_acc']
        results_keys = results_keys_train+results_keys_test+results_keys_other
             
        loc = vars()
        loc= {key:loc.get(key) for key in results_keys 
                               if loc.get(key) is not None}
        results_keys = loc.keys()        
        
        
        
        # Results Dictionary of Array for results values
        results = {key: [] if key not in results_keys_other 
                           else {k: {vi: [] for vi in np.reshape(v,(-1,))} 
                                 for k,v in data_typed['other'].items()}
                   for key in results_keys}        
        
         # Results Dictionary of Functions for results       
        results_func = {}
        for key in results_keys:
            results_func[key] = lambda feed_dict: sess_run(loc[key],feed_dict) 
        
        display(printit,timeit,'Results Initialized...')
        
        # Ability to save and restore all the variables
        #saver = tf.train.Saver(max_to_keep=self.nn_params['n_epochs'])  
        
        
        
        
        
        # Optional Other Analysis of Data
        
        # Sort Data based on unique elements in Other Data Arrays
        if other:
            # Sort data_test to {key_other: {key_test: val_test_(other_sort)} }
            data_typed['sorted'] = Data_Proc.dict_sort(data_typed['other'],
                                                       data_typed['test'])
            
            
            
        
        # Optional Plot Arguments
        if plot_args.get('y_estimate'):
            plot_args['plot_f'] = lambda **x: plot_args['y_estimate'](
                                                          [sess_run,y_est],**x)
        

        plot_args['plot_title'] = '\n'.join([str(k)+':  '+str(alg_params[k])
                           for i,k in 
                           enumerate(sorted(list(alg_params.keys())))])
        
        plot_args['fig_title'] = '%depochs_'%alg_params['n_epochs']
        
        # Initialize all tensorflow variables
        sess.run(tf.global_variables_initializer())
        
        display(printit,timeit,'Training...')
        




        # Train Model
        if train:
            
            epoch_range = range(alg_params['n_epochs'])
            dataset_range = np.arange(alg_params['n_dataset_train'])
            batch_range = range(0,alg_params['n_dataset_train'],
                                    alg_params['n_batch_train'])
            
            domain = lambda e=alg_params['n_epochs']:  {
                          key: list(range(0,e,alg_params['n_epochs_meas']))
                          for key in results_keys }
            
                        
            # Train Model over n_epochs with Gradient Descent 
            for epoch in epoch_range:
                           
                # Divide training data into batches for training 
                # with Stochastic Gradient Descent
                np.random.shuffle(dataset_range)                       
                for i_batch in batch_range:
                    
                    # Choose Random Batch of data                    
                    sess_run(train_step,{k:v[dataset_range,:][i_batch:
                                   i_batch + alg_params['n_batch_train'],:]
                                   for k,v in data_typed['train'].items()})
                
            
                # Record Results every n_epochs_meas
                if (epoch+1) % alg_params['n_epochs_meas'] == 0:
                    
                    # Record Results: Cost and Training Accuracy
                    for key,val in results.items():
                        if key in results_keys_train and train:
                            val.append(results_func[key](data_typed['train'])) 
                        
                        elif key in results_keys_test and test:
                            val.append(results_func[key](data_typed['test']))
                        
                        elif key in results_keys_other and other:
                            for k,v in data_typed['other'].items():
                                for vi in v.flatten():
                                    val[k][vi].append(results_func[key](
                                                  data_typed['sorted'][k][vi]))
                    


                     # Display Results
                    display(printit,False,'Epoch: %d'% epoch + 
                          '\n'+
                          'Training Accuracy: '+str(results['train_acc'][-1])+
                          '\n'+
                          'Testing Accuracy: '+str(results['test_acc'][-1])+
                          '\n'+
                          'Cost:             '+str(results['cost'][-1]))

            
                    # Save and Plot Data
                    Data_Proc.process(results,domain(epoch+1),data_params,
                                      results_keys_train+results_keys_test,
                                      save=save,plot=plot,**plot_args)
            
            

    
        # Make Results Class variable for later testing with final network
        data_params['results'] = results.copy
        data_params['results_func'] = results_func.copy

        self.data_params = data_params.copy
        

        # Save Figure of Results
        Data_Proc.plot_save(data_params['data_dir'],plot_args['fig_title'])


    
    def testing(self,results,data,data_func):
        
        for key,val in results.items():
            val.append(data_func.get(key,lambda x: x)(data))
            
        return results
        
    
    
    
    
    
    def layers(self,sigma=1):
        
        def neuron_var(shape,name,sigma=0.1):
            initial = tf.truncated_normal(shape,stddev=sigma)
            return tf.Variable(initial,name=name)        
        
        
        # Define Numbers In + Out + Hidden Layers
        self.nn_params['n_layers'] = np.size(self.nn_params['n_neuron']) 
        
        # Define Type of Layers (fcc: Fully Connected, cnn: Convolutional)
        if not self.nn_params.get('layers'):
            self.nn_params['layers'] = ['fcc']*(self.nn_params['n_layers']-1)
        
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
            if self.nn_params.get('layers')[i] == 'fcc':
                Wshape = [self.nn_params['n_neuron'][i],
                          self.nn_params['n_neuron'][i+1]]
                bshape = [self.nn_params['n_neuron'][i+1]]
            
                W[i] = neuron_var(Wshape,'weights_%d'%i,sigma)
                b[i] = neuron_var(bshape,'biases_%d'%i,sigma)
            
            # Calculate Activation function for ith layer and Output
            if i != self.nn_params['n_layers']:
                T[i+1] = self.nn_params['neuron_func']['layer'](
                                              tf.matmul(T[i],W[i]) + b[i])
            else:
                T[i+1] = self.nn_params['neuron_func']['output'](
                                              tf.matmul(T[i],W[i]) + b[i])

        # Define Ouput
        y_est = T[-1]
        
        return y_est,x_,y_


if __name__ == '__main__':
    
    # Data Set
    
#    from network_datasets import spiral_dataset 
#    x_train,y_train,plot_data = spiral_dataset()
#    x_test = None
#    y_test = None
#    data_format = 'values'
#    one_hot = True
#    kwargs = {'y_estimate': plot_data}


    data_files = ['x_train','y_train','x_test','y_test','T_test']
    data_types = ['x_train','y_train','x_test','y_test','T_other']
    data_format = 'npz'
    one_hot = False
    kwargs = {}
    
    network_params = {'n_neuron': [None,100,None],                  
                     'neuron_func':{
                       'layer':  tf.nn.sigmoid,
                       'output': tf.nn.sigmoid
                                  },
                       'cost_func': {
                         'cross_entropy': lambda y_label,y_est,eps= 10**(-8): 
                                         -tf.reduce_sum(
                                         y_label*tf.log(y_est+eps) +
                                         (1.0-y_label)*tf.log(1.0-y_est +eps),
                                         axis=1),
                                                 
                          'mse': lambda y_label,y_est: (1/2)*(
                                        tf.reduce_sum(tf.square(y_est-y_label),
                                        axis=1)),
                          
                          'entropy_logits': lambda y_label,y_est: 
                                  tf.nn.sigmoid_cross_entropy_with_logits(
                                                 labels=y_label, logits=y_est),
                          'L2': lambda var,eta_reg: tf.add_n([ tf.nn.l2_loss(v) 
                                             for v in var]) * eta_reg
                                },
                                      
                    'optimize_func': {'grad': lambda a,c: 
                                            tf.train.GradientDescentOptimizer(
                                                                a).minimize(c),
                                     'adam': lambda a,c: 
                                            tf.train.AdamOptimizer(
                                                                a).minimize(c)
                                },
                           
                  }
    
    data_params =    {'data_files': data_files,
                      'data_types': data_types,
                      'data_format': data_format,
                      'data_dir': 'dataset/',
                      'one_hot': one_hot
                     }

   
    alg_params = { 'alpha_learn': 0.5, 'eta_reg': 0.005,                  
                  
                  'n_epochs':100 ,'n_batch_train': 1/20, 'n_epochs_meas': 1/20,
                  
                  'cost_func': 'cross_entropy', 'optimize_func':'grad',
                  'regularize':None}
    
    # Run Neural Network   
    
    
    nn = neural_net(network_params)
    datT = nn.training(data_params,alg_params,
                train=True,test=True,other=False,
                plot=True,save=False,
                printit=True,timeit=True,**kwargs)
    
    #np.savez_compressed('dataset/' + 'T_test',a=np.reshape(dT,(-1,1)))
    
    