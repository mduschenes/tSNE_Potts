# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 11:04:42 2018

@author: Matt Duschenes
Machine Learning and Many Body Physics
PSI 2018 - Perimeter Institute
"""
import sys
sys.path.insert(0, "C:/Users/Matt/Google Drive/PSI/"+
                   "PSI Essay/PSI Essay Python Code/tSNE_Potts/")

from optimizer_methods import *
from Data_Process import Data_Process 
from ModelFunctions import display
array_sort =  Data_Process().array_sort

import numpy as np
import tensorflow as tf

tf.reset_default_graph()

seed=1234
np.random.seed(seed)
tf.set_random_seed(seed)



class optimizer(object):
    
    def __init__(self,opt_params):
                
        # Define Neural Network Properties Dictionary
        self.opt_params = opt_params  
       
        return


    def training(self,data_params = 
                     {'data_files': ['x_train','y_train','x_test','y_test'],
                      'data_sets': ['x_train','y_train','x_test','y_test'],
                      'data_types': ['train','test','other'],
                      'data_wrapper': lambda i,v: array_sort(i,v,0,'list'), 
                      'data_format': 'npz',
                      'data_dir': 'dataset/',
                      'one_hot': False
                     },
                     
                     alg_params = {'n_epochs':10, 
                                   'n_batch_train': 1/20,'n_epochs_meas': 1/20,
                                   'alpha_learn': 0.5, 'eta_reg': 0.005, 
                                   'sigma_var':0.1, 
                                   'cost_func': 'cross_entropy', 
                                   'optimize_func':'grad',
                                   'regularize': None,
                                   'method': 'layers',
                                  }, 
                     train = True, test = False, other = False,
                     timeit = True, printit=True,
                     save = False, plot = False,**plot_args):
    
        # Initialize Network Data and Parameters
        
        display(printit,timeit,'Neural Network Starting...')

        # Import Data
        Data_Proc = Data_Process()
        Data_Proc.plot_close()
        
        data, data_size = Data_Proc.importer(data_params)
               
        # Organize data by data_types (train, test, other),
        # and sorted in the the order of x,y, where it is assumed data_sets is
        # of the form ['x_train','y_train','x_test','y_test',*vars_other]
        data_typed = {t: [data[k] 
                          for k in data_params['data_sets'] if t in k]
                          for t in data_params['data_types']}
        data_sets=   {t: [k 
                          for k in data_params['data_sets'] if t in k]
                          for t in data_params['data_types']}
        
        
        
        display(printit,timeit,'Data Imported...')
                
        for t in data_params['data_types']:
            if not any(t in key for key in data.keys()):
                setattr(locals(), t, False)
        
        
        # Define Number of Neurons at Input and Output
        alg_params['n_dataset_train'],self.opt_params['n_neuron'][0] = (
                                                          data_size['x_train'])
        self.opt_params['n_neuron'][-1] = data_size['y_train'][1]
        alg_params['n_dataset_test'] = data_size.get('x_test',[None])[0]
        
        # Define Training Parameters
        alg_params['n_batch_train'] = max(1,int(alg_params['n_batch_train']*
                                                alg_params['n_dataset_train']))
        alg_params['n_epochs_meas'] = max(1,int(alg_params['n_epochs']*
                                                alg_params['n_epochs_meas']))
        
    
        # Depending on Network Type, either initialize Neural Network or 
        # Initialize Layers
        try:
            y_est,x_,y_ = getattr(self,alg_params['method'])(opt_params,alg_params)
        except:
            y_est,x_,y_ = locals().get(alg_params['method'])()
            
        
        display(printit,timeit,'Layers Initialized...')
        
               
        # Initalize Tensorflow session
        sess = tf.Session()
        
        # Session Run Function
        sess_run = lambda var,data_: sess.run(var,feed_dict={k:v 
                                            for k,v in zip([x_,y_],data_)})
                       
        # Initialize Lable, Accuracy and Cost Functions
        
        y_equiv = tf.equal(tf.argmax(y_est,axis=1),tf.argmax(y_,axis=1))
        
        train_acc = tf.reduce_mean(tf.cast(y_equiv, tf.float32))
        test_acc  = tf.reduce_mean(tf.cast(y_equiv, tf.float32))
        other_acc = tf.reduce_mean(tf.cast(y_equiv, tf.float32))
                       
        # Define Learning Rate Corrections
        #global_step = tf.Variable(0, trainable=False)
        alpha_learn = alg_params['alpha_learn']
#        tf.train.exponential_decay(self.opt_params['apha_learn'],
#                              global_step, self.opt_params['n_epochs'],
#                              0.96, staircase=True)
       
        
        # Define Cost Function (with possible Regularization)
        cost = tf.reduce_mean(self.opt_params['cost_func'][
                                            alg_params['cost_func']](y_,y_est))
        
        if alg_params['regularize']:
            cost += self.opt_params['cost_func'][
                 alg_params['regularize']]([v for v in tf.trainable_variables() 
                                                       if 'reg' in v.name],
                                           alg_params['eta_reg'])
        
        
        
        # Training Output with Learning Rate Alpha and regularization Eta
        train_step = self.opt_params['optimize_func'][
                                 alg_params['optimize_func']](alpha_learn,cost)

         
       
        
        # Initialize Results Dictionaries
        results_keys = {}
        results_keys['train'] = ['train_acc','cost']
        results_keys['test'] =  ['test_acc']
        results_keys['other'] = ['y_equiv','y_est']
        results_keys['all'] = (results_keys['train']+results_keys['test']+
                                results_keys['other'])
             
        loc = vars()
        loc = {key:loc.get(key) for key in results_keys['all'] 
                               if loc.get(key)}
        results_keys['all'] = loc.keys()        
        
        
        # Results Dictionary of Array for results values
        results = {key: [] for key in results_keys['all']}
                           
        # Results Dictionary of Functions for results       
        results_func = {}
        for key in results_keys['all']:
            results_func[key] = lambda feed_dict : sess_run(loc[key],feed_dict) 
        
        
        display(printit,timeit,'Results Initialized...')
           
        
        
        
        
        # Optional Plot Arguments
        if plot_args.get('y_estimate'):
            plot_args['plot_f'] = lambda **x: plot_args['y_estimate'](
                                                          [sess_run,y_est],**x)
        

        plot_args['plot_title'] = '\n'.join([str(k)+':  '+str(alg_params[k])
                           for i,k in 
                           enumerate(sorted(list(alg_params.keys())))])
        
        plot_args['fig_title'] = '%depochs_'%alg_params['n_epochs']
        plot_args['xlabel'] = {k: 'epochs' if k not in results_keys['other'] 
                                           else str(data_sets['other'][0]) 
                               for k in results_keys['all']}
        
        
        
        
        # Initialize all tensorflow variables
        sess.run(tf.global_variables_initializer())
        
        display(printit,timeit,'Training...')
        




        # Train Model
        if train:
            
            epoch_range = range(alg_params['n_epochs'])
            dataset_range = np.arange(alg_params['n_dataset_train'])
            batch_range = range(0,alg_params['n_dataset_train'],
                                    alg_params['n_batch_train'])
            
            def domain(i = alg_params['n_epochs'],
                       f = lambda i,*k: list(range(*i))):
                
                if not hasattr(i,'__iter__'):
                    i = (0,i,alg_params['n_epochs_meas'])
                
                if callable(f):
                    return {k: f(i,k) 
                            for k in results_keys['all'] }
                else:
                    return {k: f for k in results_keys['all']}
            
                        
            # Train Model over n_epochs with Gradient Descent 
            for epoch in epoch_range:
                           
                # Divide training data into batches for training 
                # with Stochastic Gradient Descent
                np.random.shuffle(dataset_range)                       
                for i_batch in batch_range:
                    
                    # Choose Random Batch of data                    
                    sess_run(train_step,[d[dataset_range,:][i_batch:
                                   i_batch + alg_params['n_batch_train'],:]
                                   for d in data_typed['train']])
                
            
                # Record Results every n_epochs_meas
                if (epoch+1) % alg_params['n_epochs_meas'] == 0:
                    
                    # Record Results: Cost and Training Accuracy
                    for key,val in results.items():
                        if train and key in results_keys['train']:
                            val.append(results_func[key](data_typed['train'])) 
                        
                        elif test and key in results_keys['test']:
                            val.append(results_func[key](data_typed['test']))
                        
                        elif other and key in results_keys['other']:
                            val.append(results_func[key](data_typed['test']))
                                                


                     # Display Results
                    display(printit,timeit,'\n Epoch: %d'% epoch + 
                          '\n'+
                          'Training Accuracy: '+str(results['train_acc'][-1])+
                          '\n'+
                          'Testing Accuracy: '+str(results['test_acc'][-1])+
                          '\n'+
                          'Cost:             '+str(results['cost'][-1])+
                          '\n')

            
                    # Save and Plot Data
                    Data_Proc.plotter(results,domain(epoch+1),data_params,
                                    results_keys['train']+results_keys['test'],
                                    save=save,plot=plot,**plot_args)
            
            

    
        # Make Results Class variable for later testing with final network
        data_params['results'] = results
        data_params['results_func'] = results_func

        self.data_params = data_params
        
       
        # Process Other Data with final trained values
        if other:            
            # Sort and Average over each array from other results
            results_sort_avg = {i: {k:[] for k in results_keys['other']}
                                  for i in range(np.size(data_typed['other']))}
                        
            for i,d in enumerate(data_typed['other']):

                d_sort = {}

                for k in results_keys['other']:

                    results_sort,d_sort[i] = data_params['data_wrapper'](
                                                              results[k][-1],d) 
                                               
                    results_avg = dict(zip(data_params['label_titles'][i],
                                    [np.array(list(map(lambda i: np.mean(i,0), 
                                    results_sort)))[:,i] 
                                   for i in range(np.shape(results_sort)[1])]))
                    
                    results_sort_avg[i][k] = results_avg
                    d_sort[i] = domain(f=d_sort[i])
                Data_Proc.process(results_sort_avg[i],d_sort[i],data_params,
                                      results_keys['other'],
                                      save=save,plot=True,**plot_args)
                
                
                    
        
        # Plot Final Results
        Data_Proc.process(results,domain(epoch+1),data_params,
                                    results_keys['train']+results_keys['test'],
                                    save=save,plot=True,**plot_args)
        
        # Save Final Figures
        Data_Proc.plot_save(data_params['data_dir'],plot_args['fig_title'])


    
    def testing(self,results,data,data_func):
        
        for key,val in results.items():
            val.append(data_func.get(key,lambda x: x)(data))
            
        return results
        
    
    
    
    
    
    def neural_network(self,sigma=0.1,**kwargs):
        
        def neuron_var(shape,name,sigma=0.1):
            initial = tf.truncated_normal(shape,stddev=sigma)
            return tf.Variable(initial,name=name)        
        
        
        # Define Numbers In + Out + Hidden Layers
        self.opt_params['n_layers'] = np.size(self.opt_params['n_neuron']) 
        
        # Define Type of Layers (fcc: Fully Connected, cnn: Convolutional)
        if not self.opt_params.get('layers'):
            self.opt_params['layers'] = ['fcc']*(self.opt_params['n_layers']-1)
        
        # Initialize Weights, Biases and Input/Output Placeholders
        x_ = tf.placeholder(tf.float32, [None,self.opt_params['n_neuron'][0] ])
        y_ = tf.placeholder(tf.float32, [None,self.opt_params['n_neuron'][-1]])
        
        W = [None]*(self.opt_params['n_layers']-1)
        b = [None]*(self.opt_params['n_layers']-1)
        T = [None]*(self.opt_params['n_layers'])
        
        # Define Input
        T[0] = x_
        
        # Create Neuron Parameters in Layers
        for i in range(self.opt_params['n_layers']-1):
            if self.opt_params.get('layers')[i] == 'fcc':
                Wshape = [self.opt_params['n_neuron'][i],
                          self.opt_params['n_neuron'][i+1]]
                bshape = [self.opt_params['n_neuron'][i+1]]
            
                W[i] = neuron_var(Wshape,'weights_reg_%d'%i,sigma)
                b[i] = neuron_var(bshape,'biases_%d'%i,sigma)
            
            # Calculate Activation function for ith layer and Output
            if i != self.opt_params['n_layers']:
                T[i+1] = self.opt_params['neuron_func']['layer'](
                                              tf.matmul(T[i],W[i]) + b[i])
            else:
                T[i+1] = self.opt_params['neuron_func']['output'](
                                              tf.matmul(T[i],W[i]) + b[i])

        # Define Ouput
        y_est = T[-1]
        
        return y_est,x_,y_
    

    






if __name__ == '__main__':
    

    # Data Set
    data_files = ['x_train','y_train','x_test','y_test','T_test']
    data_sets = ['x_train','y_train','x_test','y_test','T_other']
    data_format = 'npz'
    one_hot = False
    kwargs = {}

    # Neural Network Parameters    
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
    
    # Dataset Parameters
    data_params =  {'data_files': data_files,
                    'data_sets': data_sets,
                    'data_types': ['train','test','other'],
                    'data_wrapper': lambda i,v: array_sort(i,v,0,'list'),
                    'label_titles': [['T>Tc','T<Tc']],
                    'data_format': data_format,
                    'data_dir': 'dataset/',
                    'one_hot': one_hot
                   }

   
    # Algorithm Parameters
    alg_params = { 'alpha_learn': 0.0035, 'eta_reg': 0.0005,'sigma_var':0.1,                  
                  
                  'n_epochs':5 ,'n_batch_train': 1/10, 'n_epochs_meas': 1/20,
                 
                  'cost_func': 'cross_entropy', 'optimize_func':'adam',
                  'regularize':'L2',
                  'method': 'neural_network',
                  }
    
    
    
    # Run Neural Network   
    nn = optimizer(network_params)
    nn.training(data_params,alg_params,
                train=True,test=True,other=True,
                plot=True,save=False,
                printit=True,timeit=True,**kwargs)
    
    