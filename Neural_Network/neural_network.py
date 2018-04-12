# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 11:04:42 2018

@author: Matt
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from mutual_information import MUT_INFO


class neural_net(object):
    def __init__(self,n_neuron = [10,7,5,4,3],
                      alpha_learn = 0.5,neuron_activation=None,
                      n_epochs=None,n_batch_train=None,n_dataset_train=None,
                      dataset_file=None):
        
        # Define Layers

        # Array of Number of Neurons per Layer, 
        # excluding input and ouptut layers
        
        self.n_neuron = n_neuron
        self.alpha_learn = alpha_learn
        self.n_layers = len(n_neuron)     # Number of HIDDEN layers
        
        # Define Neuron Activation Functions as 
        # [Neuron Function, Output Function, Cross Entropy Function]
        if neuron_activation == None:
            self.neuron_activation = [tf.tanh,
                                      lambda x,y:
                                      tf.nn.sigmoid_cross_entropy_with_logits(
                                              labels=x, logits=y),
                                      tf.sigmoid]
        else:
            self.neuron_activation = neuron_activation
        
        
        # Define Training Parameters
        self.n_epochs = n_epochs       # Number of epochs to train
        
        
        # Define directories 
        self.data_folder = "dataset/"
        self.dataset_file = ["configs.npy","labels.npy","labels64.npy"]
        self.save_folder = ["model_data/accuracy_data/",
                            "model_data/self.session_data/"]
        
        for i,f in enumerate(self.save_folder):
            self.save_folder[i] = self.data_folder + f
            if not os.path.exists(self.save_folder[i]):
                # os.remove(data_folder+f)
                os.makedirs(self.save_folder[i])
    

        self.session_file = lambda i="": (self.save_folder[1]+
                                             "network_model_{}".format(i))
        
        
        # Train Neural Network Data with 
        # Number of SGD Batch Size and Training Dataset Size
        np.random.seed(111)
        #self.training(n_batch_train,n_dataset_train)
        
        return
        


    def training(self,n_batch_train,n_dataset_train):
    
        
        def training_parameters(n_batch_train,n_dataset_train):
            
            # Define Number of Neurons at Input and Output
            self.n_neuron.insert(0,self.n_x)
            self.n_neuron.append(self.n_y)
            
            # Define Number of training Epochs
            if self.n_epochs == None: 
                self.n_epochs = int(np.size(self.x,0)/10)
            # Define SGD Batch Size
            if n_batch_train == None:   # Batch size for SGD
                n_batch_train = int(np.size(self.x,0)/100)
            
            # Define size of training data set
            if n_dataset_train == None:
                n_dataset_train = int(np.size(self.x,0)*85/100) 
            
            return n_batch_train, n_dataset_train
        
        
        # Import Data and get Data Parameters
        self.data_import()
        
        n_batch_train, n_dataset_train =  training_parameters(
                                                 n_batch_train,n_dataset_train)
        # Initialize Layers
        self.layers()
        
         # Initialize arrays to collect accuracy and error data during training
        y_cross_entropy  = np.zeros(self.n_epochs)
        y_train_accuracy = np.zeros(self.n_epochs)
        
        test_trials = 10
        y_test_accuracy = np.zeros(int(self.n_epochs/test_trials))
        
        y_predictions = tf.equal(tf.argmax(
                self.neuron_activation[2](self.y_est),1), tf.argmax(self.y_,1))
        y_accuracy = tf.reduce_mean(tf.cast(y_predictions, tf.float32))
              
        
        # Define Cross Entropy Cost Function      
        self.cross_entropy = tf.reduce_mean(self.neuron_activation[2](
                                              self.y_,self.y_est))
        
        # Define Learning Rate Corrections
        self.global_step = tf.Variable(0, trainable=False)
        self.alpha_learn = tf.train.exponential_decay(self.alpha_learn,
                              self.global_step, self.n_epochs,
                              0.96, staircase=True)
        self.train_step = tf.train.GradientDescentOptimizer(
                                      self.alpha_learn).minimize(
                                              self.cross_entropy)
        
        # Training Output with learning rate alpha
        self.y_step = tf.train.GradientDescentOptimizer(
                self.alpha_learn).minimize(self.cross_entropy)
        
        
        # Initalize Tensorflow self.session
        self.sess = tf.Interactiveself.session()
        tf.global_variables_initializer().run()
        
        # Ability to save and restore all the variables
        self.saver = tf.train.Saver(max_to_keep=self.n_epochs)  
        
        
        # Train Model with random subset of data set over self.n_epochs
            
        for j in range(self.n_epochs):
            
            # Initilialize random subset of  n_dataset_train elements of data
            # for training versus testing (~ 85% of dataset)
            i_train = np.random.choice(range(n_batch_train),n_dataset_train)
            i_test = [i for i in range(n_batch_train) if i not in i_train]
            
            x_train = self.x[i_train]
            y_train = self.y[i_train]
            
            x_test = self.x[i_test]
            y_test = self.y[i_test]
            
            # Divide training data into batches for training 
            # with Stochastic Gradient Descent
            for i in range(0,n_dataset_train,n_batch_train):
                batch_x = x_train[i:i+n_batch_train,:]
                batch_y = y_train[i:i+n_batch_train,:]
                self.sess.run(self.y_step,
                              feed_dict={self.x_:batch_x, self.y_:batch_y}) 
                
            # Record Training Cross Entropy and Training Accuracy
            y_cross_entropy[j]  = self.sess.run(self.cross_entropy,
                                feed_dict={self.x_: batch_x, self.y_: batch_y})
            y_train_accuracy[j] = y_accuracy.eval(
                                feed_dict={self.x_: batch_x, self.y_: batch_y})

            # Print Test Data Accuracy
            if j % int(self.n_epochs/test_trials) == 0:
                y_test_accuracy[int(j/test_trials-1)] = self.sess.run(
                        y_accuracy, feed_dict={self.x_:x_test, self.y_:y_test})
                print('Epoch: ',j)
                print('Label Accuracy: ',y_test_accuracy[int(j/test_trials-1)])    
                print('\n')
                
        
        # Save and Plot Accuracy and Error Data
        self.results = [y_train_accuracy, y_test_accuracy, y_cross_entropy]
        self.data_process(self.n_epochs,save=False,plot=False)
        
        # Calculate Mutual Information from Loaded Data
        I,I_file = self.info_plane(plot=False)

        
        return
    
    
    
    def layers(self):
        
        def neuron_var(shape,sigma=0.1):
            initial = tf.truncated_normal(shape,stddev=sigma)
            return tf.Variable(initial)        
        
        
        # Initialize Weights, Biases and Input/Output Placeholders
        self.x_ = tf.placeholder(tf.float32,[None,self.n_x])
        self.y_ = tf.placeholder(tf.float32, [None,self.n_y])
        
        W = [None]*(self.n_layers+1)
        b = [None]*(self.n_layers+1)
        self.T = [None]*(self.n_layers+2)
        
        self.T[0] = self.x_
        
        # Pass data through Layers
        for i in range(self.n_layers+1):
            Wshape = [self.n_neuron[i],self.n_neuron[i+1]]
            bshape = [self.n_neuron[i+1]]
            
            W[i] = neuron_var(Wshape)
            b[i] = neuron_var(bshape)
            
            # Calculate Activation function for ith layer
            if i != self.n_layers:
                self.T[i+1] = self.neuron_activation[0](
                                              tf.matmul(self.T[i],W[i]) + b[i])
            else:
                # Cross Entropy performs activation on output
                self.T[i+1] = tf.matmul(self.T[i],W[i]) + b[i] 
        
        # Define Ouput
        self.y_est = self.T[-1]
        
        return
    
    


    
    def data_import(self,datatype = 'text',trainingdata=True):
        # Import Data
        
        def one_hot(x,n=None):
            x = x.astype(np.int32)
            if n==None:
                n = int(max(x)+1)
            y = np.zeros([len(x),n],dtype=np.int32)
            for i in range(n):
                p = np.zeros(n)
                np.put(p,i,1)
                y[x==i,:] = p
            return y
        
        
        def textimport():

#            datafile = """C:/Users/Matt/Google Drive/PSI/PSI Essay/
#                PSI Essay Python Code/MachineLearning/
#                Tensorflow/mnist/config_labels.txt""" if \
#                                                datafile == None else datafile
#                        # Import Data in collumn 0 and Labels in collumn 1
#            datapoints = np.loadtxt(datafile,dtype=str,usecols=0)
#            data = np.array([[int(i) for i in x] for x in datapoints])
#            datalabels = np.loadtxt(datafile,dtype=int,usecols=1)
            
            # Import x data
            self.x = np.load(self.dataset_file[0])

            # Import x labels
            self.y = one_hot(np.load(self.dataset_file[1]))
            
            # Define Number of Datasets, Size of x and y datapoints
            self.n_dataset,self.n_x = np.shape(self.x)
            self.n_y = np.size(self.y,1)
            
            return
        
        def mnistimport():
#            datafile = "MNIST_data/" if datafile == None else datafile
#                
#            self.x = datafile #input_data.read_data_sets(
#            datafile,one_hot=True)
#             
#            if self.n_epochs == None: 
#                self.n_epochs = int(np.size(self.x,0)/4)
#            if self.n_batch_train == None:
#                self.n_batch_train = int(np.size(self.x,0)/100)
#             
            return
        
        data_import_dict = {'text':textimport,'mnist':mnistimport}
        
        data_import_dict[datatype]()
        
        
        return
    
    
        # Plot Information Plane
    def info_plane(self, n_bins = 1000,plot=False):
        # Plot Mutual Information for each Layer from saved epoch data
        
        # Initialize Mutual Information
        m = MUT_INFO(n_bins=n_bins, dataset_file=self.dataset_file)
        I = np.empty([self.n_layers-1,2,self.n_epochs])
        
        # Plotting Properties
        colormap = plt.cm.gist_ncar
        plt.gca().set_color_cycle([colormap(i) 
                                for i in np.linspace(0, 0.9, self.n_epochs)])
        
        # Perform MI calculation [I(X,T), I(Y,T)] for each layer per epoch
        for epoch in range(self.n_epochs):
            # Import each epoch self.session file
            saver_epoch = tf.train.import_meta_graph(
                    self.session_file(epoch+1)+'.meta')
            saver_epoch.restore(
                    tf.Session(),self.session_file(epoch+1)+'.ckpt')    
            
            for i,T_layer in enumerate(self.T[1:]):
                I[i,:,epoch] = m.mut_info(
                        T_layer.eval(feed_dict={self.x_: self.x}))
            
        # Save Data
        file_name = self.data_folder+(
                               '{}epochs_Mut_Info_XT_YT'.format(self.n_epochs))
        np.savez_compressed(file_name, a=I)
        
        # Plot Information Plane
        if plot:
            for epoch in range(self.n_epochs):
                if epoch == 0:
                    plt.plot(I[:,0,epoch],I[:,1,epoch],
                             '-*',label='Layer: '+str(i))
                    plt.legend(loc=2,numpoints=1,prop={'size': 6})
                else:
                    plt.plot(I[:,0,epoch],I[:,1,epoch],'-*')
    
            plt.show()
        
        return I,file_name+'{}epochs_Mut_Info_XT_YT'.format(self.n_epochs)
    
    
    
    def data_process(self,save=True,plot=True):
        
        # Save or Load Data and Plot Data
        result_names = ['train_acc','test_acc','err_list']
        file_names = lambda i:'{}epochs_'.format(self.n_epochs)+result_names[i]
        n_results = len(self.results)
        
        
        for i in range(n_results):
            if save:    
                np.savez_compressed(file_names(i),a=self.results[i])
            elif not self.results:
                self.results[i] = np.load(file_names(i))['a']
        
            if plot:
                plt.figure(figsize=(15,4))
                
                for i in range(len(self.results)):
                    plt.subplot(1,3,1)
                    plt.plot(np.arange(len(self.results[i])),
                                           self.results[i], color='r')
                    plt.title(result_names[i])
                    plt.ylabel(result_names[i])
                    plt.xlabel('Epoch')
        



    

if __name__ == '__main__':
    nn = neural_net()