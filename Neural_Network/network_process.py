# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 18:44:58 2018

@author: Matt
"""
import numpy as np
import os

#            datafile = """C:/Users/Matt/Google Drive/PSI/PSI Essay/
#                PSI Essay Python Code/MachineLearning/
#                Tensorflow/mnist/config_labels.txt""" if \
#                                                datafile == None else datafile
#                        # Import Data in collumn 0 and Labels in collumn 1
#            datapoints = np.loadtxt(datafile,dtype=str,usecols=0)
#            data = np.array([[int(i) for i in x] for x in datapoints])
#            datalabels = np.loadtxt(datafile,dtype=int,usecols=1)



#        data_import_dict = {'txt':import_txt,'npz':import_npz,
#                            'mnist':import_mnist}
            
#        func_list = lambda obj,module: list(map(lambda x: 
#                            getattr(module,'import_'+x,lambda *args:[]),
#                            np.atleast_1d(obj)))
        # (data_files,data_types,data_format)




            # Initilialize random subset of  n_dataset_train elements of data
            # for training versus testing (~ 85% of dataset)
#            i_train = np.random.randint(0,n_batch_train,n_dataset_train)
#            i_test = [i for i in range(n_batch_train) if i not in i_train]
#            
#            x_train = data['x_train'][i_train]
#            y_train = data['y_train'][i_train]
#            
#            x_train = data['x_test'][i_test]
#            y_train = data['y_test'][i_test]


##### Initialize Network #######
def training_parameters(self,n_batch_train,n_dataset_train):
    
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

n_batch_train, n_dataset_train =  training_parameters(
                                        self.nn_params['n_batch_train'],
                                        self.nn_params['n_dataset_train'])
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
    