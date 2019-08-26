#!/usr/bin/env python
# coding: utf-8


# Import standard python and jupyter modules
import numpy as np
import matplotlib,copy,importlib,os,argparse
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
np.seterr(divide='ignore',invalid='ignore');
matplotlib.rcParams['figure.max_open_warning'] = 0
#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
#get_ipython().run_line_magic('autosave', '60')




import data_process
#importlib.reload(data_process)
from data_process import importer,exporter

file = 'template.config'
config = importer([file],options={'atleast_1d':True})[file]
print(config)




import data_process
#importlib.reload(data_process)
from data_process import importer,exporter
a = np.empty((10,10,20000))
file = 'a.json'
files = [file,'b.json','c.npz']
dat = [{'meta1':1.0,'meta2':{'1':'3','45':4},'a':a[:,:,:10]},{'b':a},a]
directory  = "./"
data = {f:d for f,d in zip(files,dat)}
exporter(data)#,options={'encode':{'wrapper':lambda x:x}})
files_imp = importer(files,directory);
np.shape(files_imp[files[1]]['b'])
np.shape(files_imp[files[2]]['arr_0'])




list(files_imp[files[2]].keys())




type(dat_imp)




np.shape([['32434','12323']])




'%s'%type({})




alist = a.tolist()




np.shape(alist)




b=((1.0,2,3),(1,2,3))




np.shape(b)




barray = np.array(b)




barray.dtype




btuple = tuple(barray)




btuple




btuple = barray.totuple()




s = 'b 1 2 3 4'
s0,sn = s.split()[0],s.split()[1:]




sn




import data_process
#importlib.reload(data_process)
from data_process import importer,exporter
directory  = "./"
datas = [np.empty((200,200,1000))]
files = ['a.dat']
data = {f:d for f,d in zip(files,datas)}
exporter(data,directory)




aimp = importer(files,directory)[files[0]]




print(aimp)




'1212'+'1212'







