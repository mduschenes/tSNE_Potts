# Import standard python modules
import numpy as np
import argparse,os,logging

# Import defined modules
from model import model
from data_process import importer, exporter


# Parser Object
parser = argparse.ArgumentParser(description = "Parse Arguments")

parser.add_argument('-d','--directory',help = 'Data Directory',
					type=str,default='.')

parser.add_argument('-f','--configs',help = 'Configuration File',
					type=str,default='main.config')


# Parse Args Command
for k,v in dict(**vars(parser.parse_args())).items():
	locals()[k.upper()] = v


# Import config file
config = importer([CONFIGS],DIRECTORY,options=True)[CONFIGS]
CONFIG = 'CONFIG'
PROPERTIES = ['DIRECTORY','FILE','FILETYPE','OBSERVABLES']
TYPE = int

assert CONFIG in config.sections(),'Incorrect main.config file'

for p in PROPERTIES:
	locals()[p] = config.get_typed(CONFIG,p,TYPE,atleast_1d=True)


# Import Files
data = importer([os.path.join(FILE,FILE+FILETYPE)],DIRECTORY)

for file,datum in data:
	print(np.shape(datum['sites']))


# print(data)