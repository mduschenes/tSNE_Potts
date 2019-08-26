# Import standard python modules
import numpy as np
import argparse,os

# Import defined modules
from model import model
from montecarlo import montecarlo
from logging_config import logging_config
from data_process import importer, exporter


# Parser Object
parser = argparse.ArgumentParser(description = "Parse Arguments")



parser.add_argument('-d','--directory',help = 'Data Directory',
					type=str,default='.')

parser.add_argument('-j','--job',help = 'Job Number',
					type=int,default=0)

parser.add_argument('-f','--file',help = 'Name Format',
					type=str,default="job_%d")


# Parse Args Command
args = parser.parse_args()

# Import simulation properties from config file
if '%d' in args.file:
	file = args.file%args.job
elif '%s' in args.file:
	file = args.file%str(args.job)
else:
	file = args.file


directory = os.path.join(args.directory,file)
if not os.path.isdir(directory):
	os.makedirs(directory)


# Setup logging
log = 'warning'
logger = logging_config(os.path.join(directory,'%s.log'%file),
										loggername='warning')
props = importer([file+'.config'],directory,
					options={'typer':int,'atleast_1d':True})[file+'.config']


# Setup model
model = model(**props['model'])




# Log model
getattr(logger,log)('\nModel Simulation: %s'%directory)
getattr(logger,log)('Model: %s'%', '.join(['%s: %s'%(str(k),str(v)) 
								for k,v in props['model'].items()]))
getattr(logger,log)('Simulation: %s'%', '.join(['%s: %s'%(str(k),str(v)) 
								for k,v in props['simulation'].items()]))


# Set plotting
if props['simulation']['plotting']:
	props['simulation']['plotting'] = props['model']
else:
	plot = False


# Add simulation parameters to props
for k in ['state_difference','state_generate','transition_probability','dtype']:
	props['simulation'][k] = getattr(model,k)

# Perform MonteCarlo simulation
data = montecarlo(model.N,model.neighbours[0], props['simulation'],file,directory)
data['model'] = props['model']

# Export data
exporter({'%s.%s'%(file,props['simulation']['filetype']):data},directory)

getattr(logger,log)('Model Simulation Complete\n')