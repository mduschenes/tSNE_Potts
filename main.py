# Import standard python modules
import numpy as np
import argparse,os

# Import defined modules
from model import model
from montecarlo import montecarlo
from plot_properties import set_plot_montecarlo
from logging_config import logging_config
from data_process import importer, exporter


# Parser Object
parser = argparse.ArgumentParser(description = "Parse Arguments")



parser.add_argument('-d','--directory',help = 'Data Directory',
					type=str,default='./Jobs/job_0')

parser.add_argument('-j','--job',help = 'Job Number',
					type=int,default=0)

parser.add_argument('-f','--file',help = 'Name Format',
					type=str,default='job_%')


# Parse Args Command
args = parser.parse_args()




# Import simulation properties from config file
file = args.file%args.job
directory = os.join(args.directory,file)
props = importer([file+'.config'],directory,
					options={'typer':int,'atleast_1d':False})[file+'.config']


# Set whether plotting and logging will occur
if props['simulation']['plotting']:
	plot =  lambda keys,i: set_plot_montecarlo(keys=keys,i=i+1,**props['model'])
else:
	plot = False
log = 'warning'


# Setup model

model = Model(**props['model'])

# Setup logging
logger = logging_config(path+'.log',loggername='warning')


# Log model
getattr(logger,log)('\nMonte Carlo Simulation: %s'%path)
getattr(logger,log)('Model: %s'%', '.join(['%s: %s'%(str(k),str(v)) 
								for k,v in props['model'].items()]))
getattr(logger,log)('Simulation: %s'%', '.join(['%s: %s'%(str(k),str(v)) 
								for k,v in props['simulation'].items()]))


# Add simulation parameters to props
for k in ['state_difference','state_generate','transition_probability','dtype']:
	props['simulation'][k] = getattr(model,k)

# Perform MonteCarlo simulation
data = montecarlo(model.N,model.neighbours[0], props['simulation'],file,directory)
data['model'] = props['model']

# Export data
exporter({'%s.%s'%(job,props['job']['filetype']):data},
					props['job']['directory'])

# Update datetime
props['job']['datetime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

getattr(logging,log)('Monte Carlo Simulation Complete\n')