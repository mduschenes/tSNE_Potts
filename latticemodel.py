# Import standard python modules
import numpy as np
from datetime import datetime
import argparse,copy,itertools,os,logging,logging.config

# Import defined modules
from Model import Model
from MonteCarlo import montecarlo
from logging_config import logging_config
from data_process import importer, exporter
from plot_properties import set_plot_montecarlo


# Parser Object
parser = argparse.ArgumentParser(description = "Parse Arguments")



parser.add_argument('-d','--directory',help = 'Data Directory',
					type=str,default='./Jobs/job_0')

parser.add_argument('-f','--file',help = 'Configuration File',
					type=str,default='job_0.config')

# Parse Args Command
args = parser.parse_args()




# Import simulation properties from config file
properties = importer([args.file],args.directory,
					options={'typer':int,'atleast_1d':True})[args.file]
props = copy.deepcopy(properties)



# From config file, generate all combinations of parameters
keys,vals = zip(*properties['model'].items())
model_sets = [dict(zip(keys,v))for v in itertools.product(*[[v] for v in vals])]

# Set whether plotting and logging will occur
if props['analysis']['plotting']:
	plot =  lambda keys,i: set_plot_montecarlo(keys=keys,i=i+1,**props['model'])
else:
	plot = False
log = 'warning'

# Iterate through all possible sets of parameters
for i,m in enumerate(model_sets):
	props['model'].update(m)


	# Setup job, model and lattice
	job = props['job']['nameformat']%props['job']['job']
	path = os.path.join(props['job']['directory'],job)
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
	data = montecarlo(model.N,model.neighbours[0], props['simulation'],
							 plot=plot,quiet=props['analysis']['quiet'],
							 job=job,directory=props['job']['directory'])
	data['model'] = m

	# Export data
	exporter({'%s.%s'%(job,props['job']['filetype']):data},
						props['job']['directory'])

	# Update datetime
	props['job']['datetime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

	getattr(logging,log)('Monte Carlo Simulation Complete\n')