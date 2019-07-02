
# Import standard python modules
import numpy as np
import itertools,argparse,os
from configparse import configparse

from datetime import datetime

# Import file exporting module
from data_process import exporter

# Parser Object
parser = argparse.ArgumentParser(description = "Parse Arguments")
parser.add_argument('-f','--configfile',help = 'Configuration File',
					type=str,default='main.config')
args = parser.parse_args()


config = configparse()
config.read(args.configfile)

# Get generate arguments
GENERATE = 'GENERATE'
TYPE = int
DATE = datetime.now().strftime('%Y-%m-%d--%H:%M:%S')
assert GENERATE in config.sections(),'Incorrect main.config file'

for p in ['NAMEFORMAT','ITERABLES','DIRECTORY','SUBMIT']:
	locals()[p] = config.get_typed(GENERATE,p,TYPE,atleast_1d=True)

config.remove_section(GENERATE)

# Get arguments from file
options = []
values = []
for section in config.sections():
		for option in config.options(section):
			options.append(option)
			if section in ITERABLES:
				values.append(config.get_typed(section,option,TYPE))
			else:
				values.append([config.get_typed(section,option,TYPE)])
			

# Get all permutations of values from lists of arguments
sets = [dict(zip(options,params))for params in  list(itertools.product(*values))]


# Create config files from permutations of arguments
# and add job section with job id directory and date
directory = [None for _ in range(len(sets))]
for i,params in enumerate(sets):
	file = configparse()
	directory[i] = os.path.join(DIRECTORY+'_'+DATE,NAMEFORMAT%i)
	for section,options in config.items(): 
		section_options = [(o,v) for o,v in params.items() if o in options]
		if section_options == []:
			continue
		file.add_section(section)
		for option,value in section_options:
			file.set_typed(section,option,value)

	file.set_typed('job','job',str(i))
	file.set_typed('job','nameformat',NAMEFORMAT)
	file.set_typed('job','directory',directory[i])
	file.set_typed('job','datetime',DATE)
	exporter({NAMEFORMAT%i+'.config': file},directory[i])



# Generate .sh file to submit jobs
if SUBMIT:
	with open('submit.sh','w') as f:
		for i,d in enumerate(directory):
			f.write('python %s --dir %s --file %s.config\nsleep 1\n\n'%(
													SUBMIT,d,NAMEFORMAT%i))