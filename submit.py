
# Import standard python modules
import numpy as np
import itertools,argparse,os
from datetime import datetime

# Import file exporting module
from data_process import importer,exporter
from configparse import configparse

# Parser Object
parser = argparse.ArgumentParser(description = "Parse Arguments")
parser.add_argument('-config','--configfile',help = 'Configuration File',
					type=str,default='template.config')
parser.add_argument('-pbs','--pbsfile',help = 'PBS File',
					type=str,default='template.PBS')
args = parser.parse_args()


config = importer([args.configfile],options=True)[args.configfile]

# Get TASK arguments
CONFIG = 'CONFIG'
PROPERTIES = ['FILE','DIRECTORY','TASK','MODULE','SOURCE','ITERABLES']
TYPE = int
DATE = datetime.now().strftime('%Y-%m-%d--%H:%M:%S')
assert CONFIG in config.sections(),'Incorrect main.config file'

for p in PROPERTIES:
	locals()[p] = config.get_typed(CONFIG,p,TYPE,atleast_1d=True)


DIRECTORY = DIRECTORY+'_'+DATE
config.remove_section(CONFIG)


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
for i,params in enumerate(sets):
	f = configparse()
	for section,options in config.items(): 
		section_options = [(o,v) for o,v in params.items() if o in options]
		if section_options == []:
			continue
		f.add_section(section)
		for option,value in section_options:
			f.set_typed(section,option,value)

	exporter({FILE%i+'.config': f},os.path.join(DIRECTORY,FILE%i))



# Write .sh and .pbs file to submit jobs

modules = {'py':'python',None:''}
sources = {'sh':'./','pbs':'qsub',None:''}
command = lambda job: '%s %s.%s --directory %s --job %s --file %s\nsleep 1\n\n'%(
							modules.get(MODULE,''),TASK,MODULE,DIRECTORY,job,FILE)


os.system('cp template.%s %s.%s'%(SOURCE,TASK,SOURCE))
with open('%s.%s'%(TASK,SOURCE),'a') as f:
	

	if SOURCE == 'sh':
		# Job submission
		f.write("\n##### SCRIPT #####\n")
		
		# Write jobs to submit
		for i in range(len(sets)):
			f.write(command(i))

		os.system('$(chmod 777 %s.%s)'%(TASK,SOURCE))
		

	elif SOURCE == 'pbs':
	
		# Job submission
		if len(sets) > 1:
			f.write('\n#PBS -t %d-%d\n'%(0,len(sets)-1))
		else:
			f.write('\n#PBS -t %d\n'%0)
		f.write('\n#PBS -N %s\n'%TASK)

		f.write("\n##### SCRIPT #####\n")
		
		# Write jobs to submit
		i = '$PBS_ARRAY_INDEX'
		f.write(command(i))



# Output Source
if __name__ == "__main__":
	for w in [sources.get(SOURCE),TASK,SOURCE,DIRECTORY]:
		print(w)






