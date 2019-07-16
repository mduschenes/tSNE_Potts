
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
DATE = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
assert CONFIG in config.sections(),'Incorrect main.config file'

for p in PROPERTIES:
	locals()[p] = config.get_typed(CONFIG,p,TYPE,atleast_1d=True)


DIRECTORY = DIRECTORY+'_'+DATE
config.remove_section(CONFIG)

print(DIRECTORY)

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
sources = {'sh':'./','pbs':'qsub','lsf':'bsub',None:''}
command = lambda job: '%s %s.%s --directory %s --job %s --file %s\nsleep 1\n\n'%(
							modules.get(MODULE,''),TASK,MODULE,DIRECTORY,job,FILE)
envvar = {'pbs':'$PBS_ARRAY_INDEX' ,'lsf':'$LSB_JOBINDEX'}

os.system('cp template.%s %s.%s'%(SOURCE,TASK,SOURCE))
with open('%s.%s'%(TASK,SOURCE),'a') as f:
	

	if SOURCE == 'sh':
		# Job submission
		f.write("\n##### SCRIPT #####\n")
		
		# Write jobs to submit
		for i in range(len(sets)):
			f.write(command(i))

		os.system('$(chmod 777 %s.%s)'%(TASK,SOURCE))
		
	elif SOURCE in ['pbs','lsf']:
		# Job submission
		if SOURCE =='pbs':
			if len(sets) > 1:
				f.write('\n#%s -t %d-%d\n'%(sources.get(SOURCE).upper(),
											 0,len(sets)-1))
			f.write('\n#%s -N %s\n'%(sources.get(SOURCE).upper(),TASK))

		elif SOURCE =='lsf':
			if len(sets) > 1:
				f.write('\n#%s -J %s %d-%d\n'%(sources.get(SOURCE).upper(),
										   TASK,0,len(sets)-1))
			else:
				f.write('\n#%s -J %s\n'%(sources.get(SOURCE).upper(),TASK))
		

		f.write("\n##### SCRIPT #####\n")
		
		# Write jobs to submit
		f.write(command(envvar.get(SOURCE,'')))



# Output Source
for w in [sources.get(SOURCE),TASK,SOURCE,DIRECTORY]:
	print(w)






