
# Import standard python modules
import numpy as np
import itertools,argparse,os
from datetime import datetime

# Import file exporting module
from data_process import importer,exporter
from configparse import configparse

# Parser Object
parser = argparse.ArgumentParser(description = "Parse Arguments")
parser.add_argument('-dir','--directory',help = 'Directory',
					type=str,default='.')
parser.add_argument('-configs','--configs',help = 'Configuration File',
					type=str,default='template.config')
parser.add_argument('-options','--options',help = 'Job Options File',
					type=str,default='template.sh')
parser.add_argument('-task','--task',help = 'Task',
					type=str,default='main')
args = dict(**vars(parser.parse_args()))

for k,v in args.items():
	locals()[k.upper()] = v


# Job Submission Defaults
modules = {'py':'python',None:''}
commands = {'sh':'.', 'pbs':'qsub', 'sbatch':'sbatch',
			'lsf':'bsub', None:''}
jobindices = {'sh':None, 'pbs':'$PBS_ARRAY_INDEX', 'sbatch':'SLURM_ARRAY_TASK_ID',
			  'lsf':'$LSB_JOBINDEX', None:''}



# Environmental Variable Defaults
DATE = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
DIRECTORY = os.path.join(DIRECTORY,DATE).replace(' ','\\ ')
SOURCE = OPTIONS.split('.')[-1]
MODULE = TASK.split('.')[-1]
TASK = '.'.join(TASK.split('.')[:-1])
COMMAND = commands.get(SOURCE,'').upper()
JOBID = TASK+'_'+DATE
JOBINDEX = jobindices.get(SOURCE,'')
config = importer([CONFIGS],options=True)[CONFIGS]

# Get TASK arguments
CONFIG = 'CONFIG'
PROPERTIES = ['FILE','ARGUMENTS']
TYPE = int

assert CONFIG in config.sections(),'Incorrect main.config file'

for p in PROPERTIES:
	locals()[p] = config.get_typed(CONFIG,p,TYPE,atleast_1d=True)
config.remove_section(CONFIG)


# Get arguments from file
options = []
values = []
for section in config.sections():
		for option in config.options(section):
			options.append(option)
			if section in ARGUMENTS:
				values.append(config.get_typed(section,option,TYPE))
			else:
				values.append([config.get_typed(section,option,TYPE)])
			

# Get all permutations of values from lists of arguments
sets = [dict(zip(options,params))for params in list(itertools.product(*values))]


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



# Write file to submit jobs
TASK_SRC = os.path.join(DIRECTORY,'%s.%s'%(TASK,SOURCE))
os.system('cp %s %s'%(OPTIONS,TASK_SRC))
jobline = lambda j:'%s %s.%s --directory %s --job %s --file %s\nsleep 1\n\n'%(
						  modules.get(MODULE,''),TASK,MODULE,DIRECTORY,j,FILE)
with open(TASK_SRC,'a') as f:
	

	if SOURCE == 'sh':
		# Job submission
		f.write("\n##### SCRIPT #####\n")
		
		# Write jobs to submit
		for i in range(len(sets)):
			f.write(jobline(i))

		os.system('$(chmod 777 %s)'%(TASK_SRC))
		
	
	elif SOURCE =='pbs':
		# Job submission
		if len(sets) > 1:
			f.write('\n#%s -t %d-%d\n'%(COMMAND,
										 0,len(sets)-1))
		f.write('\n#%s -N %s\n'%(COMMAND,JOBID))
		f.write('\n#%s -o %s\n'%(COMMAND,os.path.join(DIRECTORY,JOBID+'.log')))


		f.write("\n##### SCRIPT #####\n")
		
		# Write jobs to submit
		f.write(jobline(envvar.get(SOURCE,0)))

	elif SOURCE =='sbatch':
		# Job submission
		if len(sets) > 1:
			f.write('\n#%s --array %s %d-%d\n'%(COMMAND,
									   TASK,0,len(sets)-1))
		f.write('\n#%s --job_name %s\n'%(COMMAND,JOBID))
		f.write('\n#%s --output %s\n'%(COMMAND,os.path.join(DIRECTORY,JOBID+'.log')))


		f.write("\n##### SCRIPT #####\n")
		
		# Write jobs to submit
		f.write(jobline(envvar.get(SOURCE,0)))

	elif SOURCE =='lsf':
		# Job submission
		if len(sets) > 1:
			f.write('\n#%s -J %s %d-%d\n'%(COMMAND,
									   TASK,0,len(sets)-1))
		else:
			f.write('\n#%s -J %s\n'%(COMMAND.upper(),TASK))
		
		f.write('\n#%s -o %s\n'%(COMMAND,os.path.join(DIRECTORY,JOBID+'.log')))


		f.write("\n##### SCRIPT #####\n")
		
		# Write jobs to submit
		f.write(jobline(envvar.get(SOURCE,0)))



# Output Source
for p in [DIRECTORY,COMMAND,SOURCE,TASK_SRC]: 
	print(p)






