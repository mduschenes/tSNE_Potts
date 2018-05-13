"""
Created on Sun May 13 10:48:34 2018
@author: Matt
"""
import numpy as np
import argparse
import subprocess
import datetime
import itertools

# Parser Object
parser = argparse.ArgumentParser(description = "Parse Arguments")

# Known Arguments
parser.add_argument('-c','--command',help = 'Job Command',
					type=str,default='qsub')
parser.add_argument('-j','--job',help = 'Job Executable',
					type=str,default='python None')
parser.add_argument('-a','--script_args',help = 'Script Arguments File',
					type=str,default='args_file.txt')					

parser.add_argument('-q','--q',help = 'Job Queue',
					type=str,default='None')
parser.add_argument('-o','--o',help = 'Output File',
					type=str,default='output_file.txt')

_, unparsed = parser.parse_known_args()

# Unknown Arguments
for arg in unparsed:
    if arg.startswith(("-", "--")):
        parser.add_argument(arg, type=str,help = arg+' Argument')
		
args = parser.parse_args()


def arg_parse(kwargs):
	args = []
	for k,v in kwargs.items():
		if isinstance(v,dict):
			exit()
		if isinstance(v,(list,np.ndarray,tuple,set)):
			t = ''
			for val in v:
				t +=str_check(val)+' '
		else:
			t = str_check(v)
		
		args.append('-'+str_check(k)+' '+t)
		
	
	return ' '.join(args)

def file_write(file,
			   text='\n Job: '+
						datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')+
						'\n'):
	if file is not None and file is not 'None':
		try:
			with open(file,'a') as file_txt:
				file_txt.write(text)
			return
		except IOError:
			return

def file_read(file):
	if file is not None and file is not 'None':
		try:
			with open(file,'r') as file_txt:
				data = list(file_txt)
				data_dict = {}
				data_func = [None]*len(data)
				data_text = [None]*len(data)
				f = ''
				h = data[0].split(' ')[0]
				for i,line in enumerate(data):
					if line != '\n' and i < len(data)-1:
						if h == '{':
							data_dict.update(eval(line))
						else:
							f += line
					else:
						if h == 'def':
							f_ = {}
							exec(f.replace('(\n','('),locals(),f_)
							data_func[i] = f_
						elif h != '{':
							data_text[i] = f
						if i < len(data)-1:
							f = ''
							h = data[i+1].split(' ')[0]
			data_func = [d for d in data_func if d]
			data_text = [d for d in data_text if d]
			return data_dict,data_func,data_text
		except IOError:
			return {},[],[]
			

def str_check(v):
	if not isinstance(v,str): 
		return str(v)
	else: 
		return v

	
def cmd_run(cmd_args):
	
	# Command and Job Args
	job = cmd_args.pop('job')
	command = cmd_args.pop('command')
	
	# Read Script args from File
	script_args,script_args_func,_ = file_read(cmd_args.pop('script_args'))

	# Loop over Model Args
	script_key = script_args.keys()	
	for arg in list(itertools.product(*list(script_args.values()))):
		
		# Update Args
		script_arg = dict(zip(script_key,arg))
		for f in script_args_func:
			cmd_args.update({k: v(**script_arg) for k,v in f.items()})
	
		# Update Output File
		file_write(cmd_args.get('o'))
		
		# Bash Command
		cmd_bash = ' '.join([command,arg_parse(cmd_args),
							 job, arg_parse(script_arg)]) 
		print(cmd_bash)
		
		# process = subprocess.run(cmd_bash.split(), stdout=subprocess.PIPE)
		# output, error = process.communicate()
		

def cmd_runtime(**kwargs):
	if   kwargs['Nm'] < 1e4 and kwargs['L'] < 12 and (
		 kwargs['T'] > 1.1/(1+np.sqrt(kwargs['q']))) : return '6h'
	elif kwargs['Nm'] < 1e4 and kwargs['L'] < 12 and (
		 kwargs['T'] < 1.1/(1+np.sqrt(kwargs['q']))) : return '10h'
	
	elif kwargs['Nm'] < 1e4 and kwargs['L'] > 12 and (
		 kwargs['T'] > 1.1/(1+np.sqrt(kwargs['q']))) : return '15h'
	elif kwargs['Nm'] < 1e4 and kwargs['L'] > 12 and (
		 kwargs['T'] < 1.1/(1+np.sqrt(kwargs['q']))) : return '20h'
	
	elif kwargs['Nm'] > 1e4 and kwargs['L'] < 12 and (
		 kwargs['T'] > 1.1/(1+np.sqrt(kwargs['q']))) : return '14h'
	elif kwargs['Nm'] > 1e4 and kwargs['L'] < 12 and (
		 kwargs['T'] < 1.1/(1+np.sqrt(kwargs['q']))) : return '24h'
	
	elif kwargs['Nm'] > 1e4 and kwargs['L'] > 12 and (
		 kwargs['T'] > 1.1/(1+np.sqrt(kwargs['q']))) : return '24h'
	elif kwargs['Nm'] > 1e4 and kwargs['L'] > 12 and (
		 kwargs['T'] < 1.1/(1+np.sqrt(kwargs['q']))) : return '24h'
	   
 

# Run Command
cmd_run(vars(args))