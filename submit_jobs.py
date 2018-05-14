"""
Created on Sun May 13 10:48:34 2018
@author: Matt
"""
import numpy as np
import subprocess,argparse,datetime,time,itertools,os

# Parser Object
parser = argparse.ArgumentParser(description = "Parse Arguments")

# Known Arguments
parser.add_argument('-c','--command',help = 'Job Command',
					type=str,nargs = '*',default='')
parser.add_argument('-j','--job',help = 'Job Executable',
					type=str,nargs='*',default='')
parser.add_argument('-arg','--script_args',help = 'Script Arguments File',
					type=str,default='')					

parser.add_argument('-q','--q',help = 'Job Queue',
					type=str,default='')
parser.add_argument('-o','--o',help = 'Output File',
					type=str, default = '')

parser.add_argument('-rw','--rw',help = 'Write or Run Bash Commands',
					type=str,choices=['run','write','run-write','execute'],
					default='write')

parser.add_argument('-rw_f','--rw_file',help = 'Bash Script File',
					type=str, default='command_script.sh')
					
_, unparsed = parser.parse_known_args()

# Unknown Arguments
for arg in unparsed:
    if arg.startswith(("-", "--")):
        parser.add_argument(arg, type=str,nargs='*',help = arg+' Argument')
		
args = parser.parse_args()


def arg_parse(kwargs):

	if isinstance(kwargs,dict):
		args = []
		for k,v in [(k,v) for k,v in kwargs.items() if v not in [None, '']]:
			if isinstance(v,dict):
				exit()
			if isinstance(v,(list,np.ndarray,tuple,set)):
				t = ''
				for val in v:
					t +=str_check(val)+' '
				t = t[:-1]
			else:
				t = str_check(v)
			
			args.append('-'+str_check(k)+' '+t)
		return ' '.join(args)
		
	elif isinstance(kwargs,(list,np.ndarray,tuple,set)):
		t = ''
		for val in kwargs:
			t +=str_check(val)+' '
		return t[:-1]
		
	else:
		return str_check(kwargs)

def file_write(file='output_file.txt', text='',read_write='a'):
	if file is None or file == '':
		return
	try:
		with open(file,read_write) as f:
			f.write(text)
		return
	except IOError:
		return

def file_read(file):
	str_header = lambda s: s.replace('\t','').split(' ')[0]
	str_comment = lambda s: s.split('#')[0]+' \n'
	str_eol = lambda s: s.replace('(\n','(')
	
	if file is not None and file is not 'None':
		try:
			with open(file,'r') as file_txt:
				data = list(file_txt)
				data_dict = {}
				data_func = [None]*len(data)
				data_text = [None]*len(data)
				f = ''
				h = str_header(data[0])
				for i,line in enumerate(data):
					if h == '#':
						f = ''
						h = str_header(data[i+1])
						pass
					if line != '\n' and i < len(data)-1:
						if h == '{':
							data_dict.update(eval(str_comment(line)))
						else:
							f += str_comment(line)
					else:
						if h == 'def':
							f_ = {}
							exec(str_eol(f),locals(),f_)
							data_func[i] = f_
						elif h != '{':
							data_text[i] = f
						elif line != '\n':
							data_dict.update(eval(str_comment(line)))
						if i < len(data)-1:
							f = ''
							h = str_header(data[i+1])
			data_func = [d for d in data_func if d]
			data_text = [d for d in data_text if d]
			return data_dict,data_func,data_text
		except IOError:
			return {},[],[]
			

def str_check(v):
	if not isinstance(v,str): 
		return str(v).replace('[','').replace(']','')
	else: 
		return v.replace('[','').replace(']','')

	
def cmd_run(cmd_args):
	
	# Command and Job Args
	rw = cmd_args.pop('rw')
	file_bash = cmd_args.pop('rw_file')
	job = arg_parse(cmd_args.pop('job'))
	command = arg_parse(cmd_args.pop('command'))
	
	# Read Script args from File
	script_args,script_args_func,_ = file_read(cmd_args.pop('script_args'))

	# Job Output Header
	job_header = lambda s: '\n' + 'Job %s: '%str_check(s[0]) + (
							datetime.datetime.now().strftime(
							'%Y-%m-%d-%H-%M-%S-%f'))+(
					       '\n %s'%str_check(s[1])+'\n')
	
	# Loop over Model Args
	script_key = script_args.keys()	
	for i,arg in enumerate(itertools.product(*list(script_args.values()))):
		
		# Update Args
		script_arg = dict(zip(script_key,arg))
		for f in script_args_func:
			for k,v in f.items():
				cmd_args.update({k: v(**script_arg)})
		
		# Bash Command
		cmd_bash = ' '.join([command,arg_parse(cmd_args),
							 job, arg_parse(script_arg)]).replace('\n','') 
		
		# Update Output File
		file_write(cmd_args.get('o'),job_header([i,cmd_bash]))
		
		if rw == 'run':
			os.system(cmd_bash)
			# process = subprocess.Popen(cmd_bash.split(), stdout=subprocess.PIPE)
			# output, error = process.communicate()
			#print(cmd_bash)
		elif rw == 'write':
			file_write(file_bash, text=cmd_bash,read_write='w')
		elif rw == 'write_run':
			file_write(file_bash, text=cmd_bash,read_write='w')
			os.system(cmd_bash)
			# process = subprocess.Popen(cmd_bash.split(), stdout=subprocess.PIPE)
			# output, error = process.communicate()
			#print(cmd_bash)
		elif rw == 'execute':
			subprocess.call(['./'+file_bash])
		
		# Pause to not overload system
		time.sleep(1) 
		
		
 

# Run Command
cmd_run(vars(args))
