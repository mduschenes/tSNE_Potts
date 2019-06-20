"""
Created on Sun May 13 10:48:34 2018
@author: Matt
"""
import numpy as np
import subprocess,argparse,datetime,time,itertools,os,shutil
import os.path

VERBOCITY_TRUE = '"VERBOCITY_TRUE"'
VERBOCITY_FALSE = '"VERBOCITY_FALSE"'

date_ = lambda: datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Parser Object
parser = argparse.ArgumentParser(description = 'Parse Arguments',
								 conflict_handler='resolve')

# Known Arguments
parser.add_argument('-c','--command',help = 'Job Command',
					type=str,nargs = '*',default='sqsub')

parser.add_argument('-j','--job',help = 'Job Executable',
					type=str,nargs='*',default='python3 MagnetismModel.py')

parser.add_argument('-arg','--script_args',help = 'Script Arguments File',
					type=str,default='args_file.txt')					

parser.add_argument('-dirupdt','--dir_update',help = 'Update Directories',
					action='store_false')

parser.add_argument('-wr','--wr',help = 'Write or Run Bash Commands',
					type=str,choices=['run','write','write-run','execute'],
					default='write')

parser.add_argument('-wr_f','--wr_file',help = 'Bash Script File',
					type=str, default='command_script.sh')
					
parser.add_argument('--double_dash',help = 'Double Dash parsing',
					action='store_true')
					
parser.add_argument('--verbocity_type',help='Verbocity Type for Parsing',
					type=str,nargs='+',default=[VERBOCITY_TRUE,VERBOCITY_FALSE])
					
_, unparsed = parser.parse_known_args()

# Unknown Arguments
for arg in unparsed:
    if arg.startswith(("-", "--")):
        parser.add_argument(arg, type=str,nargs='*',help = arg+' Argument')
		
args = parser.parse_args()


def arg_parse(kwargs,dash='-',verbocity_type=[VERBOCITY_TRUE,VERBOCITY_FALSE]):
	if isinstance(kwargs,dict):
		args = []
		for k,v in sorted([(k,v) for k,v in kwargs.items() 
								 if v not in [None]],
							key=lambda k: (len(k[0]),np.size(k[1]),k[0])):		
			if isinstance(v,(list,np.ndarray,tuple,set)):
				t = ''
				for val in v:
					t +=str_check(val)+' '
				t = t[:-1]
			else:
				t = str_check(v)
			
			if t == verbocity_type[0]:
				args.append(dash+str_check(k)+' ')
			elif t == verbocity_type[1]:
				pass #args.append(dash+str_check(k))
			else:
				args.append(dash+str_check(k)+' '+t)
				
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

	parse_set = '{'
	parse_comment = '#'
	parse_func = 'def'

	str_header = lambda s: s.replace('\t','').split(' ')[0]
	str_comment = lambda s: s.split(parse_comment)[0]+' \n'
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
					if h == parse_comment:
						f = ''
						h = str_header(data[i+1])
						pass
					if line != '\n' and i < len(data)-1:
						if h == parse_set:
							data_dict.update(eval(str_comment(line)))
						else:
							f += str_comment(line)
					else:
						if h == parse_func:
							if i == len(data)-1:
								f += str_comment(line)
							f_ = {}
							exec(str_eol(f),locals(),f_)
							data_func[i] = f_
						elif h != parse_set:
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
			

def file_copy(file,directory,file_ending=''):
	file_copy = (file_ending+'.').join(file.split('.'))
	if not os.path.isdir(directory):
		os.makedirs(directory)
	if not os.path.isfile(os.path.join(directory,file_copy)):
		shutil.copy2(file,os.path.join(directory,file_copy))

def dir_update(directory,update=None):
	return os.path.join(directory,update) if update else directory


def str_check(v):
	if not isinstance(v,str): 
		return str(v).replace('[','').replace(']','')
	else: 
		return v.replace('[','').replace(']','')

	
def cmd_run(cmd_args):
	
	# System Args
	dash = '--' if cmd_args['double_dash'] else '-';cmd_args.pop('double_dash');
	verbocity = cmd_args.pop('verbocity_type')
	wr = cmd_args.pop('wr')
	bash_file = cmd_args.pop('wr_file')
	job = arg_parse(cmd_args.pop('job'),dash,verbocity)
	command = arg_parse(cmd_args.pop('command'),dash,verbocity)
	update_dir = cmd_args.pop('dir_update',True)
	
	
	# Read Script args from File
	args_file = cmd_args.pop('script_args')
	args_date = date_()
	script_args,script_args_func,_ = file_read(args_file)

	# Job Output Header
	job_header = lambda s: 'Job %s: \n%s \n%s'%(tuple(str_check(i) for i in s))

	# Loop over Model Args
	script_key = script_args.keys()
	for i,arg in enumerate(itertools.product(*list(script_args.values()))):

		c_args = cmd_args.copy()
		
		# Update Args
		script_arg = dict(zip(script_key,arg))
		
		# Update Data Directory
		if update_dir and script_arg.get('-data_dir'):
			args_dir = dir_update(script_arg['-data_dir'],args_date)
			script_arg['-data_dir'] = args_dir
		else:
			args_dir = script_arg.get('-data_dir',args_date)

		script_arg['-job_id'] = i

		for f in script_args_func:
			for k,v in f.items():
				if k.endswith('_'):
					k = '-'+k[:-1]
				if k not in c_args:
					c_args.update({k: v(**script_arg)})
		
		# Keep valid args
		for k in c_args.copy():
			if c_args.get(k) in ['',None]:
					c_args.pop(k);
		
		# Bash Command
		if job not in ['',None]:
			cmd_bash = ' '.join([command,arg_parse(c_args,dash,verbocity),
							 job, arg_parse(script_arg,dash,verbocity)
							 ]).replace('\n','') 
			
			# if i == 0 and 'sqsub' in command:
				# cmd_bash = 'module load python/intel/3.4.2' + '\n'+cmd_bash
			
		else:
			c_args.update(script_arg)
			cmd_bash = ' '.join([command,
							arg_parse(c_args,dash,verbocity)]).replace('\n','') 
								
			
		
				
		# Do Process, with Pause to not overload system
		
		# Update Output File
		if wr == 'write':		
			file_write(file=bash_file,text=cmd_bash+'\nsleep 1s \n', 
					   read_write='w' if i==0 else 'a')

		else:
			# # Update Output File with Run Header
			# file_write(file=c_args.get('o'),
			# 	       text='\n'+job_header([i,date_(),cmd_bash]),
			# 		   read_write='a')
			
			if wr == 'run':
				os.system(cmd_bash)
				# process = subprocess.Popen(cmd_bash.split(), stdout=subprocess.PIPE)
			elif wr == 'write-run':
				file_write(file=bash_file,text=cmd_bash+'\nsleep 1s \n', 
					   read_write='w' if i==0 else 'a')

				# Copy args_file and bash_file
				file_copy(args_file,args_dir)
				file_copy(bash_file,args_dir)

				os.system(cmd_bash)
			elif wr == 'execute':
				os.system('sh '+bash_file)
			
			time.sleep(1)

		
	

		
		
		 
		
		
 

# Run Command
cmd_run(vars(args))
