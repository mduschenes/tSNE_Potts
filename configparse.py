

# Configuration Parser
import numpy as np
from configparser import ConfigParser, ExtendedInterpolation,NoSectionError,NoOptionError

class configparse(ConfigParser):

	def __init__(self,interpolation=ExtendedInterpolation(),*args,**kwargs):
		ConfigParser.__init__(self,interpolation=interpolation,*args,**kwargs)
		self.optionxform=str
		return

	def get_typed(self,section,option,typer=float,filters=[],atleast_1d=False):
		values = self.get(section,option)
		values = list(filter(lambda x: x not in [None,'',*filters],
							 (y.strip() for x in values.splitlines() 
							 			for y in x.split(',') )))
		
		for format in ['range','tuple','bool','string','number']:
			if any([globals()['is_'+format](v) for v in values]):
				return make_atleast_1d(globals()['make_'+format](values,typer),
																	atleast_1d)
				
	def set_typed(self,section, option, value):
		if isinstance(value,(list,tuple,set,np.ndarray)):
			for d in ['[',']','(',')','{','}',"'",r'"',' ']:
				value = str(value).replace(d,'')
			self.set(section, option, value)
		else:
			self.set(section, option, str(value))
		return


	def get_dict(self,typer=int,atleast_1d=False):
		return {s:{o:self.get_typed(s,o,typer=typer,atleast_1d=atleast_1d) 
				for o in self.options(s)} for s in self.sections()}


def is_range(s):
	return s == '...' 

def make_range(values,typer):
	values.remove('...')
	assert len(values) == 3, "Incorrect range parameters"

	if any(['.' in x for x in values]):
		return np.linspace(*[float(x) for x in values])
	else:
		return range(*[int(x) for x in values])


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def make_number(values,typer):
	if any(['.' in x for x in values]):
		return [float(x) for x in values]
	else:
		return [typer(x) for x in values]


def is_string(s):
	return not is_number(s) or "'" in s or r'"' in s

def make_string(values,typer):

	for i,v in enumerate(values):
		if v == 'None':
			values[i] = None
		else:
			values[i] = v.replace("'",'')
	return values


def is_bool(s):
	return is_string(s) and s.lower() in ['true','false']

def make_bool(values,typer):
	for i,v in enumerate(values):
		if v.lower() == 'true':
			values[i] = True
		elif v.lower() == 'false':
			values[i] = False
		else:
			values[i] = typer(v)
	return values


def is_tuple(s):
	return any([d in s for d in ['(',')']])

def make_tuple(values,typer):
	string = ','.join(values)
	assert string.count('(') == string.count(')') !=0,"Incorrect tuple format"
	string = string.split(',')
	values = [[]]
	for i in range(len(string)):
		if '(' in string[i]:
			values[-1].append(string[i].replace('(',''))
		if ')' in string[i]:
			values[-1].append(string[i].replace(')',''))
			if i < len(string)-1:
				values.append([])
		if not '(' in string[i] and not ')' in string[i]:
			values[-1].append(string[i])

	for format in ['range','string','number']:
			if any([globals()['is_'+format](u) for v in values for u in v]):
				return [globals()['make_'+format](v,typer) for v in values]	


def make_atleast_1d(values,atleast_1d):
	if atleast_1d and len(values)  == 1:
		return values[0]
	else:
		return values