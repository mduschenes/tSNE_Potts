
# Import standard python modules
import numpy as np
import os,time,logging

# Perform logging
# from logging_config import logging_config
# if 'logger' not in globals():
# 	logger = logging_config(logfilename='log.log',loggername='warning')


### Simulation Functions ###

# Log text and time
class display(object):
	def __init__(self,display=True,timer=True):
		self.display = display
		self.timer = timer
		self.times = [time.clock()]
		return

	def log(self,message='',message_type='warning',
					time0=-2,line_break=0,line_tab=0):
	
		line_break *= '\n' 
		line_tab *= '\t'
		message = line_tab+message
		
		if self.timer and self.display:
			self.times.append(time.clock())
			getattr(logging,message_type)(message+
									' %0.5f'%(self.times[-1]-self.times[time0])) 
		elif self.display:
			getattr(logging,message_type)(message)

		return


### List Manipulation Functions ###

def flatten(x,flattenint=True):
	# Return a 1d list of all elements in inner lists of x of arbirtrary shape
	if not (isinstance(x,list)):
		return x
	elif len(x) == 1 and flattenint:
		if isinstance(x[0],tuple) or isinstance(x[0],str):
			return x
		else:
			return x[0]
	xlist = []
	for y in x:
		if isinstance(y,type([])):
			xlist.extend(flatten(y))
		else: 
			xlist.append(y)
	return xlist


def zipped_lists(lists):
	return [i for j in (zip(*l) for l in lists) for i in j]

def sort_unhashable(lists,inds=slice(0,None,1),key=lambda x:x):
	lists = zipped_lists(lists)
	seen = set()
	result = []
	for item in lists:
		if item[inds] not in [s[inds] for s in seen]:
			seen.add(item)
			result.append(item)
	return list(zip(*sorted(result,key=key)))




### String Manipulation Functions ###

# Escape Characters in Text
escape_dict={'\a':r'\a',
             '\b':r'\b',
             '\c':r'\c',
             '\f':r'\f',
             # '\n':r'\n',
             '\r':r'\r',
             '\t':r'\t',
             '\v':r'\v',
             '\'':r'\'',
             '\"':r'\"'}

# Process Raw Text with Escape characters
def raw(text):
    new_text=''
    for char in text:
        new_text += escape_dict.get(char,char)
    return new_text


def texify(word,every_word=None,sep_char=' ',split_char=' ',decimals=0,delims={}):
	word = caps(word,every_word=every_word,sep_char=sep_char,
					 split_char=split_char,decimals=decimals,
					 delims=delims)
	if '$' in word:
		return r'%s'%word
	else:
		return r"$\mathrm{%s}$"%word

def caps(word,every_word=None,sep_char=' ',split_char=' ',decimals=0,delims={}):
	def capitalize(s):
		return s[0].upper()+s[1:].lower() 

	word = str_check(word,decimals,delims)
	try:
		if every_word is False:
			return capitalize(word) 
		elif every_word is True:
			word_sep = sep_char.join([capitalize(w)
								 for w in word.split(split_char)])
			return sep_char.join([capitalize(w)
								 for w in word_sep.split(sep_char)])
		else:
			return sep_char.join([w for w in word.split(sep_char)])
	except IndexError:
		return word


def str_check(string,decimals=0,delims={}):
	
	if not isinstance(string,str): 
		if isinstance(string,(int,float)): 
			string = '%0.*f'%(decimals,string)
		else:
			string = str(string) 

	for d in delims:
		string = string.replace(delim,'')

	return string



### Function Functions ###

# Function wrapper to pass some args and kwargs
def argswrapper(function):

	def wrapper(*args0,**kwargs):
		return lambda *args,**kwargs: function(*args0,*args,**kwargs)

	return wrapper




### Calculation Functions ###


