
# Import standard python modules
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# from pdf2image import convert_from_path
import os,glob,json,zlib,base64,array,gzip,struct,csv,h5py,pickle
from PIL import Image
import networkx as nx

# Import defined modules
from configparse import configparse

# Function wrapper to pass some args and kwargs
def argswrapper(function):

	def wrapper(*args0,**kwargs0):
		return lambda *args,**kwargs: function(*args0,*args,**kwargs0,**kwargs)

	return wrapper

# Type Parsers

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def make_number(s):
	if '.' in s:
		return float(s)
	elif 'e' in s:
		return int(float(s))
	else:
		return int(s)

def is_string(s):
	return not is_number(s) or "'" in s or r'"' in s

def make_string(s):
	return str(s)

def is_bool(s):
	return is_string(s) and s.lower() in ['true','false']

def make_bool(s):
	if s.lower() == 'true':
		return True
	elif s.lower() == 'false':
		return False

def formatter(s):
	for t in ['string','number','bool']:
		if globals()['is_'+t](s):
			return globals()['make_'+t](s)
			



# Import list of files in directory and return dictionary of {file: data}
def importer(files=[],directory='.',options={}):

	# Gets name of file
	def get_name(file):
		return ''.join(file.split('.')[:-1])

	# Gets format of file
	def get_format(file):
		return file.split('.')[-1]


	# Import data from file based on format
	def get_data(file='',directory='.',format='',options={}):		
		path = os.path.join(directory,file)

		if not os.path.getsize(path) >0 or not os.path.isfile(path):
			return None
		
		elif format == 'json':
			with open(path, 'r') as f:
				return json.load(f, **options.get('load',{}),
					cls=JSONdecoder(**options.get('decode',{})))

		elif format in ['bin','dat']:
			with open(path,'rb') as f:
				metadata =str(f.readline().decode()).replace('\n','').split(' ')
				dtype = metadata[0]; dshape = (int(m) for m in metadata[1:])
				if 'float' in dtype:
					dtypeb = 'd'
				elif 'int' in dtype:
					dtypeb = 'i'
				return np.array(array.array(dtypeb).fromfile(f,np.prod(dshape)),
								dtype=dtype).reshape(dshape)

		elif format in ['data']:
			with open(path,'rb') as f:
				return pickle.load(f)

		elif format == 'npy':
			return np.load(path)
		
		elif format == 'npz':
			return np.load(path)

		elif format == 'txt':
			return np.loadtxt(path)

		elif format == 'csv':
			with open(path,newline='') as f:
				return [options.get('format',
						lambda x: [formatter(y) for y in x])(d) 
						for d in csv.reader(f,**options.get('decode',{}))]

		elif format in ['h5','hdf5']:
			with h5py.File(path,'r') as f:
				return dict(f)

		# elif format == 'pdf':
		# 	return convert_from_path(path,**options)

		elif format in ['jpg','png']:
			return Image.open(path,**options)

		elif format == 'config':
			config = configparse()
			config.read(path)
			if not isinstance(options,dict):
				return config
			else:
				return config.get_dict(**options)

		elif format in ['graphyaml']:
			path = path.replace('.yaml.graphyaml','.yaml')
			return nx.read_yaml(path)

		elif format in ['gml']:
			return nx.read_gml(path)

		elif format in ['graphml']:
			return nx.read_graphml(path)

		else:
			return None

		

	# Check if real path
	if '.' in directory or '..' in directory:
		directory = os.path.realpath(os.path.expanduser(directory))

	# Data dictionary of {file: data}
	data = {}
	i = 0
	while i < len(files):
		
		# Find files associated with name and format
		f = files[i]
		name,format = get_name(f),get_format(f)
		if '*' in name or '*' in format:
			files.remove(f)
			files += [p.split(directory)[1][1:]
					  for p in glob.glob(os.path.join(directory,f))]
			continue
		else:
			data[f] = get_data(f,directory,format,options)
			
		i += 1

	return data





# Export data  of dictionary of {file: data} to directory, 
# with options {file:options}
def exporter(data={},directory='.',options={},overwrite=True):

	# Gets name of file
	def get_name(file):
		return '.'.join(file.split('.')[:-1])

	# Gets format of file
	def get_format(file):
		return file.split('.')[-1]


	# Export data to file based on format
	def set_data(file='',data=[],directory='.',format='',options={}):		
		
		path = os.path.join(directory,file)

		if format == 'json':
			with open(path, 'w') as f:
				json.dump(data,f,**options.get('dump',{}), 
					cls=JSONencoder(**options.get('encode',{})));
			f.close();

		elif format in ['bin','dat']:
			assert isinstance(data,
						np.ndarray), "Wrong data type for binary serialization"
			with open(path,'wb') as f:
				metadata = ' '.join([str(data.dtype), 
									  *['%d'%s for s in data.shape]])+'\n'
				f.write(metadata.encode());
				# data.tofile(f);
				if 'float' in str(data.dtype):
					array.array('d',data.flatten()).tofile(f)
				elif 'int' in str(data.dtype):
					array.array('i',data.flatten()).tofile(f)
				# f.write(bytearray(data.flatten()))
			f.close();

		elif format in ['data']:
			with open(path,'rb') as f:
				pickle.dump(data,f)

		elif format == 'npy':
			np.save(path,data);
		
		elif format == 'npz':
			np.savez_compressed(path,data);

		elif format == 'txt':
			with open(path,'w') as f:
				f.write(str(data));
			f.close();

		elif format == 'csv':
			with open(path,'w',newline='') as f:
				file = csv.writer(f,**options.get('encode',{}))
				for d in data:
					file.writerow(options.get('format',
								  lambda x: [str(y) for y in x])(d))
			f.close();
		
		elif format in ['pdf','png','jpg','eps','svg']:
			options['format'] = format
			data.savefig(path,**options);

		elif format in ['gif','mp4']:
			data.save(path,**options);

		elif format in ['config','ini']:
			with open(path, 'w') as f: 
				data.write(f);
			f.close();

		elif format in ['graphyaml']:
			path = path.replace('.graphyaml','.yaml')
			return nx.write_yaml(data,path)

		elif format in ['gml']:
			nx.write_gml(data,path)

		elif format in ['graphml']:
			nx.write_graphml(data,path)


		return

	# Data dictionary of {file: data}

	# Check if real path
	if '.' in directory or '..' in directory:
		directory = os.path.realpath(os.path.expanduser(directory))

	# Check if directory exists
	if not os.path.isdir(directory):
		os.makedirs(directory);

	# Overwritten files
	files_overwitten = {file:file for file in data.keys()}
	for file,datum in data.items():
		# Get name and format associated with file
		name,format = get_name(file),get_format(file)
		
		# Ensure no overwriting of file
		i = 0
		file_end = ''
		while not overwrite and os.path.isfile(name + file_end + '.'+format):
			file_end = '_%d'%i
			i+=1
		files_overwitten[file] = name+file_end + '.'+format
		file = files_overwitten[file]

		# Export data to file based on file and format
		set_data(file,datum,directory,format,options)

	# Update data file names
	data = {file:data[file] for file in files_overwitten}

	return


# Given data {key:data}, seed existing data into new keys using 
# seeds: {key_i: [key_seed,seed_axis,
#				  seed_indices(key_seed,data[key_seed],axis_seed))]}, 
# and possibly delete seeded data
def seeder(data,seeds,delete_seed=False):
	for key,seed in seeds.items():
		key_seed = seed[0]
		axis_seed = seed[1]
		indices_seed = seed[2](key_seed,data[key_seed],axis_seed)

		shape_seed = np.shape(data[key_seed])
		shape_seed[axis_seed] = len(indices_seed)

		data[key] = np.reshape(np.take(data[key_seed],indices_seed,axis_seed),
								shape_seed)

		if delete_seed:
			data[key_seed] = np.delete(data[key_seed],indices_seed,axis_seed)
		
	return data


# Append data: {file: data} to existing file
def appender(data={},directory='.',options={}):

	# Append data by type
	def append_data(file,data_existing,data):
		if file.endswith('.config'):
			for section,options in data.items():
				for option,value in options.items():
					data_existing.set_typed(section,option,value)
		elif type(data) != type(data_existing):
			return data_existing
		elif isintance(data_existing,(list,tuple,str)):
			return data_existing + data
		elif isinstance(data_existing,np.ndarray):
			return np.concatenate((data_existing,data),0)
		elif isinstance(data_existing,dict):
			return data_existing.update(data)


	# Existing data dictionary of {file: data}
	data_existing = importer(data.keys(),directory,options)

	# Append data to data_existing
	for (file,datum_existing),datum in zip(data_existing.items(),data.values()):
		data_existing[file] = append_data(file,datum_existing,datum)

	# Export updated data to files
	exporter(data_existing,directory)		

	return





# Class ovveride for encoding objects with unserializable
# encode_types: {type: serialize_function} in json
@argswrapper
class JSONencoder(json.JSONEncoder):

	def __init__(self,encode_types={},wrapper=None,*args,**kwargs):
		json.JSONEncoder.__init__(self, *args, **kwargs)
		if wrapper is None:
			self.wrapper = lambda obj: base64.b64encode(zlib.compress(
											json.dumps(obj).encode())
											).decode('ascii')
		else:
			self.wrapper = wrapper

		self.encode_types = {dict: self.encode_dict,
							 np.ndarray: self.encode_ndarray}
		if isinstance(encode_types,dict):
			self.encode_types.update(encode_types)
		
		return

	def default(self, obj):
		return self.encode_types.get(type(obj),
								lambda o:json.JSONEncoder.default(self, o))(obj)

	def encode_ndarray(self,obj):
		return self.wrapper(obj.tolist())

	def encode_dict(self,obj):
		obj_json = {}
		for k,v in obj.items():
			k = self.encode_types.get(type(k),lambda x:x)(k)
			v = self.encode_types.get(type(v),lambda x:x)(v)
			obj_json[k] = v

		return obj_json


# Class ovveride for decoding objects with unserializable
# decode_types: {type: serialize_function} from json
@argswrapper
class JSONdecoder(json.JSONDecoder):

	def __init__(self,decode_types={},wrapper=None,*args,**kwargs):
		json.JSONDecoder.__init__(self, object_hook = self.object_hook,
										*args,**kwargs)
		if wrapper is None:
			self.wrapper = lambda obj: json.loads(zlib.decompress(
													base64.b64decode(obj)))
		else:
			self.wrapper = wrapper

		self.decode_types = {dict: self.decode_dict,
							 list: self.decode_ndarray,
							 str: lambda x:self.decode_ndarray(self.wrapper(x))}
		if isinstance(decode_types,dict):
			self.decode_types.update(decode_types)

		return

	def object_hook(self, obj):
		return self.decode_types.get(type(obj),lambda o:o)(obj)

	def decode_ndarray(self,obj):
		return np.array(obj)

	def decode_dict(self,obj):
		obj_json = {}
		for k,v in obj.items():
			try:
				v = self.wrapper(v)
				k = self.decode_types.get(type(k),lambda x:x)(k)
				v = self.decode_types.get(type(v),lambda x:x)(v)
			except:
				try:
					k = self.decode_types.get(type(k),lambda x:x)(k)
					v = self.decode_types.get(type(v),lambda x:x)(v)
				except:
					pass
			obj_json[k] = v

		return obj_json
