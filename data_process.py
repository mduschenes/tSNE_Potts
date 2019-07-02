
# Import standard python modules
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os,json

# Import defined modules
from configparse import configparse
from miscellaneous_functions import argswrapper

# Import list of files in directory and return dictionary of {file: data}
def importer(files=[],directory='.',options={}):

	# Gets name of file
	def get_name(file):
		return ''.join(file.split('.')[:-1])

	# Gets format of file
	def get_format(file):
		return file.split('.')[-1]

	# Gets files associated with name and or format
	def get_files(directory='',name=[''],format=['']):
		return [f for f in os.listdir(directory)
					if os.path.isfile(os.path.join(directory,f))
					and all([i in get_name(f) for i in name]) 
					and all([i in get_format(f) for i in format])]


	# Import data from file based on format
	def get_data(file='',directory='.',format='',options={}):		
		path = os.path.join(directory,file)

		if not os.path.getsize(path) >0 or not os.path.isfile(path):
			return None
		
		elif format == 'json':
			with open(path, 'r') as f:
				return json.load(f, **options.get('load',{}),
					cls=JSONdecoder(**options.get('decode',{})))

		elif format == 'npy':
			return np.load(path)
		
		elif format == 'npz':
			return np.load(os.path.join(directory,formatter(file,format)))

		elif format == 'txt':
			return np.loadtxt(os.path.join(directory,
											formatter(file,format)))
		elif format == 'config':
			config = configparse()
			config.read(path)
			return config.get_dict(**options)
		else:
			return None


	# Data dictionary of {file: data}
	data = {}
	i = 0
	while i < len(files):
		
		# Find files associated with name and format
		f = files[i]
		name,format = get_name(f),get_format(f)
		if '*' in name or '*' in format:
			files.remove(f)
			files += get_files(directory,name.split('*'),format.split('*'))
			print(files)
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

	# Gets files associated with name and or format
	def get_files(directory='',name=[''],format=['']):
		return [f for f in os.listdir(directory)
					if os.path.isfile(os.path.join(directory,f))
					and all([i in get_name(f) for i in name]) 
					and all([i in get_format(f) for i in format])]


	# Export data to file based on format
	def set_data(file='',data=[],directory='.',format='',options={}):		
		
		path = os.path.join(directory,file)
		
		if format == 'json':
			with open(path, 'w') as f:
				return json.dump(data,f,**options.get('dump',{}), 
					cls=JSONencoder(**options.get('encode',{})))

		elif format == 'npy':
			np.save(path,data)
		
		elif format == 'npz':
			np.savez_compressed(path,data)

		elif format == 'txt':
			with open(path,'w') as f:
				f.write(str(data));
		
		elif format in ['pdf','png','jpg','eps']:
			options['format'] = format
			data.savefig(path,**options);

		elif format in ['gif','mp4']:
			data.save(path,**options);

		elif format in ['config','ini']:
			with open(path, 'w') as f: 
				data.write(f)

		return

	# Data dictionary of {file: data}

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
def appender(data={},directory='.'):

	# Append data by type
	def append_data(data_existing,data):
		if type(data) != type(data_existing):
			return data_existing
		elif isintance(data_existing,(list,tuple,str)):
			return data_existing + data
		elif isinstance(data_existing,np.ndarray):
			return np.concatenate((data_existing,data),0)
		elif isinstance(data_existing,dict):
			return data_existing.update(data)


	# Existing data dictionary of {file: data}
	data_existing = importer(data.keys(),directory)

	# Append data to data_existing
	for (file,datum_existing),datum in zip(data_existing.items(),data.values):
		data_existing[file] = append_data(datum_existing,datum)

	# Export updated data to files
	exporter(data_existing,directory)		

	return





# Class ovveride for encoding objects with unserializable
# encode_types: {type: serialize_function} in json
@argswrapper
class JSONencoder(json.JSONEncoder):

	def __init__(self,encode_types={},*args,**kwargs):
		json.JSONEncoder.__init__(self, *args, **kwargs)

		self.encode_types = {dict: self.encode_dict,
							 np.ndarray: self.encode_ndarray}
		if isinstance(encode_types,dict):
			self.encode_types.update(encode_types)
		
		return

	def default(self, obj):
		return self.encode_types.get(type(obj),
								lambda o:json.JSONEncoder.default(self, o))(obj)

	def encode_ndarray(self,obj):
		return obj.tolist()

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

	def __init__(self,decode_types={},*args,**kwargs):
		json.JSONDecoder.__init__(self, object_hook = self.object_hook,
										*args,**kwargs)

		self.decode_types = {dict: self.decode_dict,
							 np.ndarray: self.decode_ndarray}
		if isinstance(decode_types,dict):
			self.decode_types.update(decode_types)

		return

	def object_hook(self, obj):
		if isinstance(obj,dict):
			return self.decode_types.get(type(obj),lambda o:o)(obj)
		else:
			return obj

	def decode_ndarray(self,obj):
		return np.array(obj)

	def decode_dict(self,obj):
		obj_json = {}
		for k,v in obj.items():
			k = self.decode_types.get(type(k),lambda x:x)(k)
			v = self.decode_types.get(type(v),lambda x:x)(v)
			obj_json[k] = v

		return obj_json
