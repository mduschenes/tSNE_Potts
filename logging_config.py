# Logger Tracking
import logging,logging.config,sys,os
from os import path

class StreamToLogger(object):
	"""
	Fake file-like stream object that redirects writes to a logger instance.
	"""
	def __init__(self, logger, level):
		self.logger = logger
		self.level = level
		self.linebuf = ''

	def write(self, buf):
		# self.logger.log(self.log_level,buf)
		line = '\n'.join([l.rstrip() 
									for l in buf.rstrip().splitlines()])
		if line not in ['',':']:
			self.logger.log(self.level,line)


def logging_config(logfilename='log.log',configfilename='logging.conf',
					loggername='debug'):
	configfilepath = path.join(path.dirname(path.abspath(__file__)), 
								configfilename)
	logging.config.fileConfig(configfilepath,
								defaults={'logfilename': logfilename})


	logger = logging.getLogger(loggername)

	# Log STDOUT and STDERR 
	# sys.stdout = StreamToLogger(logger, logging.DEBUG)

	sys.stderr = StreamToLogger(logger, logging.WARNING)

	return logger


# logFormatter = logging.Formatter('%(asctime)s %(message)s',
# 					datefmt='%m/%d/%Y %I:%M:%S %p')

# fileHandler = logging.FileHandler(filename='logger.log',mode='w')
# fileHandler.setFormatter(logFormatter)

# consoleHandler = logging.StreamHandler()
# consoleHandler.setFormatter(logFormatter)

# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
# logger.addHandler(fileHandler)
# logger.addHandler(consoleHandler)								