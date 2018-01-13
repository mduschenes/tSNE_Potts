# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 01:08:22 2018

@author: Matt
"""

class profiler(object):
    def __init__(self,sort='cumulative'):
        # ,file= os"MagnetismModel" if file == None else file,obj=system,path="/C:/Users/Matt/Google Drive/PSI/PSI Essay/",sort='cumultive'):
        # Import profiler modules
        import cProfile
        import pstats
        from MagnetismModel_v2 import Lattice as obj 
        # Import object to profile from file
#        import importlib.util
#        import os
#        objpath = os.path.dirname(os.path.realpath(file.__name__))
#        print(objpath)
#        spec = importlib.util.spec_from_file_location(file,"\"+objpath)
#        module = importlib.util.module_from_spec(spec)
#        spec.loader.exec_module(module)
#        obj = module.obj
#       
       # file = MagnetismModel if file == None else file
       
        
#        import sys
#        import os
#        os.path.
#        objpath = os.path.dirname(os.path.realpath())
#        sys.path.append(objpath)
# 
#        from file import obj
        
        statsfile = 'profilestats_'+obj.__name__
        
        cProfile.run(obj.__name__+'(50,3)',statsfile)
        
        p = pstats.Stats(statsfile)
        p.strip_dirs().sort_stats('cumulative').print_stats()
        
if __name__ == "__main__":
    profiler()
        