# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 01:08:22 2018

@author: Matt
"""

def import_obj(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod



class profiler(object):
    def __init__(self,module,method,*args):
        # ,file= os"MagnetismModel" if file == None else file,obj=system,path="/C:/Users/Matt/Google Drive/PSI/PSI Essay/",sort='cumultive'):
        # Import profiler modules
        import cProfile
        import pstats
        import os
        
        mod = import_obj(module)
#        obj = getattr(mod(*args),method)
#        print(mod.__class__.__name__)
#        print(obj.__name__)

#        methods = [obj]
#        for i,m in enumerate(np.atleast_1d(method)):
#            methods.append(getattr())
        
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
        profiledir = 'ProfilerStats'
        if not(os.path.isdir(profiledir)):
            os.mkdir(profiledir)
                
        statsdir = '/'.join([profiledir,('_').join(module.split('.'))])
        if not(os.path.isdir(statsdir)):
            os.mkdir(statsdir)
            
        statsfile = '/'.join([statsdir,'profilestats_'+getattr(mod(*args),method).__name__])
       
        
        
        cProfile.run(getattr(mod(*args),method).__name__,statsfile)
        
        p = pstats.Stats(statsfile)
        p.strip_dirs().sort_stats('cumulative').print_stats()
        
class profiler_graph(object):
    def __init__(self,module):
        from pycallgraph import PyCallGraph
        from pycallgraph.output import GraphvizOutput

        with PyCallGraph(output=GraphvizOutput()):
            module()
            
if __name__ == "__main__":
    T = [5,2.5,2,1.5,1,0.5]
    T0 = 0.5
    L=6
    d=2
    
    from MagnetismModel import system
    
    profiler_graph(system(L,d,T).MonteCarlo)
    
    #profiler('MagnetismModel.system','MonteCarlo',L,d,T)
        