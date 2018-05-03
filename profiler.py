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
    def __init__(self,module,num=''):
        from pycallgraph import PyCallGraph
        from pycallgraph.output import GraphvizOutput

        with PyCallGraph(output=GraphvizOutput(output_file = 'pycallgraph_%s.png'%num)):
            module()

num = 0
if __name__ == "__main__":
    L=15
    d=2
    T = [3.0,2.5,1.75,1.2,0.8,0.5,0.2]
    T0 = 0.25
    model=['potts',2,[0,1]]
    update = [True,10,10,1,1]
    observe = {'configurations': [False,'sites','cluster'],
                           'observables': [True,'temperature','energy',
                                                'order','specific_heat',
                                                'susceptibility'],
                           'observables_mean': [True]
                           }
    datasave = False
    
    
    props_iter = {'algorithm':['wolff','metropolis']}
    num +=1
#    
    from MagnetismModel import system
    
    profiler_graph(system(L,d,T,model,update,observe,datasave).MonteCarlo,num)
    
#    from neural network import foo
#    profiler_graph(foo,3)
    #profiler('MagnetismModel.system','MonteCarlo',L,d,T)