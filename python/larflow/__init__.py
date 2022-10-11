from __future__ import print_function
import ROOT,os

# check important environment variables
#for basedir in ['LARLITE_BASEDIR','LAROPENCV_BASEDIR','LARCV_BASEDIR','UBLARCVAPP_BASEDIR','LARFLOW_BASEDIR']:
for basedir in ['LARLITE_BASEDIR','LARCV_BASEDIR','UBLARCVAPP_BASEDIR','LARFLOW_BASEDIR']:
    if not basedir in os.environ:
        print(basedir+' shell env. var. not found (run configure.sh for {})'.format(basedir.split("_")[0]))
        raise ImportError


# LOAD DEPENDENCIES

if 'LARLITE_BASEDIR' in os.environ:
    from larlite import larlite
#if 'LAROPENCV_BASEDIR' in os.environ:
#    from larocv import larocv
if 'LARCV_BASEDIR' in os.environ:
    from larcv import larcv
if 'UBLARCVAPP_BASEDIR' in os.environ:
    from ublarcvapp import ublarcvapp
    
lib_dir = os.environ['LARFLOW_LIBDIR']

# LOAD LIBS
for l in [x for x in os.listdir(lib_dir) if x.endswith('.so')]:
    if "LARFLOW_PYTHONLOAD_DEBUG" in os.environ and os.environ["LARFLOW_PYTHONLOAD_DEBUG"]!='0':
        print("loading: ",l)
    ROOT.gSystem.Load(l)

import ROOT.larflow as larflow
#larflow.load_flow_contour_match()
