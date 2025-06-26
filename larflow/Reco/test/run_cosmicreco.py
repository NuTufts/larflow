#!/bin/env python3
from __future__ import print_function
import os,sys,argparse,time
import traceback
from io import StringIO

"""
Run the PCA-based clustering routine for track space-points.
Uses 3D points saved in larflow3dhit objects.
"""

parser = argparse.ArgumentParser("Run larflow3dhit clustering algorith")
# required
parser.add_argument('-i','--input-dlmerged',type=str,required=True,help="Input file containing ADC, ssnet, badch images/info")
parser.add_argument('-l','--input-larflow',type=str,required=True,help="Input file containing larlite::larflow3dhit objects")
parser.add_argument('-o','--output',type=str,required=True,help="Name of output file. Will not overwrite")
# optional
parser.add_argument('-n','--num-entries',type=int,default=None,help="Number of entries to run")
parser.add_argument('-e','--start-entry',type=int,default=0,help="Starting entry")
parser.add_argument('-tb','--tickbackwards',action='store_true',default=False,help="Input larcv images are tick-backward")
parser.add_argument("-mc",'--ismc',action='store_true',default=False,help="If true, store MC information")
parser.add_argument("-v",'--version',default=2,type=int,help="The reco version [default 2]")
parser.add_argument("-ll","--loglevel",default=1,type=int,help="log verbosity (0: debug, 1: info, 2: normal, 3: warning, 4: error)")
# just for debug/development

args = parser.parse_args()

import ROOT as rt
from ROOT import std
from larlite import larlite
from larcv import larcv
from ublarcvapp import ublarcvapp
from larflow import larflow
larlite.larflow3dhit
larlite.larflowcluster

# check we have the spline files
if not os.path.exists( os.environ["LARFLOW_BASEDIR"]+"/larflow/Reco/data/Proton_Muon_Range_dEdx_LAr_TSplines.root" ):
    print("Did not find Range and dE/dx spline file: Proton_Muon_Range_dEdx_LAr_TSplines.root")
    print("file should be in larflow/Reco/data")
    sys.exit(0)


input_spacepoint_container_name = "larmatch"

io = larlite.storage_manager( larlite.storage_manager.kBOTH )
tickdir = larcv.IOManager.kTickForward
if args.tickbackwards:
    tickdir = larcv.IOManager.kTickBackward
iolcv = larcv.IOManager( larcv.IOManager.kBOTH, "larcv", tickdir )

output_ana_filename = args.output.replace(".root","_cosmicreco.root")

print("[INPUT: DL MERGED] ",args.input_dlmerged)
print("[INPUT: LARMATCH-KPS]  ",args.input_larflow)
print("[OUTPUT]    ",args.output)
print("[OUTPUT-ANA]: ",output_ana_filename)

# RECO ALGORITHM MANAGER: larflow::reco::KPSRecoManager

recoman = larflow.reco.CosmicParticleReconstruction()
recoman.set_verbosity(args.loglevel)
recoman.logger().default_level(args.loglevel)
iolcv.set_verbosity(args.loglevel)
io.set_verbosity(args.loglevel)


# INPUT/OUTPUT SETTINGS
io.add_in_filename(  args.input_dlmerged )
io.add_in_filename(  args.input_larflow )
io.set_data_to_read( "larflow3dhit", "larmatch" )
io.set_data_to_read( "mctrack",  "mcreco" )
io.set_data_to_read( "mcshower", "mcreco" )
io.set_data_to_read( "mctruth",  "generator" )
io.set_data_to_read( "opflash",  "simpleFlashBeam" )
io.set_data_to_read( "opflash",  "simpleFlashCosmic" )

iolcv.add_in_file(   args.input_dlmerged )
iolcv.specify_data_read( "image2d", "wire" );
iolcv.specify_data_read( "image2d", "thrumu" );
iolcv.specify_data_read( "image2d", "ancestor" );
iolcv.specify_data_read( "image2d", "segment" );
iolcv.specify_data_read( "image2d", "instance" );
iolcv.specify_data_read( "image2d", "larflow" );
iolcv.specify_data_read( "chstatus", "wire" );
iolcv.specify_data_read( "image2d", "ubspurn_plane0" )
iolcv.specify_data_read( "image2d", "ubspurn_plane1" )
iolcv.specify_data_read( "image2d", "ubspurn_plane2" )
iolcv.specify_data_read( "sparseimage", "sparseuresnetout" )
iolcv.specify_data_read( "sparseimage", "sparsessnet" ) 
#iolcv.addto_storeonly_list( ... )
if args.tickbackwards:
    iolcv.reverse_all_products()

io.set_out_filename( args.output.replace(".root","_larlite.root") )
iolcv.set_out_file( args.output.replace(".root","_larcv.root") )

io.open()
iolcv.initialize()

lcv_nentries = iolcv.get_n_entries()
ll_nentries  = io.get_entries()
if lcv_nentries<ll_nentries:
    nentries = lcv_nentries
else:
    nentries = ll_nentries
    
if args.num_entries is not None:
    end_entry = args.start_entry + args.num_entries
    if end_entry>nentries:
        end_entry = nentries
else:
    end_entry = nentries

    
io.go_to( args.start_entry )

for ientry in range( args.start_entry, end_entry ):
    print("[ENTRY ",ientry,"]")

    # create storage container for user_info
    ev_userinfo = io.get_data(larlite.data.kUserInfo,'recoinfo')
    reco_ok = 1
    reco_err = ""

    tstart = time.time()
    try:
      iolcv.read_entry(ientry)
      recoman.process( iolcv, io )
    except Exception as e:
      print("reco failure",file=sys.stderr)
      print(e,file=sys.stderr)        
      print(traceback.format_exc(),sys.stderr)
      #error_string = error_buffer.getvalue()
      #print(error_string)
      print("reco failure")
      reco_ok = 0
      pass

    dt_reco = time.time()-tstart
    user_info = larlite.user_info()
    user_info.store( "dt_reco", float(dt_reco))
    user_info.store( "reco_ok", int(reco_ok))
    #user_info.store( "error", str(error_string))
    ev_userinfo.push_back( user_info )
    print("save entry")
    io.set_id( io.run_id(), io.subrun_id(), io.event_id() )
    io.next_event()
    iolcv.save_entry()
    sys.stdout.flush()
    

print("Event Loop finished")
#del kpsrecoman
sys.stdout.flush()

io.close()
iolcv.finalize()

