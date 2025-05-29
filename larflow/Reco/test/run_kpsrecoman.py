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
parser.add_argument("-p","--products",default="rerun",help="output products saved. choices: {rerun[default],min,debug}")
parser.add_argument("-f","--event-filter",default=False,action='store_true',help="If true, filter events by dev 1e1p selection [default false]")
parser.add_argument("-v",'--version',default=2,type=int,help="The reco version [default 2]")
parser.add_argument("-ll","--loglevel",default=1,type=int,help="log verbosity (0: debug, 1: info, 2: normal, 3: warning, 4: error)")
# just for debug/development
parser.add_argument('--stop-after-spacepointprep',default=False,action='store_true',help="If true, stop at Spacepoint Prep")
parser.add_argument('--stop-after-keypointreco',default=False,action='store_true',help="If true, stop at Keypoint Reco")
parser.add_argument('--stop-after-subclustering',default=False,action='store_true',help="If true, stop at subcluster reco")
parser.add_argument('--stop-after-nutracker',default=False,action='store_true',help="If true, stop at subcluster reco")
parser.add_argument("--run-perfect-mcreco",default=False,action='store_true',help="If true, and --ismc also provided, then perfecto reco module is run")
parser.add_argument("--save-all-keypoints",default=False,action="store_true",help="If flag given, all reconstructed keypoints are saved to the ana file")
parser.add_argument("--run-nuvertexshowerreco-mcana-mode", default=False, action='store_true', help="If flag given, run MC analysis for NuVertexShowerReco")

args = parser.parse_args()
if args.products not in ["rerun","min","debug"]:
    raise ValueError("--product argument must either be {rerun,min,debug}")

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

print("[INPUT: DL MERGED] ",args.input_dlmerged)
print("[INPUT: LARMATCH-KPS]  ",args.input_larflow)
print("[OUTPUT]    ",args.output)

# RECO ALGORITHM MANAGER: larflow::reco::KPSRecoManager
recoman = larflow.reco.KPSRecoManager( args.output.replace(".root","_kpsrecomanagerana.root"), args.version )
recoman.set_verbosity(args.loglevel)
recoman.logger().default_level(args.loglevel)
recoman.saveNuAttachableClusters()
# if args.loglevel == 0:
#   recoman.set_verbosity(larcv.msg.kDEBUG)
#   recoman.logger().default_level(larcv.msg.kDEBUG)
# elif args.loglevel == 1:
#   recoman.set_verbosity(larcv.msg.kINFO)
#   recoman.logger().default_level(larcv.msg.kINFO)
# elif args.loglevel == 2:
#   recoman.set_verbosity(larcv.msg.kNORMAL)
#   recoman.logger().default_level(larcv.msg.kNORMAL)
# elif args.loglevel == 3:
#   recoman.set_verbosity(larcv.msg.kWARNING)
#   recoman.logger().default_level(larcv.msg.kWARNING)
# elif args.loglevel == 4:
#   recoman.set_verbosity(larcv.msg.kERROR)
#   recoman.logger().default_level(larcv.msg.kERROR)
# else:
#   recoman.set_verbosity(larcv.msg.kINFO)
#   recoman.logger().default_level(larcv.msg.kINFO)

recoman.minimze_output_size(True)
if args.ismc:
    activate_mcanamode_nuvertexshowereco = args.run_nuvertexshowerreco_mcana_mode
    recoman.saveEventMCinfo( args.ismc, activate_mcanamode_nuvertexshowereco )
    if args.run_perfect_mcreco:
        recoman.runPerfectMCreco( True )
else:
    recoman.saveEventMCinfo( False, False )
    recoman._nuvertex_shower_reco.activateMCanalysisMode(False)
    
recoman.set_spacepoint_input_container_name( input_spacepoint_container_name )
        
if args.event_filter:
    recoman.saveSelectedNuVerticesOnly( args.event_filter )
if args.stop_after_keypointreco:
    recoman.debug_stop_at_keypoint_reco( True )
if args.stop_after_spacepointprep:
    print("STOP AFTER SPACEPOINT PREP")
    recoman.debug_stop_at_spacepoint_prep( True )
    print("[enter] to start")
    input()
if args.stop_after_subclustering:
    print("STOP AFTER SUBCLUSTERING")
    recoman.debug_stop_at_subclustering( True )
    print("[enter] to start")
    input()
if args.stop_after_nutracker:
    print("STOP AFTER MULTIPRONG RECO/NUTRACKBUILDER")
    recoman.debug_stop_at_nutracker( True )
    print("[enter] to start")
    input()
if args.save_all_keypoints:
    print("Save all reconstructed keypoints. Usually for selection development.")
    recoman.saveEventKeypoints( True )


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


    
if args.products in ["rerun"]:
    print("Save enough info to allow rerunning")
    #larcv
    iolcv.addto_storeonly_list( "image2d",  "wire" )
    iolcv.addto_storeonly_list( "image2d",  "thrumu" )
    iolcv.addto_storeonly_list( "chstatus", "wire" )          
    for p in range(3):
        iolcv.addto_storeonly_list( "image2d", "ubspurn_plane%d"%(p) )
    iolcv.addto_storeonly_list( "sparseimage", "sparseuresnetout" )
    iolcv.addto_storeonly_list( "sparseimage", "sparsessnet" )    
    for truthproduct in ["instance","segment","ancestor","larflow"]:
        iolcv.addto_storeonly_list( "image2d", truthproduct )
         
    #larlite
    io.set_data_to_write( "larflow3dhit", "larmatch" )
    io.set_data_to_write( "mctruth", "generator" )
    io.set_data_to_write( "mcshower", "mcreco" )
    io.set_data_to_write( "mctrack",  "mcreco" )

if args.products in ["rerun","min"]:

    print("Save minimal amount of data, enough to plot in vis_kpreco.py")
    
    # cosmic reco saved, since nuvertex data in ana file is does not save cosmic info
    io.set_data_to_write( "track", "boundarycosmic" )
    io.set_data_to_write( "track", "boundarycosmicnoshift" )
    io.set_data_to_write( "track", "containedcosmic" )
    io.set_data_to_write( "track", "nutrack_fitted" )
    io.set_data_to_write( "track", "cosmicproton" )        
    io.set_data_to_write( "larflowcluster", "cosmicproton" )  # out-of-time track clusters with dq/dx consistent with possible proton
    io.set_data_to_write( "pcaxis", "cosmicproton" )  # out-of-time track clusters with dq/dx consistent with possible proton

    # keypoint reco
    io.set_data_to_write( "larflow3dhit", "keypoint" ) # save reco keypoints
    io.set_data_to_write( "larflow3dhit", "showerkp" ) # save reco keypoints matched to shower clusters
    io.set_data_to_write( "larflow3dhit", "keypoint_nuvtxseed" ) # save reco keypoints, used to seed nu candidates
    io.set_data_to_write( "larflow3dhit", "keypointcosmic" ) # save reco keypoints, used to seed cosmic candidates    

    # cosmic hit clusters:  trade space for time, since can use track paths to pick up hits again
    io.set_data_to_write( "larflowcluster", "boundarycosmicnoshift" )
    io.set_data_to_write( "larflowcluster", "containedcosmic" )

    # user info
    io.set_data_to_write( "user_info", "recoinfo" )

    # cluster reco

    # mc info
    io.set_data_to_write( larlite.data.kMCShower, "mcdetectableshower" )
    io.set_data_to_write( "larflowcluster", "trackprojsplit_wcfilter" ) # in-time track clusters
    io.set_data_to_write( "larflowcluster", "showerkp" )        # in-time shower clusters, found using shower keypoints
    io.set_data_to_write( "larflowcluster", "showergoodhit" )   # in-time shower clusters
    io.set_data_to_write( "larflowcluster", "hip" )             # in-time proton tracks
    io.set_data_to_write( "larflow3dhit", "projsplitvetoed" ) # vetoed kp hits
    io.set_data_to_write( "pcaxis", "maxtrackhit_wcfilter" ) # in-time track clusters
    io.set_data_to_write( "pcaxis", "showerkp" )      # in-time shower clusters, found using shower keypoints
    io.set_data_to_write( "pcaxis", "showergoodhit" ) # in-time shower clusters
    io.set_data_to_write( "pcaxis", "hip" )           # in-time proton tracks
    
    # save flash
    io.set_data_to_write( "opflash", "simpleFlashBeam" )
    io.set_data_to_write( "opflash", "simpleFlashCosmic" )  
    
    recoman.minimze_output_size(False)

if args.products in ["debug"]:
    print("Saving all products loaded and made by reconstruction code")
    # no output specify, so io managers will default to saving everything

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


error_buffer = io.StringIO()
original_stderr = sys.stderr
sys.stderr = error_buffer
    
io.go_to( args.start_entry )
#io.next_event()
#io.go_to( args.start_entry )
for ientry in range( args.start_entry, end_entry ):
    print("[ENTRY ",ientry,"]")

    # create storage container for user_info
    ev_userinfo = io.get_data('user_info','recoinfo')
    reco_ok = 1
    reco_err = ""
    tstart = time.time()
    iolcv.read_entry(ientry)
    
    try:
        print("reco, make nu candidates, calculate selection variables")
        sys.stdout.flush()
        recoman.process( iolcv, io )

    except Exception as e:
        print("reco failure",file=sys.stderr)
        print(e,file=sys.stderr)        
        print(traceback.format_exc(),sys.stderr)
        error_string = error_buffer.getvalue()
        reco_ok = 0

    dt_reco = time.time()-tstart
    user_info = larlite.user_info()
    user_info.store( "dt_reco", float(dt_reco))
    user_info.store( "reco_ok", int(reco_ok))
    user_info.store( "error", str(error_string))
    ev_userinfo.push_back( user_info )
    
    io.set_id( io.run_id(), io.subrun_id(), io.event_id() )
    io.next_event()
    iolcv.save_entry()
    sys.stdout.flush()
    

print("Event Loop finished")
#del kpsrecoman
sys.stdout.flush()

io.close()
iolcv.finalize()
recoman.write_ana_file()
recoman.close_ana_file()

#os._exit(0)
