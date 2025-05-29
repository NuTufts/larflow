from __future__ import print_function
import os,sys,argparse,time

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

args = parser.parse_args()
if args.products not in ["rerun","min","debug"]:
    raise ValueError("--product argument must either be {rerun,min,debug}")

import ROOT as rt
from ROOT import std
from larlite import larlite
from larcv import larcv
from ublarcvapp import ublarcvapp
from larflow import larflow


io = larlite.storage_manager( larlite.storage_manager.kBOTH )
tickdir = larcv.IOManager.kTickForward
if args.tickbackwards:
    tickdir = larcv.IOManager.kTickBackward
iolcv = larcv.IOManager( larcv.IOManager.kBOTH, "larcv", tickdir )

print("[INPUT: DL MERGED] ",args.input_dlmerged)
print("[INPUT: LARMATCH-KPS]  ",args.input_larflow)
print("[OUTPUT]    ",args.output)

# ALGORITHMS
recoman = larflow.reco.KPSRecoManager( args.output.replace(".root","_kpsrecomanagerana.root"), args.version )
recoman.set_verbosity(args.loglevel)
recoman.logger().default_level(args.loglevel)
if args.loglevel == 0:
    recoman.set_verbosity(larcv.msg.kDEBUG)
    recoman.logger().default_level(larcv.msg.kDEBUG)
    io.set_verbosity(args.loglevel)
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
    recoman.saveEventMCinfo( args.ismc )
    if args.run_perfect_mcreco:
        recoman.runPerfectMCreco( True )
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
iolcv.specify_data_read( larcv.kProductSparseImage, "sparseuresnetout" )
iolcv.specify_data_read( larcv.kProductSparseImage, "sparsessnet" ) 
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
    iolcv.addto_storeonly_list( larcv.kProductSparseImage, "sparseuresnetout" )
    iolcv.addto_storeonly_list( larcv.kProductSparseImage, "sparsessnet" )    
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
    io.set_data_to_write( "track", "cosmicproton" )
    io.set_data_to_write( "track", "nutrack_fitted" )  
    io.set_data_to_write( "larflowcluster", "cosmicproton" )  # out-of-time track clusters with dq/dx consistent with possible proton
    io.set_data_to_write( "pcaxis", "cosmicproton" )  # out-of-time track clusters with dq/dx consistent with possible proton

    # keypoint reco
    io.set_data_to_write( "larflow3dhit", "keypoint" ) # save reco keypoints, used to seed nu candidates
    io.set_data_to_write( "larflow3dhit", "keypointcosmic" ) # save reco keypoints, used to seed cosmic candidates    

    # cosmic hit clusters:  trade space for time, since can use track paths to pick up hits again
    io.set_data_to_write( "larflowcluster", "boundarycosmicnoshift" )
    io.set_data_to_write( "larflowcluster", "containedcosmic" )

    # cluster reco
    io.set_data_to_write( "larflowcluster", "trackprojsplit_wcfilter" ) # in-time track clusters
    io.set_data_to_write( "larflowcluster", "showerkp" )      # in-time shower clusters, found using shower keypoints
    io.set_data_to_write( "larflowcluster", "showergoodhit" ) # in-time shower clusters
    io.set_data_to_write( "larflowcluster", "hip" )           # in-time proton tracks
    io.set_data_to_write( "pcaxis", "trackprojsplit_wcfilter" ) # in-time track clusters
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

io.enable_event_alignment(False) # hack
#io.go_to( args.start_entry )
#io.next_event(False)
#io.go_to( args.start_entry )
for ientry in range( args.start_entry, end_entry ):
    print("[ENTRY ",ientry,"]")
    iolcv.read_entry(ientry)
    io.go_to(ientry,False) # read without writing
    print("io larlite: ",io.run_id(), io.subrun_id(), io.event_id())
    print("reco, make nu candidates, calculate selection variables")
    sys.stdout.flush()

    # get empty containers to ensure we create an entry for the critical larlite and larcv trees passed
    # passed to merged dlreco file and kps larlite file
    io.get_data( larlite.data.kTrack,           "boundarycosmic" )
    io.get_data( larlite.data.kTrack,           "boundarycosmicnoshift" )
    io.get_data( larlite.data.kTrack,           "containedcosmic" )
    io.get_data( larlite.data.kTrack,           "cosmicproton" )
    io.get_data( larlite.data.kTrack,           "nutrack_fitted" )
    io.get_data( larlite.data.kLArFlow3DHit,    "keypoint" )
    io.get_data( larlite.data.kLArFlow3DHit,    "keypointcosmic" )
    io.get_data( larlite.data.kLArFlowCluster,  "hip" )
    io.get_data( larlite.data.kPCAxis,          "hip" )    
    io.get_data( larlite.data.kLArFlowCluster,  "showergoodhit" )
    io.get_data( larlite.data.kPCAxis,          "showergoodhit" )    
    io.get_data( larlite.data.kLArFlowCluster,  "showerkp" )
    io.get_data( larlite.data.kPCAxis,          "showerkp" )    
    io.get_data( larlite.data.kLArFlowCluster,  "trackprojsplit_wcfilter" )
    io.get_data( larlite.data.kPCAxis,          "trackprojsplit_wcfilter" )        
    
    recoman.process( iolcv, io )
    
    io.set_id( io.run_id(), io.subrun_id(), io.event_id() )
    io.go_to(ientry,True) # write
    iolcv.save_entry()
    sys.stdout.flush()
# write last event


print("Event Loop finished")
#del kpsrecoman
sys.stdout.flush()

io.close()
iolcv.finalize()
recoman.write_ana_file()
recoman.close_ana_file()
