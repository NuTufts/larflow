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
parser.add_argument("-d","--debug",default=False,action='store_true',help="If true, store many intermediate products")
parser.add_argument("-min","--minimal",default=False,action='store_true',help="If true, store minimal products")
parser.add_argument("-f","--event-filter",default=False,action='store_true',help="If true, filter events by dev 1e1p selection [default false]")

args = parser.parse_args()

import ROOT as rt
from ROOT import std
from larlite import larlite
from larcv import larcv
from ublarcvapp import ublarcvapp
from larflow import larflow


io = larlite.storage_manager( larlite.storage_manager.kBOTH )
iolcv = larcv.IOManager( larcv.IOManager.kBOTH, "larcv", larcv.IOManager.kTickBackward )

print("[INPUT: DL MERGED] ",args.input_dlmerged)
print("[INPUT: LARMATCH-KPS]  ",args.input_larflow)
print("[OUTPUT]    ",args.output)

# ALGORITHMS
recoman = larflow.reco.KPSRecoManager( args.output.replace(".root","_kpsrecomanagerana.root") )
recoman.set_verbosity(larcv.msg.kINFO)
recoman.minimze_output_size(True)
if args.ismc:
    recoman.saveEventMCinfo( args.ismc )
if args.event_filter:
    recoman.saveSelectedNuVerticesOnly( args.event_filter )

# INPUT/OUTPUT SETTINGS
io.add_in_filename(  args.input_dlmerged )
io.add_in_filename(  args.input_larflow )
io.set_data_to_read( larlite.data.kLArFlow3DHit, "larmatch" )
io.set_data_to_read( larlite.data.kMCTrack,  "mcreco" )
io.set_data_to_read( larlite.data.kMCShower, "mcreco" )
io.set_data_to_read( larlite.data.kMCTruth,  "generator" )
io.set_data_to_read( larlite.data.kOpFlash,  "simpleFlashBeam" )
io.set_data_to_read( larlite.data.kOpFlash,  "simpleFlashCosmic" )


iolcv.add_in_file(   args.input_dlmerged )
iolcv.specify_data_read( larcv.kProductImage2D, "wire" );
iolcv.specify_data_read( larcv.kProductImage2D, "thrumu" );
iolcv.specify_data_read( larcv.kProductImage2D, "ancestor" );
iolcv.specify_data_read( larcv.kProductImage2D, "segment" );
iolcv.specify_data_read( larcv.kProductImage2D, "instance" );
iolcv.specify_data_read( larcv.kProductImage2D, "larflow" );
iolcv.specify_data_read( larcv.kProductChStatus, "wire" );
iolcv.specify_data_read( larcv.kProductImage2D, "ubspurn_plane0" )
iolcv.specify_data_read( larcv.kProductImage2D, "ubspurn_plane1" )
iolcv.specify_data_read( larcv.kProductImage2D, "ubspurn_plane2" )
iolcv.specify_data_read( larcv.kProductSparseImage, "sparseuresnetout" ) 
#iolcv.addto_storeonly_list( ... )
iolcv.reverse_all_products()

io.set_out_filename( args.output.replace(".root","_larlite.root") )
iolcv.set_out_file( args.output.replace(".root","_larcv.root") )

if args.debug is None or args.debug==False:

    # cosmic reco saved, since nuvertex data in ana file is does not save cosmic info
    io.set_data_to_write( larlite.data.kTrack, "boundarycosmic" )
    io.set_data_to_write( larlite.data.kTrack, "boundarycosmicnoshift" )
    io.set_data_to_write( larlite.data.kTrack, "containedcosmic" )

    io.set_data_to_write( larlite.data.kLArFlowCluster, "cosmicproton" )  # out-of-time track clusters with dq/dx consistent with possible proton
    io.set_data_to_write( larlite.data.kPCAxis, "cosmicproton" )  # out-of-time track clusters with dq/dx consistent with possible proton

    print("minimal: ",args.minimal)
    if args.minimal is None or args.minimal==False:
        # keypoint reco
        io.set_data_to_write( larlite.data.kLArFlow3DHit, "keypoint" ) # save reco keypoints, used to seed nu candidates

        # cosmic hit clusters:  trade space for time, since can use track paths to pick up hits again
        io.set_data_to_write( larlite.data.kLArFlowCluster, "boundarycosmicnoshift" )
        io.set_data_to_write( larlite.data.kLArFlowCluster, "containedcosmic" )

        # cluster reco
        io.set_data_to_write( larlite.data.kLArFlowCluster, "trackprojsplit_wcfilter" ) # in-time track clusters
        io.set_data_to_write( larlite.data.kLArFlowCluster, "showerkp" )      # in-time shower clusters, found using shower keypoints
        io.set_data_to_write( larlite.data.kLArFlowCluster, "showergoodhit" ) # in-time shower clusters
        io.set_data_to_write( larlite.data.kLArFlowCluster, "hip" )           # in-time proton tracks
        io.set_data_to_write( larlite.data.kPCAxis, "trackprojsplit_wcfilter" ) # in-time track clusters
        io.set_data_to_write( larlite.data.kPCAxis, "showerkp" )      # in-time shower clusters, found using shower keypoints
        io.set_data_to_write( larlite.data.kPCAxis, "showergoodhit" ) # in-time shower clusters
        io.set_data_to_write( larlite.data.kPCAxis, "hip" )           # in-time proton tracks

        # save flash
        io.set_data_to_write( larlite.data.kOpFlash, "simpleFlashBeam" )
        io.set_data_to_write( larlite.data.kOpFlash, "simpleFlashCosmic" )  
        
        io.minimze_output_size(False)

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
#io.next_event()
#io.go_to( args.start_entry )
for ientry in range( args.start_entry, end_entry ):
    print("[ENTRY ",ientry,"]")
    iolcv.read_entry(ientry)

    print("reco, make nu candidates, calculate selection variables")
    sys.stdout.flush()
    recoman.process( iolcv, io )

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
