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
#iolcv.addto_storeonly_list( ... )
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

algo = larflow.reco.NuVertexActivityReco()
algo.set_verbosity( larcv.msg.kDEBUG )
mcdata = ublarcvapp.mctools.LArbysMC()

kpreco = larflow.reco.KeypointReco()
kpreco.set_input_larmatch_tree_name( "larmatch" );
kpreco.set_sigma( 10.0 );
kpreco.set_min_cluster_size(   50, 0 )
kpreco.set_keypoint_threshold( 0.5, 0 )
kpreco.set_min_cluster_size(   20, 1 )
kpreco.set_keypoint_threshold( 0.5, 1 )
kpreco.set_larmatch_threshold( 0.5 )


io.go_to(8)
for ientry in range(8,nentries):
    print("[ENTRY ",ientry,"]")
    iolcv.read_entry(ientry)

    algo.process( iolcv, io )
    mcdata.process( io )
    mcdata.process( iolcv, io )
    mcdata.printInteractionInfo()

    # process nu keypoints
    kpreco.set_keypoint_type( 0 )
    kpreco.set_lfhit_score_index( 13 )    
    kpreco.process( io );
    # track key points
    kpreco.set_keypoint_type( 1 )
    kpreco.set_lfhit_score_index( 14 )    
    kpreco.process( io );
    # shower key points
    kpreco.set_keypoint_type( 2)
    kpreco.set_lfhit_score_index( 15 )    
    kpreco.process( io );
    
    io.set_id( io.run_id(), io.subrun_id(), io.event_id() )
    io.next_event()
    iolcv.save_entry()
    break

io.close()
iolcv.finalize()
print("[END]")
