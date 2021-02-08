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
#parser.add_argument('-t','--input-mcinfo',type=str,default=None,help="Input file containing larlite mc truth objects")
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
#print("[INPUT (optional): MCINFO]  ",args.input_mcinfo)
print("[OUTPUT]    ",args.output)

io.add_in_filename(  args.input_dlmerged )
if args.input_dlmerged!=args.input_larflow:
    io.add_in_filename(  args.input_larflow )
#if args.input_mcinfo is not None and args.input_dlmerged!=args.input_mcinfo and args.input_larflow!=args.input_mcinfo:
#    io.add_in_filename( args.input_mcinfo )
#io.set_data_to_read( larlite.data.kLArFlow3DHit, "larmatch" )
#io.set_data_to_read( larlite.data.kMCTrack,  "mcreco" )
#io.set_data_to_read( larlite.data.kMCShower, "mcreco" )
#io.set_data_to_read( larlite.data.kMCTruth,  "generator" )
io.set_data_to_read( larlite.data.kOpFlash,  "simpleFlashBeam" )
#io.set_data_to_read( larlite.data.kOpFlash,  "simpleFlashCosmic" )
#io.set_data_to_write( larlite.data.kLArFlow3DHit, "vacand" )
#io.set_data_to_write( larlite.data.kLArFlow3DHit, "keypoint" )
#io.set_data_to_write( larlite.data.kPCAxis, "keypoint" )

iolcv.add_in_file(   args.input_dlmerged )
iolcv.specify_data_read( larcv.kProductImage2D, "wire" );
#iolcv.specify_data_read( larcv.kProductImage2D, "thrumu" );
#iolcv.specify_data_read( larcv.kProductImage2D, "ancestor" );
#iolcv.specify_data_read( larcv.kProductImage2D, "segment" );
#iolcv.specify_data_read( larcv.kProductImage2D, "instance" );
#iolcv.specify_data_read( larcv.kProductImage2D, "larflow" );
#iolcv.specify_data_read( larcv.kProductChStatus, "wire" );
#iolcv.specify_data_read( larcv.kProductImage2D, "ubspurn_plane0" )
#iolcv.specify_data_read( larcv.kProductImage2D, "ubspurn_plane1" )
#iolcv.specify_data_read( larcv.kProductImage2D, "ubspurn_plane2" )
iolcv.specify_data_read( larcv.kProductROI, "segment" ) #partroi_segment_tree
iolcv.specify_data_read( larcv.kProductSparseImage, "sparseuresnetout" )
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

algo = larflow.reco.SplitHitsByParticleSSNet()
algo.set_verbosity( larcv.msg.kDEBUG )
#algo.set_verbosity( larcv.msg.kINFO )
mcdata = ublarcvapp.mctools.LArbysMC()

#tfana = rt.TFile( args.output.replace(".root","_ana.root"), "recreate" )
#tfana.cd()
#vatree = rt.TTree("vtxactivityana","Vertex Activity Ana")
#mcdata.bindAnaVariables( vatree )
#algo.bind_to_tree( vatree )

start_entry = 0
io.go_to(start_entry)
for ientry in range(start_entry,nentries):

    rse = ( io.run_id(), io.subrun_id(), io.event_id() )
    print("[ENTRY ",ientry,"]: ",rse)
    
    iolcv.read_entry(ientry)    
    algo.process( iolcv, io )
    #if args.input_mcinfo is not None:
    #    print("RUN MC ROUTINES")
    #    mcdata.process( io )
    #    mcdata.process( iolcv, io )
    #    mcdata.printInteractionInfo()
    #    algo.calcTruthVariables( io, iolcv, mcdata )
    #else:
    #    print("No MC provided")

    io.set_id( io.run_id(), io.subrun_id(), io.event_id() )
    io.next_event()
    iolcv.save_entry()
    #vatree.Fill()
    #if ientry>=10:
    #break


io.close()
iolcv.finalize()
#tfana.cd()
#vatree.Write()
print("[END]")
