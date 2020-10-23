import os,sys,time,argparse

parser = argparse.ArgumentParser("Run CRT-MATCHING algorithms")
parser.add_argument('-dl','--input-dlmerged',type=str,required=True,help="DL merged file")
parser.add_argument('-lm','--input-larmatch',type=str,required=True,help="larmatch output")
parser.add_argument('-o',"--output",type=str,required=True,help="Output file stem")
parser.add_argument('-n','--num-entries',type=int,default=None,help="Number of entries to run")
parser.add_argument('-e','--start-entry',type=int,default=0,help="Starting entry")

args = parser.parse_args()

import ROOT as rt
from ROOT import std
from larlite import larlite
from larcv import larcv
from larflow import larflow

io = larlite.storage_manager( larlite.storage_manager.kBOTH )
iolcv = larcv.IOManager( larcv.IOManager.kBOTH, "larcv", larcv.IOManager.kTickBackward )

# INPUTS
io.add_in_filename( args.input_dlmerged )
io.add_in_filename( args.input_larmatch )
io.set_data_to_read( larlite.data.kLArFlow3DHit, "larmatch" )
io.set_data_to_read( larlite.data.kMCTrack,  "mcreco" )
io.set_data_to_read( larlite.data.kMCShower, "mcreco" )
io.set_data_to_read( larlite.data.kMCTruth,  "generator" )
io.set_data_to_read( larlite.data.kOpFlash,  "simpleFlashBeam" )
io.set_data_to_read( larlite.data.kOpFlash,  "simpleFlashCosmic" )
io.set_data_to_read( larlite.data.kCRTHit,   "crthitcorr" )


iolcv.add_in_file(  args.input_dlmerged )
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
iolcv.reverse_all_products()

# OUTPUTS
if ".root"==args.output[-5:]:
    larlite_out = args.output.replace(".root","_larlite.root")
    larcv_out   = args.output.replace(".root","_larcv.root")
else:
    larlite_out = args.output + "_larlite.root"
    larcv_out   = args.output + "_larcv.root"

io.set_out_filename( larlite_out )
io.open()

iolcv.set_out_file( larcv_out )
iolcv.initialize()

crtmm = larflow.crtmatch.CRTMatchManager( "cmm_ana.root" )

#NUM ENTRIES
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


print "Number of entries to run: ",nentries
print "Start loop."

for ientry in xrange( args.start_entry, end_entry ):

    print "[ENTRY ",ientry,"]"
    io.go_to(ientry)    
    iolcv.read_entry(ientry)


    dtload = time.time()
    
    crtmm.process( iolcv, io )
    #crtmm.store_output( outlcv, outio )
    
    #io.set_id( io.run_id(), io.subrun_id(), io.event_id() )
    #iolcv.set_id( io.run_id(), io.subrun_id(), io.event_id() )    
io.next_event()
iolcv.save_entry()


io.close()
iolcv.finalize()
