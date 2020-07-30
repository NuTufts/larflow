import os,sys,time,argparse

"""
Run the CRT-TRACK MATCHING ALGORITHM
"""

parser = argparse.ArgumentParser("Find Through-going muon paths using CRT-TRACK objects")
# required
parser.add_argument('-i','--input-dlmerged',type=str,required=True,help="Input file containing ADC, ssnet, badch images/info")
parser.add_argument('-l','--input-larflow',type=str,required=True,help="Input file containing larlite::larflow3dhit objects")
parser.add_argument('-o','--output',type=str,required=True,help="Name of output file. Will not overwrite")
# optional
parser.add_argument('-n','--num-entries',type=int,default=None,help="Number of entries to run")
parser.add_argument('-e','--start-entry',type=int,default=0,help="Starting entry")
parser.add_argument('-tb','--tickbackwards',action='store_true',default=False,help="Input larcv images are tick-backward")

args = parser.parse_args()

import ROOT as rt
from ROOT import std
from larlite import larlite
from larcv import larcv
from larflow import larflow

#print larflow.reco.cluster_t
#print larflow.reco.cluster_larflow3dhits


io = larlite.storage_manager( larlite.storage_manager.kREAD )
io.add_in_filename(  args.input_dlmerged )
io.add_in_filename(  args.input_larflow )
io.set_data_to_read( larlite.data.kLArFlow3DHit, "larmatch" )
io.set_data_to_read( larlite.data.kCRTTrack, "crttrack" )
io.set_data_to_read( larlite.data.kCRTHit,   "crthitcorr" )
io.set_data_to_read( larlite.data.kMCTrack,  "mcreco" )
io.set_data_to_read( larlite.data.kMCShower, "mcreco" )
io.set_data_to_read( larlite.data.kMCTruth,  "generator" )
io.set_data_to_read( larlite.data.kOpFlash,  "simpleFlashBeam" )
io.set_data_to_read( larlite.data.kOpFlash,  "simpleFlashCosmic" )
#io.set_data_to_read( larlite.data.kTrack,    "boundarycosmicnoshift" )
#io.set_data_to_read( larlite.data.kTrack,    "containedcosmic" )
#io.set_data_to_read( larlite.data.kTrack,    "nutrack" )
io.open()

outio = larlite.storage_manager( larlite.storage_manager.kWRITE )
outio.set_out_filename( args.output )
outio.open()

iolcv = larcv.IOManager( larcv.IOManager.kREAD, "larcv", larcv.IOManager.kTickBackward )
iolcv.add_in_file( args.input_dlmerged )
iolcv.specify_data_read( larcv.kProductImage2D, "wire" );
iolcv.specify_data_read( larcv.kProductImage2D, "thrumu" );
iolcv.specify_data_read( larcv.kProductImage2D, "ancestor" );
iolcv.specify_data_read( larcv.kProductImage2D, "instance" );
iolcv.specify_data_read( larcv.kProductImage2D, "larflow" );
iolcv.specify_data_read( larcv.kProductChStatus, "wire" );
iolcv.specify_data_read( larcv.kProductImage2D, "ubspurn_plane0" )
iolcv.specify_data_read( larcv.kProductImage2D, "ubspurn_plane1" )
iolcv.specify_data_read( larcv.kProductImage2D, "ubspurn_plane2" )
iolcv.reverse_all_products()
iolcv.initialize()

crtmatch = larflow.crtmatch.CRTTrackMatch()
crtmatch.set_max_iters( 25 )
crtmatch.make_debug_images( False )
crtmatch.set_keep_only_boundary_tracks( True )
crtmatch.set_verbosity(0)

start_entry = args.start_entry

nentries = iolcv.get_n_entries()
if args.num_entries is not None:
    end_entry = start_entry + args.num_entries
else:
    end_entry = start_entry + nentries
    
if end_entry>nentries:
    end_entry = nentries
    nentries = end_entry-start_entry

print "Run entries ",start_entry," to ",end_entry
print "Number of entries to run: ",nentries

print "Start loop."

for ientry in xrange( start_entry, end_entry ):

    print "================ ENTRY[",ientry,"] ============================"

    io.go_to(ientry)    
    iolcv.read_entry(ientry)

    dtload = time.time()
    
    # get opflashes
    beamflash   = io.get_data( larlite.data.kOpFlash, "simpleFlashBeam" )
    cosmicflash = io.get_data( larlite.data.kOpFlash, "simpleFlashCosmic" )
    
    dtload = time.time()-dtload
    print "time to load inputs: ",dtload," secs"

    dtprocess = time.time()
    crtmatch.process( iolcv, io )
    dtprocess = time.time()-dtprocess
    print "timem to process: ",dtprocess," secs"

    crtmatch.save_to_file( outio )
    outio.set_id( io.run_id(), io.subrun_id(), io.event_id() )
    outio.next_event()
    #break


io.close()
iolcv.finalize()
outio.close()
