import os,sys,time

import ROOT as rt
from ROOT import std
from larlite import larlite
from larcv import larcv
from larflow import larflow

io = larlite.storage_manager( larlite.storage_manager.kREAD )
io.add_in_filename( "merged_dlreco_extbnb_run3_821c2dfc-96b0-4725-a187-6b628cbbbea3.root" )
io.add_in_filename( "larflow_triplet_reco_extbnb_run3.root" )
io.open()

iolcv = larcv.IOManager( larcv.IOManager.kREAD, "larcv", larcv.IOManager.kTickBackward )
iolcv.add_in_file( "merged_dlreco_extbnb_run3_821c2dfc-96b0-4725-a187-6b628cbbbea3.root" )
iolcv.reverse_all_products()
iolcv.initialize()

outio = larlite.storage_manager( larlite.storage_manager.kWRITE )
outio.set_out_filename( "crt_match_reco_extbnb_run3_triplet.root" )
outio.open()

outlcv = larcv.IOManager( larcv.IOManager.kWRITE, "larcvout" )
outlcv.set_out_file( "crt_match_reco_extbnb_run3_triplet_larcv.root" )
outlcv.initialize()


crtmatch = larflow.crtmatch.CRTMatch()

nentries = iolcv.get_n_entries()
print "Number of entries: ",nentries
nentries = 5

print "Start loop."

for ientry in xrange( 0, nentries ):

    io.go_to(ientry)    
    iolcv.read_entry(ientry)

    dtload = time.time()
    
    crtmatch.process( iolcv, io )
    crtmatch.store_output( outlcv, outio )
    
    outio.set_id( io.run_id(), io.subrun_id(), io.event_id() )
    outio.next_event()

    outlcv.set_id( io.run_id(), io.subrun_id(), io.event_id() )
    outlcv.save_entry()
    #break


io.close()
iolcv.finalize()
outio.close()
outlcv.finalize()
