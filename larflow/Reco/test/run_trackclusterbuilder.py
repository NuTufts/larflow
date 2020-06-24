import os,sys,time

import ROOT as rt
from ROOT import std
from larlite import larlite
from larcv import larcv
from larflow import larflow

print larflow.reco.cluster_t
print larflow.reco.cluster_larflow3dhits


io = larlite.storage_manager( larlite.storage_manager.kREAD )
io.add_in_filename( "test_trackreco2kp_paf_lowE_larlite.root" )
io.open()

outio = larlite.storage_manager( larlite.storage_manager.kWRITE )
outio.set_out_filename( "outest_trackclusterbuilder.root" )
outio.open()

iolcv = larcv.IOManager( larcv.IOManager.kREAD, "larcv", larcv.IOManager.kTickBackward )
iolcv.add_in_file( "test_trackreco2kp_paf_lowE_larcv.root" )
iolcv.reverse_all_products()
iolcv.initialize()

tracker = larflow.reco.TrackClusterBuilder()
tracker.set_verbosity(0)

nentries = iolcv.get_n_entries()
print "Number of entries: ",nentries
nentries = 1
start_entry = 0

print "Start loop."

outio.next_event()
for ientry in xrange( start_entry, start_entry+nentries ):
    print "[ RUN ENTRY ",ientry," ]"
    io.go_to(ientry)    
    iolcv.read_entry(ientry)

    tracker.process( iolcv, io )

    outio.set_id( io.run_id(), io.subrun_id(), io.event_id() )
    outio.next_event()    
    break

io.close()
outio.close()
iolcv.finalize()
