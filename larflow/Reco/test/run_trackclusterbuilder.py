import os,sys,time

import ROOT as rt
from ROOT import std
from larlite import larlite
from larcv import larcv
from larflow import larflow

print larflow.reco.cluster_t
print larflow.reco.cluster_larflow3dhits


io = larlite.storage_manager( larlite.storage_manager.kBOTH )
io.add_in_filename( "test_trackreco2kp_paf_lowE_larlite.root" )
io.set_out_filename( "outtest_trackclusterbuilder.root" )
io.open()

iolcv = larcv.IOManager( larcv.IOManager.kREAD, "larcv", larcv.IOManager.kTickBackward )
iolcv.add_in_file( "test_trackreco2kp_paf_lowE_larcv.root" )
iolcv.reverse_all_products()
iolcv.initialize()

tracker = larflow.reco.TrackClusterBuilder()
tracker.set_verbosity(0)

cosmictracker = larflow.reco.CosmicTrackBuilder()
cosmictracker.set_verbosity(0)

nentries = iolcv.get_n_entries()
print "Number of entries: ",nentries
nentries = 10
start_entry = 0

print "Start loop."

io.next_event()
for ientry in xrange( start_entry, start_entry+nentries ):
    print "[ RUN ENTRY ",ientry," ]"
    iolcv.read_entry(ientry)

    if False:
        # for testing
        tracker.process( iolcv, io )
        tracker.buildConnections()
        startpt = std.vector("float")(3,0)
        #startpt[0] = 88.49
        #startpt[1] = 113.6
        #startpt[2] = 183.6
        startpt[0] = 272.96
        startpt[1] = 112.25
        startpt[2] = 244.75
        tracker.buildTracksFromPoint( startpt )
        
        evout_track = outio.get_data(larlite.data.kTrack,"testtrack")
        tracker.fillLarliteTrackContainer(evout_track)

    # cosmic trackers
    if True:
        cosmictracker.clear()
        cosmictracker.process( iolcv, io )

    io.set_id( io.run_id(), io.subrun_id(), io.event_id() )
    io.next_event()    

io.close()
iolcv.finalize()
