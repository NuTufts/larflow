import os,sys

import ROOT as rt
from ROOT import std
from larlite import larlite
from larcv import larcv
from larflow import larflow

print larflow.reco.cluster_t
print larflow.reco.cluster_larflow3dhits

ientry = 0
max_entries = None

io = larlite.storage_manager( larlite.storage_manager.kBOTH )
#io.add_in_filename( "larmatch-testfile.root" )
#io.add_in_filename( "merged_dlreco_larmatch_run3_bnboverlay.root" )
#io.add_in_filename( "larmatch_wctagger_example.root" )
io.add_in_filename(  "larmatch_eLEE_sample2.root" )
io.set_out_filename( "larflow_cluster_eLEE_sample2.root" )
#io.add_in_filename( "larmatch_extbnb_run3_821c2dfc-96b0-4725-a187-6b628cbbbea3.root" )
#io.set_out_filename( "larflow_reco_extbnb_run3.root" )
io.open()

iolcv = larcv.IOManager( larcv.IOManager.kREAD, "larcv", larcv.IOManager.kTickBackward )
iolcv.add_in_file( "merged_dlreco_eLEE_sample2.root" )
#iolcv.add_in_file( "merged_dlreco_extbnb_run3_821c2dfc-96b0-4725-a187-6b628cbbbea3.root" )
iolcv.reverse_all_products()
iolcv.initialize()

pcacluster = larflow.reco.PCACluster()

nentries = iolcv.get_n_entries()
if max_entries is not None and nentries>max_entries:
    nentries = max_entries
    
io.next_event()
for ientry in xrange( nentries ):
    
    iolcv.read_entry(ientry)

    ev_lfhits = io.get_data( larlite.data.kLArFlow3DHit, "larmatch" )
    print "num of hits: ",ev_lfhits.size()

    pcacluster.process( iolcv, io )

    io.set_id( io.run_id(), io.subrun_id(), io.event_id() )
    io.next_event()    

io.close()
iolcv.finalize()
