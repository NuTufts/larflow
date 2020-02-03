import os,sys

import ROOT as rt
from ROOT import std
from larlite import larlite
from larcv import larcv
from larflow import larflow

print larflow.reco
larflow.reco.ClusterFunctions()
print larflow.reco.ClusterFunctions

ientry = 0
max_entries = None
#max_entries = 1

io = larlite.storage_manager( larlite.storage_manager.kBOTH )
iolcv = larcv.IOManager( larcv.IOManager.kREAD, "larcv", larcv.IOManager.kTickBackward )

#io.add_in_filename( "larmatch-testfile.root" )
#io.add_in_filename( "merged_dlreco_larmatch_run3_bnboverlay.root" )
#io.add_in_filename( "larmatch_wctagger_example.root" )

# eLEE sample
#io.add_in_filename(  "larmatch_eLEE_sample2.root" )
#io.set_out_filename( "larflow_cluster_eLEE_sample2.root" )
#iolcv.add_in_file( "merged_dlreco_eLEE_sample2.root" )

# EXTBNB RUN 3
io.add_in_filename(  "larmatch_extbnb_run3_821c2dfc-96b0-4725-a187-6b628cbbbea3_larlite.root" )
io.set_out_filename( "larflow_reco_extbnb_run3.root" )
#io.set_out_filename( "larflow_reco_extbnb_run3.root" )
iolcv.add_in_file(   "merged_dlreco_extbnb_run3_821c2dfc-96b0-4725-a187-6b628cbbbea3.root" )

io.open()
iolcv.reverse_all_products()
iolcv.initialize()

pcacluster = larflow.reco.PCACluster()
pcacluster.set_min_larmatch_score( 0.8 )
pcacluster.set_downsample_fraction( 0.15 )
pcacluster.set_dbscan_pars( 10.0, 5, 10 )

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
