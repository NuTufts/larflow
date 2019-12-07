import os,sys

import ROOT as rt
from ROOT import std
from larlite import larlite
from larcv import larcv
from larflow import larflow

print larflow.reco.cluster_t
print larflow.reco.cluster_larflow3dhits

io = larlite.storage_manager( larlite.storage_manager.kREAD )
#io.add_in_filename( "larmatch-testfile.root" )
#io.add_in_filename( "merged_dlreco_larmatch_run3_bnboverlay.root" )
io.add_in_filename( "larmatch_wctagger_example.root" )
io.open()
io.go_to(0)

iolcv = larcv.IOManager( larcv.IOManager.kREAD, "larcv", larcv.IOManager.kTickBackward )
iolcv.add_in_file( "merged_dlreco_wctagger.root" )
iolcv.reverse_all_products()
iolcv.initialize()
iolcv.read_entry(0)


ev_lfhits = io.get_data( larlite.data.kLArFlow3DHit, "larmatch" )
print "num of hits: ",ev_lfhits.size()

pcacluster = larflow.reco.PCACluster()

pcacluster.process( iolcv, io )
