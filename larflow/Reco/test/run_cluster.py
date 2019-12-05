import os,sys

import ROOT as rt
from ROOT import std
from larlite import larlite
from larcv import larcv
from larflow import larflow

io = larlite.storage_manager( larlite.storage_manager.kREAD )
io.add_in_filename( "larmatch-testfile.root" )
io.open()

io.go_to(0)

ev_lfhits = io.get_data( larlite.data.kLArFlow3DHit, "larmatch" )
print "num of hits: ",ev_lfhits.size()

cluster_v = std.vector("larflow::reco::cluster_t")()

larflow.reco.cluster_larflow3dhits( ev_lfhits, cluster_v )
larflow.reco.cluster_runpca( cluster_v )

larflow.reco.cluster_dump2jsonfile( cluster_v, "dump.json" )
