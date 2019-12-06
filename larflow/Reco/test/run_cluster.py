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

ssnet_track_v = std.vector("larcv::Image2D")()

for p in range(3):
    ssnet_track_v.push_back( iolcv.get_data( larcv.kProductImage2D, "ubspurn_plane%d"%(p) ).at(1) )

track_hit_v  = std.vector("larlite::larflow3dhit")()
shower_hit_v = std.vector("larlite::larflow3dhit")()

larflow.reco.cluster_splitbytrackshower( ev_lfhits, ssnet_track_v, track_hit_v, shower_hit_v )


cluster_track_v   = std.vector("larflow::reco::cluster_t")()
cluster_shower_v = std.vector("larflow::reco::cluster_t")()

# Track Clusters
larflow.reco.cluster_larflow3dhits( track_hit_v, cluster_track_v )
larflow.reco.cluster_runpca( cluster_track_v )

projimg_v = std.vector("larcv::Image2D")()
ev_adc_v = iolcv.get_data( larcv.kProductImage2D, "wire" ).Image2DArray()
for p in range(3):
    projimg = larcv.Image2D( ev_adc_v[p].meta() )
    projimg.paint(0.0)
    projimg_v.push_back( projimg )

for icluster in xrange(cluster_track_v.size()):
    clust = cluster_track_v[icluster]
    # pca axis size:
    print "track cluster[",icluster,"] pca axis: [0]=",clust.pca_eigenvalues[0]," [1]=",clust.pca_eigenvalues[1]," [2]=",clust.pca_eigenvalues[2]
    if clust.pca_eigenvalues[1]>10.0:
        print "Run splitter"
        larflow.reco.cluster_imageprojection( clust, projimg_v )
        larflow.reco.cluster_getcontours( projimg_v )

larflow.reco.cluster_dump2jsonfile( cluster_track_v, "dump_track.json" )

# shower clusters
larflow.reco.cluster_larflow3dhits( shower_hit_v, cluster_shower_v )
larflow.reco.cluster_runpca( cluster_shower_v )
larflow.reco.cluster_dump2jsonfile( cluster_shower_v, "dump_shower.json" )
