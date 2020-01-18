import os,sys,time

import ROOT as rt
from ROOT import std
from larlite import larlite
from larcv import larcv
from larflow import larflow

print larflow.reco.cluster_t
print larflow.reco.cluster_larflow3dhits


io = larlite.storage_manager( larlite.storage_manager.kREAD )
io.add_in_filename( "merged_dlreco_eLEE_sample2.root" )
io.add_in_filename( "larmatch_eLEE_sample2.root" )
io.add_in_filename( "larflow_cluster_eLEE_sample2_full.root" ) 
io.open()

outio = larlite.storage_manager( larlite.storage_manager.kWRITE )
outio.set_out_filename( "larflow_vertexer_eLEE_sample2.root" )
outio.open()

iolcv = larcv.IOManager( larcv.IOManager.kREAD, "larcv", larcv.IOManager.kTickBackward )
iolcv.add_in_file( "merged_dlreco_eLEE_sample2.root" )
iolcv.reverse_all_products()
iolcv.initialize()

vtx_reco = larflow.reco.VertexReco()

nentries = iolcv.get_n_entries()
print "Number of entries: ",nentries
nentries = 1

print "Start loop."

outio.next_event()
for ientry in xrange( nentries ):

    io.go_to(ientry)    
    iolcv.read_entry(ientry)

    candidate_v = vtx_reco.findVertices( iolcv, io )

    outio.set_id( io.run_id(), io.subrun_id(), io.event_id() )
    outio.next_event()    
    break

io.close()
outio.close()
iolcv.finalize()
