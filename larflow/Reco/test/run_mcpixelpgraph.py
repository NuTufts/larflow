import os,sys
import ROOT as rt
from larcv import larcv
from larlite import larlite
from larflow import larflow

ioll = larlite.storage_manager( larlite.storage_manager.kREAD )
ioll.add_in_filename(  "merged_dlreco_eLEE_sample2.root" )
ioll.open()

iolcv = larcv.IOManager( larcv.IOManager.kREAD, "larcv", larcv.IOManager.kTickBackward )
iolcv.add_in_file( "merged_dlreco_eLEE_sample2.root" )
iolcv.reverse_all_products()
iolcv.initialize()

crtmatch = larflow.reco.MCPixelPGraph()

nentries = iolcv.get_n_entries()
print "Number of entries: ",nentries
nentries = 10

print "Start loop."

mcpg = larflow.reco.MCPixelPGraph()

for ientry in xrange( nentries ):

    print 
    print "=========================="
    print "===[ EVENT ",ientry," ]==="
    ioll.go_to(ientry)
    iolcv.read_entry(ientry)

    mcpg.buildgraph( iolcv, ioll )
    #mcpg.printAllNodeInfo()
    mcpg.printGraph()

    #node = mcpg.findTrackID(1)
    #if not node:
    #    print "not found"
    #else:
    #    print "found node with trackid=",node.tid
    #    mcpg.printNodeInfo(node)


print "=== FIN =="
