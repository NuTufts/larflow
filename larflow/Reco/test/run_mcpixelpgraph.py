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

tmp = rt.TFile("temp.root","recreate")

c = rt.TCanvas("c","c",1200,1800)
c.Divide(1,3)

for ientry in xrange( nentries ):

    print 
    print "=========================="
    print "===[ EVENT ",ientry," ]==="
    ioll.go_to(ientry)
    iolcv.read_entry(ientry)

    ev_adc = iolcv.get_data( larcv.kProductImage2D, "wire" )
    print "number of images: ",ev_adc.Image2DArray().size()
    adc_v = ev_adc.Image2DArray()
    for p in xrange(adc_v.size()):
        print " image[",p,"] ",adc_v[p].meta().dump()
    
    # make histogram
    hist_v = []
    for ih in xrange(adc_v.size()):
        h = larcv.rootutils.as_th2d( adc_v[ih], "hentry%d_plane%d"%(ientry,ih) )
        h.GetZaxis().SetRangeUser(0,100)
        hist_v.append(h)

    mcpg.buildgraph( iolcv, ioll )
    #mcpg.printAllNodeInfo()
    mcpg.printGraph()

    primaries = mcpg.getPrimaryParticles()    

    # get primary electron, make tgraph of pixels
    graph_v = []
    for i in xrange(primaries.size()):
        node = primaries.at(i)
        if node.pid==11:
            e_v = []
            for p in xrange(3):
                g = rt.TGraph( node.pix_vv[p].size()/2 )
                for j in xrange( node.pix_vv[p].size()/2 ):
                    g.SetPoint(j, node.pix_vv[p][2*j+1], node.pix_vv[p][2*j] ) # wire, tick
                g.SetMarkerStyle(20)
                g.SetMarkerColor(rt.kRed)
                e_v.append(g)
            graph_v.append(e_v)

    #draw canvas
    for p in xrange(3):
        c.cd(p+1)
        hist_v[p].Draw("colz")
        for e_v in graph_v:
            e_v[p].Draw("P")
    c.Update()

    print "[enter to continue]"
    raw_input()
    
    
    #node = mcpg.findTrackID(1)
    #if not node:
    #    print "not found"
    #else:
    #    print "found node with trackid=",node.tid
    #    mcpg.printNodeInfo(node)


print "=== FIN =="
