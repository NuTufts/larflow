import os,sys,argparse

parser = argparse.ArgumentParser("Test MCPixelPGraph")
parser.add_argument("-ill", "--input-larlite",required=True,type=str,help="Input larlite file")
parser.add_argument("-ilcv","--input-larcv",required=True,type=str,help="Input LArCV file")
parser.add_argument("-adc", "--adc",type=str,default="wire",help="Name of tree with Wire ADC values [default: wire]")
parser.add_argument("-tb",  "--tick-backward",action='store_true',default=False,help="Input LArCV data is tick-backward [default: false]")
args = parser.parse_args()

import ROOT as rt
from larcv import larcv
from larlite import larlite
from ublarcvapp import ublarcvapp

"""
test script that demos the MCPixelPGraph class.
"""

rt.gStyle.SetOptStat(0)

ioll = larlite.storage_manager( larlite.storage_manager.kREAD )
ioll.add_in_filename(  args.input_larlite )
ioll.open()

if args.tick_backward:
    iolcv = larcv.IOManager( larcv.IOManager.kREAD, "larcv", larcv.IOManager.kTickBackward )
else:
    iolcv = larcv.IOManager( larcv.IOManager.kREAD, "larcv", larcv.IOManager.kTickForward )
iolcv.add_in_file( args.input_larcv )
iolcv.reverse_all_products()
iolcv.initialize()

nentries = iolcv.get_n_entries()
print "Number of entries: ",nentries
nentries = 5

print "Start loop."

mcpg = ublarcvapp.mctools.MCPixelPGraph()
mcpg.set_adc_treename( args.adc )

wiretool = ublarcvapp.UBWireTool

tmp = rt.TFile("temp.root","recreate")

c = rt.TCanvas("c","c",1200,1800)
c.Divide(1,3)

for ientry in xrange( nentries ):

    print 
    print "=========================="
    print "===[ EVENT ",ientry," ]==="
    ioll.go_to(ientry)
    iolcv.read_entry(ientry)

    ev_adc = iolcv.get_data( larcv.kProductImage2D, args.adc )
    print "number of images: ",ev_adc.Image2DArray().size()
    adc_v = ev_adc.Image2DArray()
    for p in xrange(adc_v.size()):
        print " image[",p,"] ",adc_v[p].meta().dump()

    imgmeta = adc_v[2].meta()
        
    # make histogram
    hist_v = larcv.rootutils.as_th2d_v( adc_v, "hentry%d"%(ientry) )
    for ih in xrange(adc_v.size()):
        h = hist_v[ih]
        h.GetZaxis().SetRangeUser(0,100)

    ev_track = ioll.get_data(larlite.data.kMCTrack,"mcreco")
    ev_shower = ioll.get_data(larlite.data.kMCShower,"mcreco")
    
    mcpg.buildgraph( iolcv, ioll )
    #mcpg.printAllNodeInfo()
    #mcpg.printGraph()

    primaries = mcpg.getPrimaryParticles()    

    # get primary electron, make tgraph of pixels
    graph_v = []
    gr_v2 = []
    for i in xrange(primaries.size()):
        node = primaries.at(i)
        #print "primary pid[",node.pid,"]"
        if node.pid in [11,2212,13,-13]:
            if(node.type==0):
                trk = ev_track.at(node.vidx)
            else:
                trk = ev_shower.at(node.vidx)
            #print "making tgraph for pid=",node.pid
            e_v = []
            e_v2 = []
            for p in xrange(3):
                if node.pix_vv[p].size()==0:
                    e_v.append(None)
                    e_v2.append(None)
                    continue
                g = rt.TGraph( node.pix_vv[p].size()/2 )
                for j in xrange( node.pix_vv[p].size()/2 ):
                    g.SetPoint(j, node.pix_vv[p][2*j+1], node.pix_vv[p][2*j] ) # wire, tick
                g.SetMarkerStyle(20)
                g.SetMarkerSize(0.5)                
                if node.pid==11:
                    if node.origin==1:
                        g.SetMarkerColor(rt.kRed)
                elif node.pid in [13,-13]:
                    if node.origin==2:
                        g.SetMarkerColor(rt.kGreen)
                    elif node.origin==1:
                        g.SetMarkerColor(rt.kMagenta)
                elif node.pid in [2212]:
                    if node.origin==1:                    
                        g.SetMarkerColor(rt.kBlue)
                e_v.append(g)

                #test graph axis
                g2 = rt.TGraph(2)
                g2.SetMarkerStyle(24)
                pix = wiretool.getProjectedImagePixel(trk.Start().X(), trk.Start().Y(), trk.Start().Z(), imgmeta, 3)
                #print "start: ", pix
                g2.SetPoint(0,pix[p+1],pix[0]*6+2400)
                pix = wiretool.getProjectedImagePixel(trk.End().X(), trk.End().Y(), trk.End().Z(), imgmeta, 3)
                #print "end: ", pix
                g2.SetPoint(1,pix[p+1],pix[0]*6+2400)
                e_v2.append(g2)

                
            graph_v.append(e_v)
            gr_v2.append(e_v2)

    #draw canvas
    for p in xrange(3):
        c.cd(p+1)
        hist_v[p].Draw("colz")
        for e_v in graph_v:
            if e_v[p] is not None:
                e_v[p].Draw("P")

        for e_v2 in gr_v2:
            if e_v2[p] is not None:
                e_v2[p].Draw("pl")
            
    c.Update()

    print "[enter to continue]"
    raw_input()    


print "=== FIN =="
