import os,sys,time,argparse

import ROOT as rt
from ROOT import std
from larlite import larlite
from ublarcvapp import ublarcvapp
from larcv import larcv
from larflow import larflow

print larflow.reco.cluster_t
print larflow.reco.cluster_larflow3dhits

from visutil_showerlikelihoodbuilder import make_particle_node_tgraph

parser = argparse.ArgumentParser("Make true hits shower clusters")
parser.add_argument("-lcv","--input-larcv",type=str,required=True,help="larcv input")
parser.add_argument("-ll","--input-larlite",type=str,required=True,help="larlite input")
parser.add_argument("-o","--output",type=str,required=True,help="output stem")
parser.add_argument("-vis","--vis-2d",action='store_true',default=False,help="make 2d visualizations")
parser.add_argument("-s","--start",type=int,default=0,help="starting entry")
parser.add_argument("-n","--nentries",type=int,default=None,help="number of entries")
parser.add_argument("--invisible",action='store_false',default=True,help="if provided, graph dump shows invisible nodes")
parser.add_argument("--draw-node",type=int,default=None,help="if provided, will draw specific node")

args = parser.parse_args()

plot_2d_clusters = args.vis_2d


output_stem = args.output
if output_stem[:-5]==".root":
    output_stem = output_stem[:-5]

iolcv = larcv.IOManager( larcv.IOManager.kBOTH, "larcv", larcv.IOManager.kTickBackward )
#iolcv.add_in_file( "../../../../testdata/mcc9_v13_nueintrinsic_overlay_run1/supera-Run004999-SubRun000006.root" )
iolcv.add_in_file( args.input_larcv )
iolcv.set_out_file(output_stem+"_larcv.root")
iolcv.reverse_all_products()
iolcv.addto_storeonly_list( larcv.kProductImage2D, "trueshoweradc" )
iolcv.addto_storeonly_list( larcv.kProductImage2D, "segment" )
iolcv.initialize()

io = larlite.storage_manager( larlite.storage_manager.kBOTH )
#io.add_in_filename( "../../../../testdata/mcc9_v13_nueintrinsic_overlay_run1/reco2d-Run004999-SubRun000006.root" )
io.add_in_filename( args.input_larlite )
io.set_out_filename( output_stem+"_larlite.root")
io.set_data_to_read( larlite.data.kMCShower, "mcreco" )
io.set_data_to_read( larlite.data.kMCTrack, "mcreco" )
io.set_data_to_read( larlite.data.kMCTruth, "generator" )
io.set_data_to_read( larlite.data.kLArFlow3DHit, "larmatch" )
io.set_data_to_write( larlite.data.kLArFlow3DHit, "trueshowerhits" )
io.set_data_to_write( larlite.data.kMCShower, "truthshower" )
io.set_data_to_write( larlite.data.kPCAxis, "truthshower" )
io.set_data_to_write( larlite.data.kLArFlowCluster, "trueshowerclusters" )
io.open()

outio = larlite.storage_manager( larlite.storage_manager.kWRITE )

outana_name = output_stem+"_ana.root"
outana = rt.TFile(outana_name,"recreate")
builder = larflow.reco.ShowerLikelihoodBuilder()
mcpg = ublarcvapp.mctools.MCPixelPGraph()
mcpg.set_adc_treename("wiremc")
    
file_nentries = iolcv.get_n_entries()
print "Number of entries in file: ",file_nentries
start = args.start
if args.nentries is not None:
    nentries = args.nentries
else:
    nentries = file_nentries

print "Start loop."
#raw_input()

if plot_2d_clusters or args.draw_node is not None:
    c = rt.TCanvas("c","c",1200,1800)
    c.Divide(1,3)

io.go_to(start)
for ientry in xrange( start, start+nentries ):

    if ientry>=file_nentries:
        break

    iolcv.read_entry(ientry)

    ev_adc = iolcv.get_data( larcv.kProductImage2D, "wiremc" )
    adc_v = ev_adc.Image2DArray()

    print "build PIXEL GRAPH"
    mcpg.buildgraph( iolcv, io )
    mcpg.printAllNodeInfo() # check if sorted
    mcpg.printGraph(0,args.invisible)

    print "Run Shower Likelihood builder"
    print "[enter] continue"
    #raw_input()
    
    builder.process( iolcv, io )

    print "Update Pixel Graph"
    builder.updateMCPixelGraph( mcpg, iolcv )    

    if plot_2d_clusters:

        # make histogram
        hist_v = larcv.rootutils.as_th2d_v( adc_v, "hentry%d"%(ientry) )
        for ih in xrange(adc_v.size()):
            h = hist_v[ih]
            h.GetZaxis().SetRangeUser(0,100)

        primaries = mcpg.getNeutrinoParticles()    

        # get primary electron, make tgraph of pixels
        for i in xrange(primaries.size()):
            node = primaries.at(i)
            g_v = make_particle_node_tgraph(node,adc_v)

            print "Draw node[",i,"]: pid=",node.pid," tid=",node.tid

            #draw canvas
            ndrawn = 0
            for p in xrange(3):
                c.cd(p+1)
                hist_v[p].Draw("colz")
                if g_v[p] is not None:
                    ndrawn += 1
                    g_v[p].Draw("P")
            if ndrawn>0:
                c.Update()
                raw_input()
    elif args.draw_node is not None:

        drawnode = mcpg.node_v.at(args.draw_node)
        pix_vv = mcpg.getPixelsFromParticleAndDaughters( drawnode.tid )

        g_v = make_particle_node_tgraph(drawnode,adc_v)

        print "Draw node[",args.draw_node,"]: pid=",drawnode.pid," tid=",drawnode.tid

        #draw canvas
        hist_v = larcv.rootutils.as_th2d_v( adc_v, "hentry%d"%(ientry) )        
        ndrawn = 0
        for p in xrange(3):
            c.cd(p+1)
            hist_v[p].Draw("colz")
            if g_v[p] is not None:
                ndrawn += 1
                g_v[p].Draw("P")
            else:
                print "graph in plane[",p,"] is None!"
        if ndrawn>0:
            c.Update()
        raw_input()               
        
        
    iolcv.set_id( iolcv.event_id().run(), iolcv.event_id().subrun(), iolcv.event_id().event() )
    io.set_id( iolcv.event_id().run(), iolcv.event_id().subrun(), iolcv.event_id().event() )
    io.next_event()
    iolcv.save_entry()

outana.Write()
outana.Close()
io.close()
iolcv.finalize()
