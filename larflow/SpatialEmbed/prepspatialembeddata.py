# IMPORTANT:
#       Depending on the data file used, the "iolcv.get_data( larcv.kProductImage2D, "wire" )" will either be "wire" or "wiremc".
#       To avoid the segfault that happens, this script requires going in to change "wire" to "wiremc" or vice versa in the following places:
#           This file, when instantiating ev_adc; This file, when setting mcpg_adc_treename; PrepMatchEmbed.cxx process(); ShowerLikelihoodBuilder.cxx process(); 
#



import os,sys,argparse,time

parser = argparse.ArgumentParser("Test PrepKeypointData")
parser.add_argument("-ill", "--input-larlite",required=True,type=str,help="Input larlite file [required]")
parser.add_argument("-ilcv","--input-larcv",required=True,type=str,help="Input LArCV file [required]")
parser.add_argument("-o","--output",required=True,type=str,help="output file name [required]")
parser.add_argument("-adc", "--adc",type=str,default="wire",help="Name of tree with Wire ADC values [default: wire]")
parser.add_argument("-tb",  "--tick-backward",action='store_true',default=False,help="Input LArCV data is tick-backward [default: false]")
parser.add_argument("-vis", "--visualize", action='store_true',default=False,help="Visualize Keypoints in TCanvas [default: false]")
parser.add_argument("-bvh", "--use-bvh", action='store_true',default=False,help="Use BVH [default: false]")
parser.add_argument("-tri", "--save-triplets",action='store_true',default=False,help="Save triplet data [default: false]")
parser.add_argument("-n",   "--nentries",type=int,default=-1,help="Number of entries to run [default: -1 (all)]")
args = parser.parse_args()

import ROOT as rt
from ROOT import std
from larcv import larcv
from larlite import larlite
from larflow import larflow
from ublarcvapp import ublarcvapp

print larflow.reco.cluster_t

larcv.SetPyUtil()

rt.gStyle.SetOptStat(0)

ioll = larlite.storage_manager( larlite.storage_manager.kBOTH )
ioll.add_in_filename(  args.input_larlite )
ioll.set_out_filename( "delete.root")
ioll.set_data_to_read( larlite.data.kMCTrack,  "mcreco" );
ioll.set_data_to_read( larlite.data.kMCShower, "mcreco" );
ioll.set_data_to_read( larlite.data.kMCTruth,  "generator" );


# if args.tick_backward:
#     iolcv = larcv.IOManager( larcv.IOManager.kBOTH, "larcv", larcv.IOManager.kTickBackward )
# else:
#     iolcv = larcv.IOManager( larcv.IOManager.kBOTH, "larcv", larcv.IOManager.kTickForward )

iolcv = larcv.IOManager( larcv.IOManager.kBOTH, "larcv", larcv.IOManager.kTickBackward )

iolcv.add_in_file( args.input_larcv )
iolcv.set_out_file( "delete2.root" )
iolcv.reverse_all_products()
iolcv.specify_data_read( larcv.kProductImage2D, "wiremc" )
iolcv.specify_data_read( larcv.kProductImage2D, "segment" );
iolcv.specify_data_read( larcv.kProductImage2D, "instance" );
iolcv.specify_data_read( larcv.kProductImage2D, "ancestor" );
iolcv.specify_data_read( larcv.kProductImage2D, "larflow" );
iolcv.specify_data_read( larcv.kProductChStatus,"wire");

ioll.open()
iolcv.initialize()

nentries = iolcv.get_n_entries()
print "Number of entries: ",nentries
if args.nentries>=0 and args.nentries<nentries:
    nentries = args.nentries


root_file = rt.TFile(args.output, "RECREATE")
output_tree = rt.TTree("trainingdata", "Spatial Embed Training Data")

data = larflow.spatialembed.SpatialEmbedData()
output_tree.Branch('DataBranch', data)
# output_tree.Branch('DataBranch', 'larflow::spatialembed::SpatialEmbedData', data)

builder = larflow.reco.ShowerLikelihoodBuilder()

for ientry in xrange( nentries ):

    print 
    print "=========================="
    print "===[ EVENT ",ientry," ]==="
    ioll.go_to(ientry)
    iolcv.read_entry(ientry)

    # Process Image data    

    # ev_adc = iolcv.get_data( larcv.kProductImage2D, "wiremc" )
    ev_adc = iolcv.get_data( larcv.kProductImage2D, "wire" )

    # Process Instance Data
    mcpg = ublarcvapp.mctools.MCPixelPGraph()
    # mcpg.set_adc_treename( "wiremc" )
    mcpg.set_adc_treename( "wire" )
    mcpg.buildgraph( iolcv, ioll )

    builder.process( iolcv, ioll )
    builder.updateMCPixelGraph( mcpg, iolcv )

    data.processImageData(ev_adc, 10)

    preptriplet = larflow.prep.PrepMatchTriplets()
    prepembed = larflow.spatialembed.PrepMatchEmbed()
    prepembed.process( iolcv, ioll, preptriplet )

    
    img_meta = ev_adc.Image2DArray().at(0).meta();

    data.processLabelDataWithShower( mcpg, iolcv, img_meta)
    # data.processLabelDataWithShower( mcpg, prepembed, ev_adc)
    # data.processLabelData( mcpg, prepembed )

    for plan in xrange(3):
        print data.num_instances_plane(plan)
    
    output_tree.Fill()

root_file.cd()
output_tree.Write()

print "=== FIN =="
