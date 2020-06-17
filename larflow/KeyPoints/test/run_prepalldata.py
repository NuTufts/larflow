import os,sys,argparse,time

parser = argparse.ArgumentParser("Test PrepKeypointData")
parser.add_argument("-ill", "--input-larlite",required=True,type=str,help="Input larlite file [required]")
parser.add_argument("-ilcv","--input-larcv",required=True,type=str,help="Input LArCV file [required]")
parser.add_argument("-o","--output",required=True,type=str,help="output file name [required]")
parser.add_argument("-adc", "--adc",type=str,default="wire",help="Name of tree with Wire ADC values [default: wire]")
parser.add_argument("-tb",  "--tick-backward",action='store_true',default=False,help="Input LArCV data is tick-backward [default: false]")
parser.add_argument("-tri", "--save-triplets",action='store_true',default=False,help="Save triplet data [default: false]")
parser.add_argument("-n",   "--nentries",type=int,default=-1,help="Number of entries to run [default: -1 (all)]")
args = parser.parse_args()

import ROOT as rt
from ROOT import std
from larcv import larcv
from larlite import larlite
from larflow import larflow
from ublarcvapp import ublarcvapp

"""
test script for the PrepKeypointData class
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
if args.nentries>=0 and args.nentries<nentries:
    nentries = args.nentries

print "Start loop."
tmp = rt.TFile(args.output,"recreate")

# ALGOS
# -----------------------

# bad channel/gap channel maker
badchmaker = ublarcvapp.EmptyChannelAlgo()

# triplet proposal maker
ev_triplet = std.vector("larflow::PrepMatchTriplets")(1)

# keypoint score data
kpana = larflow.keypoints.PrepKeypointData()
kpana.setADCimageTreeName( args.adc )
tmp.cd()
kpana.defineAnaTree()

# ssnet label data
ssnet = larflow.prepflowmatchdata.PrepSSNetTriplet()
tmp.cd()
ssnet.defineAnaTree()

# affinity field data
kpflow = larflow.keypoints.PrepAffinityField()
tmp.cd()
kpflow.defineAnaTree()


if args.save_triplets:
    triptree = rt.TTree("larmatchtriplet","LArMatch triplets")
    triptree.Branch("triplet_v",ev_triplet)

start = time.time()

nrun = 0
for ientry in xrange( nentries ):

    print 
    print "=========================="
    print "===[ EVENT ",ientry," ]==="
    ioll.go_to(ientry)
    iolcv.read_entry(ientry)

    tripmaker = ev_triplet[0]
    
    ev_adc = iolcv.get_data( larcv.kProductImage2D, "wiremc" )
    print "number of images: ",ev_adc.Image2DArray().size()
    adc_v = ev_adc.Image2DArray()
    for p in xrange(adc_v.size()):
        print " image[",p,"] ",adc_v[p].meta().dump()

    ev_chstatus = iolcv.get_data( larcv.kProductChStatus, "wiremc" )
    ev_larflow = iolcv.get_data( larcv.kProductImage2D, "larflow" )
    larflow_v  = ev_larflow.Image2DArray()
    
    badch_v = badchmaker.makeGapChannelImage( adc_v, ev_chstatus,
                                              4, 3, 2400, 1008*6, 3456, 6, 1,
                                              1.0, 100, -1.0 );
    print("made badch_v, size=",badch_v.size())

    # make triplet proposals
    tripmaker.process( adc_v, badch_v, 10.0, True )

    # make good/bad triplet ground truth
    tripmaker.make_truth_vector( larflow_v )

    # make keypoint score ground truth
    kpana.process( iolcv, ioll )
    kpana.make_proposal_labels( tripmaker )
    kpana.fillAnaTree()

    # make ssnet ground truth
    ssnet.make_ssnet_labels( iolcv, ioll, tripmaker )
    # fill happens automatically (ugh so ugly)

    # make affinity field ground truth
    kpflow.process( iolcv, ioll, tripmaker )
    kpflow.fillAnaTree()    
    
    if args.save_triplets:
        triptree.Fill()
    nrun += 1    
    
    #sys.exit(0)
    #break

print "NCLOSE: ",kpana._nclose
print "NFAR: ",kpana._nfar
print "FRAC CLOSE: ",float(kpana._nclose)/float(kpana._nclose+kpana._nfar)

dtime = time.time()-start
print "Time: ",float(dtime)/float(nrun)," sec/event"

tmp.cd()
kpana.writeAnaTree()
kpana.writeHists()
ssnet.writeAnaTree()
kpflow.writeAnaTree()
if args.save_triplets:
    triptree.Write()

del kpana
del ssnet
del kpflow

print "=== FIN =="
