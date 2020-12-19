from __future__ import print_function
import os,sys,argparse,time
from ctypes import c_int

parser = argparse.ArgumentParser(description='Run Prep LArFlow Match Data')
parser.add_argument("-lcv","--input-larcv",required=True,type=str,help="Input larcv file")
parser.add_argument("-ll","--input-larlite",required=True,type=str,help="Input larlite file")
parser.add_argument('-o','--output',required=True,type=str,help="Filename for LArCV output")
parser.add_argument("-adc","--adc-name",default="wire",type=str,help="Name of Tree containing wire images")
parser.add_argument("-mc","--has-mc",default=False,action="store_true",help="Has MC information")
parser.add_argument("-n","--nentries",default=None,type=int,help="Set number of events to run [default: all in file]")
args = parser.parse_args(sys.argv[1:])

import ROOT as rt
from ROOT import std
from larlite import larlite
from larcv import larcv
larcv.load_pyutil()
from ublarcvapp import ublarcvapp
from larflow import larflow

rt.gStyle.SetOptStat(0)

badchmaker = ublarcvapp.EmptyChannelAlgo()

io = larcv.IOManager( larcv.IOManager.kREAD, "io", larcv.IOManager.kTickBackward )
io.add_in_file( args.input_larcv )
io.specify_data_read( larcv.kProductImage2D,  "wire" )
io.specify_data_read( larcv.kProductImage2D,  "wiremc" )
io.specify_data_read( larcv.kProductChStatus, "wire" )
io.specify_data_read( larcv.kProductChStatus, "wiremc" )
io.specify_data_read( larcv.kProductImage2D,  "ancestor" )
io.specify_data_read( larcv.kProductImage2D,  "instance" )
io.specify_data_read( larcv.kProductImage2D,  "segment" )
io.specify_data_read( larcv.kProductImage2D,  "larflow" )
io.reverse_all_products()
io.initialize()

ioll = larlite.storage_manager( larlite.storage_manager.kREAD )
ioll.add_in_filename( args.input_larlite )
ioll.set_data_to_read( larlite.data.kMCTrack,  "mcreco" )
ioll.set_data_to_read( larlite.data.kMCShower, "mcreco" )
ioll.set_data_to_read( larlite.data.kMCTruth,  "generator" )
ioll.open()

nentries = io.get_n_entries()
if args.nentries is not None and args.nentries<nentries:
    nentries = args.nentries

out = rt.TFile(args.output,"recreate")
outtree = rt.TTree("larmatchtriplet","triplet data")
triplet_v = std.vector("larflow::prep::PrepMatchTriplets")(1)
outtree.Branch("triplet_v",triplet_v)

for ientry in xrange(nentries):

    io.read_entry(ientry)
    ioll.go_to(ientry)

    tripmaker = triplet_v.at(0)
    tripmaker.clear()
    tripmaker.process( io, args.adc_name, args.adc_name, 10.0, True )

    ev_larflow = io.get_data( larcv.kProductImage2D, "larflow" )
    ev_instance = io.get_data( larcv.kProductImage2D, "instance" )
    ev_ancestor = io.get_data( larcv.kProductImage2D, "ancestor" )
    ev_segment = io.get_data( larcv.kProductImage2D, "segment" )    
    tripmaker.make_truth_vector(      ev_larflow.as_vector()  )
    tripmaker.make_instanceid_vector( ev_instance.as_vector() )
    tripmaker.make_ancestorid_vector( ev_ancestor.as_vector() )
    tripmaker.make_segmentid_vector(  ev_segment.as_vector() )

    truthfixer = larflow.prep.TripletTruthFixer()
    truthfixer.calc_reassignments( tripmaker, io, ioll )
    
    outtree.Fill()
    
    if True:
        continue

    

    ntriples = tripmaker._triplet_v.size()
    startidx = 0
    NUM_PAIRS = 20000
    while startidx<ntriples:
        # get indices
        npairs      = c_int()
        npairs.value = 0
        last_index  = c_int()
        last_index.value = 0
        with_truth  = False

        tstart = time.time()
        print("create matchpairs: startidx=",startidx," of ",ntriples)
        matchpair_np = tripmaker.get_chunk_2plane_matches( 4,
                                                           startidx,
                                                           NUM_PAIRS,
                                                           last_index,
                                                           npairs,
                                                           with_truth )
        startidx += NUM_PAIRS
        print("made pairs last_index=",last_index.value," npairs=",npairs.value)
        

    th2d_sparse_v = tripmaker.plot_sparse_images( adc_v, "sparse" )
    if args.has_mc:
        th2d_truth3_v = tripmaker.plot_truth_images( adc_v, "truth3" )
    csparse = rt.TCanvas("sparse","sparse",1400,1000)
    csparse.Divide(3,2)
    for p in xrange(3):
        csparse.cd(p+1)
        if p in [0,1]:
            th2d_sparse_v[p].GetXaxis().SetRangeUser(0,2400)
        th2d_sparse_v[p].Draw("colz")

        if args.has_mc:
            print("make triplet truth image plane={}".format(p))
            csparse.cd(4+p)
            if p in [0,1]:
                th2d_truth3_v[p].GetXaxis().SetRangeUser(0,2400)
            th2d_truth3_v[p].Draw("colz")
        
    
    # badch
    th2d_badch_v = larcv.rootutils.as_th2d_v( badch_v, "badch_entry%d"%(ientry))
    th2d_adcch_v = larcv.rootutils.as_th2d_v( adc_v,   "adcch_entry%d"%(ientry))    
    cbadch = rt.TCanvas("badch","badch",1400,1000)
    cbadch.Divide(3,2)
    for p in xrange(3):
        if p in [0,1]:
            th2d_badch_v[p].GetXaxis().SetRangeUser(0,2400)
            th2d_adcch_v[p].GetXaxis().SetRangeUser(0,2400)            
        cbadch.cd(p+1)
        th2d_badch_v[p].Draw("colz")
        cbadch.cd(p+4)
        th2d_adcch_v[p].Draw("colz")

    
    print("[view triple points] ENTER to continue.")
    raw_input()
    break

#out.write_file()
out.Write()
io.finalize()
ioll.close()

print("End of event loop")
print("[FIN]")
