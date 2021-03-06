from __future__ import print_function
import os,sys,argparse,time
from ctypes import c_int

parser = argparse.ArgumentParser(description='Run Prep LArFlow Match Data')
parser.add_argument('-olcv','--out-larcv',required=True,type=str,help="Filename for LArCV output")
parser.add_argument("-mc","--has-mc",default=False,action="store_true",help="Has MC information")
parser.add_argument('input_larcv',nargs='+',help="Input larcv files")


args = parser.parse_args(sys.argv[1:])

import ROOT as rt
from ROOT import std
from larcv import larcv
larcv.load_pyutil()
from ublarcvapp import ublarcvapp
from larflow import larflow

rt.gStyle.SetOptStat(0)

badchmaker = ublarcvapp.EmptyChannelAlgo()

io = larcv.IOManager( larcv.IOManager.kREAD, "io", larcv.IOManager.kTickBackward )
for f in args.input_larcv:
    io.add_in_file( f )
io.reverse_all_products()
io.initialize()

nentries = io.get_n_entries()
nentries = 1

for ientry in xrange(nentries):

    io.read_entry(ientry)

    ev_adc  = io.get_data( larcv.kProductImage2D, "wire" )
    adc_v   = ev_adc.Image2DArray()

    ev_chstatus = io.get_data( larcv.kProductChStatus, "wire" )

    if args.has_mc:
        ev_larflow = io.get_data( larcv.kProductImage2D, "larflow" )
        larflow_v  = ev_larflow.Image2DArray()
    
    badch_v = badchmaker.makeGapChannelImage( adc_v, ev_chstatus,
                                              4, 3, 2400, 1008*6, 3456, 6, 1,
                                              1.0, 100, -1.0 );
    print("made badch_v, size=",badch_v.size())

    tripmaker = larflow.PrepMatchTriplets()
    tripmaker.process( adc_v, badch_v, 10.0 )
    if args.has_mc:
        tripmaker.make_truth_vector( larflow_v )

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

