from __future__ import print_function
import os,sys,argparse

parser = argparse.ArgumentParser(description='Run Prep LArFlow Match Data')
parser.add_argument('-olcv','--out-larcv',required=True,type=str,help="Filename for LArCV output")
#parser.add_argument("-c","--config",required=True,type=str,help="Configuration file")
parser.add_argument('input_larcv',nargs='+',help="Input larcv files")

args = parser.parse_args(sys.argv[1:])

#print("inputfiles: ",args.input_larcv)
#print("type(inpufiles)=",type(args.input_larcv))

import ROOT as rt
from ROOT import std
from larcv import larcv
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
for ientry in xrange(nentries):

    io.read_entry(ientry)

    ev_adc  = io.get_data( larcv.kProductImage2D, "wire" )
    adc_v   = ev_adc.Image2DArray()

    ev_chstatus = io.get_data( larcv.kProductChStatus, "wire" )
    
    badch_v = badchmaker.makeGapChannelImage( adc_v, ev_chstatus,
                                              4, 3, 2400, 1008*6, 3456, 6, 1,
                                              1.0, 100, -1.0 );
    print("made badch_v, size=",badch_v.size())

    tripmaker = larflow.PrepMatchTriplets()
    tripmaker.process( adc_v, badch_v, 10.0 )

    th2d_sparse_v = tripmaker.plot_sparse_images( adc_v, "sparse" )
    csparse = rt.TCanvas("sparse","sparse",1400,1000)
    csparse.Divide(3,2)
    for p in xrange(3):
        csparse.cd(p+1)
        if p in [0,1]:
            th2d_sparse_v[p].GetXaxis().SetRangeUser(0,2400)
        th2d_sparse_v[p].Draw("colz")
    
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

    
    print("[view triple points")
    raw_input()
    break

