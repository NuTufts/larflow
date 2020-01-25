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
larflow.FlowTriples

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
    
    trips = larflow.FlowTriples( 2, 0, adc_v, badch_v, 10.0 )
    break

