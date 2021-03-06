from __future__ import print_function
import os,sys,argparse,time
from ctypes import c_int

parser = argparse.ArgumentParser(description='Run Prep LArFlow Match Data')
parser.add_argument('-o','--output',required=True,type=str,help="Filename stem for output files")
parser.add_argument("-mc","--has-mc",default=False,action="store_true",help="Has MC information")
parser.add_argument("-a","--adc",default="wire",type=str,required=False,help="Name of wire and chstatus producers")
parser.add_argument('input_larcv',nargs='+',help="Input larcv files")


args = parser.parse_args(sys.argv[1:])

import ROOT as rt
from ROOT import std
from larcv import larcv
larcv.load_pyutil()
from ublarcvapp import ublarcvapp
from larflow import larflow

rt.gStyle.SetOptStat(0)

io = larcv.IOManager( larcv.IOManager.kREAD, "io", larcv.IOManager.kTickBackward )
for f in args.input_larcv:
    io.add_in_file( f )
io.specify_data_read( larcv.kProductImage2D,  args.adc )
io.specify_data_read( larcv.kProductImage2D,  "larflow" )
io.specify_data_read( larcv.kProductChStatus, args.adc )
io.reverse_all_products()
io.initialize()

nentries = io.get_n_entries()
nentries = 1
print("Number of input entries to run: ",nentries)

outfile = rt.TFile(args.output,"recreate")
outtree = rt.TTree("voxelizer","Voxelization Data")
data_v = std.vector("larflow::voxelizer::VoxelizeTriplets")(1)
outtree.Branch("data_v",data_v)

for ientry in xrange(nentries):

    io.read_entry(ientry)

    data_v[0].process_fullchain( io, args.adc, args.adc, args.has_mc )

    outtree.Fill()

outfile.Write()
print("Done")

