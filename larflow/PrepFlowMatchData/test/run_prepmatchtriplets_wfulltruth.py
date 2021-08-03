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
parser.add_argument("-e","--start-entry",default=0,type=int,help="Set entry to start at [default: entry 0]")
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
start_entry = args.start_entry
if args.nentries is not None and args.nentries<nentries:
    end_entry = start_entry + args.nentries-1
    if end_entry>=nentries:
        end_entry = nentries-1
else:
    end_entry = nentries-start_entry-1

out = rt.TFile(args.output,"recreate")
outtree = rt.TTree("larmatchtriplet","triplet data")
triplet_v = std.vector("larflow::prep::PrepMatchTriplets")(1)
outtree.Branch("triplet_v",triplet_v)

for ientry in xrange(start_entry, end_entry+1):

    io.read_entry(ientry)
    ioll.go_to(ientry)

    tripmaker = triplet_v.at(0)
    tripmaker.clear()
    tripmaker.process( io, args.adc_name, args.adc_name, 10.0, True )
    tripmaker.process_truth_labels( io, ioll, args.adc_name ) 

    #truthfixer = larflow.prep.TripletTruthFixer()
    #truthfixer.calc_reassignments( tripmaker, io, ioll )
    
    outtree.Fill()
    

#out.write_file()
out.Write()
io.finalize()
ioll.close()

print("End of event loop")
print("[FIN]")
