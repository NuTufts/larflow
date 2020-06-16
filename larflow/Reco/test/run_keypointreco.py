import os,sys,argparse

"""
Run the PCA-based clustering routine for track space-points.
Uses 3D points saved in larflow3dhit objects.
"""

parser = argparse.ArgumentParser("Run larflow3dhit clustering algorith")
# required
parser.add_argument('-i','--input-dlmerged',type=str,required=True,help="Input file containing ADC, ssnet, badch images/info")
parser.add_argument('-l','--input-larflow',type=str,required=True,help="Input file containing larlite::larflow3dhit objects")
parser.add_argument('-o','--output',type=str,required=True,help="Name of output file. Will not overwrite")
# optional
parser.add_argument('-n','--num-entries',type=int,default=None,help="Number of entries to run")
parser.add_argument('-e','--start-entry',type=int,default=0,help="Starting entry")
parser.add_argument('-tb','--tickbackwards',action='store_true',default=False,help="Input larcv images are tick-backward")

args = parser.parse_args()

import ROOT as rt
from ROOT import std
from larlite import larlite
from larcv import larcv
from larflow import larflow

print larflow.reco
print larflow.reco.KeypointReco

io = larlite.storage_manager( larlite.storage_manager.kREAD )
iolcv = larcv.IOManager( larcv.IOManager.kREAD, "larcv", larcv.IOManager.kTickBackward )

print "[INPUT: DL MERGED] ",args.input_dlmerged
print "[INPUT: LARMATCH-KPS]  ",args.input_larflow
print "[OUTPUT]    ",args.output

io.add_in_filename(  args.input_dlmerged )
io.add_in_filename(  args.input_larflow )
io.set_data_to_read( larlite.data.kLArFlow3DHit, "larmatch" )
#io.set_out_filename( args.output )
iolcv.add_in_file(   args.input_dlmerged )

io.open()
if args.tickbackwards:
    iolcv.reverse_all_products()
iolcv.initialize()

out = rt.TFile(args.output,"recreate")

kpreco = larflow.reco.KeypointReco()
kpreco.setupOwnTree()
print kpreco

lcv_nentries = iolcv.get_n_entries()
ll_nentries  = io.get_entries()
if lcv_nentries<ll_nentries:
    nentries = lcv_nentries
else:
    nentries = ll_nentries
    
if args.num_entries is not None:
    end_entry = args.start_entry + args.num_entries
    if end_entry>nentries:
        end_entry = nentries
else:
    end_entry = nentries

#io.go_to( args.start_entry )
io.next_event()
#io.go_to( args.start_entry )
for ientry in xrange( args.start_entry, end_entry ):
    print "[ENTRY ",ientry,"]"
    iolcv.read_entry(ientry)

    ev_lfhits = io.get_data( larlite.data.kLArFlow3DHit, "larmatch" )
    print "num of hits: ",ev_lfhits.size()

    kpreco.process( ev_lfhits )

    #kpreco.dump2json("dump_kpreco_event%d.json"%(ientry))
    kpreco.fillTree()
    
    io.set_id( io.run_id(), io.subrun_id(), io.event_id() )
    io.next_event()

print "Event Loop finished"
kpreco.writeTree()
del kpreco
out.Close()

io.close()
iolcv.finalize()
