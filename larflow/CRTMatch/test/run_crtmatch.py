import os,sys,time,argparse

parser = argparse.ArgumentParser("Run CRT-MATCHING algorithms")
parser.add_argument('-dl','--input-dlmerged',type=str,required=True,help="DL merged file")
parser.add_argument('-cl','--input-cluster',type=str,required=True,help="PCA cluster file")
parser.add_argument('-o',"--output",type=str,required=True,help="Output file stem")
parser.add_argument('-n','--num-entries',type=int,default=None,help="Number of entries to run")
parser.add_argument('-e','--start-entry',type=int,default=0,help="Starting entry")

args = parser.parse_args()

import ROOT as rt
from ROOT import std
from larlite import larlite
from larcv import larcv
from larflow import larflow

io = larlite.storage_manager( larlite.storage_manager.kREAD )
iolcv = larcv.IOManager( larcv.IOManager.kREAD, "larcv", larcv.IOManager.kTickBackward )

# INPUTS
io.add_in_filename( args.input_dlmerged )
io.add_in_filename( args.input_cluster )
iolcv.add_in_file(  args.input_dlmerged )

io.open()
iolcv.reverse_all_products()
iolcv.initialize()

if ".root"==args.output[-5:]:
    larlite_out = args.output.replace(".root","_larlite.root")
    larcv_out   = args.output.replace(".root","_larcv.root")
else:
    larlite_out = args.output + "_larlite.root"
    larcv_out   = args.output + "_larcv.root"

outio = larlite.storage_manager( larlite.storage_manager.kWRITE )
outio.set_out_filename( larlite_out )
outio.open()

outlcv = larcv.IOManager( larcv.IOManager.kWRITE, "larcvout" )
outlcv.set_out_file( larcv_out )
outlcv.initialize()

crtmatch = larflow.crtmatch.CRTMatch()

#NUM ENTRIES
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


print "Number of entries to run: ",nentries
print "Start loop."

for ientry in xrange( args.start_entry, end_entry ):

    print "[ENTRY ",ientry,"]"

    io.go_to(ientry)    
    iolcv.read_entry(ientry)

    dtload = time.time()
    
    crtmatch.process( iolcv, io )
    crtmatch.store_output( outlcv, outio )
    
    outio.set_id( io.run_id(), io.subrun_id(), io.event_id() )
    outio.next_event()

    outlcv.set_id( io.run_id(), io.subrun_id(), io.event_id() )
    outlcv.save_entry()
    #break


io.close()
iolcv.finalize()
outio.close()
outlcv.finalize()
