from __future__ import print_function
import os,sys,argparse,time
from ctypes import c_int

parser = argparse.ArgumentParser(description='Run Prep LArFlow Match Data')
parser.add_argument("-lcv","--input-larcv",required=True,type=str,help="Input larcv file")
parser.add_argument('-o','--output',required=True,type=str,help="Filename for LArCV output")
parser.add_argument('-d','--detector',required=True,type=str,help="Choose detector. Optons: {'uboone','sbnd','icarus'}")
parser.add_argument("-adc","--adc-name",default="wire",type=str,help="Name of Tree containing wire images")
parser.add_argument("-mc","--has-mc",default=False,action="store_true",help="Has MC information")
parser.add_argument("-ll","--input-larlite",required=False,default=None,type=str,help="Input larlite file")
parser.add_argument("-n","--nentries",default=None,type=int,help="Set number of events to run [default: all in file]")
parser.add_argument("-e","--start-entry",default=0,type=int,help="Set entry to start at [default: entry 0]")
parser.add_argument("--save-mc",default=False,action='store_true',help="Save MCTrack and MCShower [default:false]")
args = parser.parse_args(sys.argv[1:])

if args.detector not in ["uboone","sbnd","icarus"]:
    raise ValueError("Invalid detector")
if args.has_mc and args.input_larlite is None:
    raise ValueError("If analyzing MC for truth (--has-mc), need to provide larlite file with truth info (--input-larlite)")

import ROOT as rt
from ROOT import std
from larlite import larlite
from larcv import larcv
from ROOT import larutil
larcv.load_pyutil()
from ublarcvapp import ublarcvapp
from larflow import larflow

rt.gStyle.SetOptStat(0)

# SET DETECTOR
if args.detector == "icarus":
    detid = larlite.geo.kICARUS
    overlap_matrix_file = os.environ["LARFLOW_BASEDIR"]+"/larflow/PrepFlowMatchData/test/output_icarus_wireoverlap_matrices.root"
elif args.detector == "uboone":
    detid = larlite.geo.kMicroBooNE
    overlap_matrix_file = os.environ["LARFLOW_BASEDIR"]+"/larflow/PrepFlowMatchData/test/output_microboone_wireoverlap_matrices.root"    
elif args.detector == "sbnd":
    detid = larlite.geo.kSBND    
larutil.LArUtilConfig.SetDetector(detid)


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

if args.has_mc:
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
out.cd()
outtree = rt.TTree("larmatchtriplet","triplet data")
triplet_v = std.vector("larflow::prep::MatchTriplets")()
outtree.Branch("triplet_v",triplet_v)

if args.save_mc:
    outio = larlite.storage_manager( larlite.storage_manager.kWRITE )
    outio.set_out_filename( args.output.replace(".root","_larlite.root") )
    outio.open()

preptripletalgo = larflow.prep.PrepMatchTriplets()
preptripletalgo.set_verbosity(0) # for debug
preptripletalgo.set_wireoverlap_filepath( overlap_matrix_file  )


for ientry in range(start_entry, end_entry+1):

    triplet_v.clear()
    
    io.read_entry(ientry)
    if args.has_mc:
        ioll.go_to(ientry)

    preptripletalgo.process( io, args.adc_name, args.adc_name, 10.0, True )
    if args.has_mc:
        print("process truth labels")
        preptripletalgo.process_truth_labels( io, ioll, args.adc_name )

    truthfixer = larflow.prep.TripletTruthFixer()
    truthfixer.calc_reassignments( preptripletalgo, io, ioll )

    for imatchdata in range(preptripletalgo._match_triplet_v.size()):
        triplet_v.push_back( preptripletalgo._match_triplet_v.at(imatchdata) )    

    if args.save_mc:
        ev_mctrack  = ioll.get_data( larlite.data.kMCTrack,  "mcreco" )
        ev_mcshower = ioll.get_data( larlite.data.kMCShower, "mcreco" )
        evout_mctrack  = outio.get_data( larlite.data.kMCTrack,  "mcreco" )
        evout_mcshower = outio.get_data( larlite.data.kMCShower, "mcreco" )
        for itrack in range( ev_mctrack.size() ):
            if ev_mctrack.at(itrack).Origin()==1:
                evout_mctrack.push_back( ev_mctrack.at(itrack) )
        for ishower in range( ev_mcshower.size() ):
            if ev_mcshower.at(ishower).Origin()==1:
                evout_mcshower.push_back( ev_mcshower.at(ishower) )
    
    outtree.Fill()
    if args.save_mc:
        outio.set_id( ioll.run_id(), ioll.subrun_id(), ioll.event_id() )
        outio.next_event()
    if False:
        break
    

#out.write_file()
out.Write()
io.finalize()
if args.has_mc:
    ioll.close()
if args.save_mc:
    outio.close()


print("End of event loop")
print("[FIN]")
