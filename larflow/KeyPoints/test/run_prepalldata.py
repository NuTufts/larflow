from __future__ import print_function
import os,sys,argparse,time

parser = argparse.ArgumentParser("Test PrepKeypointData")
parser.add_argument('-d','--detector',required=True,type=str,help="Choose detector. Optons: {'uboone','sbnd','icarus'} [required]")
parser.add_argument("-ill", "--input-larlite",required=True,type=str,help="Input larlite file [required]")
parser.add_argument("-ilcv","--input-larcv",required=True,type=str,help="Input LArCV file [required]")
parser.add_argument("-o","--output",required=True,type=str,help="output file name [required]")
parser.add_argument("-adc", "--adc",type=str,default="wire",help="Name of tree with Wire ADC values [default: wire]")
parser.add_argument("-tb",  "--tick-backward",action='store_true',default=False,help="Input LArCV data is tick-backward [default: false]")
parser.add_argument("-tri", "--save-triplets",action='store_true',default=False,help="Save triplet data [default: false]")
parser.add_argument("-n",   "--nentries",type=int,default=-1,help="Number of entries to run [default: -1 (all)]")
parser.add_argument("-e",   "--start-entry",type=int,default=0,help="Entry to start [default: 0]")
args = parser.parse_args()

if args.detector not in ["uboone","sbnd","icarus"]:
    raise ValueError("Invalid detector")


import ROOT as rt
from ROOT import std
from larcv import larcv
from larlite import larlite
from larflow import larflow
from ublarcvapp import ublarcvapp
from ROOT import larutil

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
print("Number of entries in file: ",nentries)
start_entry = args.start_entry
if start_entry>=nentries:
    print("Asking to start after last entry in file")
    sys.exit(0)

if args.nentries>0:
    end_entry = start_entry + args.nentries
else:
    end_entry = start_entry + nentries
if end_entry>=nentries:
    end_entry = nentries

# OUTPUT FILE
lmc = ublarcvapp.mctools.LArbysMC()

outfile = rt.TFile(args.output,"recreate")
lmctree = rt.TTree("LArbysMCTree","MC infomation")
lmctree.SetDirectory(outfile)
# MAKE TRUTH INFO
lmc.bindAnaVariables(lmctree)
for ientry in range(start_entry,end_entry,1):
    ioll.go_to(ientry)
    lmc.process(ioll)
    lmctree.Fill()
outfile.cd()
lmctree.Write()
del lmc

# ALGOS
# -----------------------

# bad channel/gap channel maker
badchmaker = ublarcvapp.EmptyChannelAlgo()

# triplet proposal maker
ev_triplet = std.vector("larflow::prep::PrepMatchTriplets")(1)
ev_triplet.at(0).set_wireoverlap_filepath( overlap_matrix_file  )
ev_tripdata = std.vector("larflow::prep::MatchTriplets")()

# keypoint score data
kpana = larflow.keypoints.PrepKeypointData()
kpana.set_verbosity( larcv.msg.kDEBUG )
kpana.setADCimageTreeName( args.adc )
outfile.cd()
kpana.defineAnaTree()

# ssnet label data
ssnet = larflow.prep.PrepSSNetTriplet()
outfile.cd()
ssnet.defineAnaTree()

# affinity field data
#kpflow = larflow.keypoints.PrepAffinityField()
#outfile.cd()
#kpflow.defineAnaTree()


if args.save_triplets:
    outfile.cd()
    triptree = rt.TTree("larmatchtriplet","LArMatch triplets")
    triptree.Branch("triplet_v",ev_tripdata)

start = time.time()

nrun = 0
for ientry in range(start_entry,end_entry,1):

    print(" ") 
    print("==========================")
    print("===[ EVENT ",ientry," ]===")
    sys.stdout.flush()
    
    ioll.go_to(ientry)
    iolcv.read_entry(ientry)

    ev_tripdata.clear()

    #lmc.process(ioll)

    tripmaker = ev_triplet[0]
    mcpg = ublarcvapp.mctools.MCPixelPGraph()
    mcpg.buildgraphonly( ioll )
    mcpg.printGraph(0,False)    
    sys.stdout.flush()
    
    # make triplet proposals
    tripmaker.process( iolcv, args.adc, args.adc, 10.0, True )

    # make good/bad triplet ground truth
    tripmaker.process_truth_labels( iolcv, ioll, args.adc )

    # fix up some labels
    truthfixer = larflow.prep.TripletTruthFixer()    
    truthfixer.calc_reassignments( tripmaker, iolcv, ioll )    

    # make keypoint score ground truth
    kpana.process( iolcv, ioll )
    kpana.make_proposal_labels( tripmaker )
    kpana.fillAnaTree()

    # make ssnet ground truth
    ssnet.make_ssnet_labels( iolcv, ioll, tripmaker )
    
    # fill happens automatically (ugh so ugly)

    # make affinity field ground truth
    #kpflow.process( iolcv, ioll, tripmaker )
    #kpflow.fillAnaTree()    
    
    if args.save_triplets:
        for imatch in range(tripmaker._match_triplet_v.size()):
            ev_tripdata.push_back( tripmaker._match_triplet_v.at(imatch) )
        triptree.Fill()
    nrun += 1    
    
    #sys.exit(0)
    #break

print("NCLOSE: ",kpana._nclose)
print("NFAR: ",kpana._nfar)
print("FRAC CLOSE: ",float(kpana._nclose)/float(kpana._nclose+kpana._nfar))

dtime = time.time()-start
print("Time: ",float(dtime)/float(nrun)," sec/event")

outfile.cd()
#lmc.finalize()
kpana.writeAnaTree()
ssnet.writeAnaTree()
#kpflow.writeAnaTree()
if args.save_triplets:
    triptree.Write()

#del kpana
#del ssnet
#del kpflow

print("=== FIN ==")
