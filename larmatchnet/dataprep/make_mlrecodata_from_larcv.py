from __future__ import print_function
import os,sys,argparse,time
#sys.path.append("/usr/local/lib/python3.8/dist-packages/")
sys.path.append(os.environ["LARFLOW_BASEDIR"]+"/larmatchnet")
from ctypes import c_int
import numpy as np

"""
Convert larcv/larlite information from MicroBooNE/SBND/ICARUS into mlreco data products needed for training.

example call (which runs only the first event for testing purposes)
python3 make_mlrecodata_from_larcv.py -d uboone -wo output_microboone_wireoverlap_matrices.root --input-larlite merged_dlreco_mcc9_v13_bnbnue_corsika_run00001_subrun00001.root --input-larcv dlmerged_larflowtruth_mcc9_v13_bnbnue_corsika_run00001_subrun00001.root -adc wire -tb -n 1 -e 0 -o test.root
"""
parser = argparse.ArgumentParser("Make mlreco data from larcv")
parser.add_argument('-d','--detector',required=True,type=str,help="Choose detector. Optons: {'uboone','sbnd','icarus'} [required]")
parser.add_argument("-wo",'--wire-overlap-file',required=True,type=str,help="Location of wire overlap file")
parser.add_argument("-ill", "--input-larlite",required=True,type=str,help="Input larlite file [required]")
parser.add_argument("-ilcv","--input-larcv",required=True,type=str,help="Input LArCV file [required]")
parser.add_argument("-o","--output",required=True,type=str,help="output file name [required]")
parser.add_argument("-adc", "--adc",type=str,default="wire",help="Name of tree with Wire ADC values [default: wire]")
parser.add_argument("-tf",  "--tick-forward",action='store_true',default=False,help="Input LArCV data is tick-forwards [default: false]")
parser.add_argument("-n",   "--nentries",type=int,default=-1,help="Number of entries to run [default: -1 (all)]")
parser.add_argument("-e",   "--start-entry",type=int,default=0,help="Entry to start [default: 0]")
args = parser.parse_args()

import ROOT as rt
from ROOT import std
from larlite import larlite
from larcv import larcv
from ublarcvapp import ublarcvapp
from larflow import larflow
from ROOT import larutil


# SET DETECTOR
if args.detector not in ["uboone","sbnd","icarus"]:
    raise ValueError("Invalid detector. Choices: uboone, sbnd, or icarus")

# SET DETECTOR
overlap_matrix_file = args.wire_overlap_file
if not os.path.exists(overlap_matrix_file):
    print("Overlap matrix file not found at given path: ",overlap_matrix_file)
    print("This file contains information on which combination of wires intersect")
    print("example (on TREX): /tutorial_files/output_microboone_wireoverlap_matrices.root")
    print("It is required.")
    sys.exit(0)
    
if args.detector == "icarus":
    detid = larlite.geo.kICARUS
    #overlap_matrix_file = os.environ["LARFLOW_BASEDIR"]+"/larflow/PrepFlowMatchData/test/output_icarus_wireoverlap_matrices.root"
elif args.detector == "uboone":
    detid = larlite.geo.kMicroBooNE
    #overlap_matrix_file = os.environ["LARFLOW_BASEDIR"]+"/larflow/PrepFlowMatchData/test/output_microboone_wireoverlap_matrices.root"    
elif args.detector == "sbnd":
    detid = larlite.geo.kSBND
    #raise ValueError("SBND not supported yet")
larutil.LArUtilConfig.SetDetector(detid)

rt.gStyle.SetOptStat(0)

# OPEN INPUT FILES: LARLITE
ioll = larlite.storage_manager( larlite.storage_manager.kREAD )
ioll.add_in_filename(  args.input_larlite )
ioll.open()

# OPEN INPUT FILES: LARCV
if args.tick_forward:
    iolcv = larcv.IOManager( larcv.IOManager.kREAD, "larcv", larcv.IOManager.kTickForward )
else:
    iolcv = larcv.IOManager( larcv.IOManager.kREAD, "larcv", larcv.IOManager.kTickBackward )
iolcv.add_in_file( args.input_larcv )
if not args.tick_forward:
    iolcv.reverse_all_products()
iolcv.initialize()

# GET NUMBER OF ENTRIES: LARCV
nentries = iolcv.get_n_entries()
print("Number of entries in file: ",nentries)

# CHECK: DOES NUM OF LARLITE ENTRIES MATCH?
nentries_ll = ioll.get_entries()
if nentries_ll!=nentries:
    raise ValueError("Number of LArCV entries (%d) and number of larlite entries (%d) do not match! Usually a sign of an error."%(nentries,nentries_ll))

# DID USER SPECIFY A START ENTRY?
start_entry = args.start_entry
if start_entry>=nentries:
    print("Asking to start after last entry in file")
    sys.exit(0)

# DID USER SPECIFY THE NUMBER OF ENTRIES TO PROCESS
if args.nentries>0:
    end_entry = start_entry + args.nentries
else:
    end_entry = start_entry + nentries
if end_entry>=nentries:
    end_entry = nentries

# LARBYSMC OUTPUT TREE: Contains some common high-level truth info we want
lmc = ublarcvapp.mctools.LArbysMC()

# CREATE THE OUTPUT FILE
outlcv = larcv.IOManager( larcv.IOManager.kWRITE )
outlcv.set_out_file( args.output )
outlcv.initialize()

# --------------------------
# LABEL MAKING ALGORITHMS
# --------------------------

# bad channel/gap channel maker
badchmaker = ublarcvapp.EmptyChannelAlgo()

# triplet proposal maker
tripletmaker = larflow.prep.PrepMatchTriplets()
tripletmaker.set_wireoverlap_filepath( overlap_matrix_file  )
ev_tripdata = std.vector("larflow::prep::MatchTriplets")() # container to store triplet maker output
ev_voxdata  = std.vector("larflow::voxelizer::TPCVoxelData")() # container to store voxelized labels
#triptree.Branch("triplet_v",ev_tripdata) # add the container to the triplet tree, triptree

# Keypoint maker
kpana = larflow.keypoints.PrepKeypointData()
#kpana.set_verbosity( larcv.msg.kDEBUG )
kpana.setADCimageTreeName( args.adc  )

# ssnet label data: provides particle label for each spacepoint
ssnet = larflow.prep.PrepSSNetTriplet()

# We make spacepoints from 2D information, so there are some mistakes we fix using various methods
truthfixer = larflow.prep.TripletTruthFixer()

# VOXELIZES TRIPLET SPACEPOINTS
voxelizer = larflow.voxelizer.VoxelizeTriplets()
voxelizer.set_verbosity(larcv.msg.kDEBUG)

# Conversion of mctrack and mcshower into larcv::particle
particleconvertor = ublarcvapp.mctools.ConvertMCInfoForLArCV2()
particleconvertor.doApplySCE(True)
particleconvertor.doApplyT0drift(True)
particleconvertor.set_verbosity(larcv.msg.kDEBUG)

# -------------------
# EVENT LOOP!!

start = time.time()
for ientry in range(start_entry,end_entry,1):

    print(" ") 
    print("==========================")
    print("===[ EVENT ",ientry," ]===")
    sys.stdout.flush()
    
    ioll.go_to(ientry)
    iolcv.read_entry(ientry)

    # CLEAR ANY TRIPLET DATA WE SAVED
    ev_tripdata.clear()

    print(" ") 
    print("==========================")
    print("===[ EVENT ",ientry," ]===")
    sys.stdout.flush()
    
    ioll.go_to(ientry)
    iolcv.read_entry(ientry)

    ev_adc = iolcv.get_data(larcv.kProductImage2D,args.adc)
    ev_mctrack = ioll.get_data(larlite.data.kMCTrack,"mcreco")
    ev_mcshower = ioll.get_data(larlite.data.kMCShower,"mcreco")

    # We need to convert mctrack and mcshower into larcv::particle objects
    particle_v = particleconvertor.convert( ev_mctrack, ev_mcshower )
    print("Number of larcv::particle instances: ",particle_v.size())
    

    # eventsparsetensor3d container for event
    ev_charge_v = {}
    for p in range(3):
        ev_charge_v[p] = outlcv.get_data( larcv.kProductSparseTensor3D, "charge_plane%d"%(p) )    
    ev_semantic = outlcv.get_data( larcv.kProductSparseTensor3D, "semantics_ghost" )
    ev_cluster  = outlcv.get_data( larcv.kProductSparseTensor3D, "pcluster" )
    ev_particle = outlcv.get_data( larcv.kProductParticle, "corrected" )

    ev_tripdata.clear()

    # make triplet proposals: function valid for simulation or real data
    tripletmaker.process( iolcv, args.adc, args.adc, 10.0, True )
    
    # make good/bad triplet ground truth
    tripletmaker.process_truth_labels( iolcv, ioll, args.adc )

    # fix up some labels
    truthfixer.calc_reassignments( tripletmaker, iolcv, ioll )

    # make keypoint score ground truth
    kpana.process( iolcv, ioll )
    kpana.make_proposal_labels( tripletmaker )
    #kpana.fillAnaTree()

    # make ssnet ground truth
    ssnet.make_ssnet_labels( iolcv, ioll, tripletmaker )

    # Make copy of MCPixelPGraph: can then dump out event particle graph
    mcpg = ublarcvapp.mctools.MCPixelPGraph()
    mcpg.buildgraphonly( ioll )
    mcpg.printGraph(0,False)    
    sys.stdout.flush()

    print("Match Triplet Data Made (one instance per TPC in detector): ",tripletmaker._match_triplet_v.size())
    for imatch in range(tripletmaker._match_triplet_v.size()):
        print(" MatchData[%d]: num of triplets=%d"%(imatch,tripletmaker._match_triplet_v.at(imatch)._triplet_v.size()))
        #ev_tripdata.push_back( tripletmaker._match_triplet_v.at(imatch) ) # only do this if we need to, expensive

        # NOW VOXELIZE TRIPLET DATA
        tpc_tripletdata = tripletmaker._match_triplet_v.at(imatch)
        tpc_ssnetdata   = ssnet._ssnet_labeldata_v.at(imatch)
        tpc_kpdata      = kpana._keypoint_tpclabel_v.at(imatch)
        tpc_voxdata     = voxelizer.make_voxeldata( tpc_tripletdata )

        # Make SparseTensor3D products
        data_v = voxelizer.make_mlreco_semantic_label_sparse3d( tpc_voxdata, tpc_tripletdata, tpc_ssnetdata )
        for p in range(3):
            ev_charge_v[p].merge( data_v[p] )
        ev_semantic.merge( data_v[3] )

        # Make individual particle cluster labels + refine partice list to those labeling voxels in the event
        rejected_v = std.vector("larcv::Particle")()
        cluster_v = voxelizer.make_mlreco_cluster_label_sparse3d( tpc_voxdata, tpc_tripletdata, particle_v, rejected_v )
        print("Number after voxelizer particle cluster labeler: ",particle_v.size())
        ev_cluster.merge( cluster_v[0] )

        # UBOONE HACK: one tpc at a time
        if True:
            break

    ev_particle.set( particle_v )
    # Done with the event -- Fill it!
    outlcv.set_id( ev_adc.run(), ev_adc.subrun(), ev_adc.event() )
    outlcv.save_entry()
    
    #if iiter>=4:
    #    break

print("Event Loop Finished")
print("Writing Output File")
outlcv.finalize()
print("Done")

