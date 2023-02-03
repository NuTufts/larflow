from __future__ import print_function
import os,sys,argparse,time
#sys.path.append("/usr/local/lib/python3.8/dist-packages/")
sys.path.append(os.environ["LARFLOW_BASEDIR"]+"/larmatchnet")
from ctypes import c_int
import numpy as np

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
if args.detector == "icarus":
    detid = larlite.geo.kICARUS
    overlap_matrix_file = os.environ["LARFLOW_BASEDIR"]+"/larflow/PrepFlowMatchData/test/output_icarus_wireoverlap_matrices.root"
elif args.detector == "uboone":
    detid = larlite.geo.kMicroBooNE
    overlap_matrix_file = os.environ["LARFLOW_BASEDIR"]+"/larflow/PrepFlowMatchData/test/output_microboone_wireoverlap_matrices.root"    
elif args.detector == "sbnd":
    detid = larlite.geo.kSBND
    raise ValueError("SBND not supported yet")
larutil.LArUtilConfig.SetDetector(detid)

rt.gStyle.SetOptStat(0)

# OPEN INPUT FILES: LARLITE
ioll = larlite.storage_manager( larlite.storage_manager.kREAD )
ioll.add_in_filename(  args.input_larlite )
ioll.open()

# OPEN INPUT FILES: LARCV
if args.tick_backward:
    iolcv = larcv.IOManager( larcv.IOManager.kREAD, "larcv", larcv.IOManager.kTickBackward )
else:
    iolcv = larcv.IOManager( larcv.IOManager.kREAD, "larcv", larcv.IOManager.kTickForward )
iolcv.add_in_file( args.input_larcv )
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
outfile = rt.TFile(args.output,"recreate")

# CREATE THE TTREE THAT WILL HOLD LARBYSMC INFO
lmctree = rt.TTree("LArbysMCTree","MC infomation")
lmctree.SetDirectory(outfile)
# ADD LARBYSMC BRANCH VARIABLES TO THE TREE
lmc.bindAnaVariables(lmctree)
for ientry in range(start_entry,end_entry,1):
    ioll.go_to(ientry)
    lmc.process(ioll)
    lmctree.Fill()
outfile.cd()
# WRITE THIS TREE, THEN DESTROY IT.
# For some reason, if we try to feel this info as we process the rest of the truth info,
# we get segfaults ...
# processing all events before the rest of the analysis is done is a work-around.
lmctree.Write()
del lmc

# 
#dataset = larvoxelDataset( txtfile=args.input_larmatch[0], random_access=False, voxelsize_cm=0.3 )
#NENTRIES = len(dataset)
#loader = torch.utils.data.DataLoader(dataset,batch_size=1,collate_fn=collate_fn)

# DEFINE THE OUTPUT TREE FOR THE VOXEL DATA
outtree = rt.TTree("larvoxeltrainingdata","LArMatch Voxel training data")

cryoid_v = std.vector("int")()
tpcid_v  = std.vector("int")()

coord_v = std.vector("larcv::NumpyArrayInt")()   # COORDINATE TENSOR OF PROPOSED CHARGE DEPOSITS
feat_v  = std.vector("larcv::NumpyArrayFloat")() # FEATURE TENSOR OF PROPOSED CHARGE DEPOSITS (Charge from each plane)

lm_truth_v  = std.vector("larcv::NumpyArrayInt")() # LARMATCH TRUTH: DOES VOXEL CONTAIN TRUE 3D CHARGE DEPOSIT?
lm_weight_v = std.vector("larcv::NumpyArrayFloat")() # WEIGHT FOR THIS VOXEL (for balancing positve and negative examples)

instance_truth_v = std.vector("larcv::NumpyArrayInt")() # THE GEANT TRACKID THAT DEPOSITED MOST ENERGY IN THIS VOXEL
instance_map_v = std.vector("std::vector<int>")() # SEQUENTIAL RENUMBERING OF IDs from 0 to N=Unique TrackIDs. INDEX IS RENUMBERED INDEX, VALUE IS GEANT4 TRACKID

ancestor_truth_v = std.vector("larcv::NumpyArrayInt")() # GEANT TRACKID OF PRIMARY THAT EVENTUALLY CAUSED THIS ENERGY DEPOSIT
ancestor_weight_v = std.vector("larcv::NumpyArrayFloat")() # WEIGHT: DON'T REMEMBER THE WEIGHT APPLIED HERE

ssnet_truth_v  = std.vector("larcv::NumpyArrayInt")() # PARTICLE CLASS THAT DEPOSITED THE MOST ENERGY IN THIS VOXEL (Turn this into INSTANCE<->PID map to save space?)
ssnet_weight_v = std.vector("larcv::NumpyArrayFloat")() # WEIGHT FOR BALANCING BY PARTICLE TYPE

kp_truth_v  = std.vector("larcv::NumpyArrayFloat")() # KEYPOINT SCORE WHERE 1=AT KEYPOINT LOCATION and falls off as Gaussian function of distance
kp_weight_v = std.vector("larcv::NumpyArrayFloat")() # WEIGHT balancing NEAR KEYPOINTS vs. NOT-NEAR KEYPOINTS (NOT-NEAR defined as score below 0.1)


outtree.Branch("cryoid_v",cryoid_v)
outtree.Branch("tpcid_v",tpcid_v)
outtree.Branch("coord_v",coord_v)
outtree.Branch("feat_v", feat_v)
outtree.Branch("larmatch_truth_v", lm_truth_v)
outtree.Branch("larmatch_weight_v",lm_weight_v)
outtree.Branch("ssnet_truth_v", ssnet_truth_v)
outtree.Branch("ssnet_weight_v",ssnet_weight_v)
outtree.Branch("kp_truth_v", kp_truth_v)
outtree.Branch("kp_weight_v",kp_weight_v)
outtree.Branch("instance_truth_v",instance_truth_v)
outtree.Branch("instance2trackid_v",instance_map_v)
outtree.Branch("ancestor_truth_v",ancestor_truth_v)
outtree.Branch("ancestor_weight_v",ancestor_weight_v)

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
outfile.cd()
#kpana.defineAnaTree()

# ssnet label data: provides particle label for each spacepoint
ssnet = larflow.prep.PrepSSNetTriplet()
outfile.cd()
ssnet.defineAnaTree()

# We make spacepoints from 2D information, so there are some mistakes we fix using various methods
truthfixer = larflow.prep.TripletTruthFixer()

# VOXELIZES TRIPLET SPACEPOINTS
voxelizer = larflow.voxelizer.VoxelizeTriplets()
voxelizer.set_verbosity(larcv.msg.kDEBUG)

# -------------------
# EVENT LOOP!!

start = time.time()
for ientry in range(start_entry,end_entry,1):

    # CLEAR THE OUTPUT VECTORS IN THE TREE
    for vec in [ cryoid_v, tpcid_v,
                 coord_v, feat_v, lm_truth_v, lm_weight_v,
                 ssnet_truth_v, ssnet_weight_v,                 
                 kp_truth_v, kp_weight_v,
                 instance_truth_v, instance_map_v,
                 ancestor_truth_v, ancestor_weight_v ]:
        vec.clear()

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

        # Make numpy arrays
        data = voxelizer.make_full_voxel_labelset_dict( tpc_voxdata, tpc_tripletdata, tpc_ssnetdata, tpc_kpdata )
        print("VoxelData dict keys: ",data.keys())

        # Fill the TTree branch containers
        cryoid_v.push_back( tpc_voxdata._cryoid )
        tpcid_v.push_back( tpc_voxdata._tpcid )

        print("Voxcoord tensor: ",data["voxcoord"].shape)
        coord_v.push_back( larcv.NumpyArrayInt( data["voxcoord"].astype(np.int32) ) )
        feat_v.push_back( larcv.NumpyArrayFloat( data["voxfeat"] ) )

        lm_truth_v.push_back( larcv.NumpyArrayInt( data["voxlabel"].squeeze().astype(np.int32) ) )
        ssnet_truth_v.push_back( larcv.NumpyArrayInt( data["ssnet_labels"].squeeze().astype(np.int32) ) )
        kp_truth_v.push_back( larcv.NumpyArrayFloat( data["kplabel"].squeeze() ) )
    
        lm_weight_v.push_back( larcv.NumpyArrayFloat( data["voxlmweight"].squeeze() ) )
        ssnet_weight_v.push_back( larcv.NumpyArrayFloat( data["ssnet_weights"].squeeze() ) )
        kp_weight_v.push_back( larcv.NumpyArrayFloat( data["kpweight"].squeeze() ) )

        instance_truth_v.push_back(  larcv.NumpyArrayInt( data["voxinstance"].squeeze().astype(np.int32) ) )
        for ii in data["voxinstance2id"]:
            k = ii
            v = data["voxinstance2id"][k]
            pair = std.vector("int")(2,0)
            pair[0] = int(k)
            pair[1] = int(v)
            instance_map_v.push_back(pair)
    
        ancestor_truth_v.push_back(  larcv.NumpyArrayInt( data["voxorigin"].squeeze().astype(np.int32) ) )
        ancestor_weight_v.push_back( larcv.NumpyArrayFloat( data["voxoriginweight"].squeeze().astype(np.float32) ) )

    # Done with the event -- Fill it!
    outtree.Fill()
    #if iiter>=4:
    #    break

print("Event Loop Finished")
print("Writing Output File")
outfile.Write()
print("Done")

