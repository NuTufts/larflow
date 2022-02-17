from __future__ import print_function
import os,sys,argparse,json
#sys.path.append(os.environ["LARFLOW_BASEDIR"]+"/larmatchnet/larvoxel/prepdata/")
sys.path.append(os.environ["LARFLOW_BASEDIR"]+"/larmatchnet")

parser = argparse.ArgumentParser(description='Run Prep larmatch data')
parser.add_argument('-o','--output',required=True,type=str,help="Filename stem for output files")
parser.add_argument('-i','--fileid',required=True,type=int,help="File ID number to run")
parser.add_argument('input_list',type=str,help="json file that collates triplet, mcinfo, and opreco files")
parser.add_argument('input_larmatch',nargs='+',help="Input larmatch triplet training args")

args = parser.parse_args()

from ctypes import c_int
from array import array
import numpy as np
from array import array
import torch
from larvoxel_dataset import larvoxelDataset

import ROOT as rt
from ROOT import std
from larlite import larlite
from larcv import larcv
from ublarcvapp import ublarcvapp
from larflow import larflow

if not os.path.exists(args.input_list):
    print("Could not find input list: ",args.input_list)
    sys.exit(0)

# LOAD JSON FILE
f = open(args.input_list,'r')
j = json.load(f)

# This is the file id to run
FILEIDS=[args.fileid]

input_triplet_v = []
input_mcinfo_v = []
input_opreco_v = []
for fid in FILEIDS:
    data = j["%d"%(fid)]
    tripletfile = data["triplet"]
    if not os.path.exists(tripletfile):
        print("input file given does not exist: ",tripletfile)
        sys.exit(0)
    input_triplet_v.append(tripletfile)
    for mcinfofile in data["mcinfo"]:
        if not os.path.exists(mcinfofile):
            print("mcinfo file given does not exist: ",mcinfofile)
            sys.exit(0)
        input_mcinfo_v.append(mcinfofile)
    for oprecofile in data["opreco"]:
        if not os.path.exists(oprecofile):
            print("opreco file given does not exist: ",oprecofile)
            sys.exit(0)
        input_opreco_v.append(oprecofile)

fmutil = ublarcvapp.mctools.FlashMatcher()

print("Loaded %d larmatch triplet files to process"%(len(input_triplet_v)))

ioll = larlite.storage_manager( larlite.storage_manager.kREAD )
#ioll.set_data_to_read( larlite.data.kMCTrack,  "mcreco" )
#ioll.set_data_to_read( larlite.data.kMCShower, "mcreco" )
#ioll.set_data_to_read( larlite.data.kMCTruth,  "generator" )
for f in input_mcinfo_v:
    ioll.add_in_filename( f )
ioll.open()

opio = larlite.storage_manager( larlite.storage_manager.kREAD )
for f in input_opreco_v:
    opio.add_in_filename( f )
opio.open()

def collate_fn(batch):
    return batch

# c++ class that appends keypoint and ssnet labels to triplet truth information
f_v = std.vector("std::string")()
for f in input_triplet_v:
    f_v.push_back(f)
labeler = larflow.keypoints.LoaderKeypointData( f_v )

#dataset = larvoxelDataset( txtfile=args.input_larmatch[0], random_access=False, voxelsize_cm=0.3 )
dataset = larvoxelDataset( filelist=input_triplet_v, random_access=False, voxelsize_cm=0.3 )
loader = torch.utils.data.DataLoader(dataset,batch_size=1,collate_fn=collate_fn)

# c++ class that provides voxels and labels using data in the labeler class
#voxelsize_cm = 0.3 # 3 mm, the wire-pitch
#voxelizer = larflow.voxelizer.VoxelizeTriplets()
#voxelizer.set_voxel_size_cm( voxelsize_cm )

# Get the number of entries in the tree
nentries = labeler.GetEntries()
print("nentries",nentries)
ll_nentries = ioll.get_entries()
if nentries!=ll_nentries:
    raise ValueError("Mismatch in triplet and larlite entries: labeler=%d larlite=%d"%(nentries,ll_nentries))
print("Input ready to go!")

# we're going to loop through the larlite file to make a rse to entry map
# do we need to?
rse_map = {}
for i in range(ll_nentries):
    ioll.go_to(i)
    rse = ( int(ioll.run_id()),int(ioll.subrun_id()),int(ioll.event_id()) )
    rse_map[rse] = i

# make tree of stuff we want to keep
outfile = rt.TFile("test.root","recreate")
outfile.cd()
outtree = rt.TTree("larvoxeltrainingdata","Flashmatched Voxel Tree")
# Run, subrun, event
#treevars = dict(run    = array('i',[0]),
#                subrun = array('i',[0]),
#                event  = array('i',[0]) )
run    = array('i',[0])
subrun = array('i',[0])
event  = array('i',[0])
ancestorID = array('i',[0])
clusterTick = array('d',[0])
flashTick = array('d',[0])
origin = array('i',[0])

coord_v = std.vector("larcv::NumpyArrayInt")()
feat_v  = std.vector("larcv::NumpyArrayFloat")()

lm_truth_v  = std.vector("larcv::NumpyArrayInt")()
lm_weight_v = std.vector("larcv::NumpyArrayFloat")()

instance_truth_v = std.vector("larcv::NumpyArrayInt")()
instance_map_v = std.vector("std::vector<int>")()

ancestor_truth_v = std.vector("larcv::NumpyArrayInt")()
ancestor_weight_v = std.vector("larcv::NumpyArrayFloat")()

ssnet_truth_v  = std.vector("larcv::NumpyArrayInt")()
ssnet_weight_v = std.vector("larcv::NumpyArrayFloat")()

kp_truth_v  = std.vector("larcv::NumpyArrayFloat")()
kp_weight_v = std.vector("larcv::NumpyArrayFloat")()

#outtree.Branch("run",    treevars["run"],    "run/I")
#outtree.Branch("subrun", treevars["subrun"], "subrun/I")
#outtree.Branch("event",  treevars["event"],  "event/I")
outtree.Branch("run", run, "run/I")
outtree.Branch("subrun", subrun, "subrun/I")
outtree.Branch("event",  event,  "event/I")
outtree.Branch("ancestorID",  ancestorID,  "ancestorID/I")
outtree.Branch("clusterTick",  clusterTick,  "clusterTick/D")
outtree.Branch("flashTick",  flashTick,  "flashTick/D")
outtree.Branch("origin",  origin,  "origin/I")
outtree.Branch("coord_v",  coord_v)
outtree.Branch("feat_v",   feat_v)
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

listy = []

# MAIN LOOP
# NOTE: Only works with tracks for now! Implement shower part too
for ientry in range(1):

    data = next(iter(loader))[0]

    print()
    print("==========================")
    print("===[ EVENT ",ientry," ]===")
    labeler.load_entry(ientry)
    ioll.go_to(ientry)
    opio.go_to(ientry)

    numTracks = fmutil.numTracks( ioll )

    for i in range( 0, 20, 1 ):
        track_tick = fmutil.grabTickFromMCTrack( ioll, i )

        producer = fmutil.producer
        isCosmic = fmutil.isCosmic

        print("Now isCosmic has been set to: ",fmutil.isCosmic)
        print(producer)

        op_tick = fmutil.grabTickFromOpflash( opio )
        #    fmtrack_tick = vtxutil.getImageCoords( ioll )
        print("track_tick",track_tick)
        print("op_tick",op_tick)
        #    iolcv.read_entry(ientry)
        match = fmutil.matchTicks( track_tick, op_tick )
        print("Found match: ", match )

        # Found a flash match! Now fill the tree
        if match != -999.999 and match != 999.999 and track_tick != -999.997:

            for vec in [ coord_v, feat_v, lm_truth_v, lm_weight_v,
                     ssnet_truth_v, ssnet_weight_v,
                     kp_truth_v, kp_weight_v,
                     instance_truth_v, instance_map_v,
                     ancestor_truth_v, ancestor_weight_v ]:
                     vec.clear()

            data = next(iter(loader))[0]

            print("Tree entry: ",data["tree_entry"])
            print(" keys: ",data.keys())
            for name,d in data.items():
                if type(d) is np.ndarray:
                    print("  ",name,": ",d.shape)
                else:
                    print("  ",name,": ",type(d))

            print("wtf: ",data["voxcoord"].shape)

            fmutil.process( ioll )
            run[0] = ioll.run_id()
            subrun[0] = ioll.subrun_id()
            event[0] = ioll.event_id()
            #ancestorID[0] = fmutil.trackAncestorID(ioll,i)
            ancestorID[0] = fmutil.trackAncestorID()
            clusterTick[0] = track_tick
            flashTick[0] = match
            origin[0] = fmutil.trackOrigin()


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


            outtree.Fill()

            #voxelizer.make_voxeldata( labeler.triplet_v[0] )
            #voxdata = voxelizer.get_full_voxel_labelset_dict( labeler )
            #print("voxdata keys: ",voxdata.keys())

        if isCosmic == 1 and match != -999.999 and track_tick != -999.997:
            #listy.append( match - track_tick )
            listy.append( track_tick )



    # Get the first entry (or row) in the tree (i.e. table)
    #labeler.load_entry(ientry)
    #trip_rse = ( int(labeler.run()), int(labeler.subrun()), int(labeler.event()) )

    #if trip_rse not in rse_map:
    #    raise ValueError("triplet rse not in larlite RSE",trip_rse)

    #llentry = rse_map[trip_rse]
    #ioll.go_to(llentry)

print("list: ", listy)
fmutil.finalize()

outfile.Write()

#for name,f in outfiles.items():
#    print("Writing file for ",name)
#    f.Write()
