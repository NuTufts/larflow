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
from ROOT import std, TFile, TTree
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
iomc = larlite.storage_manager( larlite.storage_manager.kWRITE ) # for saving mctracks amd opflash
#ioop = larlite.storage_manager( larlite.storage_manager.kWRITE ) # for saving opflash
#ioll.set_data_to_read( larlite.data.kMCTrack,  "mcreco" )
#ioll.set_data_to_read( larlite.data.kMCShower, "mcreco" )
#ioll.set_data_to_read( larlite.data.kMCTruth,  "generator" )
for f in input_mcinfo_v:
    ioll.add_in_filename( f )
ioll.open()

iomc.set_out_filename( "filtered_MCTracks_opflash_beamONLY_MODTEST.root" )
iomc.open()

#ioop.set_out_filename( "filtered_opflashes.root" )
#ioop.open()

# for creating filtered mctrack tree
#f = TFile(input_mcinfo_v[0],"READ")
#t = f.Get('mctrack_mcreco_tree')
#newF = TFile("mctracks.root","recreate")
#newT = TTree("mctrack_mcreco_tree", "mctrack_mcreco_tree")
#newT = t.CloneTree(0)

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
outfile = rt.TFile("test_beamONLY_MODTEST.root","recreate")
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
##outtree.Branch("ancestorID",  ancestorID,  "ancestorID/I")
##outtree.Branch("clusterTick",  clusterTick,  "clusterTick/D")
##outtree.Branch("flashTick",  flashTick,  "flashTick/D")
##outtree.Branch("origin",  origin,  "origin/I")
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

#listy = []

def voxelizeIntrxn(iid_v):
    print("This is iid_v: ",iid_v)

    # create array with T/F boolean
    indexmatch_v = [ data["voxinstance"]==iid for iid in iid_v ]
    #print("indexmatch_v: ",indexmatch_v)

    # filter for matching instances only
    iicoord_v = [ data["voxcoord"][indexmatch[:],:] for indexmatch in indexmatch_v ]
    iifeat_v = [ data["voxfeat"][indexmatch[:],:] for indexmatch in indexmatch_v ]

    iilm_truth_v = [ data["voxlabel"][indexmatch[:]] for indexmatch in indexmatch_v ]
    iissnet_truth_v = [ data["ssnet_labels"][indexmatch[:]] for indexmatch in indexmatch_v ]
    iikp_truth_v = [ data["kplabel"][:,indexmatch[:]] for indexmatch in indexmatch_v ]
    iilm_weight_v = [ data["voxlmweight"][indexmatch[:]] for indexmatch in indexmatch_v ]
    iissnet_weight_v = [ data["ssnet_weights"][indexmatch[:]] for indexmatch in indexmatch_v ]
    iikp_weight_v = [ data["kpweight"][:,indexmatch[:]] for indexmatch in indexmatch_v ]
    #iiinstance_truth_v = [ data["voxinstance"][indexmatch[:],:] for indexmatch in indexmatch_v ]
    iiorigin_truth_v = [ data["voxorigin"][indexmatch[:]] for indexmatch in indexmatch_v ]
    iiorigin_weight_v = [ data["voxoriginweight"][indexmatch[:]] for indexmatch in indexmatch_v ]

    print("data[voxcoord]",data["voxcoord"])
    print("data[voxlabel]",data["voxlabel"])
    #print("iicoord_v",iicoord_v)
    #print("iicoord_v array",np.array(iicoord_v)) # do NOT do this! use concatenate (below)

    iicoord = np.concatenate( iicoord_v )
    iifeat = np.concatenate( iifeat_v )
    iilm_truth = np.concatenate( iilm_truth_v )
    iissnet_truth = np.concatenate( iissnet_truth_v )
    iikp_truth = np.concatenate( iikp_truth_v )
    iilm_weight = np.concatenate( iilm_weight_v )
    iissnet_weight = np.concatenate( iissnet_weight_v )
    iikp_weight = np.concatenate( iikp_weight_v )
    #iiinstance_truth = np.concatenate( iiinstance_truth_v )
    iiorigin_truth = np.concatenate( iiorigin_truth_v )
    iiorigin_weight = np.concatenate( iiorigin_weight_v )

    print("iicoord",iicoord)
    print("iicoord.shape",iicoord.shape)
    print("iifeat",iifeat)
    print("iifeat.shape",iifeat.shape)
    print("iilm_truth",iilm_truth)
    print("iilm_truth.shape",iilm_truth.shape)

    coord_v.push_back( larcv.NumpyArrayInt(iicoord.astype(np.int32)))
    feat_v.push_back( larcv.NumpyArrayFloat(iifeat))

    lm_truth_v.push_back( larcv.NumpyArrayInt( iilm_truth.astype(np.int32) ) )
    #lm_truth_v.push_back( larcv.NumpyArrayInt( iilm_truth.squeeze().astype(np.int32) ) )


    ssnet_truth_v.push_back( larcv.NumpyArrayInt( iissnet_truth.astype(np.int32) ) )
    kp_truth_v.push_back( larcv.NumpyArrayFloat( iikp_truth ) )

    lm_weight_v.push_back( larcv.NumpyArrayFloat( iilm_weight ) )
    ssnet_weight_v.push_back( larcv.NumpyArrayFloat( iissnet_weight ) )
    kp_weight_v.push_back( larcv.NumpyArrayFloat( iikp_weight ) )

    #instance_truth_v.push_back(  larcv.NumpyArrayInt( iiinstance_truth.astype(np.int32) ) )

    ancestor_truth_v.push_back(  larcv.NumpyArrayInt( iiorigin_truth.astype(np.int32) ) )
    ancestor_weight_v.push_back( larcv.NumpyArrayFloat( iiorigin_weight.astype(np.float32) ) )

    instance_truth_v.push_back(  larcv.NumpyArrayInt( data["voxinstance"].squeeze().astype(np.int32) ) )

# MAIN LOOP
# NOTE: Only works with tracks for now! Implement shower part too
#iomc.next_event()
for ientry in range(2):

    ancestorList = [] # keep track of intrxn ancestor IDs in the event
    trackList = []

    data = next(iter(loader))[0]

    print()
    print("==========================")
    print("===[ EVENT ",ientry," ]===")
    labeler.load_entry(ientry)
    ioll.go_to(ientry)

    ev_mctrack = ioll.get_data(larlite.data.kMCTrack,"mcreco")
    ev_mcshower = ioll.get_data(larlite.data.kMCShower,"mcreco")

    opio.go_to(ientry)

    ev_opflash_cosmic = opio.get_data(larlite.data.kOpFlash,"simpleFlashCosmic")
    ev_opflash_beam = opio.get_data(larlite.data.kOpFlash,"simpleFlashBeam")

    print("ev_mctrack.size() is:", ev_mctrack.size() )
    print("ev_mcshower.size() is:", ev_mcshower.size() )
    print("ev_opflash_cosmic.size() is:", ev_opflash_cosmic.size() )
    print("ev_opflash_beam.size() is:", ev_opflash_beam.size() )

    #mcpg = ublarcvapp.mctools.MCPixelPGraph()
    #mcpg.buildgraphonly( ioll  )
    #mcpg.printGraph(0,False)

    numTracks = fmutil.numTracks( ioll )
    numShowers = fmutil.numShowers( ioll )

    counter = 0

    iidList = [] # push back each iid_v list into here

    # loop thru tracks in event
    for i in range( 0, numTracks, 1 ):

        print("NOW IN THE TRACK LOOP!")
        print("There are ",numTracks," in this event")

        track_tick = fmutil.grabTickFromMCTrack( ioll, i )

        producer = fmutil.producer
        isCosmic = fmutil.isCosmic

        print("Now isCosmic has been set to: ",isCosmic)
        if (isCosmic == 1):
            continue

        ancestor = fmutil.trackAncestorID()
        trackid = fmutil.getTrackID()

        print("TrackID here is: ", trackid)
        print("AncestorID here is: ", ancestor)

        trackList.append(trackid)

        if ancestor in ancestorList:
            continue # this is a secondary; skip for now

        ancestorList.append( ancestor )
        print("ancestorList: ",ancestorList)

        print(producer)

        op_tick = fmutil.grabTickFromOpflash( opio )
        #    fmtrack_tick = vtxutil.getImageCoords( ioll )
        print("track_tick",track_tick)
        print("op_tick",op_tick) #should be a vector of pairs of tick and index
        #    iolcv.read_entry(ientry)
        match = fmutil.matchTicks( track_tick, op_tick )
        print("Found match: ", match ) #should be a pair of tick and index

        # Found a flash match! Now fill the tree
        if match[0] != -999.999 and match[0] != 999.999 and track_tick != -999.997:

            ##for vec in [ coord_v, feat_v, lm_truth_v, lm_weight_v,
            ##         ssnet_truth_v, ssnet_weight_v,
            ##         kp_truth_v, kp_weight_v,
            ##         instance_truth_v, instance_map_v,
            ##         ancestor_truth_v, ancestor_weight_v ]:
            ##         vec.clear()

            #data = next(iter(loader))[0]

            print("Tree entry: ",data["tree_entry"])
            print(" keys: ",data.keys())
            for name,d in data.items():
                if type(d) is np.ndarray:
                    print("  ",name,": ",d.shape)
                else:
                    print("  ",name,": ",type(d))

            print("wtf: ",data["voxcoord"].shape)

            #fmutil.process( ioll )
            ##run[0] = ioll.run_id()
            ##subrun[0] = ioll.subrun_id()
            ##event[0] = ioll.event_id()
            #ancestorID[0] = fmutil.trackAncestorID(ioll,i)
            ##ancestorID[0] = fmutil.trackAncestorID()
            ##clusterTick[0] = track_tick
            ##flashTick[0] = match[0]
            ##origin[0] = fmutil.trackOrigin()

            iid_v = [] # list of instance ids to collect

            ##print("ancestorID: ",ancestorID[0])

            #print("voxinstance2id",data["voxinstance2id"])

            for ii in data["voxinstance2id"]:
                k = ii
                v = data["voxinstance2id"][k]
                pair = std.vector("int")(2,0)
                pair[0] = int(k)
                pair[1] = int(v)
                #print("trackID and instance: ", pair[0], " ", pair[1])
                instance_map_v.push_back(pair)
                #print("instance_map_v: ",instance_map_v)
                if (k == ancestor):
                    print("FOUND a track that matches ancestorID!")
                    print("It is track ID: ",k)
                    print("For instance: ",v)
                    iid_v.append(v)

            print("This is iid_v: ",iid_v)
            if not iid_v:
                print("List is empty")
                continue

            iidList.append(iid_v)

    # should be in event loop

    print("This is event NO: ", ientry)
    print("IIDLOST: ",iidList)
    print("ancestorList: ", ancestorList)
    print("trackList: ", trackList)

    run[0] = ioll.run_id()
    subrun[0] = ioll.subrun_id()
    event[0] = ioll.event_id()

    for i in range(len( iidList )):
        print("size of the iidList at the END: ",len( iidList ))
        print("i, iidList[i]: ",i, iidList[i])

        for vec in [ coord_v, feat_v, lm_truth_v, lm_weight_v,
                 ssnet_truth_v, ssnet_weight_v,
                 kp_truth_v, kp_weight_v,
                 instance_truth_v, instance_map_v,
                 ancestor_truth_v, ancestor_weight_v ]:
                 vec.clear()

        voxelizeIntrxn(iidList[i])

            ##out_mctrack  = iomc.get_data( larlite.data.kMCTrack, "mcreco" )
            ##out_mctrack.push_back( ev_mctrack.at(i) )


            ##out_opflash  = iomc.get_data( larlite.data.kOpFlash, producer )
            ##if (fmutil.isCosmic == 1):
            ##    print("ev_opflash_cosmic: ",ev_opflash_cosmic)
            ##    out_opflash.push_back( ev_opflash_cosmic.at( match[1] ) )
            ##else:
            ##    print("ev_opflash_beam: ",ev_opflash_beam)
            ##    out_opflash.push_back( ev_opflash_beam.at( match[1] ) )

            ##counter = counter + 1

            # need to increment event id so final filtered tree entries don't all have same eventid
            # (assumes <100 clusters in an event)
            ##iomc.set_id( ioll.run_id(), ioll.subrun_id(), (ioll.event_id())*100+counter )
            ##iomc.next_event()
#            ioop.next_event()

            ##print("CHOSEN TICK ACTUAL TIME WAS: ", ((match[0]-3200.0)*0.5) )

        outtree.Fill()
            #newT.Fill()


            #voxelizer.make_voxeldata( labeler.triplet_v[0] )
            #voxdata = voxelizer.get_full_voxel_labelset_dict( labeler )
            #print("voxdata keys: ",voxdata.keys())

        #if isCosmic == 1 and match != -999.999 and track_tick != -999.997:
            #listy.append( match - track_tick )
            #listy.append( track_tick )



    # Get the first entry (or row) in the tree (i.e. table)
    #labeler.load_entry(ientry)
    #trip_rse = ( int(labeler.run()), int(labeler.subrun()), int(labeler.event()) )

    #if trip_rse not in rse_map:
    #    raise ValueError("triplet rse not in larlite RSE",trip_rse)

    #llentry = rse_map[trip_rse]
    #ioll.go_to(llentry)

#print("list: ", listy)
fmutil.finalize()

iomc.close()
#ioop.close()

outfile.Write()
#newF.Write()

#for name,f in outfiles.items():
#    print("Writing file for ",name)
#    f.Write()
