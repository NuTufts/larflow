import os,sys
from ctypes import c_int
from array import array
import numpy as np
import ROOT as rt
from ROOT import std
from larcv import larcv
from larflow import larflow


input_rootfile_v = ["../testdata/larmatchtriplet_bnbnue_0492.root"]
f_v = rt.std.vector("std::string")()
for f in input_rootfile_v:
    f_v.push_back( f )

# c++ extension that provides spacepoint labels
kploader = larflow.keypoints.LoaderKeypointData( f_v )
kploader.set_verbosity( larcv.msg.kDEBUG )
kploader.exclude_false_triplets( False )

# Get the number of entries in the tree
nentries = kploader.GetEntries()

# output container for data
outfile = rt.TFile("temp.root","recreate")
outtree = rt.TTree("larmatchtrainingdata","LArMatch training data")

# Run, subrun, event
run    = array('i',[0])
subrun = array('i',[0])
event  = array('i',[0])
nueccfile = array('i',[1])

# 2D Wire Plane Images, as sparse matrices
coord_v = std.vector("larcv::NumpyArrayInt")()
feat_v  = std.vector("larcv::NumpyArrayFloat")()

# 2D Pixel Indicies (i.e. MatchTriplet) for each spacepoint
matchtriplet_v = std.vector("larcv::NumpyArrayInt")()

# ghost/true label for each spacepoint
lm_truth_v  = std.vector("larcv::NumpyArrayInt")()
lm_weight_v = std.vector("larcv::NumpyArrayFloat")()

# SSNet label and loss weights
ssnet_truth_v  = std.vector("larcv::NumpyArrayInt")()
ssnet_weight_v = std.vector("larcv::NumpyArrayFloat")()

# KP label and loss weights
kp_truth_v  = std.vector("larcv::NumpyArrayFloat")()
kp_weight_v = std.vector("larcv::NumpyArrayFloat")()

outtree.Branch("run", run, "run/I")
outtree.Branch("subrun", subrun, "subrun/I")
outtree.Branch("event",  event,  "event/I")
outtree.Branch("isnueccfile", nueccfile, "isnueccfile/I")
outtree.Branch("coord_v",coord_v)
outtree.Branch("feat_v", feat_v)
outtree.Branch("matchtriplet_v",matchtriplet_v)
outtree.Branch("larmatch_truth_v", lm_truth_v)
outtree.Branch("larmatch_weight_v",lm_weight_v)
outtree.Branch("ssnet_truth_v", ssnet_truth_v)
outtree.Branch("ssnet_weight_v",ssnet_weight_v)
outtree.Branch("kp_truth_v", kp_truth_v)
outtree.Branch("kp_weight_v",kp_weight_v)

for ientry in range(nentries):

    # clear entry data containers
    coord_v.clear()
    feat_v.clear()
    lm_truth_v.clear()
    lm_weight_v.clear()
    ssnet_truth_v.clear()
    ssnet_weight_v.clear()
    kp_truth_v.clear()
    kp_weight_v.clear()
    matchtriplet_v.clear()
    
    # Get the first entry (or row) in the tree (i.e. table)
    kploader.load_entry(ientry)

    # turn shuffle off (to do, function should be kploader function)
    tripdata = kploader.triplet_v.at(0).setShuffleWhenSampling( False )

    # 2d images
    wireimg_dict = {}
    for p in range(3):
        wireimg = kploader.triplet_v.at(0).make_sparse_image( p )
        wireimg_coord = wireimg[:,:2].astype(np.long)
        wireimg_feat  = wireimg[:,2]
        wireimg_dict["wireimg_coord%d"%(p)] = wireimg_coord
        wireimg_dict["wireimg_feat%d"%(p)] = wireimg_feat        

    # get 3d spacepoints (to do, function should be kploader function)
    tripdata = kploader.triplet_v.at(0).get_all_triplet_data( True )
    spacepoints = kploader.triplet_v.at(0).make_spacepoint_charge_array()    
    nfilled = c_int(0)
    ntriplets = tripdata.shape[0]    
    
    data = kploader.sample_data( ntriplets, nfilled, True )
    data.update(spacepoints)
    data.update( wireimg_dict )

    # to do: the commands here are still awfully wonky
    
    print("numpy arrays in tripdata: ",tripdata.shape)
    print("numpy arrays from kploader: ",data.keys())

    max_u = np.max( data["matchtriplet"][:,0] )
    max_v = np.max( data["matchtriplet"][:,1] )
    max_y = np.max( data["matchtriplet"][:,2] )
    print("sanity check, max indices: ",max_u,max_v,max_y)
    print("matchtriplet shape: ",data["matchtriplet"].shape,data["matchtriplet"].dtype)

    # Store data
    # run, subrun, event indices for entry
    run[0]    = kploader.run()
    subrun[0] = kploader.subrun()
    event[0]  = kploader.event()
    
    # wire plane images
    for p in range(3):
        imgcoord = larcv.NumpyArrayInt()
        print("store ",wireimg_dict["wireimg_coord%d"%(p)].shape)
        imgcoord.store( wireimg_dict["wireimg_coord%d"%(p)].astype(np.int32) )
        imgfeat  = larcv.NumpyArrayFloat()
        imgfeat.store( wireimg_dict["wireimg_feat%d"%(p)].astype(np.float32) )
        coord_v.push_back( imgcoord )
        feat_v.push_back( imgfeat )

    # matchtriplet matrix: how we go from spacepoint to wireplane indices
    matchtriplets = larcv.NumpyArrayInt()
    matchtriplets.store(data["matchtriplet"].astype(np.int32))
    matchtriplet_v.push_back(matchtriplets)

    outtree.Fill()
    if ientry>=4:
        break


outfile.Write()
    

    
