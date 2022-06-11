from __future__ import print_function
import os,sys,argparse
sys.path.append(os.environ["LARFLOW_BASEDIR"]+"/larmatchnet")

parser = argparse.ArgumentParser(description='Run Prep larmatch data')
parser.add_argument('-o','--output',required=True,type=str,help="Filename stem for output files")
parser.add_argument('-s','--single',default=False,action='store_true',help='If flag given, input_list argument is interpretted as a triplet file')
parser.add_argument('input_list',type=str,help="text file with paths to larmatch triplet files to distill")

args = parser.parse_args()

from ctypes import c_int
from array import array
import numpy as np
import ROOT as rt
from ROOT import std
from larcv import larcv
from larflow import larflow


if not os.path.exists(args.input_list):
    print("Could not fine input list: ",args.input_list)
    sys.exit(0)
if os.path.exists(args.output):
    print("output file already exists. do not overwrite")
    sys.exit(0)

if not args.single:
    f = open(args.input_list,'r')
    lf = f.readlines()
    input_rootfile_v = []
    for l in lf:
        if not os.path.exists(l.strip()):
            print("input file given does not exist: ",l.strip())
            sys.exit(0)
        input_rootfile_v.append(l.strip())
    print("Loaded %d larmatch triplet files to process"%(len(input_rootfile_v)))
else:
    print("Given a single larmatch triplet file to proces")
    input_rootfile_v = [ args.input_list ]

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
outfile = rt.TFile(args.output,"recreate")
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
ssnet_class_weight_v = std.vector("larcv::NumpyArrayFloat")()
ssnet_top_weight_v = std.vector("larcv::NumpyArrayFloat")()

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
outtree.Branch("ssnet_class_weight_v",ssnet_class_weight_v)
outtree.Branch("ssnet_top_weight_v",ssnet_top_weight_v)
outtree.Branch("kp_truth_v", kp_truth_v)
outtree.Branch("kp_weight_v",kp_weight_v)

for ientry in range(nentries):

    # clear entry data containers
    coord_v.clear()
    feat_v.clear()
    lm_truth_v.clear()
    lm_weight_v.clear()
    ssnet_truth_v.clear()
    ssnet_class_weight_v.clear()
    ssnet_top_weight_v.clear()    
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

    TPCID = 0
    CRYOID = 0
    data = kploader.sample_data( ntriplets, nfilled, True, TPCID, CRYOID )
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
    matchtriplets.store(data["matchtriplet"][:,:3].astype(np.int32))
    matchtriplet_v.push_back(matchtriplets)

    # truth data for larmatch tasks
    lm_truth_v.push_back( data["matchtriplet"][:,3].astype(np.int32) )
    lm_weight_v.push_back( data["match_weight"].astype(np.float32) )

    # ssnet truth data
    ssnet_truth_v.push_back( data["ssnet_label"].astype(np.int32) )
    ssnet_class_weight_v.push_back( data["ssnet_class_weight"].astype(np.float32) )
    ssnet_top_weight_v.push_back( data["ssnet_top_weight"].astype(np.float32) )

    # keypoint truth data
    kp_truth_v.push_back(  data["kplabel"].astype(np.float32) )
    kp_weight_v.push_back( data["kplabel_weight"].astype(np.float32) )    

    outtree.Fill()
    if True and ientry>=9:
        # For debug
        break


outfile.Write()
    

    
