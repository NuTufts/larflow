from __future__ import print_function
import os,sys,argparse,time
#sys.path.append("/usr/local/lib/python3.8/dist-packages/")
sys.path.append(os.environ["LARFLOW_BASEDIR"]+"/larmatchnet")
from ctypes import c_int

parser = argparse.ArgumentParser(description='Run Prep LArFlow Match Data')
parser.add_argument('-o','--output',required=True,type=str,help="Filename stem for output files")
parser.add_argument('input_larmatch',nargs='+',help="Input larmatch triplet training args")

args = parser.parse_args(sys.argv[1:])

import numpy as np
from array import array
import torch
from larvoxel_dataset import larvoxelDataset

import ROOT as rt
from ROOT import std
from larcv import larcv
from larflow import larflow
rt.gStyle.SetOptStat(0)

print("args.input_larmatch: ",args.input_larmatch)

def collate_fn(batch):
    return batch

dataset = larvoxelDataset( txtfile=args.input_larmatch[0], random_access=False, voxelsize_cm=0.3 )
NENTRIES = len(dataset)

loader = torch.utils.data.DataLoader(dataset,batch_size=1,collate_fn=collate_fn)

f_v = rt.std.vector("std::string")()
f_v.push_back( args.input_larmatch[0] )
#for f in input_rootfile_v:
#    f_v.push_back( f )

kploader = larflow.keypoints.LoaderKeypointData( f_v )
kptree = kploader.load_tree()
#kploader.set_verbosity( larcv.msg.kDEBUG )
#kploader.exclude_false_triplets( False )

print("kploader.GetEntries(): ",kploader.GetEntries())

outfile = rt.TFile(args.output,"recreate")
outtree = rt.TTree("larvoxeltrainingdata","LArMatch Voxel training data")

# Run, subrun, event                                                                                                                                                                   
run    = array('i',[0])
subrun = array('i',[0])
event  = array('i',[0])

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


outtree.Branch("run", run, "run/I")
outtree.Branch("subrun", subrun, "subrun/I")
outtree.Branch("event",  event,  "event/I")
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

start = time.time()
for iiter in range(NENTRIES):

    for vec in [ coord_v, feat_v, lm_truth_v, lm_weight_v,
                 ssnet_truth_v, ssnet_weight_v,                 
                 kp_truth_v, kp_weight_v,
                 instance_truth_v, instance_map_v,
                 ancestor_truth_v, ancestor_weight_v ]:
        vec.clear()
        
    data = next(iter(loader))[0]
    kploader.load_entry(iiter)
    print("Tree entry: ",data["tree_entry"])
    print(" keys: ",data.keys())
    for name,d in data.items():
        if type(d) is np.ndarray:
            print("  ",name,": ",d.shape)
        else:
            print("  ",name,": ",type(d))

    print("wtf: ",data["voxcoord"].shape)
    
    run[0]    = kploader.run()
    subrun[0] = kploader.subrun()
    event[0]  = kploader.event()
    
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
    #if iiter>=4:
    #    break
    
    
outfile.Write()
print("Done")

