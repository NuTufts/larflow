#!/bin/env python
import os,sys,argparse,time

"""
TRAINING SCRIPT FOR 3D VOXEL SPATIAL EMBED CLUSTERING
"""

"""
This script is to test the flow of information for training.
From reading in the data file, running the forward pass,
calculating the loss, and running backward.
"""
# ARGUMENTS
parser = argparse.ArgumentParser("inference script for 3D spatial embedding network")
parser.add_argument("-i","--input-file",type=str,required=True,help="file produced by 'prep_spatialembed.py'")
parser.add_argument("-w","--weight-file",type=str,required=True,help="weight file")
parser.add_argument("-o","--output-file",type=str,required=True,help="output root file")
parser.add_argument("-b","--batchsize",type=int,default=5,help="batchsize [default: 4]")
parser.add_argument("-s","--start-entry",type=int,default=0,help="starting entry [default: 0]")
parser.add_argument("-n","--num-entries",type=int,default=-1,help="number of entries [default: -1 (all)]")
parser.add_argument("-v","--verbose",action='store_true',default=False,help="verbose operation in cluster formation")
parser.add_argument("-t","--test-perfect",action='store_true',default=False,help="Use truth to make perfect output to test inference output")

args = parser.parse_args()

# NUMPY/TORCH
import numpy as np
import torch
import torch.nn as nn
from torch import autograd

# NETWORK/LOSS
from spatialembednet import SpatialEmbedNet
from loss_spatialembed import SpatialEmbedLoss

device = torch.device("cpu")
#device = torch.device("cuda")
NET_VERBOSE = False
CHECKPOINT_MAP_LOCATIONS={"cuda:0":"cpu",
                          "cuda:1":"cpu"}


# python,numpy
import os,sys,commands
import shutil
import time
import traceback
import numpy as np

# create model, mark it to run on the device
voxel_dims = (2048, 1024, 4096)
model = SpatialEmbedNet(3, voxel_dims,
                        input_nfeatures=3,
                        nclasses=7,
                        num_unet_layers=6,
                        nsigma=3,
                        embedout_shapes=1,
                        stem_nfeatures=32,
                        smooth_inference=True).to(device)

checkpoint = torch.load( args.weight_file, map_location=CHECKPOINT_MAP_LOCATIONS ) # load weights to gpuid
model.load_state_dict( checkpoint["state_embed"] )
model.eval()

if False:
    # DUMP MODEL (for debugging)
    print model
    sys.exit(0)

# LOAD DATA FILE
print "Import ROOT"
import ROOT as rt
from larlite import larlite
from larcv import larcv
from ublarcvapp import ublarcvapp
from larflow import larflow
larcv.SetPyUtil()

input_file_v = rt.std.vector("std::string")()
input_file_v.push_back( args.input_file )
voxelloader = larflow.spatialembed.Prep3DSpatialEmbed(input_file_v)
nentries = voxelloader.getTree().GetEntries()
voxel_dims_v = rt.std.vector("int")(3)
for i in range(3): voxel_dims_v[i] = voxel_dims[i]
print("NENTRIES: ",nentries)

# output
outfile = rt.TFile( args.output_file, "recreate" )
tree = rt.TTree("se3dcluster","Clusters from Spatial Embedding 3D network")
# class to help fill output
datafiller = larflow.spatialembed.SpatialEmbed3DNetProducts()
datafiller.bindSimpleOutputVariables(tree)
datafiller.set_verbosity(1)

# Event loop
entry = args.start_entry
voxelloader.getTreeEntry(entry)
if args.num_entries>0:
    nentries = entry + args.num_entries
else:
    nentries = voxelloader.getTree().GetEntries()
    
while entry<nentries:

    if not args.test_perfect:
        start_entry = voxelloader.getCurrentEntry()        
        data = voxelloader.getTrainingDataBatch(int(args.batchsize))
        after_entry = voxelloader.getCurrentEntry()
        print("loaded entry: ",[start_entry,after_entry-1])

        # convert into torch tensors
        coord_t    = torch.from_numpy( data["coord_t"] ).to(device)
        feat_t     = torch.from_numpy( data["feat_t"] ).to(device)
        instance_t = torch.from_numpy( data["instance_t"] ).to(device)
        coord_t.requires_grad = False
        feat_t.requires_grad = False
        instance_t.requires_grad = False        
        
        # run network
        start = time.time()
        if not args.test_perfect:
            embed_v,seed_t = model( coord_t, feat_t, device, verbose=NET_VERBOSE )
        dt_forward = time.time()-start
        print("embed_v: len=",len(embed_v))
        print("seed_t: ",seed_t.shape," mean=",seed_t.mean()," min=",seed_t.min()," max=",seed_t.max())

        # apply convolutional kernal to sum and smooth seed map
        
        
    else:
        voxdata = voxelloader.getTreeEntry(entry)
        data = voxelloader.makePerfectNetOutput( voxdata, voxel_dims_v )
        print("loaded entry: ",entry)
        coord_t = torch.from_numpy( data["coord_t"] ).to(device)
        embed_v = [torch.from_numpy( data["embed_t"] ).to(device)]
        seed_t  = torch.from_numpy( data["seed_t"]  ).to(device)
        start_entry = entry
        after_entry = entry
    
    batch_clusters = model.make_clusters2( coord_t, embed_v, seed_t, verbose=args.verbose, sigma_scale=5.0, seed_threshold=0.75 )

    if not args.test_perfect:
        entry += args.batchsize
        nreturn = args.batchsize        
    else:
        entry += 1
        nreturn = 1


    if after_entry<start_entry:
        # wrapped around        
        nreturn = args.batchsize-after_entry

    for ib in range(nreturn):
        bmask = coord_t[:,3].eq(ib)
        print batch_clusters[ib][2].numpy().shape
        print batch_clusters[ib][2].numpy()
        datafiller.fillVoxelClusterID( coord_t[bmask,:].numpy(),
                                       batch_clusters[ib][0].numpy(),
                                       batch_clusters[ib][1].numpy(),
                                       batch_clusters[ib][2].numpy() )

    if False:
        break # for debug
    
    if after_entry<start_entry:
        break

outfile.Write()
print "[FIN]"
