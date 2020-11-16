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
                        nclasses=1,
                        num_unet_layers=5,
                        stem_nfeatures=32).to(device)

checkpoint = torch.load( args.weight_file, map_location=CHECKPOINT_MAP_LOCATIONS ) # load weights to gpuid
model.load_state_dict( checkpoint["state_embed"] )
model.train()

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
print("NENTRIES: ",nentries)


# Event loop
entry = 0
while entry<nentries:

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
    embed_t,seed_t = model( coord_t, feat_t, device, verbose=NET_VERBOSE )
    dt_forward = time.time()-start
    print("embed_t: ",embed_t.shape)
    print("seed_t: ",seed_t.shape)

    model.make_clusters( coord_t, embed_t, seed_t, verbose=True )

    entry += args.batchsize

    if True:
        break # for debug
    
    if after_entry<start_entry:
        break

