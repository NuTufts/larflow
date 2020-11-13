import os,sys,argparse


"""
This script is to test the flow of information for training.
From reading in the data file, running the forward pass,
calculating the loss, and running backward.
"""
# ARGUMENTS
parser = argparse.ArgumentParser("Test SpatialEmbedNet data flow")
parser.add_argument("input_file",type=str,help="file produced by 'prep_spatialembed.py'")
args = parser.parse_args()

# NUMPY/TORCH
import numpy as np
import torch
import torch.nn as nn

# DATA IO
import ROOT as rt
from larlite import larlite
from larcv import larcv
from ublarcvapp import ublarcvapp
from larflow import larflow
larcv.SetPyUtil()

# NETWORK/LOSS
from spatialembednet import SpatialEmbedNet
from loss_spatialembed import SpatialEmbedLoss

device = torch.device("cpu")

# LOAD NET and LOSS
voxel_dims = (2048, 1024, 4096)
net = SpatialEmbedNet(3, voxel_dims,
                      input_nfeatures=3,
                      nclasses=1,
                      stem_nfeatures=16).to(device)

criterion = SpatialEmbedLoss( dim_nvoxels=voxel_dims ).to(device)

# LOAD TREES
infile = rt.TFile(args.input_file)
io = infile.Get("s3dembed")
nentries = io.GetEntries()
print("NENTRIES: ",nentries)

voxelloader = larflow.spatialembed.Prep3DSpatialEmbed()
voxelloader.loadTreeBranches( io )

    
for ientry in range(nentries):

    # get entry data (numpy arrays)
    data = voxelloader.getTreeEntryDataAsArray(ientry)
    print("voxel entries: ",data["coord_t"].shape)

    # convert into torch tensors
    coord_t    = torch.from_numpy( data["coord_t"] ).to(device)
    feat_t     = torch.from_numpy( data["feat_t"] ).to(device)
    instance_t = torch.from_numpy( data["instance_t"] ).to(device)
    coord_t.requires_grad = False
    feat_t.requires_grad = False
    instance_t.requires_grad = False
    print "max(coord_t): ",(torch.max( coord_t[:,0]),torch.max(coord_t[:,1]),torch.max(coord_t[:,2]))
    print "coord_t nan inf: ",torch.isnan(coord_t).sum()," ",torch.isinf(coord_t).sum()
    print "feat_t nan inf: ",torch.isnan(feat_t).sum()," ",torch.isinf(feat_t).sum()
    print "instance_t nan: ",torch.isnan(instance_t).sum()

    del data

    # forward
    embed_t,seed_t = net( coord_t, feat_t, verbose=True )

    # loss
    loss,ninstances,iou_out = criterion( coord_t, embed_t, seed_t, instance_t, verbose=True, calc_iou=True )
    
    break
print("[FIN]")

