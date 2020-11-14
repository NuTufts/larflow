import os,sys,argparse,time


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

# NETWORK/LOSS
from spatialembednet import SpatialEmbedNet
from loss_spatialembed import SpatialEmbedLoss

device = torch.device("cpu")
#device = torch.device("cuda")
verbose = False

# random tensor option for debugging
use_random_tensor = False
nsamples = 5
nfake_instances = 3
    
# LOAD NET and LOSS
voxel_dims = (2048, 1024, 4096)
net = SpatialEmbedNet(3, voxel_dims,
                      input_nfeatures=3,
                      nclasses=1,
                      num_unet_layers=5,
                      stem_nfeatures=32).to(device)

criterion = SpatialEmbedLoss( dim_nvoxels=voxel_dims ).to(device)

# LOAD DATA

# ROOT DATA
if not use_random_tensor:
    # DATA IO
    print "Import ROOT"
    import ROOT as rt
    from larlite import larlite
    from larcv import larcv
    from ublarcvapp import ublarcvapp
    from larflow import larflow
    larcv.SetPyUtil()
    
    infile = rt.TFile(args.input_file)
    io = infile.Get("s3dembed")
    nentries = io.GetEntries()
    print("NENTRIES: ",nentries)

    voxelloader = larflow.spatialembed.Prep3DSpatialEmbed()
    voxelloader.loadTreeBranches( io )
# RANDOM DATA
else:
    print "NO ROOT"
    coord_np = np.zeros( (nsamples,4), dtype=np.int )
    coord_np[:,0] = np.random.randint(0, high=voxel_dims[0]-1, size=nsamples, dtype=np.int)
    coord_np[:,1] = np.random.randint(0, high=voxel_dims[1]-1, size=nsamples, dtype=np.int)
    coord_np[:,2] = np.random.randint(0, high=voxel_dims[2]-1, size=nsamples, dtype=np.int)
    instance_np = np.zeros( nsamples, dtype=np.int )
    instance_np[:] = np.random.randint(0,high=nfake_instances,size=nsamples,dtype=np.int)
    feat_np = np.zeros( (nsamples,3), dtype=np.float32 )
    for i in range(3):
        feat_np[:,i] = np.random.rand(nsamples)
    random_data = {}
    random_data["coord_t"] = coord_np
    random_data["feat_t"] = feat_np
    random_data["instance_t"] = instance_np
    nentries = 1


dt_loader = 0.
dt_forward = 0.
dt_loss = 0.
nrun = 0
for ientry in range(nentries):

    start = time.time()
    if not use_random_tensor:
        # get entry data (numpy arrays)
        data = voxelloader.getTreeEntryDataAsArray(ientry)
        print("voxel entries: ",data["coord_t"].shape)
        
        # convert into torch tensors
        coord_t    = torch.from_numpy( data["coord_t"] ).to(device)
        feat_t     = torch.from_numpy( data["feat_t"] ).to(device)
        instance_t = torch.from_numpy( data["instance_t"] ).to(device)
    else:
        coord_t    = torch.from_numpy( coord_np ).to(device)
        feat_t     = torch.from_numpy( feat_np ).to(device)
        instance_t = torch.from_numpy( instance_np ).to(device)
    
    coord_t.requires_grad = False
    feat_t.requires_grad = False
    instance_t.requires_grad = False

    dt_loader += (time.time()-start)
        
    print "max(coord_t): ",(torch.max( coord_t[:,0]),torch.max(coord_t[:,1]),torch.max(coord_t[:,2]))
    print "coord_t nan inf: ",torch.isnan(coord_t).sum()," ",torch.isinf(coord_t).sum()
    print "feat_t nan inf: ",torch.isnan(feat_t).sum()," ",torch.isinf(feat_t).sum()
    print "instance_t nan: ",torch.isnan(instance_t).sum()
    
    # forward
    start = time.time()    
    embed_t,seed_t = net( coord_t, feat_t, device, verbose=verbose )
    dt_forward += time.time()-start

    # loss
    start = time.time()
    loss,ninstances,iou_out = criterion( coord_t, embed_t, seed_t, instance_t, verbose=verbose, calc_iou=True )
    dt_loss += time.time()-start

    nrun += 1
print("Loading time: ",dt_loader," sec total ",dt_loader/nrun," sec/entry")
print("Loading net-forward: ",dt_forward," sec total ",dt_forward/nrun," sec/entry")
print("Loading loss: ",dt_loss," sec total ",dt_loss/nrun," sec/entry")
print("[FIN]")

