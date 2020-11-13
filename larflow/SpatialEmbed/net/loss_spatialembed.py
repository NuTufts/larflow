import os,sys

import os,sys
import torch
import torch.nn as nn

class SpatialEmbedLoss(nn.Module):
    def __init__(self, dim_nvoxels=(1,1,1) ):
        super(SpatialEmbedLoss,self).__init__()
        self.dim_nvoxels = np.array( dim_nvoxels, dtype=np.float32 )
        self.dim_nvoxels_t = torch.from_numpy( self.dim_nvoxels )
        self.dim_nvoxels_t.requires_grad = False
        print "dim_nvoxels_t: ",self.dim_nvoxels_t
        
    def forward(self, coord_t, embed_t, seed_t, instance_t, verbose=False):
        batch_size = coord_t[:,3].max()+1

        fcoord_t = coord_t.to(torch.float32)
        fcoord_t[:,0] /= self.dim_nvoxels_t[0]
        fcoord_t[:,1] /= self.dim_nvoxels_t[1]
        fcoord_t[:,2] /= self.dim_nvoxels_t[2]

        loss_var = 0
        loss_instance = 0
        loss_seed = 0
        obj_count = 0
        
        for b in range(batch_size):
            bmask = coord_t[:,3].eq(b)
            if verbose: print "bmask: ",bmask.shape,bmask.sum()
            coord = fcoord_t[bmask,:]
            embed = embed_t[bmask,:]
            seed  = seed_t[bmask,:]
            instance = instance_t[bmask]
            if verbose: print "coord: ",coord.shape

            num_instances = instance.max()
            if verbose: print "num instances: ",num_instances

            # calc embeded position
            spembed = coord[:,0:3]+embed[:,0:3] # coordinate + shift
            

            for i in range(1,num_instances+1):
                print "INSTANCE[",i,"]================"
                idmask = instance.eq(i)
                coord_i = coord[idmask,:]
                embed_i = embed[idmask,:]
                seed_i  = seed[idmask,:]
                spembed_i = spembed[idmask,:]
                if verbose: print "  instance coord.shape: ",coord_i.shape

                # get sigmas
                sigma_i = embed_i[:,3]
                # mean
                s = sigma_i.mean() # 3 dimensions
                if verbose: print "  mean(sigma): ",s

                # calculate instance centroid
                center_i = spembed_i.mean(0).view(1,3)
                if verbose: print "  centroid: ",center_i

                # variance loss, want the values to be similar
                loss_var = loss_var + torch.mean(torch.pow(sigma_i - s.detach(), 2))
                print "  variance loss: ",loss_var

                # gaus score from this instance centroid and sigma
                s = torch.exp(s*10)
                diff = spembed-center_i
                diff[:,0] *= self.dim_nvoxels_t[0]
                diff[:,1] *= self.dim_nvoxels_t[1]
                diff[:,2] *= self.dim_nvoxels_t[2]
                print "  max diff[0]: ",diff[:,0].max()
                print "  max diff[1]: ",diff[:,1].max()
                print "  max diff[2]: ",diff[:,2].max()                
                dist = torch.sum(torch.pow(spembed - center_i, 2),1,keepdim=True)
                gaus = torch.exp(-1*dist*s)
                print "  gaus: ",dist.shape
                print "  ave instance dist and gaus: ",dist[idmask].mean()," ",gaus[idmask].mean()
                print "  ave not-instance dist and gaus: ",dist[~idmask].mean()," ",gaus[~idmask].mean()
                

if __name__=="__main__":

    import os,sys,argparse,json
    from ctypes import c_int
    from math import log

    #voxel_dims = (1109, 800, 3457)
    voxel_dims = (2048, 1024, 4096)
    
    parser = argparse.ArgumentParser("Visuzalize Voxel Data")
    parser.add_argument("input_file",type=str,help="file produced by 'prep_spatialembed.py'")
    args = parser.parse_args()
    
    import numpy as np
    import ROOT as rt
    from larlite import larlite
    from larcv import larcv
    from ublarcvapp import ublarcvapp
    from larflow import larflow
    larcv.SetPyUtil()

    # LOAD TREES
    infile = rt.TFile(args.input_file)
    io = infile.Get("s3dembed")
    nentries = io.GetEntries()
    print("NENTRIES: ",nentries)

    voxelloader = larflow.spatialembed.Prep3DSpatialEmbed()
    voxelloader.loadTreeBranches( io )

    print("make loss module")
    loss = SpatialEmbedLoss( dim_nvoxels=voxel_dims )
    voxdims = rt.std.vector("int")(3,0)
    for i in range(3):
        voxdims[i] = voxel_dims[i]

    for ientry in range(nentries):
        data = voxelloader.getTreeEntry(ientry)
        print("number of voxels: ",data.size())

        netin  = voxelloader.makeTrainingDataDict( data )
        netout = voxelloader.makePerfectNetOutput( data, voxdims )
        coord_t    = torch.from_numpy( netin["coord_t"] )
        instance_t = torch.from_numpy( netin["instance_t"] )        
        embed_t    = torch.from_numpy( netout["embed_t"] )
        seed_t     = torch.from_numpy( netout["seed_t"] )
        
        print("voxel entries: ",coord_t.shape)

        loss.forward(coord_t, embed_t, seed_t, instance_t, verbose=True )
        
        break
    
    print("[FIN]")


        
                
