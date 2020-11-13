import os,sys

import os,sys
import numpy as np
import torch
import torch.nn as nn
from lovasz_losses import lovasz_hinge

class SpatialEmbedLoss(nn.Module):
    def __init__(self, dim_nvoxels=(1,1,1), w_embed=1.0, w_seed=0.1, w_sigma_var=0.01 ):
        super(SpatialEmbedLoss,self).__init__()
        self.dim_nvoxels = np.array( dim_nvoxels, dtype=np.float32 )
        self.dim_nvoxels_t = torch.from_numpy( self.dim_nvoxels )
        self.dim_nvoxels_t.requires_grad = False
        self.foreground_weight = 1.0
        
        print "dim_nvoxels_t: ",self.dim_nvoxels_t
        
    def forward(self, coord_t, embed_t, seed_t, instance_t, verbose=False, calc_iou=None):
        batch_size = coord_t[:,3].max()+1

        fcoord_t = coord_t.to(torch.float32)
        fcoord_t[:,0] /= self.dim_nvoxels_t[0]
        fcoord_t[:,1] /= self.dim_nvoxels_t[1]
        fcoord_t[:,2] /= self.dim_nvoxels_t[2]

        loss_var = 0
        loss_instance = 0
        loss_seed = 0
        ave_iou = 0.
        batch_ninstances = 0
        
        for b in range(batch_size):
            # get entries a part of current batch index
            bmask = coord_t[:,3].eq(b)
            if verbose: print "bmask: ",bmask.shape,bmask.sum()

            # get data for batch
            coord = fcoord_t[bmask,:]
            embed = embed_t[bmask,:]
            seed  = seed_t[bmask,:]
            instance = instance_t[bmask]
            if verbose: print "coord: ",coord.shape

            num_instances = instance.max()
            if verbose: print "num instances: ",num_instances

            # calc embeded position
            spembed = coord[:,0:3]+embed[:,0:3] # coordinate + shift

            obj_count = 0
            seed_pix_count = 0

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

                loss_instance = loss_instance + lovasz_hinge( gaus*2-1, idmask )

                # L2 loss for gaussian prediction
                loss_seed += self.foreground_weight*torch.sum(torch.pow(seed_i[idmask]-dist[idmask].detach(), 2))

                if calc_iou:
                    instance_iou += self.calculate_iou(gaus.detach()>0.5, idmask)
                    if verbose:
                        print "   iou: ",instance_iou
                    ave_iou += instance_iou

                obj_count += 1
                seed_pix_count += idmask.sum()

            # end of instance loop

            # normalize by number of instances
            if obj_count > 0:
                loss_instance /= float(obj_count)
                loss_var /= float(obj_count)
            if seed_pix_count>0:
                loss_seed /= float(seed_pix_count)

            loss += self.w_embed * loss_instance + self.w_seed * loss_seed + self.w_sigma_var * loss_var
            batch_ninstances += obj_count


        # normalize per batch
        loss = loss / float(b+1)

        # ave iou
        if calc_iou and instance_iou>0:
            ave_iou /= float(batch_ninstances)

        return loss,batch_ninstances,ave_iou
                
    def calculate_iou(pred, label):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | (pred == 1)).sum()
        if not union:
            return 0
        else:
            iou = intersection.item() / union.item()
        return iou
    
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


        
                
