import os,sys,time
import numpy as np
import torch
import torch.nn as nn
from lovasz_losses import lovasz_hinge

class SpatialEmbedLoss(nn.Module):
    def __init__(self, dim_nvoxels=(1,1,1), w_embed=1.0, w_seed=1.0, w_sigma_var=10.0 ):
        super(SpatialEmbedLoss,self).__init__()
        self.dim_nvoxels = np.array( dim_nvoxels, dtype=np.float32 )
        self.dim_nvoxels_t = torch.from_numpy( self.dim_nvoxels )
        self.dim_nvoxels_t.requires_grad = False
        self.foreground_weight = 1.0
        self.w_embed = w_embed
        self.w_seed  = w_seed
        self.w_sigma_var = w_sigma_var
        self.bce = torch.nn.BCELoss()
        
        print "dim_nvoxels_t: ",self.dim_nvoxels_t
        
    def forward(self, coord_t, embed_t, seed_t, instance_t, verbose=False, calc_iou=None):
        with torch.no_grad():
            batch_size = coord_t[:,3].max()+1

            fcoord_t = coord_t.to(torch.float32)
            fcoord_t[:,0] /= self.dim_nvoxels_t[0]
            fcoord_t[:,1] /= self.dim_nvoxels_t[1]
            fcoord_t[:,2] /= self.dim_nvoxels_t[2]

        ave_iou = 0.
        batch_ninstances = 0
        loss = 0
        _loss_var = 0.
        _loss_seed = 0.
        _loss_instance = 0.
        
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
            sigma   = torch.tanh(embed[:,3])

            obj_count = 0
            loss_var = 0
            loss_instance = 0

            seed_pix_count = 0            
            loss_seed = 0

            for i in range(1,num_instances+1):
                if verbose: print "== BATCH[",b,"]-INSTANCE[",i,"] ================"
                idmask = instance.eq(i)
                if idmask.sum()==0: continue
                if verbose: print "  idmask: ",idmask.shape
                coord_i = coord[idmask,:]
                sigma_i = sigma[idmask]
                seed_i  = seed[idmask,0]
                spembed_i = spembed[idmask,:]
                if verbose: print "  instance coord.shape: ",coord_i.shape
                if verbose: print "  sigma_i: ",sigma_i.shape
                if verbose: print "  seed_i: ",seed_i.shape
                
                # mean
                s = sigma_i.mean() # 1 dimensions
                if verbose: print "  mean(ln(0.5/sigma^2): ",s

                # calculate instance centroid
                center_i = spembed_i.mean(0).view(1,3)
                if verbose: print "  centroid: ",center_i

                # variance loss, want the values to be similar. orig author notes to find loss first
                loss_var = loss_var + torch.mean(torch.pow(sigma_i - s.detach(), 2))
                if verbose: print "  sigma variance loss: ",loss_var.detach().item()

                # gaus score from this instance centroid and sigma
                s = torch.exp(s*10.0) # tends to blow up
                #s = 0.5/torch.pow(s,2)
                #s = torch.clamp(s,min=0, max=2.0)

                if verbose:
                    diff = spembed[:,0:3].detach()-center_i.detach()
                    diff[:,0] *= self.dim_nvoxels_t[0]
                    diff[:,1] *= self.dim_nvoxels_t[1]
                    diff[:,2] *= self.dim_nvoxels_t[2]
                    print "  ave and max diff[0]: ",diff[:,0].mean()," ",diff[:,0].max()
                    print "  ave and max diff[1]: ",diff[:,1].mean()," ",diff[:,1].max()
                    print "  ave and max diff[2]: ",diff[:,2].mean()," ",diff[:,2].max()                
                dist = torch.sum(torch.pow(spembed - center_i, 2),1)
                gaus = torch.exp(-1*dist*s)
                if verbose: print "  gaus: ",dist.shape
                if verbose: print "  ave instance dist and gaus: ",dist.detach()[idmask].mean()," ",gaus.detach()[idmask].mean()
                if verbose: print "  ave not-instance dist and gaus: ",dist.detach()[~idmask].mean()," ",gaus.detach()[~idmask].mean()
                if verbose: print "  min=",gaus.detach().min().item()," max=",gaus.detach().max().item()," inf=",torch.isinf(gaus.detach()).sum().item()


                #print " idmask.float(): ",idmask.float().sum()
                loss_i = self.bce( gaus, idmask.float() )
                #loss_i = lovasz_hinge( gaus*2-1, idmask.long() )
                if verbose: print "  instance loss: ",loss_i.detach().item()
                loss_instance = loss_instance +  loss_i

                # L2 loss for gaussian prediction
                if verbose: print "  seed_i [min,max]=[",seed_i.detach().min().item(),",",seed_i.detach().max().item(),"]"
                if verbose: print "  gaus_i [min,max]=[",gaus[idmask].detach().min().item(),",",gaus[idmask].detach().max().item(),"]"
                dist_s = torch.pow(seed_i-gaus[idmask].detach(), 2)
                #dist_s = seed_i-gaus[idmask].detach()
                if verbose: print "  dist_s: ",dist_s.detach().shape," ",dist_s.detach().mean().item()
                #if verbose: print "  dist_s: ",dist_s
                loss_s = self.foreground_weight*dist_s.mean()
                if verbose: print "  loss_s = ",loss_s.detach().item()
                if verbose:
                    if idmask.sum()>0:
                        print "  seed loss: ",loss_s.detach().item()/float(idmask.sum().item())
                    else:
                        print "  seed loss: no instance?"
                loss_seed += loss_s

                if calc_iou:
                    instance_iou = self.calculate_iou(gaus.detach()>0.5, idmask.detach())
                    if verbose:
                        print "   iou: ",instance_iou
                    ave_iou += instance_iou
                if verbose:
                    print "  npix-instance inside margin: ",(gaus[idmask].detach()>0.5).sum().item()," of ",gaus[idmask].detach().shape[0]                    
                    print "  npix-not-instance inside margin: ",(gaus[~idmask].detach()>0.5).sum().item()," of ",gaus[~idmask].detach().shape[0]

                obj_count += 1
                seed_pix_count += idmask.detach().sum()

            # end of instance loop

            # normalize by number of instances
            if obj_count > 0:
                loss_instance /= float(obj_count)
                loss_var /= float(obj_count)
                loss_seed /= float(obj_count)
                
            if verbose: print "_loss_seed=",loss_seed.detach().item()
            loss += self.w_embed * loss_instance + self.w_seed * loss_seed + self.w_sigma_var * loss_var                
            _loss_instance += self.w_embed * loss_instance.detach().item()
            _loss_seed     += self.w_seed * loss_seed.detach().item()
            _loss_var      += self.w_sigma_var * loss_var.detach().item()
            #loss += self.w_embed * loss_instance + self.w_seed * loss_seed
            batch_ninstances += obj_count

        # end of batch loop

        # normalize per batch        
        loss = loss / float(batch_size)
        _loss_instance /= float(batch_size)
        _loss_seed     /= float(batch_size)
        _loss_var      /= float(batch_size)

        # ave iou
        if calc_iou and instance_iou>0:
            ave_iou /= float(batch_ninstances)
            if ave_iou>1.0:
                ave_iou = 1.0

        if verbose: print "total loss=",loss.detach().item()," ave-iou=",ave_iou

        return loss,batch_ninstances,ave_iou,(_loss_instance,_loss_seed,_loss_var)
                
    def calculate_iou(self, pred, label):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | (pred == 1)).sum()
        if not union:
            print "no union for iou"
            return 0
        else:
            iou = float(intersection.item()) / float(union.item())
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

        loss.forward(coord_t, embed_t, seed_t, instance_t, verbose=True, calc_iou=True )
        
        break
    
    print("[FIN]")


        
                
