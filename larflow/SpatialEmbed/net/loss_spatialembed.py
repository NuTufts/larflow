import os,sys,time
import numpy as np
import torch
import torch.nn as nn
from lovasz_losses import lovasz_hinge

class SpatialEmbedLoss(nn.Module):
    def __init__(self, dim_nvoxels=(1,1,1), nsigma=3, w_embed=1.0, w_seed=1.0, w_sigma_var=10.0 ):
        super(SpatialEmbedLoss,self).__init__()
        self.dim_nvoxels = np.array( dim_nvoxels, dtype=np.float32 )
        self.dim_nvoxels_t = torch.from_numpy( self.dim_nvoxels )
        self.dim_nvoxels_t.requires_grad = False
        self.foreground_weight = 1.0
        self.w_embed = w_embed
        self.w_seed  = w_seed
        self.w_sigma_var = w_sigma_var
        self.nsigma = nsigma
        self.bce = torch.nn.BCELoss(reduction='none')
        
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
        totpix_instances = 0
        loss = torch.zeros(1).to(embed_t.device) # total loss over all batches
        _loss_var = torch.zeros(1).to(embed_t.device) # contribution from sigma-variation, all batches
        _loss_seed = torch.zeros(1).to(embed_t.device) # contribution from seed map, all batches
        _loss_instance = torch.zeros(1).to(embed_t.device) # contribution from proper clustering
        
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
            #sigma   = torch.tanh(embed[:,3:3+self.nsigma])
            sigma   = torch.sigmoid(embed[:,3:3+self.nsigma]) # keep range between [0,1]

            obj_count = 0
            seed_pix_count = torch.zeros(1).to(embed_t.device)
            
            bloss_var      = torch.zeros(1).to(embed_t.device)
            bloss_instance = torch.zeros(1).to(embed_t.device)
            bloss_seed     = torch.zeros(1).to(embed_t.device)

            for i in range(1,num_instances+1):
                if verbose: print "== BATCH[",b,"]-INSTANCE[",i,"] ================"
                idmask = instance.detach().eq(i)
                if idmask.sum()<10: continue
                if verbose: print "  idmask: ",idmask.shape
                coord_i = coord[idmask,:]
                sigma_i = sigma[idmask,:]
                seed_i  = seed[idmask,0]
                spembed_i = spembed[idmask,:]
                if verbose: print "  instance coord.shape: ",coord_i.shape
                if verbose: print "  sigma_i: ",sigma_i.shape
                if verbose: print "  seed_i: ",seed_i.shape
                
                # mean
                s = 1.0e-5+sigma_i.mean(0).view(1,3) # 1 dimensions
                if verbose: print "  mean(ln(0.5/sigma^2): ",s

                # calculate instance centroid
                center_i = spembed_i.mean(0).view(1,3)
                if verbose: print "  centroid: ",center_i

                # variance loss, want the values to be similar. orig author notes to find loss first
                #bloss_var = bloss_var + torch.mean(torch.pow(sigma_i - s.detach(), 2))
                bloss_var = bloss_var + torch.mean(torch.pow( (sigma_i - s.detach())/(s.detach()), 2)) # scale-free variance
                if verbose: print "  sigma variance loss: ",bloss_var.detach().item()

                # gaus score from this instance centroid and sigma
                #s = torch.exp(s*10.0) # paper's activation. tends to blow up
                #s = 0.5/torch.pow(s,2) # used by SLAC
                #s = torch.clamp(s,min=0, max=2.0) # used by SLAC

                if verbose:
                    diff = spembed[:,0:3].detach()-center_i.detach()
                    diff[:,0] *= self.dim_nvoxels_t[0]
                    diff[:,1] *= self.dim_nvoxels_t[1]
                    diff[:,2] *= self.dim_nvoxels_t[2]
                    print "  ave and max diff[0]: ",diff[:,0].mean()," ",diff[:,0].max()
                    print "  ave and max diff[1]: ",diff[:,1].mean()," ",diff[:,1].max()
                    print "  ave and max diff[2]: ",diff[:,2].mean()," ",diff[:,2].max()
                dist = torch.pow(spembed - center_i, 2)
                gaus = torch.exp(-1.0e3*torch.sum(dist*s,1))
                if verbose: print "  dist: [",dist[idmask].detach().min(),",",dist[idmask].detach().max(),"] mean=",dist[idmask].detach().mean(),"]"
                if verbose: print "  gaus: ",gaus.shape
                if verbose: print "  ave instance dist and gaus: ",dist.detach()[idmask].mean()," ",gaus.detach()[idmask].mean()
                if verbose: print "  ave not-instance dist and gaus: ",dist.detach()[~idmask].mean()," ",gaus.detach()[~idmask].mean()
                if verbose: print "  min=",gaus.detach().min().item()," max=",gaus.detach().max().item()," inf=",torch.isinf(gaus.detach()).sum().item()


                #print " idmask.float(): ",idmask.float().sum()
                loss_i = self.bce( gaus, idmask.float() )                
                if idmask.sum()!=idmask.shape[0]:
                    # there is a mix of instance and not-instance (should usually be the case)
                    n_pos = idmask.float().sum()
                    n_neg = idmask.shape[0]-n_pos
                    w_pos = 0.5/n_pos
                    w_neg = 0.5/n_neg
                    if verbose: print "  w_pos=",w_pos.item()," w_neg=",w_neg.item()," N_pos=",n_pos.item()," N_neg=",n_neg.item()
                    w_v = idmask.float()*w_pos + (1.0-idmask.float())*w_neg
                    if verbose: print "  instance loss-unweighted=",loss_i.detach().mean().item()
                    loss_i *= w_v
                    loss_i = loss_i.sum()
                else:
                    loss_i = loss_i.mean()
                    
                #loss_i = lovasz_hinge( gaus*2-1, idmask.long() )
                if verbose: print "  instance loss-weighted: ",loss_i.detach().item()
                bloss_instance = bloss_instance +  loss_i

                # L2 loss for gaussian prediction
                if verbose: print "  seed_i [min,max]=[",seed_i.detach().min().item(),",",seed_i.detach().max().item(),"]"
                if verbose: print "  gaus_i [min,max]=[",gaus[idmask].detach().min().item(),",",gaus[idmask].detach().max().item(),"]"
                # positive case
                gaus_pos = torch.exp(-1.0e3*torch.sum( (dist.detach()[idmask])*s.detach(), 1 )) # note larger sigma scale factor.
                if verbose or True:
                    print "  gaus_pos: ",gaus_pos.detach().shape," mean=",gaus_pos.detach().mean().item(),
                    print " range=[",gaus_pos.detach().min().item(),",",gaus_pos.detach().max().item(),"]"  
                dist_s = torch.pow(seed_i-gaus_pos.detach(), 2) # positive case
                if verbose or True:
                    print "  dist_s: ",dist_s.detach().shape," mean=",dist_s.detach().mean().item(),
                    print " range=[",dist_s.detach().min().item(),",",dist_s.detach().max().item(),"]"
                loss_s = self.foreground_weight*dist_s.sum()
                if verbose or True: print "  loss_s(instance) = ",dist_s.detach().mean().item()
                bloss_seed += loss_s

                if calc_iou:
                    instance_iou = self.calculate_iou(gaus.detach()>0.5, idmask.detach())
                    if verbose:
                        print "   iou: ",instance_iou
                    ave_iou += instance_iou*idmask.float().sum().item()
                    totpix_instances += idmask.float().sum()
                if verbose:
                    print "  npix-instance inside margin: ",(gaus[idmask].detach()>0.5).sum().item()," of ",gaus[idmask].detach().shape[0]                    
                    print "  npix-not-instance inside margin: ",(gaus[~idmask].detach()>0.5).sum().item()," of ",gaus[~idmask].detach().shape[0]

                obj_count += 1
                seed_pix_count += idmask.detach().sum()

            # end of instance loop
            noidmask = instance.detach().eq(0)
            n_seed_neg = noidmask.float().sum()
            n_seed_pos = seed_pix_count.float()
            if n_seed_pos>0 and n_seed_neg>0:
                w_seed_pos = 0.5/n_seed_pos
                w_seed_neg = 0.5/n_seed_neg
            elif n_seed_pos==0:
                w_seed_pos = 0.0
                w_seed_neg = 1.0/n_seed_neg
            elif n_seed_neg==0:
                w_seed_neg = 0.0
                w_seed_pos = 1.0/n_seed_pos

            if n_seed_neg>0:
                bloss_seed_neg = torch.pow( seed[noidmask,0], 2 ).sum()
            else:
                bloss_seed_neg = torch.zeros(1).to(embed_t.device)

            # normalize by number of instances
            if obj_count > 0:
                bloss_instance /= float(obj_count)
                bloss_var /= float(obj_count)
                bloss_seed /= float(obj_count)
                
            if verbose: print "batch loss_seed=",bloss_seed.detach().item()
            loss += self.w_embed * bloss_instance + self.w_seed * (w_seed_pos*bloss_seed+w_seed_neg*bloss_seed_neg) + self.w_sigma_var * bloss_var
            _loss_instance += self.w_embed * bloss_instance.detach()
            _loss_seed     += self.w_seed * (w_seed_pos * bloss_seed.detach() + w_seed_neg * bloss_seed_neg.detach() )
            _loss_var      += self.w_sigma_var * bloss_var.detach()
            #loss += self.w_embed * loss_instance + self.w_seed * loss_seed
            batch_ninstances += obj_count

        # end of batch loop

        # normalize per batch        
        loss = loss / float(batch_size)
        _loss_instance /= float(batch_size)
        _loss_seed     /= float(batch_size)
        _loss_var      /= float(batch_size)

        # ave iou
        if calc_iou:
            if totpix_instances.item()>0.0:
                ave_iou /= totpix_instances.item()
            else:
                ave_iou = None

        if verbose: print "total loss=",loss.detach().item()," ave-iou=",ave_iou

        return loss,batch_ninstances,ave_iou,(_loss_instance.detach().item(),_loss_seed.detach().item(),_loss_var.detach().item())
                
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
    
    parser = argparse.ArgumentParser("Debug loss routine")
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
    loss = SpatialEmbedLoss( dim_nvoxels=voxel_dims, nsigma=3 )
    voxdims = rt.std.vector("int")(3,0)
    for i in range(3):
        voxdims[i] = voxel_dims[i]

    for ientry in range(nentries):
        data = voxelloader.getTreeEntry(ientry)
        print("number of voxels: ",data.size())

        netin  = voxelloader.makeTrainingDataDict( data )
        netout = voxelloader.makePerfectNetOutput( data, voxdims, 3 )
        coord_t    = torch.from_numpy( netin["coord_t"] )
        instance_t = torch.from_numpy( netin["instance_t"] )        
        embed_t    = torch.from_numpy( netout["embed_t"] )
        seed_t     = torch.from_numpy( netout["seed_t"] )
        
        print("voxel entries: ",coord_t.shape)

        loss_tot,ninstancs,ave_iou,_ = loss.forward(coord_t, embed_t, seed_t, instance_t, verbose=True, calc_iou=True )
        print("loss total=",loss_tot)
        print("ave iou=",ave_iou)
    
    print("[FIN]")


        
                
