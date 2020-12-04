import os,sys,time
import numpy as np
import torch
import torch.nn as nn
from lovasz_losses import lovasz_hinge

class SpatialEmbedLoss(nn.Module):
    def __init__(self, dim_nvoxels=(1,1,1), nsigma=3, nclasses=7,
                 w_embed=1.0, w_seed=1.0, w_sigma_var=1.0, w_discr=1.0, sigma_scale=1.0 ):
        super(SpatialEmbedLoss,self).__init__()
        self.dim_nvoxels = np.array( dim_nvoxels, dtype=np.float32 )
        self.dim_nvoxels_t = torch.from_numpy( self.dim_nvoxels )
        self.dim_nvoxels_t.requires_grad = False
        self.foreground_weight = 1.0
        self.w_embed = w_embed
        self.w_seed  = w_seed
        self.w_sigma_var = w_sigma_var
        self.w_discr = w_discr
        self.nsigma = nsigma
        self.nclasses = nclasses
        self.sigma_scale = sigma_scale
        self.bce = torch.nn.BCELoss(reduction='none')
        
        print "dim_nvoxels_t: ",self.dim_nvoxels_t

    def get_instance_centroids( self, embed_t, instance_t, ndims=3 ):
        """
        embed_t: embedding coordinates
        instance_t: instance labels
        """
        num_instances = instance_t.max()
        centroids = []
        masks = []
        for iid in range(1,num_instances+1):
            idmask = instance_t.eq(iid)
            if idmask.sum()<50.0:
                continue
            icentroid = embed_t[idmask,:ndims].mean(0)
            centroids.append(icentroid)
            masks.append(idmask)
        return centroids,masks

    def get_instance_probs( self, embed_t, sigma_t, instance_t, verbose=False, ndims=3, nsigma=3 ):
        """
        """
        centroids,masks = self.get_instance_centroids(embed_t,instance_t,ndims=ndims)
        sigmas = []
        probs = []
        for iid,(center_i,idmask) in enumerate(zip(centroids,masks)):
            sigma_i = sigma_t[idmask,:]
            if verbose or True: print "instance[",iid,"] npixs=",idmask.sum().item()
            if verbose or True: print "  centroid[",iid,"]: ",center_i.detach().cpu().numpy().tolist()
            
            # mean
            s = sigma_i.mean(0).view(1,nsigma) # 1 dimensions
            if verbose or True: print "  margin[",iid,"]: ",s.detach().cpu().numpy().tolist()

            # calculate instance centroid
            dist = torch.pow(embed_t[:,0:ndims] - center_i, 2)
            gaus = torch.exp(-self.sigma_scale*torch.sum(dist*s,1))

            sigmas.append(s)
            probs.append(gaus)
        return probs,centroids,sigmas,masks

    def calc_discr_cluster_loss(self, centroids, margins, device, verbose=False ):

        ninstances = len(centroids)
        if ninstances<2:
            return torch.zeros( 1 ).to(device)

        inter_loss = torch.zeros( margins[0].shape ).to( device )
        
        #define margin as mean_s per dimension
        s_mean = torch.zeros( margins[0].shape ).to(device)
        for s in margins:
            s_mean += s
            #s_mean += s.detach()
        s_mean /=float(ninstances)
        
        if verbose: print "s_mean: ",s_mean.detach()
        # convert s into a margin distance-x such that gaus=0.5
        # our s is related to sigma by: s*self.sigma_scale=0.5/sigma^2
        # distance that makes gaus=0.5 is
        # margin = sigma * sqrt( -2.0*ln(0.5) )
        # so: margin = sqrt( -2*ln(0.5) / 2.0*self.sigma_scale*s )
        # margin = torch.clamp( torch.sqrt( 0.69/(1.0e-2+self.sigma_scale*s_mean) ), min=1.0e-2 )
        inv_margin = torch.sqrt( self.sigma_scale*s_mean/0.69 )
        if verbose: print "inv_margin: ",inv_margin.detach().cpu().numpy().tolist()

        for i in range(ninstances):
            for j in range(i+1,ninstances):
                #pair_dist = torch.sqrt( torch.pow( centroids[i]-centroids[j], 2 ) )
                #pair_dist = torch.sqrt( torch.pow( (centroids[i]-centroids[j])*inv_margin.detach(), 2 ) ) # L2 norm
                pair_dist = torch.sqrt( torch.pow( (centroids[i]-centroids[j])*inv_margin, 2 ) ) # L2 norm
                hinge = torch.clamp( 2.0 - pair_dist, min=0.0 )
                if verbose: print " pair[",(i,j),"]-hinge: ",hinge.detach()," scaled-pair-dist=",pair_dist.detach()
                inter_loss += torch.pow( hinge, 2 ).mean()
        inter_loss /= float(ninstances)*float(ninstances-1)
        return inter_loss.sum()
                

    def forward(self, coord_t, embed_t, seed_t, instance_t, class_t, verbose=False, calc_iou=None):
        with torch.no_grad():
            batch_size = coord_t[:,3].max()+1

            fcoord_t = coord_t.to(torch.float32)
            fcoord_t[:,0] /= self.dim_nvoxels_t[0]
            fcoord_t[:,1] /= self.dim_nvoxels_t[1]
            fcoord_t[:,2] /= self.dim_nvoxels_t[2]

        ave_iou = 0.
        batch_ninstances = 0
        ftot_pixels = 0.0

        _loss_var      = torch.zeros(1).to(embed_t.device) # contribution from sigma-variation, all batches
        _loss_seed     = torch.zeros(1).to(embed_t.device) # contribution from seed map, all batches, all classes
        _loss_seed_pos = torch.zeros(self.nclasses).to(embed_t.device) # contribution from seed map, all batches, positive examples
        _loss_seed_neg = torch.zeros(self.nclasses).to(embed_t.device) # contribution from seed map, all batches, negative examples
        _loss_instance = torch.zeros(1).to(embed_t.device) # contribution from proper clustering
        _loss_discr    = torch.zeros(1).to(embed_t.device) # contribution to push centroids apart

        seed_pospix_count = torch.zeros(self.nclasses).to(embed_t.device) # number of total pixels used in seed count
        seed_negpix_count = torch.zeros(self.nclasses).to(embed_t.device) # number of total pixels used in seed count
        npix_posinstance  = torch.zeros(1).to(embed_t.device)        # total number of positive instances
        
        for b in range(batch_size):
            if verbose: print "==== BATCH ",b," =================="
            # get entries a part of current batch index
            bmask = coord_t[:,3].eq(b)
            if verbose: print "bmask: ",bmask.shape,bmask.sum()

            # get data for batch
            coord    = fcoord_t[bmask,:]
            embed    = embed_t[bmask,:]
            seed     = seed_t[bmask,:]
            instance = instance_t[bmask]
            particle = class_t[bmask]
            if verbose: print "coord: ",coord.shape

            num_instances = instance.max().detach().item()
            if verbose: print "num instances: ",num_instances
            npix_particles = []
            for c in range(self.nclasses):
                npix_class = particle.eq(c+1).sum().item()
                npix_particles.append(npix_class)
            if verbose: print "Num of pix per particle type: ",npix_particles

            # calc embeded position
            spembed = coord[:,0:3]+embed[:,0:3] # coordinate + shift
            #sigma   = torch.sigmoid(embed[:,3:3+self.nsigma]) # keep range between [0,1]
            sigma   = torch.exp(embed[:,3:3+self.nsigma]) # keep range between [0,1]
            #sigma = 0.5/(1.0e-2 + torch.pow(embed[:,3:3+self.nsigma],2))
            obj_count = 0
            probs,centroids,sigmas,masks = self.get_instance_probs(spembed,sigma,instance,verbose=verbose)

            if verbose: print "Instance Losses"
            ninstances = len(masks)
            if verbose: print "Number of instances: ",ninstances
            fnpix_tot  = 0.0
            bloss_instance = 0.0
            bloss_var      = 0.0
            closs          = torch.zeros( self.nclasses ).to(embed_t.device)
            batch_ninstances += ninstances            
            for iid in range(ninstances):
                # from get-instance-probs
                idmask   = masks[iid]
                prob     = probs[iid]
                centroid = centroids[iid]
                s        = sigmas[iid]

                # from batch tensors
                iclass   = particle[idmask]
                iseed    = seed[idmask]

                fnpix_pos = idmask.float().sum()     # number of positive examples
                fnpix_neg = (~idmask).float().sum() # number of negative examples
                iloss    = self.bce( prob, idmask.float() )
                iloss_pos = iloss[idmask].sum()
                iloss_neg = iloss[~idmask].sum()
                fw_pos = 0.0
                fw_neg = 0.0
                if fnpix_pos>10.0 and fnpix_neg>10.0:
                    fw_pos = 0.5/fnpix_pos
                    fw_neg = 0.5/fnpix_neg
                elif fnpix_pos>10.0:
                    fw_pos = 1.0/fnpix_pos
                elif fnpix_neg>10.0:
                    fw_neg = 1.0/fnpix_neg
                iloss = fw_pos*iloss_pos + fw_neg*iloss_neg
                
                if verbose: print "instance[",iid,"] npixels(pos)=",fnpix_pos.item()," npixels(neg)=",fnpix_neg.item()
                if verbose: print "  prob loss: ",iloss.detach().item(),
                if verbose and fnpix_pos>0: print " loss(pos)=",iloss_pos.detach().item()/fnpix_pos.item(),
                if verbose and fnpix_neg>0: print " loss(neg)=",iloss_neg.detach().item()/fnpix_neg.item()
                bloss_instance += iloss
                
                # variance loss, want the values to be similar. orig author notes to find loss first
                iloss_var = torch.pow( sigma[idmask,:] - s.detach(), 2).mean()
                if verbose: print "  variance loss: ",iloss_var.detach().item()
                #iloss_var = torch.mean(torch.pow( (sigma_i - s.detach())/(s.detach()), 2)) # scale-free variance
                bloss_var += iloss_var
                fnpix_tot += fnpix_pos

                instance_iou = self.calculate_iou(prob.detach()>0.5, idmask.detach())
                if verbose:
                    print "  iou=",instance_iou," weight=",fnpix_pos.item()
                    print "  npix-instance inside=",(prob[idmask].detach()>0.5).sum().float().item()/fnpix_pos.item()
                if verbose and fnpix_neg>0:
                    print "  outside=: ",(prob[~idmask].detach()>0.5).sum().float().item()/fnpix_neg.item()
                    
                ave_iou += instance_iou*fnpix_pos.item()
                ftot_pixels += fnpix_pos.item()

                # add to seed loss for pixels inside index of certain class
                iprob = prob[idmask] # prob for this instance
                for c in range(self.nclasses):
                    cmask = iclass.eq(c+1)
                    if cmask.sum()>0:
                        _loss_seed_pos[c] += torch.pow( iprob[cmask].detach()-iseed[cmask,c], 2 ).sum()
                        seed_pospix_count[c] += cmask.sum().float()
                        if verbose: print " class[",c+1,"] cmask.sum(): ",cmask.sum().item()

            _loss_instance += bloss_instance
            _loss_var      += bloss_var

            # inter-cluster discrimitative loss
            bloss_inter = self.calc_discr_cluster_loss( centroids, sigmas, embed_t.device, verbose=verbose )
            if verbose: "batch inter-cluster-loss: ",bloss_inter.detach().item()
            _loss_discr += bloss_inter

            # class losses: get contribution from the negative examples -- should be zero
            if verbose: print "neg-class losses"
            for c in range(self.nclasses):
                cmask = particle.eq(c+1)
                if (~cmask).sum()>0:
                    closs_neg = torch.pow( seed[~cmask,c], 2).sum()
                    _loss_seed_neg[c] += closs_neg 
                    seed_negpix_count[c] += (~cmask).sum().float()
                    if verbose: print "  neg-class[",c,"]: ",closs_neg.detach().item()/(~cmask).float().sum().item()," npix=",(~cmask).sum().item()
                else:
                    if verbose: print "  neg-class[",c,"]: ",0.0," npix=",(~cmask).sum().item()

        # normalize batch loss
        if batch_ninstances>0:
            _loss_instance *= self.w_embed/float(batch_ninstances)
            _loss_var      *= self.w_sigma_var/float(batch_ninstances)
            _loss_discr    *= self.w_discr/float(batch_ninstances)

        # normalize/accumulate class sum
        for c in range(self.nclasses):
            fclass_w_pos = 0.0
            fclass_w_neg = 0.0
            if seed_negpix_count[c]>0 and seed_pospix_count[c]>0:
                fclass_w_pos = 0.5/seed_pospix_count[c]
                fclass_w_neg = 0.5/seed_negpix_count[c]
            elif seed_negpix_count[c]>0:
                fclass_w_neg = 1.0/seed_negpix_count[c]
            elif seed_pospix_count[c]>0:
                fclass_w_pos = 1.0/seed_pospix_count[c]
            closs = self.w_seed*(fclass_w_pos*_loss_seed_pos[c] + fclass_w_neg*_loss_seed_neg[c])/float(self.nclasses)
            if verbose: print "class[",c,"] batch-loss: ",closs.detach().item()," npos=",seed_pospix_count[c].detach().item()," nneg=",seed_negpix_count[c].detach().item()
            _loss_seed += closs
        if verbose: print "total class batch-loss: ",_loss_seed.detach().item()

        loss = _loss_instance + _loss_var + _loss_seed + _loss_discr
        ave_iou /= ftot_pixels
        return loss,batch_ninstances,ave_iou,(_loss_instance.detach().item(),_loss_seed.detach().item(),_loss_var.detach().item(),_loss_discr.detach().item())

        
                
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
    loss = SpatialEmbedLoss( dim_nvoxels=voxel_dims, nsigma=3, sigma_scale=1.0 )
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
        class_t = torch.from_numpy( netin["class_t"] )
        embed_t    = torch.from_numpy( netout["embed_t"] )
        seed_t     = torch.from_numpy( netout["seed_t"] )

        # set sigma
        embed_t[:,3:] = 10.0
        
        print("voxel entries: ",coord_t.shape)

        loss_tot,ninstancs,ave_iou,_ = loss.forward(coord_t, embed_t, seed_t, instance_t, class_t, verbose=True, calc_iou=True )
        print("loss total=",loss_tot)
        print("ave iou=",ave_iou)
        print("loss components=",_)
        break
    
    print("[FIN]")


        
                
