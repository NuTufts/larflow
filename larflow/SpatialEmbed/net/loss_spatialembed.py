import os,sys,time
import numpy as np
import torch
import torch.nn as nn
from lovasz_losses import lovasz_hinge

class SpatialEmbedLoss(nn.Module):
    def __init__(self, dim_nvoxels=(1,1,1), nsigma=3, nclasses=7,
                 w_embed=1.0, w_seed=1.0, w_sigma_var=1.0, sigma_scale=1.0 ):
        super(SpatialEmbedLoss,self).__init__()
        self.dim_nvoxels = np.array( dim_nvoxels, dtype=np.float32 )
        self.dim_nvoxels_t = torch.from_numpy( self.dim_nvoxels )
        self.dim_nvoxels_t.requires_grad = False
        self.foreground_weight = 1.0
        self.w_embed = w_embed
        self.w_seed  = w_seed
        self.w_sigma_var = w_sigma_var
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

        seed_pospix_count = torch.zeros(self.nclasses).to(embed_t.device) # number of total pixels used in seed count
        seed_negpix_count = torch.zeros(self.nclasses).to(embed_t.device) # number of total pixels used in seed count
        npix_posinstance  = torch.zeros(1).to(embed_t.device)        # total number of positive instances
        
        for b in range(batch_size):
            print "==== BATCH ",b," =================="
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
            if verbose or True: print "Num of pix per particle type: ",npix_particles

            # calc embeded position
            spembed = coord[:,0:3]+embed[:,0:3] # coordinate + shift
            #sigma   = torch.sigmoid(embed[:,3:3+self.nsigma]) # keep range between [0,1]
            sigma   = torch.exp(embed[:,3:3+self.nsigma]) # keep range between [0,1]
            #sigma = 0.5/(1.0e-2 + torch.pow(embed[:,3:3+self.nsigma],2))
            obj_count = 0
            probs,centroids,sigmas,masks = self.get_instance_probs(spembed,sigma,instance,verbose=verbose)

            if verbose: print "Instance Losses"
            ninstances = len(masks)
            if verbose or True: print "Number of instances: ",ninstances
            fnpix_tot  = 0.0
            bloss_instance = 0.0
            bloss_var      = 0.0
            closs          = torch.zeros( self.nclasses ).to(embed_t.device)
            batch_ninstances += ninstances
            for iid in range(ninstances):
                idmask   = masks[iid]
                prob     = probs[iid]
                centroid = centroids[iid]
                iclass   = particle[iid]
                iseed    = seed[iid]
                s        = sigmas[iid]
                fnpix    = idmask.float().sum()
                iloss    = self.bce( prob, idmask.float() ).mean()
                if verbose: print "instance[",iid,"] npixels=",idmask.sum().item()
                if verbose: print "  prob loss: ",iloss.detach().item()
                bloss_instance += iloss*fnpix
                
                # variance loss, want the values to be similar. orig author notes to find loss first
                iloss_var = torch.pow( sigma[idmask,:] - s.detach(), 2).mean()
                if verbose: print "  variance loss: ",iloss_var.detach().item()
                #iloss_var = torch.mean(torch.pow( (sigma_i - s.detach())/(s.detach()), 2)) # scale-free variance
                bloss_var += iloss_var*fnpix
                fnpix_tot += fnpix

                instance_iou = self.calculate_iou(prob.detach()>0.5, idmask.detach())
                if verbose:
                    print "  iou=",instance_iou," weight=",fnpix.item()
                    print "  npix-instance inside=",(prob[idmask].detach()>0.5).sum().float().item()/float(idmask.sum().item())
                if verbose and (~idmask).detach().sum()>0:
                    print "  outside=: ",(prob[~idmask].detach()>0.5).sum().float().item()/float((~idmask).detach().sum().item())
                    
                ave_iou += instance_iou*fnpix.item()
                ftot_pixels += fnpix.item()

                # add to seed loss for pixels inside index of certain class
                for c in range(self.nclasses):
                    cmask = iclass.eq(c+1)
                    if cmask.sum()>0:
                        closs[c] += torch.pow( prob[cmask]-iseed[cmask,c], 2 ).sum()

            if fnpix_tot>0:
                bloss_instance /= fnpix_tot
                bloss_var /= fnpix_tot

            _loss_instance += bloss_instance
            _loss_var      += bloss_var

            # class losses: get contribution from the negative examples -- should be zero
            print "class losses"
            closs_tot = 0.0
            for c in range(self.nclasses):
                cmask = particle.eq(c+1)
                closs_neg = torch.pow( seed[~cmask,c], 2).sum()
                closs[c] += closs_neg
                closs[c] /= float(particle.shape[0])
                print "  class[",c,"]: ",closs[c].detach().item()
                closs_tot += closs[c]
            _loss_seed += closs_tot/float(self.nclasses)

        _loss_instance *= self.w_embed/float(batch_size)
        _loss_var      *= self.w_sigma_var/float(batch_size)
        _loss_seed     *= self.w_seed/float(batch_size)

        loss = _loss_instance + _loss_var + _loss_seed
        ave_iou /= ftot_pixels
        return loss,batch_ninstances,ave_iou,(_loss_instance.detach().item(),_loss_seed.detach().item(),_loss_var.detach().item())    

        
    def forward_old(self, coord_t, embed_t, seed_t, instance_t, class_t, verbose=False, calc_iou=None):
        with torch.no_grad():
            batch_size = coord_t[:,3].max()+1

            fcoord_t = coord_t.to(torch.float32)
            fcoord_t[:,0] /= self.dim_nvoxels_t[0]
            fcoord_t[:,1] /= self.dim_nvoxels_t[1]
            fcoord_t[:,2] /= self.dim_nvoxels_t[2]

        ave_iou = 0.
        batch_ninstances = 0

        _loss_var      = torch.zeros(1).to(embed_t.device) # contribution from sigma-variation, all batches
        _loss_seed     = torch.zeros(1).to(embed_t.device) # contribution from seed map, all batches, all classes
        _loss_seed_pos = torch.zeros(self.nclasses).to(embed_t.device) # contribution from seed map, all batches, positive examples
        _loss_seed_neg = torch.zeros(self.nclasses).to(embed_t.device) # contribution from seed map, all batches, negative examples
        _loss_instance = torch.zeros(1).to(embed_t.device) # contribution from proper clustering

        seed_pospix_count = torch.zeros(self.nclasses).to(embed_t.device) # number of total pixels used in seed count
        seed_negpix_count = torch.zeros(self.nclasses).to(embed_t.device) # number of total pixels used in seed count
        npix_posinstance  = torch.zeros(1).to(embed_t.device)        # total number of positive instances
        
        for b in range(batch_size):
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

            num_instances = instance.max()
            if verbose: print "num instances: ",num_instances
            npix_particles = []
            for c in range(self.nclasses):
                npix_class = particle.eq(c+1).sum()
                npix_particles.append(npix_class)
            if verbose or True: print "Num of pix per particle: ",npix_particles

            # calc embeded position
            spembed = coord[:,0:3]+embed[:,0:3] # coordinate + shift
            #sigma   = torch.sigmoid(embed[:,3:3+self.nsigma]) # keep range between [0,1]
            sigma   = torch.exp(embed[:,3:3+self.nsigma]) # keep range between [0,1]
            obj_count = 0
            
            for i in range(1,num_instances+1):
                if verbose: print "== BATCH[",b,"]-INSTANCE[",i,"] ================"
                idmask = instance.detach().eq(i)
                pid    = particle[idmask][0]  # get the class for this instance
                fpos_i = idmask.float().sum() # total pixels in this instance (for weighting)
                if verbose: print "  idmask: ",idmask.shape," particle-class=",pid.item()
                coord_i = coord[idmask,:]
                sigma_i = sigma[idmask,:]
                seed_i  = seed[idmask,pid-1] # class indices are 1-indexed
                spembed_i = spembed[idmask,:]
                if verbose: print "  instance coord.shape: ",coord_i.shape
                if verbose: print "  sigma_i: ",sigma_i.shape
                if verbose: print "  seed_i: ",seed_i.shape
                if verbose or True: print "  embed: ",torch.sqrt( torch.pow( embed[idmask,0:3].detach(), 2 ) ).mean().item()
                

                # add to pixel total
                npix_posinstance += fpos_i

                # increment number of instances evaluated
                obj_count += 1                
                
                # mean
                s = 1.0e-6+sigma_i.mean(0).view(1,3) # 1 dimensions
                if verbose or True: print "  mean(ln(0.5/sigma^2): ",s.detach().cpu().numpy().tolist()

                # calculate instance centroid
                center_i = spembed_i.mean(0).view(1,3)
                if verbose: print "  centroid: ",center_i

                # variance loss, want the values to be similar. orig author notes to find loss first
                iloss_var = torch.mean(torch.pow( sigma_i - s.detach(), 2))
                #iloss_var = torch.mean(torch.pow( (sigma_i - s.detach())/(s.detach()), 2)) # scale-free variance
                _loss_var += fpos_i*iloss_var
                if verbose: print "  instance margin variance loss: ",iloss_var.detach().item()

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
                gaus = torch.exp(-self.sigma_scale*torch.sum(dist*s,1))
                if verbose: print "  dist: [",dist[idmask].detach().min(),",",dist[idmask].detach().max(),"] mean=",dist[idmask].detach().mean(),"]"
                if verbose: print "  gaus: ",gaus.shape
                if verbose: print "  ave instance dist and gaus: ",dist.detach()[idmask].mean()," ",gaus.detach()[idmask].mean()
                if verbose: print "  ave not-instance dist and gaus: ",dist.detach()[~idmask].mean()," ",gaus.detach()[~idmask].mean()
                if verbose: print "  min=",gaus.detach().min().item()," max=",gaus.detach().max().item()," inf=",torch.isinf(gaus.detach()).sum().item()

                # calculating loss for this instance, balancing weight of pos and neg pixels
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
                    loss_i = loss_i.sum()
                    
                #loss_i = lovasz_hinge( gaus*2-1, idmask.long() )
                if verbose: print "  instance loss-weighted: ",loss_i.detach().item()
                # contribute to total loss of batch, weighted by size of instance
                _loss_instance += fpos_i*loss_i
                
                # L2 loss for gaussian prediction
                if verbose: print "  seed_i [min,max]=[",torch.min(seed_i.detach()),",",torch.max(seed_i.detach()),"]"

                # gaussian score for true instance pixels (detached)
                gaus_pos = torch.exp(-self.sigma_scale*torch.sum( (dist.detach()[idmask])*s.detach(), 1 )) # (n,)
                if verbose:
                    print "  gaus_pos: ",gaus_pos.detach().shape," mean=",gaus_pos.detach().mean().item(),
                    print " range=[",gaus_pos.detach().min().item(),",",gaus_pos.detach().max().item(),"]"

                dist_s = torch.pow(seed_i-gaus_pos.detach(), 2) # (n,)
                if verbose:
                    print "  dist_s: ",dist_s.detach().shape," mean=",dist_s.detach().mean().item(),
                    print " range=[",dist_s.detach().min().item(),",",dist_s.detach().max().item(),"]"
                loss_s = self.foreground_weight*dist_s.sum() # (1,)
                if verbose: print "  loss_s(instance) = ",dist_s.detach().mean().item()

                # add to seed loss for class
                _loss_seed_pos[pid-1] += loss_s
                seed_pospix_count[pid-1] += fpos_i

                if calc_iou:
                    instance_iou = self.calculate_iou(gaus.detach()>0.5, idmask.detach())
                    if verbose:
                        print "  IOU: ",instance_iou," weight=",fpos_i.item()
                    ave_iou += instance_iou*fpos_i.item()
                if verbose or True:
                    #print "  npix-instance inside margin: ",(gaus[idmask].detach()>0.5).sum().item()," of ",gaus[idmask].detach().shape[0]                    
                    #print "  npix-not-instance inside margin: ",(gaus[~idmask].detach()>0.5).sum().item()," of ",gaus[~idmask].detach().shape[0]
                    print "  npix-instance inside=",(gaus[idmask].detach()>0.5).sum().float().item()/float(gaus[idmask].detach().shape[0]),
                    print "  outside=: ",(gaus[~idmask].detach()>0.5).sum().float().item()/float(gaus[~idmask].detach().shape[0])
                    print "  npix-instance=",idmask.sum().item()



            # end of instance loop
            
            # calculate loss for seed score over background pixels for each class (want to regress to near zero)
            if verbose: print "NEG-SEED EXAMPLES BATCH[",b,"]"
            for c in range(self.nclasses):
                notclassmask = ~particle.eq(c+1)
                n_seed_neg   = notclassmask.float().sum()

                if n_seed_neg>0:
                    bloss_seed_neg = torch.pow( seed[notclassmask,c], 2 ).sum() # (1,)
                    if verbose: print "class[",c+1,"] seed-neg loss: ",bloss_seed_neg.detach().item()/n_seed_neg.detach().item()," weight=",n_seed_neg
                    _loss_seed_neg[c]    += bloss_seed_neg
                    seed_negpix_count[c] += n_seed_neg
            
            batch_ninstances += obj_count

        # end of batch loop

        # normalize the different loss components

        # instance and variance loss, weighted by num of pixels in an instance
        if npix_posinstance>0:
            _loss_instance = _loss_instance/npix_posinstance
            _loss_var      = _loss_var/npix_posinstance
            if calc_iou:
                if verbose: print "ave_iou before removing weight: ",ave_iou," totweight=",npix_posinstance.item()
                ave_iou /= npix_posinstance.item()
        else:
            ave_iou = None

        # seed losses, per class want to weight by pos and neg examples
        num_nonzero_classes = (~seed_pospix_count.eq(0)).sum() # (c,)
        if verbose or True: print "number of non-zero classes: ",num_nonzero_classes
        if num_nonzero_classes>0:
            for i in range(self.nclasses):
                if seedd_pospix_count[i]>0 and particle.shape[0]>0:
                    _loss_seed += (_loss_seed_pos[i] + _loss_seed_neg[i])/float(particle.shape[0])
            _loss_seed /= num_nonzero_classes.float()
                
        loss = self.w_embed*_loss_instance + self.w_sigma_var*_loss_var + self.w_seed*_loss_seed
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


        
                
