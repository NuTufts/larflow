import os,sys,time
from array import array

import numpy as np

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# ROOT
import ROOT as rt
from larcv import larcv

from func_intersect_ub import IntersectUB

class LArFlow3DConsistencyLoss(nn.Module):
    def __init__(self,ncols, nrows, batchsize, intersectiondata=None, larcv_version=None, nsource_wires=3456, ntarget_wires=2400, goodrange=None):
        super(LArFlow3DConsistencyLoss,self).__init__()
        IntersectUB.load_intersection_data(intersectiondatafile=intersectiondata,larcv_version=larcv_version,nsource_wires=nsource_wires,ntarget_wires=ntarget_wires)
        IntersectUB.set_img_dims( nrows, ncols )
        if goodrange is not None:
            self.goodrange_t = torch.zeros( (ncols,nrows), dtype=torch.float )
            self.goodrange_t[goodrange[0]:goodrange[1],:] = 1.0
        else:
            self.goodrange_t = None
        
        
    def forward(self,flow1_predict,flow2_predict,fmask1,fmask2,
                source_originx, targetu_originx, targetv_originx):
                  
        """
        input
        -----
        flow[x]_predict: output prediction for flow with {x:0=Y2U, 1=Y2V}
        visi[x]_predict: output prediction for visibility with {x:0=Y2U, 1=Y2V}
        """

        # build the mask
        hasmask = False
        mask = torch.ones( flow1_predict.shape, dtype=torch.float ).to(device=flow1_predict.device)
        if fmask1 is not None:
            mask *= fmask1
            hasmask = True
        if fmask2 is not None:
            mask *= fmask2
            hasmask = True            
        if self.goodrange_t is not None:
            self.goodrange_t = self.goodrange_t.to(device=flow1_predict.device)
            for b in xrange( flow1_predict.size()[0] ):
                mask[b,:] *= self.goodrange_t
            hasmask = True            

        posyz_target1_t,posyz_target2_t = IntersectUB.apply( flow1_predict, flow2_predict, source_originx, targetu_originx, targetv_originx )

        if hasmask:
            #print posyz_target1_t.size()," vs. mask=",mask.size()            
            posyz_target1_t *= mask
            posyz_target1_t *= mask
            posyz_target2_t *= mask
            posyz_target2_t *= mask

        #print "posyz 1: ",np.argwhere( np.isnan( posyz_target1_t.detach().cpu().numpy() ) )
        #print "posyz 2: ",np.argwhere( np.isnan( posyz_target2_t.detach().cpu().numpy() ) )

        # calculate the squared difference between the points
        diff_yz = posyz_target1_t-posyz_target2_t # take diff
        l2 = diff_yz[:,0,:,:]*diff_yz[:,0,:,:] + diff_yz[:,1,:,:]*diff_yz[:,1,:,:] # square

        #print "diffyz: ",np.argwhere( np.isnan( diff_yz.detach().cpu().numpy() ) )
        #print "mask.sum: ",np.argwhere( np.isnan( mask.sum().detach().cpu().numpy() ) )
        loss = l2.sum()
        
        # loss is the mean loss per non-masked pixel
        if mask.sum()>0:
            loss = l2.sum()/mask.sum() # divide by number of non-masked pixels

        return loss
