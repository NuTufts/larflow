import os,sys

import numpy as np

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# taken from torch.nn.modules.loss
def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as not requiring gradients"

class LArFlowLoss:
    def __init__(self,maxdist_err, visi_weight, weight=None, size_average=True, ignore_index=-100 ):
        self.ignore_index = ignore_index
        self.reduce = False
        self.visi_weight = visi_weight
        self.maxdist_err = maxdist_err
        self.maxdist_np  = np.zeros( (1,1,1,1), dtype=np.float32 )
        self.maxdist_np[:] = self.maxdist_err*self.maxdist_err
        self.min_nvis_np = np.ones( (1,1,1,1), dtype=np.float32 )
        self.gpud = None
        self.use_cuda = False
        self.classweight_np = np.ones( 2, dtype=np.float32 )
        self.classweight_np[0] = 1.0e-2 # up weight visible pixels
        self.classweight_t = torch.from_numpy( self.classweight_np )
        #self.crossentropy = nn.BCEWithLogitsLoss(reduce=False)
        #self.crossentropy  = nn.CrossEntropyLoss( weight=self.classweight_t )
        #self.crossentropy  = nn.NLLLoss2d( weight=self.classweight_t )
        #self.crossentropy  = nn.NLLLoss2d( weight=self.classweight_t, reduce=False )
        self.crossentropy  = nn.NLLLoss2d( reduce=False )
        self.smoothl1     = nn.SmoothL1Loss( reduce=False ) # other possibility

    def cuda(self,gpuid):
        self.use_cuda = True
        self.gpuid = gpuid
        self.crossentropy.cuda(gpuid)
 
    def calc_loss(self,flow_predict,visi_predict,flow_truth,visi_truth,fvisi_truth):
        """
        flow_predict: (b,1,h,w) tensor with flow prediction
        flow_truth:   (b,1,h,w) tensor with correct flow value
        visi_predict: (b,1,h,w) tensor with visibility prediction. values between [0,1].
        visi_truth:   (b,h,w) tensor with correct visibility. values either 0 or 1 (long)
        fvisi_truth:  (b,1,h,w) tensor with correct visibility. values either 0 or 1 (float)
        """
        _assert_no_grad(flow_truth)
        _assert_no_grad(visi_truth)
        _assert_no_grad(fvisi_truth)

        if self.use_cuda:
            self.maxdist_t = torch.from_numpy( self.maxdist_np ).cuda( self.gpuid )
            self.min_nvis_t = torch.from_numpy( self.min_nvis_np ).cuda( self.gpuid )
        else:
            self.maxdist_t = torch.from_numpy( self.maxdist_np )
            self.min_nvis_t = torch.from_numpy( self.min_nvis_np )            
            
        self.maxdist_var  = torch.autograd.Variable( self.maxdist_t )
        self.min_nvis_var = torch.autograd.Variable( self.min_nvis_t )


        nvisible  = fvisi_truth.sum()
        nelements = float(fvisi_truth.numel())
        ninvis    = nelements-nvisible

        if nvisible.data[0]>0:
            w_vis     = 1.0/nvisible
        else:
            w_vis     = 0.0
        w_invis   = 1.0/ninvis

        # FLOW LOSS
        # ---------
        # Smoothed L1
        flow_err = 4.0*self.smoothl1( 0.25*flow_predict, 0.25*flow_truth )

        # mask portions
        flow_err *= fvisi_truth
        if nvisible.data[0]>0:
            flow_loss = w_vis*flow_err.sum()
        else:
            flow_loss = 0.0
        
        # VISIBILITY/MATCHABILITY LOSS
        # ----------------------------
        # cross-entropy loss for two classes
        if visi_predict is not None:
            # softmax loss per pixel
            visi_loss = self.crossentropy( visi_predict, visi_truth ) # weighted already
            # visi=1 pixels
            vis1_loss = fvisi_truth*visi_loss
            vis1_loss = vis1_loss.sum()
            # visi=0 pixels
            vis0_loss = (fvisi_truth-1.0)*-1.0*visi_loss
            vis0_loss = vis0_loss.sum()
            vistot_loss = w_vis*vis1_loss + 0.01*w_invis*vis0_loss
            #print vistot_loss, torch.mean(visi_loss)
            totloss = flow_loss + self.visi_weight*vistot_loss
        else:
            totloss = flow_loss
            vistot_loss = 0.0

        #return totloss, flow_loss, self.visi_weight*vistot_loss
        return totloss
