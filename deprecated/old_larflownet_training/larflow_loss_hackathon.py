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
class testLArFlowLoss(nn.Module):
    def __init__(self, visi_weight, weight=None, size_average=True, ignore_index=-100 ):
	super(testLArFlowLoss, self).__init__()
	self.smoothl1   = nn.SmoothL1Loss()

    def forward(self, pred1, pred2, target1, target2):
	loss1 = self.smoothl1( pred1, target1)
	loss2 = self.smoothl1( pred2, target2)
	return loss1+loss2.cuda(loss1.device)
	

class LArFlowLoss(nn.Module):
    def __init__(self, visi_weight, weight=None, size_average=True, ignore_index=-100 ):
	super(LArFlowLoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduce = False
        self.visi_weight = visi_weight
        self.crossentropy  = nn.NLLLoss2d( reduce=False )
        self.smoothl1     = nn.SmoothL1Loss( reduce=False )

    #TODO def calc_loss(self,flow_predict,flow2_predict,visi_predict,flow_truth,flow2_truth,visi_truth,fvisi_truth,fvisi2_truth):
    def calc_loss(self,flow_predict,flow2_predict,flow_truth,flow2_truth,fvisi_truth,fvisi2_truth):
        """
        flow_predict: (b,1,h,w) tensor with flow prediction
        flow_truth:   (b,1,h,w) tensor with correct flow value
        visi_predict: (b,1,h,w) tensor with visibility prediction. values between [0,1].
        visi_truth:   (b,h,w) tensor with correct visibility. values either 0 or 1 (long)
        fvisi_truth:  (b,1,h,w) tensor with correct visibility. values either 0 or 1 (float)
        """
        _assert_no_grad(flow_truth)
        _assert_no_grad(flow2_truth)
        #_assert_no_grad(visi_truth)
        _assert_no_grad(fvisi_truth)
        _assert_no_grad(fvisi2_truth)

        nvisible  = fvisi_truth.sum()
        nelements = float(fvisi_truth.numel())
        ninvis    = nelements-nvisible
	visi_predict = None

        if nvisible.item()>0:
            w_vis     = 1.0/nvisible
        else:
            w_vis     = 0.0
        w_invis   = 1.0/ninvis

        # FLOW LOSS
        # ---------
        # Smoothed L1
        flow_err = 4.0*self.smoothl1( 0.25*flow_predict, 0.25*flow_truth )
        flow2_err = 4.0*self.smoothl1( 0.25*flow2_predict, 0.25*flow2_truth )

        # mask portions
        flow_err *= fvisi_truth
        flow2_err *= fvisi2_truth
        # if nvisible.data[0]>0:
        if nvisible.item()>0:
            flow_loss = w_vis*flow_err.sum()
        else:
            flow_loss = 0.0
        #if fvisi2_truth.sum().data[0]>0:
        if fvisi2_truth.sum().item()>0:
            flow2_loss = flow2_err.sum()/fvisi2_truth.sum()
        else:
            flow2_loss = 0.0
        
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
            totloss = flow_loss + flow2_loss + self.visi_weight*vistot_loss
        else:
            totloss = flow_loss + flow2_loss
            vistot_loss = 0.0
	
        return totloss, flow_loss, flow2_loss, self.visi_weight*vistot_loss
        #return totloss
