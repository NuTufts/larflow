import os,sys,time
from array import array

import ROOT as rt
from larcv import larcv

import numpy as np

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

class LArFlowFlowLoss(nn.Module):
    def __init__(self,maxdist_err):
        super(LArFlowFlowLoss,self).__init__()
        # flow loss

        # define max loss
        self.maxdist_err = maxdist_err
        self.maxloss_t = torch.ones( (1,1,1,1), dtype=torch.float )*(maxdist_err*maxdist_err)
        self.smoothl1_flow = nn.SmoothL1Loss( reduce=False )

    def forward(self,flow_predict,flow_truth,visi_truth):

        # FLOW LOSS
        # ---------
        # Smoothed L1 (why the subweights?)
        flow_err = self.smoothl1_flow( flow_predict, flow_truth ).clamp(0,self.maxdist_err*self.maxdist_err)

        # number of visible
        fvis = visi_truth.float().clamp(0,1).reshape( flow_predict.shape )
        nvis = fvis.sum()
        
        # mask by the visibility: we don't want to penalize for pixels with no info
        flow_err *= fvis

        # loss
        flow_loss = flow_err.sum()
        if nvis>0:
            flow_loss /= nvis

        return flow_loss


