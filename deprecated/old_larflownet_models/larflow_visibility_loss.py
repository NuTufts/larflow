import os,sys,time
from array import array

import ROOT as rt
from larcv import larcv

import numpy as np

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F


class LArFlowVisibilityLoss(nn.Module):
    def __init__(self,nonvisi_weight=1.0e-2):
        super(LArFlowVisibilityLoss,self).__init__()
        
        # visibility parameters
        self.classweight_t    = torch.ones( 2, dtype=torch.float )
        self.classweight_t[0] = nonvisi_weight
        self.softmax          = nn.LogSoftmax(dim=1)
        self.crossentropy     = nn.NLLLoss2d( reduce=False )


    def forward(self,visi_predict,visi_truth):

        # VISIBILITY/MATCHABILITY LOSS
        # ----------------------------
        # cross-entropy loss for two classes

        sm = self.softmax(visi_predict)
        visi_loss = self.crossentropy( visi_predict, visi_truth ) # weighted already

        # reweight by class
        visi_loss[:,0,:,:] *= self.classweight_t[0]
        visi_loss[:,1,:,:] *= self.classweight_t[1]

        # weight by pixel
        fvisi = visi_truth.float().clamp(0,1).reshape( flow_predict.shape )        
        nvis = fvisi.sum()

        visi_loss *= fvisi
        visi_loss = visi_loss.sum()
        if nvis.item()>0:
            visi_loss /= visi_loss

        return visi_loss
    

