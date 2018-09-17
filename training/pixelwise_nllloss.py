import os,sys

# torch
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F

# -------------------------------------------------------------------------
# PixelWiseNLLLoss
#
# This loss allows for a matrix to apply pixel-level weights.
# We use this to help balance out the number of classes per image.
# Also points of interest, e.g. around a vertex, are up-weighted as well.
# Modeled after NLLLoss
#
# Note!! the expected input is logsoftmax not just softmax
# -------------------------------------------------------------------------

# taken from torch.nn.modules.loss
def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as not requiring gradients"
        
class PixelWiseNLLLoss(nn.modules.loss._WeightedLoss):
    def __init__(self,weight=None, size_average=True, ignore_index=-100 ):
        super(PixelWiseNLLLoss,self).__init__(weight,size_average)
        self.ignore_index = ignore_index
        self.reduce = False
        #self.mean = torch.mean.cuda()

    def forward(self,predict,target,pixelweights):
        """
        predict: (b,c,h,w) tensor with output from logsoftmax
        target:  (b,h,w) tensor with correct class
        pixelweights: (b,h,w) tensor with weights for each pixel
        """
        _assert_no_grad(target)
        _assert_no_grad(pixelweights)
        #print "target: ",target.shape
        #print "predict: ",predict.shape
        
        # calculate loss with class weights. don't reduce
        pixelloss = F.nll_loss(predict,target, self.weight, self.size_average, self.ignore_index, self.reduce)
        #L1loss = nn.L1Loss(reduce=False)
        #pixelloss = L1loss(predict,target)
        
        # apply pixel weights, then reduce. returns mean over entire batch
        #print "ploss: ",pixelloss.shape
        #print "pweights: ",pixelweights.shape
        weightedpixelloss=pixelloss * pixelweights
        
        # note: probably need to take weight total, not just simple mean
        loss = weightedpixelloss.sum()/pixelweights.sum()
        #print "loss size",loss.type
        return loss
