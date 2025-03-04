import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedFocalLoss(nn.Module):
    """
    Basic focal loss. Down weights events that are already correct.
    """
    NAME="WeightedFocalLoss"
    def __init__(self, gamma=2.0):
        super(WeightedFocalLoss,self).__init__()
        self.gamma = gamma

    def forward(self,pred_logit,target):

        # Start by first standard BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(pred_logit, target, reduction='none')
        #print('pred=',pred.shape,"   target=",target.shape)

        # calculate weights: (1) to balance classes and (2) to shift focus away from correct examples
        w_freq  = torch.ones_like(target,requires_grad=False)
        w_focal = torch.ones_like(target,requires_grad=False)
        with torch.no_grad():
            # freq weights
            pos_mask = (target == 1.0)
            neg_mask = (target == 0.0)
            npos = pos_mask.sum().to(torch.float)
            nneg = neg_mask.sum().to(torch.float)
            if npos>0:
                w_freq[pos_mask] = 1.0/npos
            if nneg>0:
                w_freq[neg_mask] = 1.0/nneg

            # focal weights
            pred_prob = torch.sigmoid(pred_logit)
            w_focal[pos_mask] = torch.pow(1.0-pred_prob[pos_mask],self.gamma)
            w_focal[neg_mask] = torch.pow(pred_prob[neg_mask],self.gamma)

        weighted_loss = (0.5*bce_loss*w_freq*w_focal).sum()
        return weighted_loss
            
    def from_config(loss_config):
        
        if "Loss" in loss_config:
            cfg = loss_config['Loss']
        else:
            cfg = loss_config

        if WeightedFocalLoss.NAME not in cfg:
            # use default gamma = 2.0
            return WeightedFocalLoss(gamma=2.0)

        gamma = cfg['gamma']

        return WeightedFocalLoss(gamma=gamma)
