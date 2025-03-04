import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedBCELoss(nn.Module):
    """
    Wrapper around your standard binary cross entry loss with logits.
    Weights by positive and negative true labels.
    """

    NAME="WeightedBCELoss"

    def __init__(self):
        super(WeightedBCELoss,self).__init__()

    def forward(pred, target):

        # Calculate standard BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        #print('pred=',pred.shape,"   target=",target.shape)

        # Apply lower weight to examples with target = 0.5 (unknown)
        weights = torch.ones_like(target,requires_grad=False)
        with torch.no_grad():
            pos_mask = (target == 1.0)
            neg_mask = (target == 0.0)
            npos = pos_mask.sum().to(torch.float)
            nneg = neg_mask.sum().to(torch.float)
            if npos>0:
                weights[pos_mask] = 1.0/npos
            if nneg>0:
                weights[neg_mask] = 1.0/nneg

        
        # Apply weights and take mean
        weighted_loss = 0.5*(bce_loss * weights).sum()
        return weighted_loss

    def from_config(loss_config):
        """
        This loss doesn't have any config options
        """

        if "Loss" in loss_config:
            cfg = loss_config['Loss']
        else:
            cfg = loss_config

        return WeightedBCELoss()