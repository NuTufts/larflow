import os,sys
import torch
import torch.nn as nn

class SparseLArMatchKPSLoss(nn.Module):
    def __init__(self):
        super(SparseLArMatchKPSLoss,self).__init__()
        
    def forward(self,larmatch_pred,ssnet_pred,kplabel_pred,kpshift_pred,
                larmatch_label,ssnet_label,kp_label,kpshift_label,
                ssnet_weight,
                verbose=False):

        if verbose:
            print "[SparseLArMatchKPSLoss]"
        
        fmatchlabel = larmatch_label.type(torch.float).requires_grad_(False)
        
        # LARMATCH LABEL
        weight    = torch.ones( (1,), requires_grad=False, dtype=torch.float ).to(larmatch_pred.device)
        weight[0] = float(larmatch_pred.shape[0])/float(larmatch_label.sum())
        bce       = torch.nn.BCEWithLogitsLoss( pos_weight=weight, reduction='mean' )
        loss      = bce( larmatch_pred, larmatch_label.type(torch.float) )

        # SSNET
        if verbose:
            print "  ssnet_weight: ",ssnet_weight.shape
            print "  ssnet_pred: ",ssnet_pred.shape
            print "  ssnet_label: ",ssnet_label.shape
        ssnet_weight.requires_grad_(False)
        w_ssnet  = (ssnet_weight*fmatchlabel).requires_grad_(False)
        fn_ssnet = torch.nn.CrossEntropyLoss( reduction='none' )
        ssnet_loss = (fn_ssnet( ssnet_pred, ssnet_label )*w_ssnet).sum()/w_ssnet.sum()

        # KPLABEL
        if verbose:
            print "  kplabel_pred: ",kplabel_pred.shape," ",kplabel_pred.dtype
            print "  kp_label: ",kp_label.shape," ",kp_label.dtype
        fn_kp    = torch.nn.BCEWithLogitsLoss( reduction='none' )
        kp_loss  = (fn_kp( kplabel_pred, kp_label.type(torch.float) )*fmatchlabel).sum()/(fmatchlabel.sum())

        # KPSHIFT
        if verbose:
            print "  kpshift_pred: ",kpshift_pred.shape
            print "  kpshift_label: ",kpshift_label.shape        
        fn_kpshift = torch.nn.MSELoss( reduction='none' )
        kpshift_loss = fn_kpshift(kpshift_pred,kpshift_label)
        kpshift_loss[:,0] *= fmatchlabel
        kpshift_loss[:,1] *= fmatchlabel
        kpshift_loss[:,2] *= fmatchlabel
        kpshift_loss = torch.clamp( kpshift_loss, 0., 1000.0 )
        kpshift_loss = 0.1*kpshift_loss.sum()/(3.0*fmatchlabel.sum())

        if verbose:
            print " loss-larmatch: ",loss.item()
            print " loss-ssnet: ",ssnet_loss.item()
            print " loss-kplabel: ",kp_loss.item()
            print " loss-kpshift: ",kpshift_loss.item()

        totloss = loss+ssnet_loss+kp_loss+kpshift_loss
        
        return totloss,loss,ssnet_loss,kp_loss,kpshift_loss
