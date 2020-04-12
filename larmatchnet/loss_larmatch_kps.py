import os,sys
import torch
import torch.nn as nn

class SparseLArMatchKPSLoss(nn.Module):
    def __init__(self):
        super(SparseLArMatchKPSLoss,self).__init__()
        
    def forward(self,larmatch_pred, ssnet_pred, kplabel_pred, kpshift_pred,
                larmatch_label, ssnet_label, kp_label, kpshift_label,
                truematch_index,
                larmatch_weight, ssnet_weight, kplabel_weight,
                verbose=False):

        if verbose:
            print "[SparseLArMatchKPSLoss]"
        
        fmatchlabel = larmatch_label.type(torch.float).requires_grad_(False)
        if verbose:
            print "  truematch_index: ",truematch_index.shape," ",truematch_index.dtype
        
        # LARMATCH LABEL
        if verbose:
            print "  larmatch weight: ",larmatch_weight.shape
        bce       = torch.nn.BCEWithLogitsLoss( reduction='none' )
        loss      = (bce( larmatch_pred, fmatchlabel )*larmatch_weight).sum()

        # SSNET
        # only evalulate loss on pixels where true label
        if ssnet_label.shape[0]!=ssnet_pred.shape[0]:
            sel_ssnet_pred   = torch.index_select( ssnet_pred,   0, truematch_index )
        else:
            sel_ssnet_pred   = ssnet_pred
        if verbose:
            print "  sel_ssnet_pred: ",sel_ssnet_pred.shape
            print "  ssnet_label: ",ssnet_label.shape
            print "  ssnet_weight: ",ssnet_weight.shape
        fn_ssnet = torch.nn.CrossEntropyLoss( reduction='none' )
        ssnet_loss = (fn_ssnet( sel_ssnet_pred, ssnet_label )*ssnet_weight).sum()

        # KPLABEL
        # only evaluate on true match points
        if kp_label.shape[0]!=kplabel_pred.shape[0]:
            sel_kplabel_pred = torch.index_select( kplabel_pred, 0, truematch_index )
        else:
            sel_kplabel_pred = kplabel_pred
        if verbose:
            print "  kplabel_pred: ",sel_kplabel_pred.shape," ",sel_kplabel_pred.requires_grad
            print "  kp_label: ",kp_label.shape," ",kp_label[:10]
            print "  kp_weight: ",kplabel_weight.shape," ",kplabel_weight[:10]
        fn_kp    = torch.nn.MSELoss( reduction='none' )
        kp_loss  = (fn_kp( sel_kplabel_pred, kp_label )*kplabel_weight).sum()


        # KPSHIFT
        if verbose:
            print "  kpshift_pred: ",kpshift_pred.shape
            print "  kpshift_label: ",kpshift_label.shape
        #fn_kpshift = 
        #kpshift_loss = fn_kpshift(kpshift_pred,kpshift_label)
        #kpshift_loss[:,0] *= fmatchlabel
        #kpshift_loss[:,1] *= fmatchlabel
        #kpshift_loss[:,2] *= fmatchlabel
        #kpshift_loss = torch.clamp( kpshift_loss, 0., 1000.0 )
        #kpshift_loss = 0.1*kpshift_loss.sum()/(3.0*fmatchlabel.sum())

        if verbose:
            print " loss-larmatch: ",loss.item()
            print " loss-ssnet: ",ssnet_loss.item()
            print " loss-kplabel: ",kp_loss.item()
            #print " loss-kpshift: ",kpshift_loss.item()

        #totloss = loss+ssnet_loss+kp_loss+kpshift_loss
        totloss = loss + 0.5*ssnet_loss + 2.0*kp_loss
        
        return totloss,loss.detach().item(),0.5*ssnet_loss.detach().item(),2.0*kp_loss.detach().item(),None
