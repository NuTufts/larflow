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

        npairs     = larmatch_pred.shape[0]
        ntruematch = truematch_index.shape[0]
        if verbose:
            print "[SparseLArMatchKPSLoss]"
            print "  pred num triplets: ",npairs
            print "  truematch_index: ",truematch_index.shape," ",truematch_index.dtype
            
        fmatchlabel = larmatch_label.type(torch.float).requires_grad_(False)

        
        # LARMATCH LABEL
        if verbose:
            print "  larmatch weight: ",larmatch_weight.shape
        bce       = torch.nn.BCEWithLogitsLoss( reduction='none' )
        loss      = (bce( larmatch_pred, fmatchlabel[:npairs] )*larmatch_weight[:npairs]).sum()
        lm_floss = loss.detach().item()
        if verbose:
            print " loss-larmatch: ",lm_floss
                
        # SSNET
        if ssnet_pred is not None:
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
            loss += ssnet_loss
            ssnet_floss = ssnet_loss.detach().item()
            if verbose:
                print " loss-ssnet: ",ssnet_floss
        else:
            ssnet_floss = 0.0

        # KPLABEL
        if kplabel_pred is not None:
            # only evaluate on true match points
            if kp_label.shape[0]!=kplabel_pred.shape[0]:
                sel_kplabel_pred = torch.index_select( kplabel_pred, 0, truematch_index )
                sel_kpweight     = torch.index_select( kplabel_weight, 0, truematch_index )
                sel_kplabel      = torch.index_select( kp_label, 0, truematch_index )
            else:
                sel_kplabel_pred = kplabel_pred
                sel_kpweight     = kplabel_weight[:npairs]
                sel_kplabel      = kp_label[:npairs]
            if verbose:
                print "  kplabel_pred: (sel)",sel_kplabel_pred.shape," ",sel_kplabel_pred.requires_grad
                print "  kp_label:  (orig) ",kp_label.shape," (sel) ",sel_kplabel.shape," ",sel_kplabel[:10]
                print "  kp_weight: (orig) ",kplabel_weight.shape," (sel)",sel_kpweight.shape," ",sel_kpweight[:10]
            fn_kp    = torch.nn.MSELoss( reduction='none' )
            kp_loss  = (fn_kp( sel_kplabel_pred, sel_kplabel )*sel_kpweight).sum()
            loss += kp_loss
            kp_floss = kp_loss.detach().item()
            if verbose:
                print " loss-kplabel: ",kp_floss
        else:
            kp_floss = 0.0

        # KPSHIFT
        if kpshift_pred is not None:
            if verbose:
                print "  kpshift_pred: ",kpshift_pred.shape
                print "  kpshift_label: ",kpshift_label.shape

            #kpshift_loss = fn_kpshift(kpshift_pred,kpshift_label)
            #kpshift_loss[:,0] *= fmatchlabel
            #kpshift_loss[:,1] *= fmatchlabel
            #kpshift_loss[:,2] *= fmatchlabel
            #kpshift_loss = torch.clamp( kpshift_loss, 0., 1000.0 )
            #kpshift_loss = 0.1*kpshift_loss.sum()/(3.0*fmatchlabel.sum())

            if verbose:
                print " loss-kpshift: ",kpshift_loss.item()
        else:
            kpshift_floss = 0.0

        
        return loss,lm_floss,ssnet_floss,kp_floss,kpshift_floss
