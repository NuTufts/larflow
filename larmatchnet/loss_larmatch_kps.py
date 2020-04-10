import os,sys
import torch
import torch.nn as nn

class SparseLArMatchKPSLoss(nn.Module):
    def __init__(self):
        super(SparseLArMatchKPSLoss,self).__init__()
        
    def forward(self,larmatch_pred,ssnet_pred,kplabel_pred,kpshift_pred,
                larmatch_label,ssnet_label,kp_label,kpshift_label,
                truematch_index, ssnet_weight,
                verbose=False):

        if verbose:
            print "[SparseLArMatchKPSLoss]"
        
        fmatchlabel = larmatch_label.type(torch.float).requires_grad_(False)
        if verbose:
            print "  truematch_index: ",truematch_index.shape," ",truematch_index.dtype
        
        # LARMATCH LABEL
        bce       = torch.nn.BCEWithLogitsLoss( reduction='mean' )
        loss      = bce( larmatch_pred, fmatchlabel )

        # SSNET
        # only evalulate loss on pixels where true label
        sel_ssnet_pred   = torch.index_select( ssnet_pred,   0, truematch_index )
        sel_ssnet_label  = torch.index_select( ssnet_label,  0, truematch_index )
        sel_ssnet_weight = torch.index_select( ssnet_weight, 0, truematch_index )
        if verbose:
            print "  ssnet_pred: ",sel_ssnet_pred.shape
            print "  ssnet_label: ",sel_ssnet_label.shape
            print "  ssnet_weight: ",sel_ssnet_weight.shape            
        
        fn_ssnet = torch.nn.CrossEntropyLoss( reduction='none' )
        ssnet_loss = (fn_ssnet( sel_ssnet_pred, sel_ssnet_label )*sel_ssnet_weight).sum()/sel_ssnet_weight.sum()

        # KPLABEL
        # only evaluate on true match points
        sel_kplabel_pred = torch.index_select( kplabel_pred, 0, truematch_index )
        sel_kp_label     = torch.index_select( kp_label, 0, truematch_index )
        if verbose:
            print "  kplabel_pred: ",sel_kplabel_pred.shape," ",sel_kplabel_pred.dtype
            print "  kp_label: ",sel_kp_label.shape," ",sel_kp_label.dtype
        fn_kp    = torch.nn.BCEWithLogitsLoss( reduction='mean' )
        kp_loss  = fn_kp( sel_kplabel_pred, sel_kp_label.type(torch.float) )


        # KPSHIFT
        if verbose:
            print "  kpshift_pred: ",kpshift_pred.shape
            print "  kpshift_label: ",kpshift_label.shape        
        #fn_kpshift = torch.nn.MSELoss( reduction='none' )
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
        totloss = loss + ssnet_loss+kp_loss
        
        return totloss,loss,ssnet_loss,kp_loss,None
