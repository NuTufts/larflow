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
        w_pos   = float(larmatch_label.shape[0])/fmatchlabel.sum() # weight for pos examples
        w_neg   = float(larmatch_label.shape[0])/float(larmatch_label.shape[0]-fmatchlabel.sum()) # weight for neg examples
        w_norm  = fmatchlabel.sum()*w_pos + (float(larmatch_label.shape[0])-fmatchlabel.sum())*w_neg
        w_match = torch.ones( (larmatch_label.shape[0]), dtype=torch.float, requires_grad=False, device=larmatch_label.device )
        w_match[ larmatch_label.eq(1) ] = w_pos/w_norm
        w_match[ larmatch_label.eq(0) ] = w_neg/w_norm
        if verbose:
            print w_match[0:10], w_match.requires_grad
            print larmatch_label[0:10]
        
        bce       = torch.nn.BCEWithLogitsLoss( reduction='none' )
        loss      = (bce( larmatch_pred, fmatchlabel )*w_match).sum()

        # SSNET
        # only evalulate loss on pixels where true label
        sel_ssnet_pred   = torch.index_select( ssnet_pred,   0, truematch_index )
        sel_ssnet_label  = torch.index_select( ssnet_label,  0, truematch_index ).requires_grad_(False)
        sel_ssnet_weight = torch.index_select( ssnet_weight, 0, truematch_index ).requires_grad_(False)
        n_ss_tot    = float(sel_ssnet_label.shape[0])
        n_ss_bg     = float(sel_ssnet_label[ sel_ssnet_label==0 ].sum()) 
        n_ss_track  = float(sel_ssnet_label[ sel_ssnet_label==1 ].sum())
        n_ss_shower = float(sel_ssnet_label[ sel_ssnet_label==2 ].sum())
        w_ss_bg     = n_ss_tot/n_ss_bg if n_ss_bg else 0.0
        w_ss_track  = n_ss_tot/n_ss_track if n_ss_track else 0.0
        w_ss_shower = n_ss_tot/n_ss_shower if n_ss_shower else 0.0
        w_ss_norm   = n_ss_bg*w_ss_bg + n_ss_track*w_ss_track + n_ss_shower*w_ss_shower
        w_ssnet = torch.ones( (sel_ssnet_label.shape[0]), dtype=torch.float, requires_grad=False, device=sel_ssnet_label.device )
        w_ssnet[ sel_ssnet_label==0 ] = w_ss_bg/w_ss_norm
        w_ssnet[ sel_ssnet_label==1 ] = w_ss_track/w_ss_norm
        w_ssnet[ sel_ssnet_label==2 ] = w_ss_shower/w_ss_norm
        if verbose:
            print "  ssnet_pred: ",sel_ssnet_pred.shape," grad=",sel_ssnet_pred.requires_grad
            print "  ssnet_label: ",sel_ssnet_label.shape
            print "  ssnet_weight: ",sel_ssnet_weight.shape
            print "  NUM(bg,track,shower)=(",n_ss_bg,",",n_ss_track,",",n_ss_shower,")"
        
        
        fn_ssnet = torch.nn.CrossEntropyLoss( reduction='none' )
        ssnet_loss = (fn_ssnet( sel_ssnet_pred, sel_ssnet_label )*sel_ssnet_weight*w_ssnet).sum()

        # KPLABEL
        # only evaluate on true match points
        sel_kplabel_pred = torch.index_select( kplabel_pred, 0, truematch_index )
        sel_kp_label     = torch.index_select( kp_label, 0, truematch_index )
        n_kp_tot = float(sel_kp_label.shape[0])
        n_kp_pos = float(sel_kp_label.sum())
        n_kp_neg = n_kp_tot-n_kp_pos
        w_kp_pos = n_kp_tot/n_kp_pos if n_kp_pos>0 else 0.0
        w_kp_neg = n_kp_tot/n_kp_neg if n_kp_neg>0 else 0.0
        w_kp_norm = n_kp_pos*w_kp_pos + n_kp_neg*w_kp_neg
        w_kplabel = torch.ones( (sel_kp_label.shape[0]), dtype=torch.float, requires_grad=False, device=sel_kplabel_pred.device )
        w_kplabel[ sel_kp_label==0 ] = w_kp_neg/w_kp_norm
        w_kplabel[ sel_kp_label==1 ] = w_kp_pos/w_kp_norm
        if verbose:
            print "  kplabel_pred: ",sel_kplabel_pred.shape," ",sel_kplabel_pred.requires_grad
            print "  kp_label: ",sel_kp_label.shape," ",sel_kp_label.dtype
        fn_kp    = torch.nn.BCEWithLogitsLoss( reduction='none' )
        kp_loss  = (fn_kp( sel_kplabel_pred, sel_kp_label.type(torch.float) )*w_kplabel).sum()


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
