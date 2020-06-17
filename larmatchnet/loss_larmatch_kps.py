import os,sys
import torch
import torch.nn as nn

class SparseLArMatchKPSLoss(nn.Module):
    def __init__(self):
        super(SparseLArMatchKPSLoss,self).__init__(eval_ssnet=False,
                                                   eval_keypoint_label=True,
                                                   eval_keypoint_shift=False,
                                                   eval_affinity_field=True ):
        self.eval_ssnet = eval_ssnet
        self.eval_keypoint_label = eval_keypoint_label
        self.eval_keypoint_shift = eval_keypoint_shift
        self.eval_affinity_field = eval_affinity_field
        
    def forward(self, larmatch_pred, ssnet_pred, kplabel_pred, kpshift_pred, affinity_pred,
                larmatch_label, ssnet_label, kp_label, kpshift_label, affinity_label,
                truematch_index,
                larmatch_weight, ssnet_weight, kplabel_weight, affinity_weight,
                verbose=False):

        npairs     = larmatch_pred.shape[0]
        ntruematch = truematch_index.shape[0]

        loss = self.larmatch_loss( larmatch_pred, larmatch_label, larmatch_weight, truematch_index, verbose )
                
        # SSNET
        if self.eval_ssnet:
            loss += self.ssnet_loss( ssnet_pred, ssnet_label, ssnet_weight, truematch_index, verbose )

        # KPLABEL
        if self.eval_keypoint_label:
            loss += self.keypoint_loss( kplabel_pred, kp_label, kplabel_weight, truematch_index, verbose )

        # KPSHIFT
        if self.eval_keypoint_shift:
            loss += self.keypoint_shift_loss( kpshift_pred, kpshift_label, kpshift_weight, truematch_index, verbose )

        # AFFINITY FIELD
        if self.eval_affinity_field:
            loss += self.affinity_field_loss( affinity_pred, affinity_label, affinity_weight, truematch_index, verbose )
        
        return loss

    def larmatch_loss( self, larmatch_pred,
                       larmatch_truth,
                       larmatch_weight,
                       truematch_index,
                       verbose=False ):

        # number of spacepoint goodness predictions to evaluate
        npairs     = larmatch_pred.shape[0]

        # number of entries indicating if the spacepoint proposal is good
        ntruematch = truematch_index.shape[0]
        if verbose:
            print "[SparseLArMatchKPSLoss::larmatch_loss]"
            print "  pred num triplets: ",npairs
            print "  truematch_index: ",truematch_index.shape," ",truematch_index.dtype
            print "  larmatch weight: ",larmatch_weight.shape
            
        # convert int to float for subsequent calculations
        fmatchlabel = larmatch_label[:npairs].type(torch.float).requires_grad_(False)
        
        # calculate loss using binary cross entropy
        bce       = torch.nn.BCEWithLogitsLoss( reduction='none' )
        loss      = (bce( larmatch_pred, fmatchlabel )*larmatch_weight[:npairs]).sum()
        if verbose:
            lm_floss = loss.detach().item()            
            print "  loss-larmatch: ",lm_floss
        return loss
                       

    def keypoint_loss( self, keypoint_score_pred,
                       keypoint_score_truth,
                       keypoint_weight,
                       truematch_index,
                       verbose=False):
        npairs = keypoint_score_pred.shape[0]
        # only evaluate on true match points
        if keypoint_score_truth.shape[0]!=keypoint_score_pred.shape[0]:
            # when truth and prediction have different lengths,
            # the truth already has removed bad points
            raise RuntimeError("dont trust this mode of calculation right now")
            sel_kplabel_pred = torch.index_select( keypoint_score_pred, 0, truematch_index[:npairs] )
            sel_kpweight     = torch.index_select( keypoint_weight, 0, truematch_index[:npairs] )
            sel_kplabel      = torch.index_select( keypoint_score_truth, 0, truematch_index[:npairs] )
        else:
            sel_kplabel_pred = keypoint_score_pred
            sel_kpweight     = keypoint_weight[:npairs]
            sel_kplabel      = keypoint_score_truth[:npairs]
        if verbose:
            print "  keypoint_score_pred:  (sel) ",sel_kplabel_pred.shape," ",sel_kplabel_pred.requires_grad
            print "  keypoint_score_truth: (orig) ",keypoint_score_truth.shape," (sel) ",sel_kplabel.shape," ",sel_kplabel[:10]
            print "  kp_weight: (orig) ",keypoint_weight.shape," (sel)",sel_kpweight.shape," ",sel_kpweight[:10]
        fn_kp    = torch.nn.MSELoss( reduction='none' )
        kp_loss  = (fn_kp( sel_kplabel_pred, sel_kplabel )*sel_kpweight).sum()
        kp_floss = kp_loss.detach().item()
        if verbose:
            print " loss-kplabel: ",kp_floss

        return kp_loss

    def keypoint_shift_loss( self, keypoint_shift_pred,
                             shift_truth,
                             shift_weight,
                             truematch_index,
                             verbose=False ):
        if verbose:
            print "  kpshift_pred: ",kpshift_pred.shape
            print "  kpshift_label: ",kpshift_label.shape
        raise RuntimeError("dont trust this mode of calculation right now")
        #kpshift_loss = fn_kpshift(kpshift_pred,kpshift_label)
        #kpshift_loss[:,0] *= fmatchlabel
        #kpshift_loss[:,1] *= fmatchlabel
        #kpshift_loss[:,2] *= fmatchlabel
        #kpshift_loss = torch.clamp( kpshift_loss, 0., 1000.0 )
        #kpshift_loss = 0.1*kpshift_loss.sum()/(3.0*fmatchlabel.sum())

        if verbose:
            print " loss-kpshift: ",kpshift_loss.item()
            
        return kpshift_loss

    def ssnet_loss( self, ssnet_pred,
                    ssnet_truth,
                    ssnet_weight,
                    truematch_index,
                    verbose=False):
        npairs = ssnet_pred.shape[0]
        # only evalulate loss on pixels where true label
        if ssnet_truth.shape[0]!=ssnet_pred.shape[0]:
            raise RuntimeError("dont trust this mode of calculation right now")            
            sel_ssnet_pred   = torch.index_select( ssnet_pred, 0, truematch_index )
        else:
            sel_ssnet_pred   = ssnet_pred
        if verbose:
            print "  sel_ssnet_pred: ",sel_ssnet_pred.shape
            print "  ssnet_truth: ",ssnet_truth.shape
            print "  ssnet_weight: ",ssnet_weight.shape
        fn_ssnet = torch.nn.CrossEntropyLoss( reduction='none' )
        ssnet_loss = (fn_ssnet( sel_ssnet_pred, ssnet_truth )*ssnet_weight).sum()
        if verbose:
            ssnet_floss = ssnet_loss.detach().item()            
            print " loss-ssnet: ",ssnet_floss

        return ssnet_loss
                    
    

    def affinity_field_loss( self, affinity_field_pred,
                             affinity_field_truth,
                             affinity_field_weight,
                             truematch_index,
                             verbose=False):
        npairs = affinity_field_pred.shape[0]
        if affinity_field_pred.shape[0]!=affinity_field_truth.shape[0]:
            raise RuntimeError("dont trust this mode of calculation right now")            
            sel_pred   = torch.index_select( affinity_field_pred, 0, truematch_index )
            sel_weight = torch.index_select( affinity_field_weight, 0, truematch_index )
            sel_truth  = torch.index_select( affinity_field_truth, 0, truematch_index )
        else:
            sel_pred   = affinity_field_pred
            sel_weight = affinity_field_weight
            sel_truth  = affinity_field_truth

        fn_mse = torch.nn.MSELoss( reduction='none' )
        af_loss = (torch.sum(fn_mse( sel_pred, sel_truth ),1)*sel_weight).sum()
        if verbose:
            af_floss = af_loss.detach().item()
            print " loss-affinity-field: ",af_floss
        return af_loss
