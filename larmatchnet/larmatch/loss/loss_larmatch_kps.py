from __future__ import print_function
import os,sys
import torch
import torch.nn as nn
from .lovasz_losses import lovasz_softmax

class SparseLArMatchKPSLoss(nn.Module):
    def __init__(self, eval_lm=True,
                 eval_ssnet=False,
                 eval_keypoint_label=False,
                 eval_keypoint_shift=False,
                 eval_affinity_field=False,
                 larmatch_name="lm",
                 ssnet_name="ssnet",
                 keypoint_name="kp",
                 affinity_name="paf"):
        super(SparseLArMatchKPSLoss,self).__init__( )
        self.eval_lm = eval_lm
        self.eval_ssnet = eval_ssnet
        self.eval_keypoint_label = eval_keypoint_label
        self.eval_affinity_field = eval_affinity_field
        self.larmatch_softmax = torch.nn.Softmax( dim=1 )
        self.focal_loss_gamma = 2
        self.larmatch_use_focal_loss = True
        self.ssnet_use_lovasz_loss = True
        self.larmatch_name = larmatch_name
        self.ssnet_name = ssnet_name
        self.keypoint_name = keypoint_name
        self.affinity_name = affinity_name

    def forward( self, predictions, truthlabels, weights, batch_size, device, verbose=False  ):
        loss = {"tot":None,
                self.larmatch_name:0.0,
                self.ssnet_name:0.0,
                self.keypoint_name:0.0,
                self.affinity_name:0.0}
        
        for ib in range(batch_size):
            losses = self.forward_onebatch( predictions[ib], truthlabels[ib], weights[ib], device, verbose=verbose )
            for k in losses:
                if loss[k] is None:
                    loss[k] = losses[k]
                else:
                    loss[k] += losses[k]

        for k in loss:
            loss[k] /= float(batch_size)
            
        return loss
        
    def forward_onebatch(self, predictions, truthlabels, weights, device, verbose=False ):

        loss = {"tot":None,
                self.larmatch_name:0.0,
                self.ssnet_name:0.0,
                self.keypoint_name:0.0,
                self.affinity_name:0.0}
        
        # LARMATCH
        if self.eval_lm:
            if self.larmatch_name not in predictions:
                raise ValueError("Asked to eval larmatch loss, but prediction diction does not contain key",self.larmatch_name)
            larmatch_pred = predictions[self.larmatch_name]                        
            npairs     = larmatch_pred.shape[0]
            #ntruematch = truematch_index.shape[0]
            larmatch_weight = weights[self.larmatch_name]
            larmatch_label  = truthlabels[self.larmatch_name]        
            lm_loss = self.larmatch_loss( larmatch_pred, larmatch_label, larmatch_weight, verbose )
            if loss["tot"] is None:
                loss["tot"] = lm_loss
            else:
                loss["tot"] += lm_loss
            loss[self.larmatch_name] = lm_loss.detach().item()
                
        # # SSNET
        # if self.eval_ssnet:
        #     ssloss = self.ssnet_loss( ssnet_pred, ssnet_label, ssnet_weight, truematch_index, verbose )
        #     loss += ssloss
        #     fssloss = ssloss.detach().item()
        # else:
        #     fssloss = 0.0

        # # KPLABEL
        # if self.eval_keypoint_label:
        #     kploss = self.keypoint_loss( kplabel_pred, kp_label, kplabel_weight, truematch_index, verbose )
        #     loss += kploss
        #     fkploss = kploss.detach().item()
        # else:
        #     fkploss = 0.0

        # # KPSHIFT
        # if self.eval_keypoint_shift:
        #     shiftloss = self.keypoint_shift_loss( kpshift_pred, kpshift_label, kpshift_weight, truematch_index, verbose )
        #     loss += shiftloss
        #     fshiftloss = shiftloss.detach().item()
        # else:
        #     fshiftloss = 0.0

        # # AFFINITY FIELD
        # if self.eval_affinity_field:
        #     pafloss = self.affinity_field_loss( affinity_pred, affinity_label, affinity_weight, truematch_index, verbose )
        #     loss += pafloss
        #     fpafloss = pafloss.detach().item()
        # else:
        #     fpafloss = 0.0
        
        return loss

    def larmatch_loss( self, larmatch_pred,
                       larmatch_truth,
                       larmatch_weight,
                       verbose=False ):

        # number of spacepoint goodness predictions to evaluate
        if verbose:
            print("[SparseLArMatchKPSLoss::larmatch_loss]")
            print("  larmatch pred: ",larmatch_pred.shape)
            print("  larmatch weight: ",larmatch_weight.shape)
            
        # convert int to float for subsequent calculations
        true_lm = larmatch_truth.eq(1)
        false_lm = larmatch_truth.eq(0)

        if self.larmatch_use_focal_loss:
            # p_t for focal loss

            if verbose: print("  larmatch_pred shape: ",larmatch_pred.squeeze().shape)            
            p = torch.softmax( larmatch_pred.squeeze(), dim=0 )
            if verbose: print("  softmaxout shape: ",p.shape)                        

            p_t_true  = p[1,true_lm]
            loss_true = -larmatch_weight[true_lm]*torch.log(p_t_true+1.0e-4)*torch.pow(1-p_t_true,self.focal_loss_gamma)
            loss_true = loss_true.sum()

            p_t_false = p[0,false_lm]
            loss_false = -larmatch_weight[false_lm]*torch.log(p_t_false+1.0e-4)*torch.pow(1-p_t_false,self.focal_loss_gamma)
            loss_false = loss_false.sum()

            loss = loss_true+loss_false

            if verbose:
                print("  true-match loss: ",loss_true.detach().item())
                print("  false-match loss: ",loss_false.detach().item())            
                print("  tot focal loss: ",loss.detach().item())
            
            #print("p_t shape: ",p_t.shape)
            #p_t = fmatchlabel*p_t + (1-fmatchlabel)*(1-p_t) # p if y==1; 1-p if y==0        
            #loss = (-larmatch_weight[:npairs]*torch.log( p_t+1.0e-4 )*torch.pow( 1-p_t, self.focal_loss_gamma )).sum()
        else:
            # calculate loss using binary cross entropy
            bce       = torch.nn.BCEWithLogitsLoss( reduction='none' )
            loss      = (bce( larmatch_pred, fmatchlabel )*larmatch_weight[:npairs]).sum()
            
        if verbose:
            lm_floss = loss.detach().item()            
            print("  loss-larmatch: ",lm_floss)
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
            sel_kplabel_pred = torch.index_select( keypoint_score_pred, 0, truematch_index )
            sel_kpweight     = torch.index_select( keypoint_weight, 0, truematch_index )
            sel_kplabel      = torch.index_select( keypoint_score_truth, 0, truematch_index )
        else:
            sel_kplabel_pred = keypoint_score_pred
            sel_kpweight     = keypoint_weight[:npairs,:]
            sel_kplabel      = keypoint_score_truth[:npairs,:]
        if verbose:
            print("  keypoint_score_pred:  (sel) ",sel_kplabel_pred.shape," ",sel_kplabel_pred[:10])
            print("  keypoint_score_truth: (orig) ",keypoint_score_truth.shape," (sel) ",sel_kplabel.shape," ",sel_kplabel[:10])
            print("  keypoint_weight: (orig) ",keypoint_weight.shape," (sel)",sel_kpweight.shape," ",sel_kpweight[:10])
        fn_kp    = torch.nn.MSELoss( reduction='none' )
        fnout = fn_kp( sel_kplabel_pred, sel_kplabel )
        if verbose:
            print("  fnout shape: ",fnout.shape)
        kp_loss  = (fnout*sel_kpweight).sum()
        kp_floss = kp_loss.detach().item()
        if verbose:
            print(" loss-kplabel: ",kp_floss)

        return kp_loss

    def keypoint_shift_loss( self, keypoint_shift_pred,
                             shift_truth,
                             shift_weight,
                             truematch_index,
                             verbose=False ):
        if verbose:
            print("  kpshift_pred: ",kpshift_pred.shape)
            print("  kpshift_label: ",kpshift_label.shape)
        raise RuntimeError("dont trust this mode of calculation right now")
        #kpshift_loss = fn_kpshift(kpshift_pred,kpshift_label)
        #kpshift_loss[:,0] *= fmatchlabel
        #kpshift_loss[:,1] *= fmatchlabel
        #kpshift_loss[:,2] *= fmatchlabel
        #kpshift_loss = torch.clamp( kpshift_loss, 0., 1000.0 )
        #kpshift_loss = 0.1*kpshift_loss.sum()/(3.0*fmatchlabel.sum())

        if verbose:
            print(" loss-kpshift: ",kpshift_loss.item())
            
        return kpshift_loss

    def ssnet_loss( self, ssnet_pred,
                    ssnet_truth,
                    ssnet_weight,
                    truematch_index,
                    verbose=False):
        npairs = ssnet_pred.shape[0]
        nclasses = ssnet_pred.shape[1]
        # only evalulate loss on pixels where true label
        if ssnet_truth.shape[0]!=ssnet_pred.shape[0]:
            raise RuntimeError("dont trust this mode of calculation right now")            
            sel_ssnet_pred   = torch.index_select( ssnet_pred, 0, truematch_index )
        else:
            sel_ssnet_pred   = ssnet_pred
        if verbose:
            print("  sel_ssnet_pred: ",sel_ssnet_pred.shape)
            print("  ssnet_truth: ",ssnet_truth.shape)
            print("  ssnet_weight: ",ssnet_weight.shape)

        fn_ssnet = torch.nn.CrossEntropyLoss( reduction='none' )
        ssnet_loss = (fn_ssnet( sel_ssnet_pred, ssnet_truth )*ssnet_weight).sum()/20.0
            
        if self.ssnet_use_lovasz_loss:
            ssnet_pred_x  = torch.transpose( sel_ssnet_pred,1,0).reshape( (1,nclasses,npairs,1) )
            ssnet_truth_y = ssnet_truth.reshape( (1,npairs,1) )
            ssnet_loss += lovasz_softmax( ssnet_pred_x, ssnet_truth_y )
            
        if verbose:
            ssnet_floss = ssnet_loss.detach().item()            
            print(" loss-ssnet: ",ssnet_floss)

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

        if verbose:
            print("  affinity pred: ",sel_pred.shape," ",sel_pred[:20,:])#,torch.sum(sel_pred*sel_pred,1)[:20]
            print("  affinity truth: ",sel_truth.shape," ",torch.sum(sel_truth*sel_truth,1)[:20])
            print("  affinity weight: ",sel_weight.shape,"  ",sel_weight[:20])

        fn_mse = torch.nn.MSELoss( reduction='none' )
        fn_out = torch.sum(fn_mse( sel_pred, sel_truth ),1)
        if verbose:
            print("  affinity fn: ",fn_out.shape)
        af_loss = (fn_out*sel_weight).sum()
        if verbose:
            af_floss = af_loss.detach().item()
            print(" loss-affinity-field: ",af_floss)
        return af_loss


if __name__ == "__main__":

    # a test of the functions above

    import ROOT
    from ROOT import std
    from larflow import larflow
    from ctypes import c_int
    import numpy as np
    
    # use the loader function for KPS data
    from load_larmatch_kps import load_larmatch_kps

    # test file
    input_files = ["output_alldata.root"]
    input_v = std.vector("string")()
    for i in input_files:
        input_v.push_back(i)

    loaders = {"kps":larflow.keypoints.LoaderKeypointData( input_v ),
               "affinity":larflow.keypoints.LoaderAffinityField( input_v )}
    for name,loader in loaders.items():
        loader.exclude_false_triplets( False )
    nentries = loaders["kps"].GetEntries()
    print("num entries: ",nentries)

    device  = torch.device("cpu")    
    nmax    = c_int()
    nfilled = c_int()
    nmax.value = 50000
    batchsize = 1
    
    lossfn = SparseLArMatchKPSLoss( eval_ssnet=False,
                                    eval_keypoint_label=True,
                                    eval_keypoint_shift=False,
                                    eval_affinity_field=True )

    for ientry in xrange(0,nentries,batchsize):
        print("[LOAD ENTRY ",ientry,"]")
        data = load_larmatch_kps( loaders, ientry, batchsize,
                                  npairs=10000,
                                  exclude_neg_examples=False,
                                  verbose=True,
                                  single_batch_mode=True )
        if ientry==0:
            print("data contents:")
            for name in data:
                print("  ",name)

        # we copy the truth to make the "predictions"
        print("num positive examples: ",data["positive_indices"].shape[0])

        # larmatch
        larmatch_truth   = torch.from_numpy( data["larmatchlabels"] )
        larmatch_predict = torch.from_numpy( np.copy( larmatch_truth ) ).type(torch.float)
        lmloss = lossfn.larmatch_loss(  larmatch_predict,
                                        larmatch_truth,
                                        torch.from_numpy( data["match_weight"] ),
                                        verbose=True )
        
        # keypoint
        keypoint_truth   = torch.from_numpy( data["kplabel"] )
        keypoint_predict = torch.from_numpy( np.copy( keypoint_truth ) )
        kploss = lossfn.keypoint_loss(  keypoint_predict,
                                        keypoint_truth,
                                        torch.from_numpy( data["kplabel_weight"] ),
                                        torch.from_numpy( data["positive_indices"]),
                                        verbose=True )

        # affinity
        affinity_truth   = torch.from_numpy( data["paf_label"] )
        affinity_predict = torch.from_numpy( np.copy( affinity_truth ) )
        pafloss = lossfn.affinity_field_loss(  affinity_predict,
                                               affinity_truth,
                                               torch.from_numpy( data["paf_weight"] ),
                                               torch.from_numpy( data["positive_indices"]),
                                               verbose=True )

        
        break
    
