from __future__ import print_function
import os,sys
import torch
import torch.nn as nn
from .lovasz_losses import lovasz_softmax

class SparseLArMatchKPSLoss(nn.Module):
    def __init__(self, eval_lm=True,
                 eval_ssnet=True,
                 learnable_weights=True,
                 eval_keypoint_label=False,
                 eval_keypoint_shift=False,
                 eval_affinity_field=False,
                 init_lm_weight=0.0,
                 init_kp_weight=0.0,
                 init_ssnet_weight=0.0,
                 larmatch_name="lm",
                 ssnet_name="ssnet",
                 keypoint_name="kp",
                 affinity_name="paf",
                 lm_loss_type='bce'):
        super(SparseLArMatchKPSLoss,self).__init__( )
        if lm_loss_type not in ['bce','mse','focal-soft-bse']:
            raise ValueError("lm_loss_type parmater must be 'bce'=binary cross entry or 'mse'=L2 loss")
        
        self.eval_lm = eval_lm
        self.eval_ssnet = eval_ssnet
        self.eval_keypoint_label = eval_keypoint_label
        self.eval_affinity_field = eval_affinity_field
        self.larmatch_softmax = torch.nn.Softmax( dim=1 )
        self.focal_loss_gamma = 2
        self.larmatch_use_focal_loss = False
        self.larmatch_use_regression_loss = True

        self.larmatch_loss_type = lm_loss_type
        if lm_loss_type in ['bce','mse']:
            self.use_this_lm_loss = self.larmatch_loss
        elif lm_loss_type in ['focal-soft-bse']:
            self.use_this_lm_loss = self.larmatch_focal_softclassifier
        self.use_this_lm_loss = self.larmatch_loss_mse

        self.use_this_ssnet_loss = self.ssnet_loss_focal
        #self.use_this_kp_loss    = self.keypoint_loss_focal_relative_entropy
        self.use_this_kp_loss    = self.keypoint_loss # MSE loss        
            
        self.ssnet_use_lovasz_loss = False
        self.larmatch_name = larmatch_name
        self.ssnet_name = ssnet_name
        self.keypoint_name = keypoint_name
        self.affinity_name = affinity_name
        self.learnable_weights = learnable_weights
        # using multi-task adaptive weighting: arXiv:1705.07115
        # we learn a parameter s = log( sigma^2 ), which is monotonically related to the "uncertainty" in the prediction
        self.task_weights = {larmatch_name:nn.Parameter(torch.ones(1)*init_lm_weight),
                             ssnet_name:nn.Parameter( torch.ones(1)*init_ssnet_weight ),
                             keypoint_name:nn.Parameter(torch.ones(1)*init_kp_weight),
                             affinity_name:nn.Parameter(torch.zeros(1))}
        for k,w in self.task_weights.items():
            
            if k==ssnet_name and not eval_ssnet:
                continue
            elif k==keypoint_name and not eval_keypoint_label:
                continue
            elif k==affinity_name and not eval_affinity_field:
                continue
            
            if self.learnable_weights:
                w.require_grad = True
            setattr(self,k,w)
        

    def forward( self, predictions, truthlabels, weights, batch_size, device, verbose=False, whole_batch=False  ):
        loss = {"tot":None,
                self.larmatch_name:0.0,
                self.ssnet_name:0.0,
                self.keypoint_name:0.0,
                self.affinity_name:0.0}

        if verbose:
            print("type(predictions)=",type(predictions))
            
        if not whole_batch:
            for ib in range(batch_size):
                # get the batch
                batch_predicts = {}
                batch_truth = {}
                batch_weight = {}
                for k in predictions:
                    batch_predicts[k] = predictions[k][ib]
                for k in truthlabels:
                    batch_truth[k] = truthlabels[k][ib].unsqueeze(0)
                for k in weights:
                    batch_weight[k] = weights[k][ib].unsqueeze(0)
                
                losses = self.forward_onebatch( batch_predicts, batch_truth, batch_weight, device, verbose=verbose, whole_batch=whole_batch )
                for k in losses:
                    if loss[k] is None:
                        loss[k] = losses[k]
                    else:
                        loss[k] += losses[k]
            for k in loss:
                loss[k] /= float(batch_size)            
        else:
            # we need to cat the truth and weights
            whole_batch_labels = {}
            whole_batch_weights = {}
            if verbose:
                print("======== Make whole_batch truth and weights ==========")
            for k in [self.larmatch_name,self.ssnet_name,self.keypoint_name,self.affinity_name]:
                if k not in truthlabels:
                    continue
                whole_batch_labels[k]  = truthlabels[k]
                whole_batch_weights[k] = weights[k]
                if verbose:
                    print("whole_batch_labels [",k,"]: ",whole_batch_labels[k].shape)
                    print("whole_batch_weights [",k,"]: ",whole_batch_weights[k].shape)
            losses = self.forward_onebatch( predictions, whole_batch_labels, whole_batch_weights, device, verbose=verbose, whole_batch=True )
            for k in losses:
                if loss[k] is None:
                    loss[k] = losses[k]
                else:
                    loss[k] += losses[k]
        return loss
        
    def forward_onebatch(self, predictions, truthlabels, weights, device, verbose=False, whole_batch=True ):

        loss = {"tot":None,
                self.larmatch_name:0.0,
                self.ssnet_name:0.0,
                self.keypoint_name:0.0,
                self.affinity_name:0.0}
        
        # LARMATCH
        if self.eval_lm:
            if self.larmatch_name not in predictions:
                raise ValueError("Asked to eval larmatch loss, but prediction diction does not contain key",self.larmatch_name)
            if verbose:
                print("eval LARMATCH loss")            
            larmatch_pred = predictions[self.larmatch_name]                        
            npairs     = larmatch_pred.shape[0]
            #ntruematch = truematch_index.shape[0]
            larmatch_weight = weights[self.larmatch_name]
            larmatch_label  = truthlabels[self.larmatch_name]
            lm_loss = self.use_this_lm_loss( larmatch_pred, larmatch_label, larmatch_weight, verbose=verbose )

            if self.learnable_weights:
                if verbose:
                    print("  lm loss weight: ",self.task_weights[self.larmatch_name]," exp(w)=",torch.exp(-self.task_weights[self.larmatch_name].detach()))
                    print("  lm loss: ",lm_loss.detach().item())
                    print("  lm loss shape: ",lm_loss.shape)
                weighted_lm_loss = lm_loss*torch.exp(-self.task_weights[self.larmatch_name]) + 0.5*self.task_weights[self.larmatch_name]
                if loss["tot"] is None:
                    loss["tot"] = weighted_lm_loss
                else:
                    loss["tot"] += weighted_lm_loss
            else:
                if loss["tot"] is None:
                    loss["tot"] = lm_loss
                else:
                    loss["tot"] += lm_loss

            loss[self.larmatch_name] = lm_loss.detach().item()
                
        # SSNET
        if self.eval_ssnet:
            if verbose:
                print("eval SSNET loss")
            if self.ssnet_name not in predictions:
                raise ValueError("Asked to eval larmatch loss, but prediction diction does not contain key",self.larmatch_name)            
            ssnet_pred   = predictions[self.ssnet_name]
            ssnet_label  = truthlabels[self.ssnet_name]
            ssnet_weight = weights[self.ssnet_name]
            larmatch_label = truthlabels[self.larmatch_name]
            ssloss = self.use_this_ssnet_loss( ssnet_pred, ssnet_label, ssnet_weight, larmatch_label, verbose=verbose )
            if not self.learnable_weights:
                if loss["tot"] is None:
                    loss["tot"] = ssloss
                else:
                    loss["tot"] += ssloss
            else:
                if verbose:
                    print("  ssnet loss weight: ",self.task_weights[self.ssnet_name]," exp(w)=",torch.exp(-self.task_weights[self.ssnet_name].detach()))
                    print("  ssnet loss shape: ",ssloss.shape)
                    print("  task-weight shape: ",self.task_weights[self.ssnet_name].shape)
                weighted_ssloss = ssloss*torch.exp(-self.task_weights[self.ssnet_name]) + 0.5*self.task_weights[self.ssnet_name]
                if loss["tot"] is None:
                    loss["tot"] = weighted_ssloss
                else:
                    loss["tot"] += weighted_ssloss
                
            loss[self.ssnet_name] = ssloss.detach().item()

        # KPLABEL
        if self.eval_keypoint_label:
            if verbose:
                print("eval KEYPOINT loss")
            kplabel_pred = predictions[self.keypoint_name]
            kp_label     = truthlabels[self.keypoint_name]
            kp_weight    = weights[self.keypoint_name]
            kploss = self.use_this_kp_loss( kplabel_pred, kp_label, kp_weight, verbose=verbose )
            if not self.learnable_weights:
                if loss["tot"] is None:
                    loss["tot"] = kploss
                else:
                    loss["tot"] += kploss
            else:
                if verbose:
                    print("keypoint loss weight: ",self.task_weights[self.keypoint_name]," exp(-w)=",torch.exp(-self.task_weights[self.keypoint_name].detach()))
                weighted_kploss = kploss*torch.exp(-self.task_weights[self.keypoint_name]) + 0.5*self.task_weights[self.keypoint_name]
                if loss["tot"] is None:
                    loss["tot"] = weighted_kploss
                else:
                    loss["tot"] += weighted_kploss
            loss[self.keypoint_name] = kploss.detach().item()

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
            # Expect shapes: (B,C,N)
            # B = batch
            # C = output channels (2 for binary cross entryp, 1 for regression output)
            # N = number of spacepoint examples
            # prediction: (B,C,N)
            # truth  (B,1,N)
            # weight (B,1,N)
            print("[SparseLArMatchKPSLoss::larmatch_loss]")
            print("  larmatch pred: ",larmatch_pred.shape)
            print("  larmatch truth: ",larmatch_truth.shape)            
            print("  larmatch weight: ",larmatch_weight.shape)
            
        # convert int to float for subsequent calculations
        true_lm = larmatch_truth.gt(0.5)
        false_lm = larmatch_truth.lt(0.5)
        if verbose:
            print("  true_lm: ",true_lm.shape," sum=",true_lm.sum())

        if self.larmatch_loss_type=='bce':
            # BINARY CROSS ENTROPY LOSS TYPE
            if self.larmatch_use_focal_loss:
                # p_t for focal loss
                p = torch.softmax( larmatch_pred, dim=1 )
                if verbose: print("  LM softmaxout shape: ",p.shape)                        
                
                p_t_true  = p[:,1,:][true_lm.squeeze()]
                if verbose: print("  p_t_true: ",p_t_true.shape)
                loss_true = -larmatch_weight[true_lm]*torch.log(p_t_true+1.0e-4)*torch.pow(1-p_t_true,self.focal_loss_gamma)
                loss_true = loss_true.sum()
                
                p_t_false = p[:,0,:][false_lm.squeeze()]
                if verbose: print("  p_t_false: ",p_t_false.shape)
                loss_false = -larmatch_weight[false_lm]*torch.log(p_t_false+1.0e-4)*torch.pow(1-p_t_false,self.focal_loss_gamma)
                loss_false = loss_false.sum()

                loss = (loss_true+loss_false)/float(larmatch_pred.shape[0])

                if verbose:
                    print("larmatch loss, binary cross entry w/ focal loss")
                    print("  true-match loss: ",loss_true.detach().item())
                    print("  false-match loss: ",loss_false.detach().item())            
                    print("  tot focal loss: ",loss.detach().item())
            else:
                # calculate loss using standard binary cross entropy, not really used
                bce       = torch.nn.BCEWithLogitsLoss( reduction='none' )
                loss      = (bce( larmatch_pred, larmatch_truth )*larmatch_weight[:npairs]).sum()
        elif self.larmatch_loss_type=='mse':
            # regressing the score value -- used for soft labels
            fn_lm  = torch.nn.MSELoss( reduction='none' )
            fn_out = fn_lm( larmatch_pred, larmatch_truth )
            loss   = (fn_out*larmatch_weight).sum()
            
        if verbose:
            lm_floss = loss.detach().item()            
            print("  loss-larmatch: ",lm_floss)
        return loss

    def larmatch_focal_softclassifier( self, larmatch_pred,
                                       larmatch_truth,
                                       larmatch_weight,
                                       verbose=False ):

        # number of spacepoint goodness predictions to evaluate
        if verbose:
            # Expect shapes: (B,C,N)
            # B = batch
            # C = output channels (2 for binary cross entryp, 1 for regression output)
            # N = number of spacepoint examples
            # prediction: (B,C,N)
            # truth  (B,1,N)
            # weight (B,1,N)
            print("[SparseLArMatchKPSLoss::larmatch_focal_softclassifier]")
            print("  larmatch pred: ",larmatch_pred.shape)
            print("  larmatch truth: ",larmatch_truth.shape)            
            print("  larmatch weight: ",larmatch_weight.shape)
            
        # p_t for focal loss
        p = torch.softmax( larmatch_pred, dim=1 )
        if verbose: print("  LM softmaxout shape: ",p.shape)                        
                
        p_t_true  = p[:,1,:].unsqueeze(1)
        if verbose: print("  p_t_true: ",p_t_true.shape)
        loss_true = larmatch_truth*torch.log(p_t_true+1.0e-9)
                
        p_t_false = p[:,0,:].unsqueeze(1)
        if verbose: print("  p_t_false: ",p_t_false.shape)
        #loss_false = -larmatch_weight[false_lm]*torch.log(p_t_false+1.0e-4)*torch.pow(1-p_t_false,self.focal_loss_gamma)
        loss_false = (1.0-larmatch_truth)*torch.log(p_t_false+1.0e-9)

        #loss = -(torch.pow(larmatch_truth-p_t_true,self.focal_loss_gamma)*(loss_false+loss_true)*larmatch_weight).sum()/float(larmatch_pred.shape[0])
        loss = -(torch.pow(larmatch_truth-p_t_true,self.focal_loss_gamma)*(loss_false+loss_true)*larmatch_weight).sum()

        if verbose:
            print("larmatch loss, binary cross entry w/ focal loss")
            #print("  true-match loss: ",(loss_true.detach()*larmatch_weight).sum().item()/float(larmatch_pred.shape[0]))
            #print("  false-match loss: ",(loss_false.detach()*larmatch_weight).sum().item()/float(larmatch_pred.shape[0]))
            print("  true-match loss: ",(loss_true.detach()*larmatch_weight).sum().item())
            print("  false-match loss: ",(loss_false.detach()*larmatch_weight).sum().item())
            print("  tot focal loss: ",loss.detach().item())
            
        return loss

    def larmatch_loss_mse( self, larmatch_pred,
                           larmatch_truth,
                           larmatch_weight,
                           verbose=False ):

        # number of spacepoint goodness predictions to evaluate
        if verbose:
            # Expect shapes: (B,C,N)
            # B = batch
            # C = output channels (2 for binary cross entryp, 1 for regression output)
            # N = number of spacepoint examples
            # prediction: (B,C,N)
            # truth  (B,1,N)
            # weight (B,1,N)
            print("[SparseLArMatchKPSLoss::larmatch_focal_softclassifier]")
            print("  larmatch pred: ",larmatch_pred.shape)
            print("  larmatch truth: ",larmatch_truth.shape)            
            print("  larmatch weight: ",larmatch_weight.shape)
            
        # p_t for focal loss
        p = torch.softmax( larmatch_pred, dim=1 )        
        #p = larmatch_pred # no softmax applied
        if verbose:
            print("  LM softmaxout shape: ",p.shape)
            
        fn_lm_mse = torch.nn.MSELoss( reduction='none' )
        fnout = fn_lm_mse( p[:,1,:], larmatch_truth.squeeze(1) )
        #fnout = fn_lm_mse( p[:,0,:], larmatch_truth.squeeze(1) ) # non-softmax output
        
        loss = (fnout*larmatch_weight).sum()

        if verbose:
            print("larmatch MSE-loss: ",loss.detach().item())
            
        return loss
    

    def keypoint_loss( self, keypoint_score_pred,
                       keypoint_score_truth,
                       keypoint_weight,
                       verbose=False):
        npairs = keypoint_score_pred.shape[0]
        # only evaluate on true match points
        # if keypoint_score_truth.shape[0]!=keypoint_score_pred.shape[0]:
        #     # when truth and prediction have different lengths,
        #     # the truth already has removed bad points
        #     raise RuntimeError("dont trust this mode of calculation right now")
        #     sel_kplabel_pred = torch.index_select( keypoint_score_pred, 0, truematch_index )
        #     sel_kpweight     = torch.index_select( keypoint_weight, 0, truematch_index )
        #     sel_kplabel      = torch.index_select( keypoint_score_truth, 0, truematch_index )
        # else:
        if verbose:
            print("  keypoint_score_pred:  ",keypoint_score_pred.shape)
            print("  keypoint_score_truth: ",keypoint_score_truth.shape)
            print("  keypoint_weight: ",keypoint_weight.shape)
        fn_kp = torch.nn.MSELoss( reduction='none' )
        fnout = fn_kp( keypoint_score_pred, keypoint_score_truth )
        if verbose:
            print("  fnout shape: ",fnout.shape)
        #kp_loss  = (fnout*keypoint_weight).sum()/float(keypoint_score_pred.shape[0])
        kp_loss  = (fnout*keypoint_weight).sum()
        kp_floss = kp_loss.detach().item()
        if verbose:
            print(" loss-kplabel: ",kp_floss)

        return kp_loss

    def keypoint_loss_focal_relative_entropy( self, keypoint_score_pred,
                                              keypoint_score_truth,
                                              keypoint_weight,
                                              verbose=False):
        """
        keypoint_score_pred:   (B,C,N)
        keypoint_score_truth:  (B,C,N)
        keypoint_score_weight: (B,C,N)
        """
        npairs = keypoint_score_pred.shape[0]
        # only evaluate on true match points
        # if keypoint_score_truth.shape[0]!=keypoint_score_pred.shape[0]:
        #     # when truth and prediction have different lengths,
        #     # the truth already has removed bad points
        #     raise RuntimeError("dont trust this mode of calculation right now")
        #     sel_kplabel_pred = torch.index_select( keypoint_score_pred, 0, truematch_index )
        #     sel_kpweight     = torch.index_select( keypoint_weight, 0, truematch_index )
        #     sel_kplabel      = torch.index_select( keypoint_score_truth, 0, truematch_index )
        # else:
        if verbose:
            print("  keypoint_score_pred:  ",keypoint_score_pred.shape)
            print("  keypoint_score_truth: ",keypoint_score_truth.shape)
            print("  keypoint_weight: ",keypoint_weight.shape)

        loss_true  = keypoint_score_truth*torch.log(keypoint_score_pred+1.0e-9)
        loss_false = (1.0-keypoint_score_truth)*torch.log(1.0-keypoint_score_pred+1.0e-9)
        loss_focal = torch.pow( keypoint_score_truth-keypoint_score_pred, self.focal_loss_gamma )
        kp_loss = (-loss_focal*(loss_true+loss_false)*keypoint_weight).sum()
        if verbose:
            kp_floss = kp_loss.detach().item()            
            print(" loss-kplabel (focal relative entropy): ",kp_floss)
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
                    larmatch_label,
                    verbose=False):
        npairs = ssnet_pred.shape[0]
        nclasses = ssnet_pred.shape[1]
        # # only evalulate loss on pixels where true label
        # if ssnet_truth.shape[0]!=ssnet_pred.shape[0]:
        #     raise RuntimeError("dont trust this mode of calculation right now")            
        #     sel_ssnet_pred   = torch.index_select( ssnet_pred, 0, truematch_index )
        # else:
        #     sel_ssnet_pred   = ssnet_pred

        if verbose:
            print("  ssnet_pred: ",ssnet_pred.shape)
            print("  ssnet_truth: ",ssnet_truth.shape)
            print("  ssnet_weight: ",ssnet_weight.shape)

        ssnet_weight[ ssnet_truth==0 ]    = 0.0 # zero out background class which can be noisy
        ssnet_weight[ larmatch_label==0 ] = 0.0 # zero out non-true spacepoints
        fn_ssnet = torch.nn.CrossEntropyLoss( reduction='none', ignore_index=0 )
        ssnet_loss = (fn_ssnet( ssnet_pred, ssnet_truth.squeeze() )*ssnet_weight).mean()
            
        if self.ssnet_use_lovasz_loss:
            ssnet_pred_x  = torch.transpose( ssnet_pred,1,0).reshape( (1,nclasses,npairs,1) )
            ssnet_truth_y = ssnet_truth.reshape( (1,npairs,1) )
            ssnet_loss += lovasz_softmax( ssnet_pred_x, ssnet_truth_y )
            
        if verbose:
            ssnet_floss = ssnet_loss.detach().item()            
            print(" loss-ssnet: ",ssnet_floss)

        return ssnet_loss


    def ssnet_loss_focal( self, ssnet_pred,
                          ssnet_truth,
                          ssnet_weight,
                          larmatch_label,
                          verbose=False):
        """
        ssnet_pred: (B,C,N)
        ssnet_truth: (B,1,N)
        """
        npairs = ssnet_pred.shape[0]
        nclasses = ssnet_pred.shape[1]
        # # only evalulate loss on pixels where true label
        # if ssnet_truth.shape[0]!=ssnet_pred.shape[0]:
        #     raise RuntimeError("dont trust this mode of calculation right now")            
        #     sel_ssnet_pred   = torch.index_select( ssnet_pred, 0, truematch_index )
        # else:
        #     sel_ssnet_pred   = ssnet_pred

        if verbose:
            print("  ssnet_pred: ",ssnet_pred.shape)
            print("  ssnet_truth: ",ssnet_truth.shape)
            print("  ssnet_weight: ",ssnet_weight.shape)
        if len(ssnet_truth.shape)==2:
            ssnet_truth = ssnet_truth.unsqueeze(0)

        pred = torch.softmax( ssnet_pred, dim=1 )

        loss = None
        for c in range(1,ssnet_pred.shape[1]):
            c_mask = ssnet_truth[:,0,:]
            print("c mask: ",c_mask.shape," sum=",c_mask.sum())
            if c_mask.sum()==0:
                continue
            
            c_pred = pred[:,c,:]
            print("class_pred[",c,"]: ",c_pred.shape)
            c_true_pred = c_pred[ c_mask ]
            print("c_truth_pred[",c,"]: ",c_true_pred.shape)
            c_true_w    = ssnet_weight[:,0,:][c_mask ]
            print("c_true_w: ",c_true_w.shape)
            loss_entropy = -torch.log( c_true_pred+1.0e-9 )
            if verbose: print("  entropy[",c,"]: ",loss_entropy.shape," sum=",loss_entropy.detach().sum().item())
            loss_focal   = torch.pow( 1.0-c_true_pred, self.focal_loss_gamma )
            if verbose: print("  focal[",c,"]: ",loss_focal.shape," sum=",loss_focal.detach().sum().item())
            if loss is None:
                loss = (loss_entropy*loss_focal*c_true_w).sum()
            else:
                loss += (loss_entropy*loss_focal*c_true_w).sum()
            print("class[",c,"] loss: ",loss.detach().item())
        
        if verbose:
            ssnet_floss = loss.detach().item()            
            print("TOTAL loss-ssnet (focal): ",ssnet_floss)
        #print("[enter to keep going]")
        #input()
        return loss

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
    
