import os,sys
import numpy as np
import torch
import MinkowskiEngine as ME
from larmatch.utils.common import prepare_me_sparsetensor, make_batch_truth, make_batch_weights

def larmatch_hard_example_mining( model, batchdata, DEVICE, return_num_samples, verbose=False, return_eval=False ):

    model.eval()
    
    batch_size = len(batchdata["matchtriplet_v"]) 
    wireplane_sparsetensors, matchtriplet_v, batch_truth_v, batch_weight_v \
        = prepare_me_sparsetensor( batchdata, DEVICE, verbose=verbose)


    # we remake the batch, prioritizing examplaes of mistakes, then sampling randomly
    sampled_matchtriplet_v = []
    sampled_lm_v  = []
    sampled_ssnet_v  = []
    sampled_kp_v     = []
    sampled_hardexample_v = []
    sampled_eval_v = []
    with torch.no_grad():
        pred_dict = model( wireplane_sparsetensors, matchtriplet_v, batch_size )
        pred = torch.softmax( pred_dict["lm"], dim=1 )
        # num correct
        for b in range(batch_size):

            # incorrect examples
            false_pos = pred[b,1,:].ge(0.5)*batch_truth_v["lm"][b,0,:].lt(0.5) # 1 at each false pos entry
            false_neg = pred[b,1,:].lt(0.5)*batch_truth_v["lm"][b,0,:].ge(0.5) # 1 at each false neg entry
            false_ex  = false_pos+false_neg # 1 at each false entry
            false_idx = torch.arange(0,pred.shape[-1]).to(DEVICE)[ false_ex ] # try to keep everying on the gpu?
            if verbose:
                print("batch[",b,"] false_pos=",false_pos.sum()," false_neg=",false_neg.sum()," tot=",false_ex.sum())
                print("max-score=",pred[b,1,:].max())
                print("min-score=",pred[b,1,:].min())

            # Prepare re-sampled data
            num_false = np.minimum( false_idx.shape[0], return_num_samples )
            num_else = return_num_samples-num_false            
            shufidx = np.arange(0,pred.shape[2])
            np.random.shuffle(shufidx)
            shufidx = torch.from_numpy(shufidx).to(DEVICE) # for filling remainder
            if verbose:
                print("hard-example mine: batch[",b,"] false frac =",float(false_ex.sum())/float(pred.shape[-1]),
                      " num false=",num_false," num else=",num_else)
            

            hard_example_flag = torch.zeros( return_num_samples, dtype=torch.int32 )
            
            # match triplet
            #print("matchtriplet_v: ",matchtriplet_v.shape)
            sampled_matchtriplet = torch.zeros( (return_num_samples, 3), dtype=torch.long ).to(DEVICE)
            sampled_matchtriplet[:num_false,:] = matchtriplet_v[b, false_idx[:num_false], : ]
            hard_example_flag[:num_false] = 1
            if num_else>0:
                sampled_matchtriplet[num_false:,:] = matchtriplet_v[b, shufidx[:num_else], : ]

            if return_eval:
                if verbose: print("returning evaluation of spacepoints")
                out_pred = torch.zeros( return_num_samples, dtype=torch.float )
                out_pred[:num_false] = pred[b,1,false_idx[:num_false]]
                if num_else>0:
                    out_pred[num_false:] = pred[b,1,shufidx[:num_else]]
                sampled_eval_v.append( out_pred )
            
            # larmatch truth
            #print("lm_truth: ",batch_truth_v["lm"].shape)
            sampled_larmatch_truth = torch.zeros( return_num_samples, dtype=torch.float ).to(DEVICE)
            sampled_larmatch_truth[:num_false] = batch_truth_v["lm"][ b, 0, false_idx[:num_false] ]
            if num_else>0:
                sampled_larmatch_truth[num_false:] = batch_truth_v["lm"][b, 0, shufidx[:num_else] ]

            # ssnet truth
            if "ssnet" in batch_truth_v:
                if verbose: print("ssnet_truth: ",batch_truth_v["ssnet"].shape)                
                sampled_ssnet_truth = torch.zeros( return_num_samples, dtype=torch.long ).to(DEVICE)
                sampled_ssnet_truth[:num_false] = batch_truth_v["ssnet"][ b, 0, false_idx[:num_false] ]
                if num_else:
                    sampled_ssnet_truth[num_false:] = batch_truth_v["ssnet"][ b, 0, shufidx[:num_else] ]

            # keypoint truth
            if "kp" in batch_truth_v:
                if verbose: print("kptruth: ",batch_truth_v["kp"].shape)                      
                sampled_kp_truth = torch.zeros( (6,return_num_samples), dtype=torch.float ).to(DEVICE)
                sampled_kp_truth[:,:num_false] = batch_truth_v["kp"][ b, :, false_idx[:num_false] ]
                if num_else:
                    sampled_kp_truth[:,num_false:] = batch_truth_v["kp"][ b, :, shufidx[:num_else] ]

            sampled_matchtriplet_v.append( sampled_matchtriplet )
            sampled_lm_v.append(  sampled_larmatch_truth )
            sampled_ssnet_v.append( sampled_ssnet_truth )
            sampled_kp_v.append( sampled_kp_truth )
            sampled_hardexample_v.append( hard_example_flag )

    batch_truth = {"matchtriplet_v":sampled_matchtriplet_v,
                   "larmatch_label":sampled_lm_v,
                   "ssnet_truth":sampled_ssnet_v,
                   "keypoint_truth":sampled_kp_v}
    out_batched_truth   = make_batch_truth( batch_truth, DEVICE, verbose )
    if verbose:
        for k in out_batched_truth:
            print("output[",k,"]: ",out_batched_truth[k].shape)
        print("LM false examples in batch: ",out_batched_truth["larmatch_label"].lt(0.5).sum())
        print("LM true examples in batch: ",out_batched_truth["larmatch_label"].ge(0.5).sum())    

    batch_truth = {"lm":out_batched_truth["larmatch_label"].to(DEVICE),
                   "ssnet":out_batched_truth["ssnet_truth"].to(DEVICE),
                   "kp":out_batched_truth["keypoint_truth"].to(DEVICE)}
    
    out_batched_weights = make_batch_weights( batch_truth, DEVICE, verbose )

    if not return_eval:
        return wireplane_sparsetensors, out_batched_truth["matchtriplet_v"], batch_truth, out_batched_weights
    else:
        return wireplane_sparsetensors, out_batched_truth["matchtriplet_v"], batch_truth, out_batched_weights, sampled_hardexample_v, sampled_eval_v
