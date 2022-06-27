import os,sys
import numpy as np
import torch
import MinkowskiEngine as ME
from larmatch.utils.common import prepare_me_sparsetensor

def larmatch_hard_example_mining( model, batchdata, DEVICE, return_num_samples, verbose=False ):

    model.eval()
    
    wireplane_sparsetensors = prepare_me_sparsetensor( batchdata, DEVICE)
    batch_size = len(batchdata)

    # we also need the metadata associating possible 3d spacepoints
    # to the wire image location they project to
    matchtriplet_v = []
    for b,data in enumerate(batchdata):
        matchtriplet_v.append( torch.from_numpy(data["matchtriplet_v"]).to(DEVICE) )
        if verbose:
            print("batch ",b," matchtriplets: ",matchtriplet_v[b].shape)
        sys.stdout.flush()

    # # get the truth
    batch_truth_v = []
    batch_weight_v = []
    for b,data in enumerate(batchdata):
        lm_truth_t = torch.from_numpy(data["larmatch_truth"]).to(DEVICE)
        lm_weight_t = torch.from_numpy(data["larmatch_weight"]).to(DEVICE)
        lm_truth_t.requires_grad = False
        lm_weight_t.requires_grad = False
        if verbose:
            print("  truth: ",lm_truth_t.shape)
            print("  weight: ",lm_weight_t.shape)

        ssnet_truth_t  = torch.from_numpy(data["ssnet_truth"]).to(DEVICE)
        ssnet_weight_t = torch.from_numpy(data["ssnet_class_weight"]).to(DEVICE)
        ssnet_truth_t.requires_grad = False
        ssnet_weight_t.requires_grad = False

        kp_truth_t  = torch.from_numpy(data["keypoint_truth"]).to(DEVICE)
        kp_weight_t = torch.from_numpy(data["keypoint_weight"]).to(DEVICE)
        kp_truth_t.requires_grad = False
        kp_weight_t.requires_grad = False

        truth_data = {"lm":lm_truth_t,"ssnet":ssnet_truth_t,"kp":kp_truth_t}
        weight_data = {"lm":lm_weight_t,"ssnet":ssnet_weight_t,"kp":kp_weight_t}
    
        batch_truth_v.append( truth_data )
        batch_weight_v.append( weight_data )

    # we remake the batch, prioritizing errors, then sampling randomly
    sampled_matchtriplet_v = []
    sampled_batch_truth_v  = []
    sampled_batch_weight_v = []
    with torch.no_grad():
        pred_dict = model( wireplane_sparsetensors, matchtriplet_v, batch_size )
        pred = torch.softmax( pred_dict["lm"], dim=1 )
        # num correct
        for b in range(batch_size):
            pos_num_correct = pred[b,1,:].gt(0.5)[ batch_truth_v[b]["lm"].eq(1) ].sum()
            pos_num_eval    = batch_truth_v[b]["lm"].eq(1).sum()
            neg_num_correct = pred[b,1,:].lt(0.5)[ batch_truth_v[b]["lm"].eq(0) ].sum()
            neg_num_eval    = batch_truth_v[b]["lm"].eq(0).sum()


            # incorrect examples
            false_pos = pred[b,1,:].gt(0.5)*batch_truth_v[b]["lm"].eq(0) # 1 at each false pos entry
            false_neg = pred[b,1,:].lt(0.5)*batch_truth_v[b]["lm"].eq(1) # 1 at each false neg entry
            false_ex  = false_pos+false_neg # 1 at each false entry
            false_idx = torch.arange(0,pred.shape[2]).to(DEVICE)[ false_ex ] # try to keep everying on the gpu?
            if verbose:
                print("hard-example mine: batch[",b,"] true positive =",float(pos_num_correct+neg_num_correct)/float(pos_num_eval+neg_num_eval),            
                      " false examples: ",false_idx.shape)

            # Prepare re-sampled data
            num_false = np.minimum( false_idx.shape[0], return_num_samples )
            num_else = return_num_samples-num_false            
            shufidx = np.arange(0,pred.shape[2])
            np.random.shuffle(shufidx)
            shufidx = torch.from_numpy(shufidx) # for filling remainder

            # match triplet
            sampled_matchtriplet = torch.zeros( (return_num_samples, 3), dtype=torch.long ).to(DEVICE)
            sampled_matchtriplet[:num_false,:] = matchtriplet_v[b][ false_idx[:num_false], : ]
            if num_else>0:
                sampled_matchtriplet[num_false:] = matchtriplet_v[b][ shufidx[:num_else], : ]
            
            # larmatch truth
            sampled_larmatch_truth = torch.zeros( return_num_samples, dtype=torch.long ).to(DEVICE)
            sampled_larmatch_truth[:num_false] = batch_truth_v[b]["lm"][ false_idx[:num_false] ]
            if num_else>0:
                sampled_larmatch_truth[num_false:] = batch_truth_v[b]["lm"][ shufidx[:num_else] ]

            # larmatch weight
            sampled_larmatch_weight = ((1.0/float(return_num_samples))*torch.ones( return_num_samples, dtype=torch.float )).to(DEVICE)
            # we need to rebalance
            num_pos_ex = (sampled_larmatch_truth.eq(1)).sum()
            num_neg_ex = (sampled_larmatch_truth.eq(2)).sum()
            if num_pos_ex>0:
                sampled_larmatch_weight[ sampled_larmatch_truth==1 ] = 0.5/float(num_pos_ex)
            if num_neg_ex>0:
                sampled_larmatch_weight[ sampled_larmatch_truth==0 ] = 0.5/float(num_neg_ex)
            #if num_else:
            #    sampled_larmatch_weight[num_false:] = batch_weight_v[b]["lm"][ shufidx[:num_else] ]

            # ssnet truth
            if "ssnet" in batch_truth_v[b]:
                sampled_ssnet_truth = torch.zeros( return_num_samples, dtype=torch.long ).to(DEVICE)
                sampled_ssnet_truth[:num_false] = batch_truth_v[b]["ssnet"][ false_idx[:num_false] ]
                if num_else:
                    sampled_ssnet_truth[num_false:] = batch_truth_v[b]["ssnet"][ shufidx[:num_else] ]

                sampled_ssnet_weight = torch.zeros( return_num_samples, dtype=torch.float ).to(DEVICE)
                sampled_ssnet_weight[:num_false] = batch_weight_v[b]["ssnet"][ false_idx[:num_false] ]
                if num_else:
                    sampled_ssnet_weight[num_false:] = batch_weight_v[b]["ssnet"][ shufidx[:num_else] ]

            # keypoint truth
            if "kp" in batch_truth_v[b]:
                sampled_kp_truth = torch.zeros( (6,return_num_samples), dtype=torch.float ).to(DEVICE)
                sampled_kp_truth[:,:num_false] = batch_truth_v[b]["kp"][ :, false_idx[:num_false] ]
                if num_else:
                    sampled_kp_truth[:,num_false:] = batch_truth_v[b]["kp"][ :, shufidx[:num_else] ]

                sampled_kp_weight = torch.zeros( (6,return_num_samples), dtype=torch.float).to(DEVICE)
                sampled_kp_weight[:,:num_false] = batch_weight_v[b]["kp"][ :, false_idx[:num_false] ]
                if num_else:
                    sampled_kp_weight[:,num_false:] = batch_weight_v[b]["kp"][ :, shufidx[:num_else] ]

            sampled_truth_data  = {"lm":sampled_larmatch_truth,
                                   "ssnet":sampled_ssnet_truth,
                                   "kp":sampled_kp_weight}
            sampled_weight_data = {"lm":sampled_larmatch_weight,
                                   "ssnet":sampled_ssnet_weight,
                                   "kp":sampled_kp_weight}

            sampled_matchtriplet_v.append( sampled_matchtriplet )
            sampled_batch_truth_v.append(  sampled_truth_data )
            sampled_batch_weight_v.append( sampled_weight_data )
            
                            
    return wireplane_sparsetensors, sampled_matchtriplet_v, sampled_batch_truth_v, sampled_batch_weight_v
