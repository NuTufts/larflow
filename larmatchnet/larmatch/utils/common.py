import os,sys
import torch
import numpy as np
import MinkowskiEngine as ME

def prepare_me_sparsetensor( batchdata, DEVICE, num_wireplanes=3, verbose=False ):
    """
    batchdata: output of dataloader employing `larcv_dataset`.
    returns: list with ME.SparseTensor for each wire plane
    """
    wireplane_sparsetensors = []    
    for p in range(num_wireplanes):
        if verbose:
            print("plane ",p)
            for b,data in enumerate(batchdata["coord_%d"%(p)]):
                print(" coord plane[%d] batch[%d]"%(p,b),": ",data.shape)
    
        coord_v = [ torch.from_numpy(x).to(DEVICE) for x in batchdata["coord_%d"%(p)] ]
        feat_v  = [ torch.from_numpy(x).to(DEVICE) for x in batchdata["feat_%d"%(p)] ]

        for x in coord_v:
            x.requires_grad = False
        for x in feat_v:
            x.requires_grad = False
        
        coords, feats = ME.utils.sparse_collate(coord_v, feat_v)
        wireplane_sparsetensors.append( ME.SparseTensor(features=feats, coordinates=coords) )


    batched = make_batch_truth( batchdata, DEVICE, verbose )
    matchtriplet_v = batched["matchtriplet_v"].to(DEVICE)    
    batch_truth = {"lm-hardlabel":batched["larmatch_truth"].to(DEVICE),
                   "lm":batched["larmatch_label"].to(DEVICE),
                   "ssnet":batched["ssnet_truth"].to(DEVICE),
                   "kp":batched["keypoint_truth"].to(DEVICE)}
    
    batch_weight = make_batch_weights( batch_truth, DEVICE, verbose )
    
    return wireplane_sparsetensors, matchtriplet_v, batch_truth, batch_weight

def make_batch_truth( batchdata, DEVICE, verbose ):
    """
    move data to gpu and concat batch there. faster?
    """
    if verbose: print("common.make_batch_truth")
    truth_tensors = {}
    for k in ["larmatch_truth","larmatch_label","ssnet_truth","keypoint_truth","matchtriplet_v"]:
        if k not in batchdata:
            continue
        label_batch = []
        for data in batchdata[k]:
            if len(data.shape)==1:
                x = data.reshape( 1, 1, data.shape[0] ) # (N) -> (1,1,N)
            elif len(data.shape)==2:
                x = data.reshape( 1, data.shape[0], data.shape[1] ) # (C,N) -> (1,C,N)
            x = torch.from_numpy(x).to(DEVICE)
            x.requires_grad = False
            label_batch.append( x )
        truth_tensors[k] = torch.cat( label_batch, dim=0 )
    return truth_tensors
    

def make_batch_weights( batch_truth, DEVICE, verbose ):
    """
    Class balancing across the batch. Best to do after tensors are on GPU
    """
    if verbose: print("common.make_batch_weights: start")
    weight_tensors = {}
    
    # -- larmatch, hard labels --
    lm_truth = batch_truth["lm"]
    lm_weight = torch.zeros( lm_truth.shape, dtype=torch.float ).to(DEVICE)
    lm_pos = lm_truth.gt(0.5)
    lm_neg = lm_truth.lt(0.5)    
    num_lm_pos = lm_pos.sum().cpu().item()
    num_lm_neg = lm_neg.sum().cpu().item()

    lm_norm = 0.0
    if num_lm_pos>0:
        lm_norm += 1.0
    if num_lm_neg>0:
        lm_norm += 1.0
    if num_lm_pos>0:
        lm_weight[ lm_pos ] = lm_norm/num_lm_pos
    if num_lm_neg>0:
        lm_weight[ lm_neg ] = lm_norm/num_lm_neg        
    weight_tensors["lm"] = lm_weight
    
    # -- ssnet --
    ss_truth = batch_truth["ssnet"]
    ss_norm = 0.0
    ss_examples = []
    ss_class = [ ss_truth.eq(c) for c in range(7) ]
    for c in range(7):
        if c==0:
            # bg class: we're ignoring this class
            ss_examples.append(0)
            continue
        c_examples = ss_class[c].sum().cpu().item()
        if c_examples>0:
            ss_norm += 1.0
        ss_examples.append(c_examples)
    
    ss_weight = torch.zeros( ss_truth.shape, dtype=torch.float )
    for c in range(6):
        if ss_examples[c]>0:
            ss_weight[ ss_class[c] ] = ss_norm/ss_examples[c]
    weight_tensors["ssnet"] = ss_weight
    
    # -- kplabel --
    kptruth = batch_truth["kp"]
    num_examples = kptruth.shape[0]*kptruth.shape[2] # per class
    kp_norm = 0.0
    kp_pos = [ kptruth[:,c,:].gt(0.1) for c in range(6) ]
    kp_neg = [ kptruth[:,c,:].lt(0.1) for c in range(6) ]    
    kp_examples = []
    for c in range(6):
        c_examples = kp_pos[c].sum().cpu().item()
        if c_examples>0:
            kp_norm += 1.0
        if c_examples<num_examples:
            kp_norm += 1.0
        kp_examples.append(c_examples)
    
    kp_weight = torch.zeros( kptruth.shape, dtype=torch.float )
    for c in range(6):
        if kp_examples[c]>0:
            kp_weight[:,c,:][ kp_pos[c] ] = kp_norm/kp_examples[c]
        if kp_examples[c]<num_examples:
            kp_weight[:,c,:][ kp_neg[c] ] = 2.0/(num_examples-kp_examples[c])
    weight_tensors["kp"] = kp_weight
    return weight_tensors

def make_random_images( input_shape, num_samples, batchsize, num_wireplanes=3, verbose=False ):
    """
    Make a random coord and feature tensor for testing
    """
    wireplane_sparsetensors = []
    for p in range(num_wireplanes):
        coord_v = []
        feat_v = []
        for b in range(batchsize):
            fake_coord = np.random.randint( 0, high=input_shape[1], size=(num_samples,2) )
            coord_v.append( torch.from_numpy(fake_coord).to(DEVICE) )
            fake_feat  = np.random.rand( num_samples, 1 )
            feat_v.append( torch.from_numpy(fake_feat.astype(np.float32)).to(DEVICE) )

            for x in coord_v:
                x.requires_grad = False
            for x in feat_v:
                x.requires_grad = False
                
        coords, feats = ME.utils.sparse_collate(coord_v, feat_v)            
        if verbose:
            print(" coords: ",coords.shape)
            print(" feats: ",feats.shape)
        wireplane_sparsetensors.append( ME.SparseTensor(features=feats, coordinates=coords) )
    
    return wireplane_sparsetensors


