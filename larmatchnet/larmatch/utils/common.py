import os,sys
import torch
import numpy as np
import MinkowskiEngine as ME

def prepare_me_sparsetensor( batchdata, DEVICE, make_batch_tensor=True, num_wireplanes=3, verbose=False ):
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

    batched = make_batch_truth( batchdata, DEVICE, make_batch_tensor, verbose )
    if make_batch_tensor:
        matchtriplet_v = batched["matchtriplet_v"].to(DEVICE)    
        batch_truth = {"lm-hardlabel":batched["larmatch_truth"].to(DEVICE),
                       "lm":batched["larmatch_label"].to(DEVICE),
                       "ssnet":batched["ssnet_truth"].to(DEVICE),
                       "kp":batched["keypoint_truth"].to(DEVICE)}
    else:
        matchtriplet_v = batched["matchtriplet_v"]
        for x in matchtriplet_v:
            x.to(DEVICE)
            
        batch_truth = {"lm-hardlabel":batched["larmatch_truth"],
                       "lm":batched["larmatch_label"],
                       "ssnet":batched["ssnet_truth"],
                       "kp":batched["keypoint_truth"]}
        for x in batched:
            for arr in batched[x]:
                if type(arr) is torch.tensor:
                    arr.to(DEVICE)

    if make_batch_tensor:
        batch_weight = make_batch_weights( batch_truth, DEVICE, verbose )
    else:
        batch_weight = make_list_weights( batch_truth, DEVICE, verbose )
    
    return wireplane_sparsetensors, matchtriplet_v, batch_truth, batch_weight

def make_batch_truth( batchdata, DEVICE, make_batch_tensor, verbose ):
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
            if make_batch_tensor:
                if len(data.shape)==1:
                    x = data.reshape( 1, 1, data.shape[0] ) # (N) -> (1,1,N)
                elif len(data.shape)==2:
                    x = data.reshape( 1, data.shape[0], data.shape[1] ) # (C,N) -> (1,C,N)
            else:
                if len(data.shape)==1:
                    x = data.reshape( 1, data.shape[0] ) # (N) -> (1,N)
                elif len(data.shape)==2:
                    x = data
            
            if type(x) is np.ndarray:
                x = torch.from_numpy(x).to(DEVICE)
            else:
                x = x.to(DEVICE)
            x.requires_grad = False
            label_batch.append( x )
        if make_batch_tensor:
            truth_tensors[k] = torch.cat( label_batch, dim=0 )
        else:
            truth_tensors[k] = label_batch
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
    lm_pos = lm_truth.ge(0.5)
    lm_neg = lm_truth.lt(0.5)    
    num_lm_pos = lm_pos.sum().cpu().item()
    num_lm_neg = lm_neg.sum().cpu().item()

    lm_norm = 0.0
    if num_lm_pos>0:
        lm_norm += 1.0
    if num_lm_neg>0:
        lm_norm += 1.0
    if num_lm_pos>0:
        lm_weight[ lm_pos ] = 1.0/(lm_norm*num_lm_pos)
    if num_lm_neg>0:
        lm_weight[ lm_neg ] = 1.0/(lm_norm*num_lm_neg)
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
    
    ss_weight = torch.zeros( ss_truth.shape, dtype=torch.float ).to(DEVICE)
    for c in range(7):
        if ss_examples[c]>0:
            ss_weight[ ss_class[c] ] = 1.0/(ss_norm*ss_examples[c])
    weight_tensors["ssnet"] = ss_weight
    
    # -- kplabel --
    kptruth = batch_truth["kp"]
    num_examples = kptruth.shape[0]*kptruth.shape[2] # per class (B,C,N)
    kp_pos = [ kptruth[:,c,:].ge(0.1) for c in range(6) ]
    kp_neg = [ kptruth[:,c,:].lt(0.1) for c in range(6) ]    
    kp_examples = []
    kp_norms = []
    for c in range(6):
        c_examples = kp_pos[c].sum().cpu().item()
        kp_norm = 0.0
        if c_examples>0:
            kp_norm += 1.0
        if c_examples<num_examples:
            kp_norm += 1.0
        kp_examples.append(c_examples)
        kp_norms.append(kp_norm)
    
    kp_weight = torch.zeros( kptruth.shape, dtype=torch.float ).to(DEVICE)
    for c in range(6):
        if kp_examples[c]>0:
            kp_weight[:,c,:][ kp_pos[c] ] = 1.0/(kp_norms[c]*kp_examples[c])
        if kp_examples[c]<num_examples:
            kp_weight[:,c,:][ kp_neg[c] ] = 1.0/(kp_norms[c]*(num_examples-kp_examples[c]))
    weight_tensors["kp"] = kp_weight
    return weight_tensors


def make_list_weights( batch_truth, DEVICE, verbose ):
    """
    Class balancing across the batch. Best to do after tensors are on GPU
    """
    if verbose: print("common.make_batch_weights: start")
    weight_tensors = {}
    
    # -- larmatch, hard labels --
    weight_tensors["lm"] = []
    for b,lm_truth in enumerate(batch_truth["lm"]):
        lm_weight = torch.zeros( lm_truth.shape, dtype=torch.float ).to(DEVICE)
        lm_pos = lm_truth.ge(0.5)
        lm_neg = lm_truth.lt(0.5)    
        num_lm_pos = lm_pos.sum().cpu().item()
        num_lm_neg = lm_neg.sum().cpu().item()

        lm_norm = 0.0
        if num_lm_pos>0:
            lm_norm += 1.0
        if num_lm_neg>0:
            lm_norm += 1.0
        if num_lm_pos>0:
            lm_weight[ lm_pos ] = 1.0/(lm_norm*num_lm_pos)
        if num_lm_neg>0:
            lm_weight[ lm_neg ] = 1.0/(lm_norm*num_lm_neg)
        weight_tensors["lm"].append( lm_weight )
    
    # -- ssnet --
    weight_tensors["ssnet"] = []
    for b,ss_truth in enumerate(batch_truth["ssnet"]):
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
    
        ss_weight = torch.zeros( ss_truth.shape, dtype=torch.float ).to(DEVICE)
        for c in range(7):
            if ss_examples[c]>0:
                ss_weight[ ss_class[c] ] = 1.0/(ss_norm*ss_examples[c])
        weight_tensors["ssnet"].append( ss_weight )
    
    # -- kplabel --
    weight_tensors["kp"] = []
    num_examples = 0
    for b,kptruth in enumerate(batch_truth["kp"]):
        num_examples += kptruth.shape[-1]
        print("kptruth[",b,"]: ",kptruth.shape)
    for b,kptruth in enumerate(batch_truth["kp"]):
        kp_pos = [ kptruth[c,:].ge(0.1) for c in range(6) ]
        kp_neg = [ kptruth[c,:].lt(0.1) for c in range(6) ]    
        kp_examples = []
        kp_norms = []
        for c in range(6):
            c_examples = kp_pos[c].sum().cpu().item()
            kp_norm = 0.0
            if c_examples>0:
                kp_norm += 1.0
            if c_examples<num_examples:
                kp_norm += 1.0
            kp_examples.append(c_examples)
            kp_norms.append(kp_norm)
    
        kp_weight = torch.zeros( kptruth.shape, dtype=torch.float ).to(DEVICE)
        for c in range(6):
            if kp_examples[c]>0:
                kp_weight[c,:][ kp_pos[c] ] = 1.0/(kp_norms[c]*kp_examples[c])
            if kp_examples[c]<num_examples:
                kp_weight[c,:][ kp_neg[c] ] = 1.0/(kp_norms[c]*(num_examples-kp_examples[c]))
        weight_tensors["kp"].append(kp_weight)
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


