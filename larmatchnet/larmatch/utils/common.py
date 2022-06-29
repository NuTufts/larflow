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
            for b,data in enumerate(batchdata):
                print(" coord plane[%d] batch[%d]"%(p,b),": ",data["coord_%d"%(p)].shape)
    
        coord_v = [ torch.from_numpy(data["coord_%d"%(p)]).to(DEVICE) for data in batchdata ]
        feat_v  = [ torch.from_numpy(data["feat_%d"%(p)]).to(DEVICE) for data in batchdata ]

        for x in coord_v:
            x.requires_grad = False
        for x in feat_v:
            x.requires_grad = False
        
        coords, feats = ME.utils.sparse_collate(coord_v, feat_v)
        wireplane_sparsetensors.append( ME.SparseTensor(features=feats, coordinates=coords) )

    # we also need the metadata associating possible 3d spacepoints
    # to the wire image location they project to
    matchtriplet_v = []
    for b,data in enumerate(batchdata):
        matchtriplet_v.append( torch.from_numpy(data["matchtriplet_v"]).to(DEVICE) )
        if verbose:
            print("batch ",b," matchtriplets: ",matchtriplet_v[b].shape)
            sys.stdout.flush()
    

    # # get the truth
    batch_truth = []
    batch_weight = []
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
            
        batch_truth.append( truth_data )
        batch_weight.append( weight_data )
    
    return wireplane_sparsetensors, matchtriplet_v, batch_truth, batch_weight

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


