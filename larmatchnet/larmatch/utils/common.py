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
    return wireplane_sparsetensors

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


