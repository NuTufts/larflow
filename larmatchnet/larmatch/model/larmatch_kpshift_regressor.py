import os,sys
from collections import OrderedDict
import torch
import torch.nn as nn

class LArMatchKPShiftRegressor(nn.Module):

    def __init__(self,features_per_layer=16,
                 kpshift_nfeatures=[64,64],
                 ninput_planes=3,
                 spatial_dim=3,                 
                 device=torch.device("cpu")):
        super(LArMatchKPShiftRegressor,self).__init__()

        # REGRESSION: 3D-SHIFT-TO-NEAREST-KEYPOINT
        kpshift_layers = OrderedDict()
        kpshift_layers["kpshift0conv"] = torch.nn.Conv1d(ninput_planes*features_per_layer,
                                                         kpshift_nfeatures[0],1)
        kpshift_layers["kpshift0relu"] = torch.nn.ReLU()
        for ilayer,nfeats in enumerate(kpshift_nfeatures[1:]):
            kpshift_layers["kpshift%dconv"%(ilayer+1)] = torch.nn.Conv1d(nfeats,nfeats,1)
            kpshift_layers["kpshift%drelu"%(ilayer+1)] = torch.nn.ReLU()
        kpshift_layers["kpshiftout"] = torch.nn.Conv1d(nfeats,spatial_dim,1)
        self.kpshift = torch.nn.Sequential( kpshift_layers )
        
        
    def forward(self,triplet_feat_t):
        """
        classify triplet of (u,v,y) wire plane pixel combination as being (background,track,shower)
        use information from concat feature vectors.

        inputs:
        triplet_feat_t : tensor where each row is concatenated feature vector        
        """
        shift    = self.kpshift(triplet_feat_t)
        return shift
    
