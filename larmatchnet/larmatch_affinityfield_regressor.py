import os,sys
from collections import OrderedDict
import torch
import torch.nn as nn

class LArMatchAffinityFieldRegressor(nn.Module):

    def __init__(self,input_features=16,
                 layer_nfeatures=[64,64],
                 ninput_planes=3,
                 output_dim=3,
                 device=torch.device("cpu")):
        super(LArMatchAffinityFieldRegressor,self).__init__()

        # REGRESSION: 3D-SHIFT-TO-NEAREST-KEYPOINT
        layers = OrderedDict()
        layers["paf_conv0"] = torch.nn.Conv1d(ninput_planes*input_features,
                                                 layer_nfeatures[0],1)
        #layers["paf_bn0"]   = torch.nn.BatchNorm1d(layer_nfeatures[0])
        layers["paf_relu0"] = torch.nn.LeakyReLU()
        for ilayer,nfeats in enumerate(layer_nfeatures[1:]):
            layers["paf_conv%d"%(ilayer+1)] = torch.nn.Conv1d(nfeats,nfeats,1)
            #layers["paf_bn%d"%(ilayer+1)]   = torch.nn.BatchNorm1d(nfeats)
            layers["paf_relu%d"%(ilayer+1)] = torch.nn.LeakyReLU()
        layers["paf_out"] = torch.nn.Conv1d(nfeats,output_dim,1)
        self.paf_layers = torch.nn.Sequential( layers )
        
        
    def forward(self,triplet_feat_t):
        """
        classify triplet of (u,v,y) wire plane pixel combination as being (background,track,shower)
        use information from concat feature vectors.

        inputs:
        triplet_feat_t : tensor where each row is concatenated feature vector        
        """
        shift = self.paf_layers(triplet_feat_t)
        return shift
    
