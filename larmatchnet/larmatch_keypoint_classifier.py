import os,sys
from collections import OrderedDict
import torch
import torch.nn as nn

class LArMatchKeypointClassifier(nn.Module):

    def __init__(self,features_per_layer=16,
                 keypoint_nfeatures=[32,32],                 
                 ninput_planes=3,
                 device=torch.device("cpu")):
        super(LArMatchKeypointClassifier,self).__init__()

        # CLASSIFER: CLOSE-TO-KEYPOINT
        keypoint_layers = OrderedDict()
        keypoint_layers["keypoint0conv"] = torch.nn.Conv1d(ninput_planes*features_per_layer,
                                                           keypoint_nfeatures[0],1)
        keypoint_layers["keypoint0relu"] = torch.nn.ReLU()
        for ilayer,nfeats in enumerate(keypoint_nfeatures[1:]):
            keypoint_layers["keypoint%dconv"%(ilayer+1)] = torch.nn.Conv1d(nfeats,nfeats,1)
            keypoint_layers["keypoint%drelu"%(ilayer+1)] = torch.nn.ReLU()
        keypoint_layers["keypointout"] = torch.nn.Conv1d(nfeats,1,1)
        self.keypoint = torch.nn.Sequential( keypoint_layers )
        
    def forward(self,triplet_feat_t):
        """
        classify triplet of (u,v,y) wire plane pixel combination as being (background,track,shower)
        use information from concat feature vectors.

        inputs:
        triplet_feat_t : tensor where each row is concatenated feature vector        
        """
        pred = self.keypoint(triplet_feat_t)
        return pred
    
