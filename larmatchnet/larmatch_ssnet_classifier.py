import os,sys
from collections import OrderedDict
import torch
import torch.nn as nn

class LArMatchSSNetClassifier(nn.Module):

    def __init__(self,features_per_layer=16,
                 ssnet_classifier_nfeatures=[32,32],                 
                 ninput_planes=3,
                 num_classes=7,
                 device=torch.device("cpu")):
        super(LArMatchSSNetClassifier,self).__init__()

        # CLASSIFER: SSNET (background,track,shower)
        ssnet_classifier_layers = OrderedDict()
        ssnet_classifier_layers["ssnet0conv"] = torch.nn.Conv1d(ninput_planes*features_per_layer,
                                                                ssnet_classifier_nfeatures[0],1)
        ssnet_classifier_layers["ssnet0relu"] = torch.nn.ReLU()
        for ilayer,nfeats in enumerate(ssnet_classifier_nfeatures[1:]):
            ssnet_classifier_layers["ssnet%dconv"%(ilayer+1)] = torch.nn.Conv1d(nfeats,nfeats,1)
            ssnet_classifier_layers["ssnet%drelu"%(ilayer+1)] = torch.nn.ReLU()
        ssnet_classifier_layers["ssnetout"] = torch.nn.Conv1d(nfeats,num_classes,1)
        self.ssnet_classifier = torch.nn.Sequential( ssnet_classifier_layers )
        
    def forward(self,triplet_feat_t):
        """
        classify triplet of (u,v,y) wire plane pixel combination as being (background,track,shower)
        use information from concat feature vectors.

        inputs:
        triplet_feat_t : tensor where each row is concatenated feature vector
        """
        pred = self.ssnet_classifier(triplet_feat_t)
        return pred
    
