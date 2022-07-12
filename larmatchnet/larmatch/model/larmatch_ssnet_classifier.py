import os,sys
from collections import OrderedDict
import torch
import torch.nn as nn

class LArMatchSSNetClassifier(nn.Module):

    def __init__(self,features_per_layer=16,
                 ssnet_classifier_nfeatures=[64,64],
                 ninput_planes=3,
                 num_classes=7,
                 norm_layer='batchnorm'):
        super(LArMatchSSNetClassifier,self).__init__()

        # CLASSIFER: SSNET (background,track,shower)
        ssnet_classifier_layers = OrderedDict()
        ssnet_classifier_layers["ssnet0conv"] = torch.nn.Conv1d(ninput_planes*(features_per_layer+1),
                                                                ssnet_classifier_nfeatures[0],1)
        if norm_layer in ['instance','stableinstance']:
            ssnet_classifier_layers["ssnet0bn"]   = torch.nn.InstanceNorm1d(ssnet_classifier_nfeatures[0])
        elif norm_layer=='batchnorm':
            ssnet_classifier_layers["ssnet0bn"]   = torch.nn.BatchNorm1d(ssnet_classifier_nfeatures[0])
        ssnet_classifier_layers["ssnet0relu"] = torch.nn.ReLU()
        last_layer_nfeats = ssnet_classifier_nfeatures[0]
        for ilayer,nfeats in enumerate(ssnet_classifier_nfeatures[1:]):
            ssnet_classifier_layers["ssnet%dconv"%(ilayer+1)] = torch.nn.Conv1d(nfeats,nfeats,1)
            if norm_layer in ['instance','stableinstance']:
                ssnet_classifier_layers["ssnet%dbn"%(ilayer+1)]   = torch.nn.InstanceNorm1d(nfeats)
            elif norm_layer=='batchnorm':
                ssnet_classifier_layers["ssnet%dbn"%(ilayer+1)]   = torch.nn.BatchNorm1d(nfeats)
            ssnet_classifier_layers["ssnet%drelu"%(ilayer+1)] = torch.nn.ReLU()
            last_layer_nfeats = nfeats
        ssnet_classifier_layers["ssnetout"] = torch.nn.Conv1d(last_layer_nfeats,num_classes,1, bias=True)
        ssnet_classifier_layers["ssnetout"].bias.data.fill_(0.0)
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
    
