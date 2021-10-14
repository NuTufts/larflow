import os,sys
from collections import OrderedDict
import torch
import torch.nn as nn

class LArMatchKeypointClassifier(nn.Module):

    def __init__(self,features_per_layer=16,
                 keypoint_nfeatures=[32,32],                 
                 ninput_planes=3,
                 nclasses=6,
                 use_bn=True):
        super(LArMatchKeypointClassifier,self).__init__()

        # SCORE PREDICTION
        self.class_layers = {}
        for iclass in range(nclasses):
            keypoint_layers = OrderedDict()
            keypoint_layers["keypoint0conv_class%d"%(iclass)] = torch.nn.Conv1d(ninput_planes*features_per_layer,
                                                                                keypoint_nfeatures[0],1)
            if use_bn:
                keypoint_layers["keypoint0_bn_class%d"%(iclass)]  = torch.nn.BatchNorm1d(keypoint_nfeatures[0])
            else:
                keypoint_layers["keypoint0_bn_class%d"%(iclass)]  = torch.nn.InstanceNorm1d(keypoint_nfeatures[0])
            keypoint_layers["keypoint0relu_class%d"%(iclass)] = torch.nn.LeakyReLU()
            for ilayer,nfeats in enumerate(keypoint_nfeatures[1:]):
                keypoint_layers["keypoint%dconv_class%d"%(ilayer+1,iclass)] = torch.nn.Conv1d(nfeats,nfeats,1)
                if use_bn:
                    keypoint_layers["keypoint%d_bn_class%d"%(ilayer+1,iclass)]  = torch.nn.BatchNorm1d(nfeats)
                else:
                    keypoint_layers["keypoint%d_bn_class%d"%(ilayer+1,iclass)]  = torch.nn.InstanceNorm1d(nfeats)
                keypoint_layers["keypoint%drelu_class%d"%(ilayer+1,iclass)] = torch.nn.LeakyReLU()
            keypoint_layers["keypointout_class%d"%(iclass)] = torch.nn.Conv1d(nfeats,1,1)
            self.class_layers[iclass] = torch.nn.Sequential( keypoint_layers )
            setattr( self, "keypoint_class%d_layers"%(iclass), self.class_layers[iclass] )
        
    def forward(self,triplet_feat_t):
        """
        classify triplet of (u,v,y) wire plane pixel combination as being (background,track,shower)
        use information from concat feature vectors.

        inputs:
        triplet_feat_t : tensor where each row is concatenated feature vector        
        """
        classout = []
        for iclass,layers in self.class_layers.items():
            classpred = layers(triplet_feat_t)
            classout.append(classpred)
        pred = torch.cat( classout, dim=1 )
        pred = torch.sigmoid(pred)
        return pred
    
