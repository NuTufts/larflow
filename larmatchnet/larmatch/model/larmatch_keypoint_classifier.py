import os,sys
from collections import OrderedDict
import torch
import torch.nn as nn

class LArMatchKeypointClassifier(nn.Module):

    def __init__(self,features_per_layer=16,
                 keypoint_nfeatures=[32,32,32],
                 ninput_planes=3,
                 nclasses=6,
                 norm_layer='batchnorm'):
        super(LArMatchKeypointClassifier,self).__init__()
        assert norm_layer in ['batchnorm','instance','stableinstance']
                
        # SCORE PREDICTION
        self.class_layers = {}
        for iclass in range(nclasses):
            keypoint_layers = OrderedDict()
            keypoint_layers["keypoint0conv_class%d"%(iclass)] = torch.nn.Conv1d(ninput_planes*features_per_layer,
                                                                                keypoint_nfeatures[0],1,bias=True)
            if norm_layer=='batchnorm':
                keypoint_layers["keypoint0_bn_class%d"%(iclass)]  = torch.nn.BatchNorm1d(keypoint_nfeatures[0])
            elif norm_layer in ['instance','stableinstance']:
                keypoint_layers["keypoint0_bn_class%d"%(iclass)]  = torch.nn.InstanceNorm1d(keypoint_nfeatures[0])
                
            keypoint_layers["keypoint0relu_class%d"%(iclass)] = torch.nn.LeakyReLU()
            last_layer_nfeats = keypoint_nfeatures[0]
            for ilayer,nfeats in enumerate(keypoint_nfeatures[1:]):
                keypoint_layers["keypoint%dconv_class%d"%(ilayer+1,iclass)] = torch.nn.Conv1d(nfeats,nfeats,1,bias=True)
                
                #if norm_layer=='batchnorm':                    
                #    keypoint_layers["keypoint%d_bn_class%d"%(ilayer+1,iclass)]  = torch.nn.BatchNorm1d(nfeats)
                #elif norm_layer in ['instance','stableinstance']:
                #    keypoint_layers["keypoint%d_bn_class%d"%(ilayer+1,iclass)]  = torch.nn.InstanceNorm1d(nfeats)
                    
                keypoint_layers["keypoint%drelu_class%d"%(ilayer+1,iclass)] = torch.nn.LeakyReLU()
                last_layer_nfeats = nfeats
            keypoint_layers["keypointout_class%d"%(iclass)] = torch.nn.Conv1d(last_layer_nfeats,1,1,bias=True)
            # set the bias to zero so that initial output is zero
            keypoint_layers["keypointout_class%d"%(iclass)].bias.data[0] = 0.0
            
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
        #pred = torch.sigmoid(pred) # not good for gradients?
        return pred
    
