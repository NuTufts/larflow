import os,sys
from collections import OrderedDict
import torch
import torch.nn as nn

class LArMatchKeypointClassifier(nn.Module):

    def __init__(self,features_per_layer=16,
                 keypoint_nfeatures=[32,32],                 
                 ninput_planes=3,
                 nclasses=3,
                 use_bn=True,
                 device=torch.device("cpu")):
        super(LArMatchKeypointClassifier,self).__init__()

        # CLASSIFER: CLOSE-TO-KEYPOINT TRACK ENDS
        keypoint_trkEnds_layers = OrderedDict()
        keypoint_trkEnds_layers["keypoint0conv"] = torch.nn.Conv1d(ninput_planes*features_per_layer,
                                                           keypoint_nfeatures[0],1)
        keypoint_trkEnds_layers["keypoint0relu"] = torch.nn.ReLU()
        for ilayer,nfeats in enumerate(keypoint_nfeatures[1:]):
            keypoint_trkEnds_layers["keypoint%dconv"%(ilayer+1)] = torch.nn.Conv1d(nfeats,nfeats,1)
            keypoint_trkEnds_layers["keypoint%drelu"%(ilayer+1)] = torch.nn.ReLU()
        keypoint_trkEnds_layers["keypointout"] = torch.nn.Conv1d(nfeats,1,1)
        self.keypoint_trkEnds = torch.nn.Sequential( keypoint_trkEnds_layers )

        # CLASSIFER: CLOSE-TO-KEYPOINT SHOWER START
        keypoint_shwrStart_layers = OrderedDict()
        keypoint_shwrStart_layers["keypoint0conv"] = torch.nn.Conv1d(ninput_planes*features_per_layer,
                                                           keypoint_nfeatures[0],1)
        keypoint_shwrStart_layers["keypoint0relu"] = torch.nn.ReLU()
        for ilayer,nfeats in enumerate(keypoint_nfeatures[1:]):
            keypoint_shwrStart_layers["keypoint%dconv"%(ilayer+1)] = torch.nn.Conv1d(nfeats,nfeats,1)
            keypoint_shwrStart_layers["keypoint%drelu"%(ilayer+1)] = torch.nn.ReLU()
        keypoint_shwrStart_layers["keypointout"] = torch.nn.Conv1d(nfeats,1,1)
        self.keypoint_shwrStart = torch.nn.Sequential( keypoint_shwrStart_layers )

        # CLASSIFER: CLOSE-TO-KEYPOINT NEUTRINO VERTEX
        keypoint_nuVtx_layers = OrderedDict()
        keypoint_nuVtx_layers["keypoint0conv"] = torch.nn.Conv1d(ninput_planes*features_per_layer,
                                                           keypoint_nfeatures[0],1)
        keypoint_nuVtx_layers["keypoint0relu"] = torch.nn.ReLU()
        for ilayer,nfeats in enumerate(keypoint_nfeatures[1:]):
            keypoint_nuVtx_layers["keypoint%dconv"%(ilayer+1)] = torch.nn.Conv1d(nfeats,nfeats,1)
            keypoint_nuVtx_layers["keypoint%drelu"%(ilayer+1)] = torch.nn.ReLU()
        keypoint_nuVtx_layers["keypointout"] = torch.nn.Conv1d(nfeats,1,1)
        self.keypoint_nuVtx = torch.nn.Sequential( keypoint_nuVtx_layers )

    def forward(self,triplet_feat_t):
        """
        classify triplet of (u,v,y) wire plane pixel combination as being (background,track,shower)
        use information from concat feature vectors.

        inputs:
        triplet_feat_t : tensor where each row is concatenated feature vector        
        """
        pred_trkEnds = self.keypoint_trkEnds(triplet_feat_t)
        pred_shwrStart = self.keypoint_shwrStart(triplet_feat_t)
        pred_nuVtx = self.keypoint_nuVtx(triplet_feat_t)
        pred = torch.cat((pred_trkEnds, pred_shwrStart, pred_nuVtx), 1)
        return pred
    

if __name__=="__main__":
    print "test"
    a = LArMatchKeypointClassifier()
    b = torch.zeros(1,48,100)
    print b.shape
    c = a.forward(b)
    print c.shape
