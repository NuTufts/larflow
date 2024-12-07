from collections import OrderedDict
import torch
import torch.nn as nn
import MinkowskiEngine as ME

class LArMatchSpacepointClassifier( nn.Module ):

    def __init__(self, num_input_feats, classifier_nfeatures=[32,32], ndimensions=2, norm="batchnorm" ):
        super(LArMatchSpacepointClassifier,self).__init__()

        # larmatch classifier
        self.final_vec_nfeats = num_input_feats
        lm_class_layers = OrderedDict()
        for i,nfeat in enumerate(classifier_nfeatures):
            if i==0:
                lm_class_layers["lmclassifier_layer%d"%(i)] = torch.nn.Conv1d(num_input_feats,nfeat,1)
            else:
                lm_class_layers["lmclassifier_layer%d"%(i)] = torch.nn.Conv1d(classifier_nfeatures[i-1],nfeat,1)
            if norm=="instance":
                lm_class_layers["lmclassifier_norm%d"%(i)] = torch.nn.InstanceNorm1d(nfeat)
            elif norm=="batchnorm":
                lm_class_layers["lmclassifier_norm%d"%(i)] = torch.nn.BatchNorm1d(nfeat)
            else:
                raise ValueError("invalid norm option: ",norm," options=['batchnorm','instance']")
            lm_class_layers["lmclassifier_relu%d"%(i)] = torch.nn.ReLU()
        lm_class_layers["lmclassifier_out"] = torch.nn.Conv1d(classifier_nfeatures[-1],2,1)
        self.lm_classifier = nn.Sequential( lm_class_layers )

    def _init_weights(self):
        for module in self.lm_classifier.modules():
            if isinstance(module,torch.nn.Conv1d):
                print("set to kaiming normal by default")                
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(module.bias)
        # for the last layer, we set the bias in anticipation of a large class inbalance.
        # we know we have many more keypoint values that should be zero
        # ratio is probably at least 100:1 spacepoints not near keypoints to spacepoints near keypoints, if not worse
        last_layer = self.lm_classifier[-1]
        negative_to_positive_ratio = 100.0
        bias = -torch.log(torch.tensor(negative_to_positive_ratio))
        nn.init.constant_(last_layer.bias,bias)
            
        
    def forward( self, triplet_feat_t ):
        """
        classify triplet of (u,v,y) wire plane pixel locations as being a true or false position.
        use information from concat feature vectors.

        inputs:
        triplet_feat_t [torch tensor (1,3C,N)] concat spacepoint feature tensor, output of extract_features

        output:
        torch tensor (1,2,N)
        """
        #print("lm input: ",triplet_feat_t)
        pred = self.lm_classifier(triplet_feat_t)
        #print("lm spacepoint classifier: ",pred)
        return pred
        
        
