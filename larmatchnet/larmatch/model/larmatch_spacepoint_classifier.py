from collections import OrderedDict
import torch
import torch.nn as nn
import MinkowskiEngine as ME

class LArMatchSpacepointClassifier( nn.Module ):

    def __init__(self, num_input_feats, classifier_nfeatures=[32,32], ndimensions=2 ):
        super(LArMatchSpacepointClassifier,self).__init__()

        # larmatch classifier
        self.final_vec_nfeats = num_input_feats
        lm_class_layers = OrderedDict()
        for i,nfeat in enumerate(classifier_nfeatures):
            if i==0:
                lm_class_layers["lmclassifier_layer%d"%(i)] = torch.nn.Conv1d(num_input_feats,nfeat,1)
            else:
                lm_class_layers["lmclassifier_layer%d"%(i)] = torch.nn.Conv1d(classifier_nfeatures[i-1],nfeat,1)
            lm_class_layers["lmclassifier_norm%d"%(i)] = torch.nn.InstanceNorm1d(nfeat)
            lm_class_layers["lmclassifier_relu%d"%(i)] = torch.nn.ReLU()
        lm_class_layers["lmclassifier_out"] = torch.nn.Conv1d(classifier_nfeatures[-1],2,1)
        self.lm_classifier = nn.Sequential( lm_class_layers )

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
        
        
