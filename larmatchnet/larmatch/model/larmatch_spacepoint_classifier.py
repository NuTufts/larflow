

from collections import OrderedDict
import torch
import torch.nn as nn
import MinkowskiEngine as ME

class LArMatchSpacepointClassifier( nn.Module ):

    def __init__(self, num_input_feats, classifier_nfeatures=[64,64], ndimensions=2, norm_layer='batchnorm' ):
        super(LArMatchSpacepointClassifier,self).__init__()
        assert norm_layer in ['instance','batchnorm','stableinstance']        

        output_classes = 2 # for classification scheme
        #output_classes = 1 # for score matching
        
        # larmatch classifier
        self.final_vec_nfeats = num_input_feats+3
        lm_class_layers = OrderedDict()
        for i,nfeat in enumerate(classifier_nfeatures):
            if i==0:
                lm_class_layers["lmclassifier_layer%d"%(i)] = torch.nn.Conv1d(self.final_vec_nfeats,nfeat,1,bias=False)
            else:
                lm_class_layers["lmclassifier_layer%d"%(i)] = torch.nn.Conv1d(classifier_nfeatures[i-1],nfeat,1,bias=False)
            if norm_layer in ['instance','stableinstance']:
                lm_class_layers["lmclassifier_norm%d"%(i)] = torch.nn.InstanceNorm1d(nfeat)
            elif norm_layer=='batchnorm':
                lm_class_layers["lmclassifier_norm%d"%(i)] = torch.nn.BatchNorm1d(nfeat)
            lm_class_layers["lmclassifier_relu%d"%(i)] = torch.nn.ReLU()
        lm_class_layers["lmclassifier_out"] = torch.nn.Conv1d(classifier_nfeatures[-1],output_classes,1,bias=True)
        # set bias assuming a 1:2 pos:neg class imbalance
        #if output_classes == 1:
        #    lm_class_layers["lmclassifier_out"].bias.data[:] = 0.0
        #lm_class_layers["lmclassifier_out"].bias.data[1] = -1.0        
        self.lm_classifier = nn.Sequential( lm_class_layers )

    def forward( self, triplet_feat_t : torch.FloatTensor ):
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
        
        
