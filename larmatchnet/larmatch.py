from collections import OrderedDict
import torch
import torch.nn as nn
import sparseconvnet as scn
from utils_sparselarflow import create_resnet_layer

class LArMatch(nn.Module):

    def __init__(self,ndimensions=2,inputshape=(3456,1024),
                 nlayers=8,features_per_layer=16,
                 input_nfeatures=1,
                 stem_nfeatures=16,
                 classifier_nfeatures=[32,32],
                 keypoint_nfeatures=[32,32],
                 kpshift_nfeatures=[64,64,64],
                 leakiness=0.1,
                 neval=20000,
                 ninput_planes=3,
                 device=torch.device("cpu")):
        super(LArMatch,self).__init__()

        # INPUT LAYERS: converts torch tensor into scn.SparseMatrix
        self.ninput_planes = ninput_planes        
        self.source_inputlayer  = scn.InputLayer(ndimensions,inputshape,mode=0)
        self.target1_inputlayer = scn.InputLayer(ndimensions,inputshape,mode=0)
        self.target2_inputlayer = scn.InputLayer(ndimensions,inputshape,mode=0)

        # STEM
        self.stem = scn.Sequential() 
        self.stem.add( scn.SubmanifoldConvolution(ndimensions, input_nfeatures, stem_nfeatures, 3, False ) )
        self.stem.add( scn.BatchNormLeakyReLU(stem_nfeatures,leakiness=leakiness) )
        self.stem.add( scn.SubmanifoldConvolution(ndimensions, stem_nfeatures, stem_nfeatures, 3, False ) )
        self.stem.add( scn.BatchNormLeakyReLU(stem_nfeatures,leakiness=leakiness) )
        self.stem.add( scn.SubmanifoldConvolution(ndimensions, stem_nfeatures, stem_nfeatures, 3, False ) )

        # RESNET BLOCK
        self.resnet_layers = create_resnet_layer(10, stem_nfeatures, features_per_layer, leakiness=leakiness )

        # OUTPUT FEATURES
        self.nfeatures = features_per_layer
        self.feature_layer = scn.SubmanifoldConvolution(ndimensions, features_per_layer, self.nfeatures, 1, True )

        # from there, we move back into a tensor
        self.source_outlayer  = scn.OutputLayer(ndimensions)
        self.target1_outlayer = scn.OutputLayer(ndimensions)
        self.target2_outlayer = scn.OutputLayer(ndimensions)

        # CLASSIFER: MATCH/NO-MATCH
        classifier_layers = OrderedDict()
        classifier_layers["class0conv"] = torch.nn.Conv1d(self.ninput_planes*features_per_layer,classifier_nfeatures[0],1)
        classifier_layers["class0relu"] = torch.nn.ReLU()
        #classifier_layers["class0bn"]   = torch.nn.BatchNorm1d
        for ilayer,nfeats in enumerate(classifier_nfeatures[1:]):
            classifier_layers["class%dconv"%(ilayer+1)] = torch.nn.Conv1d(nfeats,nfeats,1)
            classifier_layers["class%drelu"%(ilayer+1)] = torch.nn.ReLU()
        classifier_layers["classout"] = torch.nn.Conv1d(nfeats,1,1)
        #classifier_layers["sigmoid"]  = torch.nn.Sigmoid()
        self.classifier = torch.nn.Sequential( classifier_layers )
        
        # POINTS TO EVAL PER IMAGE
        self.neval = neval

    def forward_features( self, coord_src_t, src_feat_t,
                          coord_tar1_t, tar1_feat_t,
                          coord_tar2_t, tar2_feat_t,
                          batchsize, verbose=False ):

        """
        run the feature generating portion of network only. get feature vector at each coordinate.
        For deploy only. By saving feature layers, can reduce time run.
        """
        if verbose:
            print "[larmatch] "
            print "  coord[src]=",coord_src_t.shape," feat[src]=",coord_src_t.shape
        
        srcx = ( coord_src_t,  src_feat_t,  batchsize )
        tar1 = ( coord_tar1_t, tar1_feat_t, batchsize )
        tar2 = ( coord_tar2_t, tar2_feat_t, batchsize )

        xsrc = self.source_inputlayer(srcx)
        xsrc = self.stem( xsrc )
        if verbose:
            print "  stem[src]=",xsrc.features.shape
        xsrc = self.resnet_layers( xsrc )
        if verbose:
            print "  resnet[src]=",xsrc.features.shape
        xsrc = self.feature_layer( xsrc )

        xtar1 = self.target1_inputlayer(tar1)
        xtar1 = self.stem( xtar1 )
        xtar1 = self.resnet_layers( xtar1 )
        xtar1 = self.feature_layer( xtar1 )

        xtar2 = self.target2_inputlayer(tar2)
        xtar2 = self.stem( xtar2 )
        xtar2 = self.resnet_layers( xtar2 )
        xtar2 = self.feature_layer( xtar2 )

        xsrc  = self.source_outlayer(  xsrc )
        xtar1 = self.target1_outlayer( xtar1 )
        xtar2 = self.target2_outlayer( xtar2 )

        if verbose:
            print "  outfeat[src]=",xsrc.shape
        
        return xsrc,xtar1,xtar2
                                
    def extract_features(self, feat_u_t, feat_v_t, feat_y_t, index_t, npts, DEVICE, verbose=False ):
        """ 
        take in index list and concat the triplet feature vector
        """
        if verbose:
            print "[larmatch::extract_features]"
            print "  index-shape=",index_t.shape," feat-u-shape=",feat_u_t.shape

        feats   = [ feat_u_t, feat_v_t, feat_y_t ]
        for f in feats:
            f = f.to(DEVICE)
        feats_t = [ torch.index_select( feats[x], 0, index_t[:npts,x] ) for x in xrange(0,3) ]
        if verbose:
            print "  index-selected feats_t[0]=",feats_t[0].shape
        
        veclen = feats_t[0].shape[1]+feats_t[1].shape[1]+feats_t[2].shape[1]
        matchvec = torch.transpose( torch.cat( (feats_t[0],feats_t[1],feats_t[2]), dim=1 ), 1, 0 ).reshape(1,veclen,npts)
        if verbose:
            print "  output-triplet-tensor: ",matchvec.shape

        return matchvec
    
    def classify_triplet(self,triplet_feat_t):
        """
        classify triplet of (u,v,y) wire plane pixel locations as being a true or false position.
        use information from concat feature vectors.

        inputs:
        feat_u_t 
        """
        pred = self.classifier(triplet_feat_t)
        return pred

    def forward_keypoint(self,triplet_feat_t):
        """
        classify triplet of (u,v,y) wire plane pixel combination as being (background,track,shower)
        use information from concat feature vectors.

        inputs:
        feat_u_t 
        """
        pred    = self.keypoint_classifier(matchvec)
        shift = self.kpshift(matchvec)
        return pred,shift
    
