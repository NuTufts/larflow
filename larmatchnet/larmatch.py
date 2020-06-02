from collections import OrderedDict
import torch
import torch.nn as nn
import sparseconvnet as scn
from utils_sparselarflow import create_resnet_layer

class LArMatch(nn.Module):

    def __init__(self,ndimensions=2,inputshape=(1024,3456),
                 nlayers=8,features_per_layer=16,
                 input_nfeatures=1,
                 stem_nfeatures=16,
                 classifier_nfeatures=[32,32],
                 leakiness=0.01,
                 neval=20000,
                 ninput_planes=3,
                 use_unet=False,
                 nresnet_blocks=10,
                 device=torch.device("cpu")):
        super(LArMatch,self).__init__()

        self.use_unet = use_unet
        
        # INPUT LAYERS: converts torch tensor into scn.SparseMatrix
        self.ninput_planes = ninput_planes        
        self.source_inputlayer  = scn.InputLayer(ndimensions,inputshape,mode=0)
        self.target1_inputlayer = scn.InputLayer(ndimensions,inputshape,mode=0)
        self.target2_inputlayer = scn.InputLayer(ndimensions,inputshape,mode=0)

        # STEM
        self.stem = scn.Sequential() 
        self.stem.add( scn.SubmanifoldConvolution(ndimensions, input_nfeatures, stem_nfeatures, 3, False ) )
        self.stem.add( scn.BatchNormLeakyReLU(stem_nfeatures,leakiness=leakiness) )
        if self.use_unet:
            self.stem.add( scn.SubmanifoldConvolution(ndimensions, stem_nfeatures, stem_nfeatures, 3, False ) )
            self.stem.add( scn.BatchNormLeakyReLU(stem_nfeatures,leakiness=leakiness) )
            self.stem.add( scn.SubmanifoldConvolution(ndimensions, stem_nfeatures, stem_nfeatures, 3, False ) )

        # UNET BLOCK
        if self.use_unet:
            self.unet_layers = scn.UNet( 2, 2,
                                         [stem_nfeatures,
                                          stem_nfeatures*2,
                                          stem_nfeatures*3,
                                          stem_nfeatures*4,
                                          stem_nfeatures*5],
                                         residual_blocks=True,
                                         downsample=[2, 2] )

        # RESNET BLOCK
        if not self.use_unet:
            self.resnet_layers = create_resnet_layer(nresnet_blocks, stem_nfeatures, features_per_layer, leakiness=leakiness )
        else:
            self.resnet_layers = create_resnet_layer(1,  stem_nfeatures, features_per_layer, leakiness=leakiness )
            self.resnet_layers.add( scn.BatchNormLeakyReLU(stem_nfeatures,leakiness=leakiness) )

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
        #classifier_layers["class0bn"]   = torch.nn.BatchNorm1d        
        classifier_layers["class0relu"] = torch.nn.ReLU()
        for ilayer,nfeats in enumerate(classifier_nfeatures[1:]):
            classifier_layers["class%dconv"%(ilayer+1)] = torch.nn.Conv1d(nfeats,nfeats,1)
            #classifier_layers["class0bn"]   = torch.nn.BatchNorm1d                    
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
            print "  coord[src]=",coord_src_t.shape," feat[src]=",src_feat_t.shape
            print "  coord dump: "
            print coord_tar2_t
        
        srcx = ( coord_src_t,  src_feat_t,  batchsize )
        tar1 = ( coord_tar1_t, tar1_feat_t, batchsize )
        tar2 = ( coord_tar2_t, tar2_feat_t, batchsize )

        xsrc = self.source_inputlayer(srcx)
        xsrc = self.stem( xsrc )
        if verbose: print "  stem[src]=",xsrc.features.shape            
        if self.use_unet: xsrc =self.unet_layers(xsrc)            
        if verbose: print "  unet[src]=",xsrc.features.shape
        xsrc = self.resnet_layers( xsrc )
        if verbose: print "  resnet[src]=",xsrc.features.shape
        xsrc = self.feature_layer( xsrc )
        if verbose: print "  feature[src]=",xsrc.features.shape
        
        xtar1 = self.target1_inputlayer(tar1)
        xtar1 = self.stem( xtar1 )
        if verbose: print "  stem[tar1]=",xtar1.features.shape                    
        if self.use_unet: xtar1 =self.unet_layers(xtar1)
        if verbose: print "  unet[tar1]=",xtar1.features.shape        
        xtar1 = self.resnet_layers( xtar1 )
        if verbose: print "  resnet[tar1]=",xtar1.features.shape        
        xtar1 = self.feature_layer( xtar1 )
        if verbose: print "  feature[tar1]=",xtar1.features.shape        

        xtar2 = self.target2_inputlayer(tar2)
        xtar2 = self.stem( xtar2 )
        if verbose: print "  stem[tar2]=",xtar2.features.shape                            
        if self.use_unet: xtar2 =self.unet_layers(xtar2)
        if verbose: print "  unet[tar2]=",xtar2.features.shape        
        xtar2 = self.resnet_layers( xtar2 )
        if verbose: print "  resnet[tar2]=",xtar2.features.shape                
        xtar2 = self.feature_layer( xtar2 )
        if verbose: print "  feature[tar2]=",xtar2.features.shape                

        xsrc  = self.source_outlayer(  xsrc )
        xtar1 = self.target1_outlayer( xtar1 )
        xtar2 = self.target2_outlayer( xtar2 )

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
        catvec = torch.cat( (feats_t[0],feats_t[1],feats_t[2]), dim=1 )
        if verbose: print "  concat out: ",catvec.shape
        matchvec = torch.transpose( catvec, 1, 0 ).reshape(1,veclen,npts)
        if verbose: print "  output-triplet-tensor: ",matchvec.shape
            

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

