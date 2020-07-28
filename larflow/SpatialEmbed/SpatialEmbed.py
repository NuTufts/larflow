from collections import OrderedDict
import sys
import torch
import torch.nn as nn
import sparseconvnet as scn

sys.path.insert(1, "/home/jhwang/ubdl/larflow/larmatchnet/")
from utils_sparselarflow import create_resnet_layer

class SpatialEmbed(nn.Module):
    def __init__(self, ndimensions=2,
                 inputshape=(1024, 3456),
                 input_nfeatures=1,
                 stem_nfeatures=16,
                 features_per_layer=16,
                 classifier_nfeatures=[32,32],
                 leakiness=0.01,
                 ninput_planes=3,
                 use_unet=False,
                 nresnet_blocks=10 ):
        
        super(SpatialEmbed, self).__init__()

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
        classifier_layers["class0conv"] = torch.nn.Conv1d(self.ninput_planes*features_per_layer,classifier_nfeatures[0],1) #in channels, out channels, kernel size
        classifier_layers["class0relu"] = torch.nn.ReLU()
        
        for ilayer,nfeats in enumerate(classifier_nfeatures[1:]):
            classifier_layers["class%dconv"%(ilayer+1)] = torch.nn.Conv1d(nfeats,nfeats,1)
            classifier_layers["class%drelu"%(ilayer+1)] = torch.nn.ReLU()
        
        classifier_layers["classout"] = torch.nn.Conv1d(nfeats,1,1)

        self.classifier = torch.nn.Sequential( classifier_layers )


    def forward_features( self, coord_plane0_t, plane0_feat_t,
                          coord_plane1_t, plane1_feat_t,
                          coord_plane2_t, plane2_feat_t,
                          batchsize, verbose=False ):

        """
        run the feature generating portion of network only. 
        get feature vector at each pixel coordinate.
        expects three planes
        input
        -----
        coord_plane0_t [torch tensor (N_0,3)] list of N pixel coordinates (row,col,batch)
        plane0_feat_t  [torch tensor (N_0,1)] list of pixel values for each N pixel coordinate
        coord_plane1_t [torch tensor (N_1,3)] list of N pixel coordinates (row,col,batch)
        plane1_feat_t  [torch tensor (N_1,1)] list of pixel values for each N pixel coordinate
        coord_plane2_t [torch tensor (N_2,3)] list of N pixel coordinates (row,col,batch)
        plane2_feat_t  [torch tensor (N_2,1)] list of pixel values for each N pixel coordinate        
        output
        ------
        tuple containing 3 feature tensors [torch tensors with shapes ((N_0,C), (N_1,C), (N_2,C))]
        """
        if verbose:
            print "[larmatch::make feature vectors] "
            print "  coord[plane0]=",coord_plane0_t.shape," feat[plane0]=",plane0_feat_t.shape
            print "  coord[plane1]=",coord_plane1_t.shape," feat[plane1]=",plane1_feat_t.shape
            print "  coord[plane2]=",coord_plane2_t.shape," feat[plane2]=",plane2_feat_t.shape            

        # Form input tuples for input layers
        # adds torch tensors to SparseConvTensor object
        input_plane0 = ( coord_plane0_t, plane0_feat_t, batchsize )
        input_plane1 = ( coord_plane1_t, plane1_feat_t, batchsize )
        input_plane2 = ( coord_plane2_t, plane2_feat_t, batchsize )

        # generate plane0 features
        xplane0 = self.source_inputlayer(input_plane0)
        xplane0 = self.stem( xplane0 )         
        if self.use_unet: 
            xplane0 = self.unet_layers(xplane0)         
        xplane0 = self.resnet_layers( xplane0 )
        xplane0 = self.feature_layer( xplane0 )
        
        # generate plane1 features
        xplane1 = self.target1_inputlayer(input_plane1)
        xplane1 = self.stem( xplane1 )
        if verbose: print "  stem[plane1]=",xplane1.features.shape                    
        if self.use_unet: xplane1 =self.unet_layers(xplane1)
        if verbose: print "  unet[plane1]=",xplane1.features.shape        
        xplane1 = self.resnet_layers( xplane1 )
        if verbose: print "  resnet[plane1]=",xplane1.features.shape        
        xplane1 = self.feature_layer( xplane1 )
        if verbose: print "  feature[plane1]=",xplane1.features.shape        

        # generate plane2 features
        xplane2 = self.target2_inputlayer(input_plane2)
        xplane2 = self.stem( xplane2 )
        if verbose: print "  stem[plane2]=",xplane2.features.shape                            
        if self.use_unet: xplane2 =self.unet_layers(xplane2)
        if verbose: print "  unet[plane2]=",xplane2.features.shape        
        xplane2 = self.resnet_layers( xplane2 )
        if verbose: print "  resnet[plane2]=",xplane2.features.shape                
        xplane2 = self.feature_layer( xplane2 )
        if verbose: print "  feature[plane2]=",xplane2.features.shape                

        # extracts torch tensors from SparseConvTensor object        
        xplane0  = self.source_outlayer( xplane0 )
        xplane1 = self.target1_outlayer( xplane1 )
        xplane2 = self.target2_outlayer( xplane2 )

        return xplane0,xplane1,xplane2