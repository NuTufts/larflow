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
                 leakiness=0.01,
                 ninput_planes=3,
                 use_unet=False,
                 nresnet_blocks=10 ):
        
        super(SpatialEmbed, self).__init__()

        self.use_unet = use_unet

        self.ninput_planes = ninput_planes  
              
        # OFFSET
        self.source_inputlayer_offset  = scn.InputLayer(ndimensions,inputshape,mode=0)

            # STEM
        self.stem_offset = scn.Sequential() 
        self.stem_offset.add( scn.SubmanifoldConvolution(ndimensions, input_nfeatures, stem_nfeatures, 3, False ) )
        self.stem_offset.add( scn.BatchNormLeakyReLU(stem_nfeatures,leakiness=leakiness) )

        if self.use_unet:
            self.stem_offset.add( scn.SubmanifoldConvolution(ndimensions, stem_nfeatures, stem_nfeatures, 3, False ) )
            self.stem_offset.add( scn.BatchNormLeakyReLU(stem_nfeatures,leakiness=leakiness) )
            self.stem_offset.add( scn.SubmanifoldConvolution(ndimensions, stem_nfeatures, stem_nfeatures, 3, False ) )

            # UNET BLOCK    
        if self.use_unet:
            self.unet_layers_offset = scn.UNet( 2, 2,
                                         [stem_nfeatures,
                                          stem_nfeatures*2,
                                          stem_nfeatures*3,
                                          stem_nfeatures*4,
                                          stem_nfeatures*5],
                                         residual_blocks=True,
                                         downsample=[2, 2] )

            # RESNET BLOCK
        if not self.use_unet:
            self.resnet_layers_offset = create_resnet_layer(nresnet_blocks, stem_nfeatures, 4, leakiness=leakiness )
        else:
            self.resnet_layers_offset = create_resnet_layer(1,  stem_nfeatures, 4, leakiness=leakiness )
            self.resnet_layers_offset.add( scn.BatchNormLeakyReLU(stem_nfeatures,leakiness=leakiness) )

        # INPUT LAYERS: converts torch tensor into scn.SparseMatrix
        self.source_inputlayer_seed = scn.InputLayer(ndimensions,inputshape,mode=0)


        # SEED
            # STEM
        self.stem_seed = scn.Sequential() 
        self.stem_seed.add( scn.SubmanifoldConvolution(ndimensions, input_nfeatures, stem_nfeatures, 3, False ) )
        self.stem_seed.add( scn.BatchNormLeakyReLU(stem_nfeatures,leakiness=leakiness) )

        if self.use_unet:
            self.stem_seed.add( scn.SubmanifoldConvolution(ndimensions, stem_nfeatures, stem_nfeatures, 3, False ) )
            self.stem_seed.add( scn.BatchNormLeakyReLU(stem_nfeatures,leakiness=leakiness) )
            self.stem_seed.add( scn.SubmanifoldConvolution(ndimensions, stem_nfeatures, stem_nfeatures, 3, False ) )

            # UNET BLOCK
        if self.use_unet:

            self.unet_layers_seed = scn.UNet( 2, 2,
                                         [stem_nfeatures,
                                          stem_nfeatures*2,
                                          stem_nfeatures*3,
                                          stem_nfeatures*4,
                                          stem_nfeatures*5],
                                         residual_blocks=True,
                                         downsample=[2, 2] )
 
            # RESNET BLOCK
        if not self.use_unet:
            self.resnet_layers_seed = create_resnet_layer(nresnet_blocks, stem_nfeatures, features_per_layer, leakiness=leakiness )
        else:
            self.resnet_layers_seed = create_resnet_layer(1,  stem_nfeatures, features_per_layer, leakiness=leakiness )
            self.resnet_layers_seed.add( scn.BatchNormLeakyReLU(stem_nfeatures,leakiness=leakiness) )


        # OUTPUT FEATURES
        self.nfeatures = features_per_layer
        self.feature_layer_offset = scn.SubmanifoldConvolution(ndimensions, 4, 4, 1, True ) # 4 output channels: x, y, sigma_x, sigma_y
        self.feature_layer_seed = scn.SubmanifoldConvolution(ndimensions, features_per_layer, self.nfeatures, 1, True ) 

        # from there, we move back into a tensor
        self.source_outlayer_offset  = scn.OutputLayer(ndimensions)
        self.source_outlayer_seed = scn.OutputLayer(ndimensions)


    def forward_features( self, coord_plane0_t, plane0_feat_t,
                          batchsize, verbose=False ):

        """
        run the feature generating portion of network only. 
        get feature vector at each pixel coordinate.
        expects 1 planes
        input
        -----
        coord_plane0_t [torch tensor (N_0,3)] list of N pixel coordinates (row,col,batch)
        plane0_feat_t  [torch tensor (N_0,1)] list of pixel values for each N pixel coordinate   
        output
        ------
        tuple containing 3 feature tensors [torch tensors with shapes ((N_0,C), (N_1,C), (N_2,C))]
        """
        if verbose:
            print "[SpatialEmbed::make feature vectors] "
            print "  coord[plane0]=",coord_plane0_t.shape," feat[plane0]=",plane0_feat_t.shape   

        # Form input tuples for input layers
        # adds torch tensors to SparseConvTensor object
        input_plane0 = ( coord_plane0_t, plane0_feat_t, batchsize )

        # generate offset map features
        offset_map = self.source_inputlayer_offset(input_plane0)
        offset_map = self.stem_offset( offset_map )         
        if verbose: print "stem_offset=   ",offset_map.features.shape
        if self.use_unet: 
            offset_map = self.unet_layers_offset(offset_map)        
            if verbose: print "unet_offset=   ",offset_map.features.shape 
        offset_map = self.resnet_layers_offset( offset_map )
        if verbose: print "resnet_offset= ",offset_map.features.shape
        offset_map = self.feature_layer_offset( offset_map )
        if verbose: print "feature_offset=",offset_map.features.shape
                    
        # generate seed map features
        seed_map = self.source_inputlayer_seed(input_plane0)
        seed_map = self.stem_seed( seed_map )         
        if verbose: print "stem_seed=   ",seed_map.features.shape
        if self.use_unet: 
            seed_map = self.unet_layers_seed(seed_map)         
            if verbose: print "unet_seed=   ",seed_map.features.shape
        seed_map = self.resnet_layers_seed( seed_map )
        if verbose: print "resnet_seed= ",seed_map.features.shape
        seed_map = self.feature_layer_seed( seed_map )
        if verbose: print "feature_seed=",seed_map.features.shape

        # extracts torch tensors from SparseConvTensor object        
        offset_map  = self.source_outlayer_offset( offset_map )
        seed_map  = self.source_outlayer_seed( seed_map )

        return offset_map, seed_map