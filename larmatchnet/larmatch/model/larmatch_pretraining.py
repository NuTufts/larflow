import os,sys
import torch
import torch.nn as nn
from collections import OrderedDict
import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck

from .backbone_resunetme import MinkEncode6LayerBasicBlock, MinkAEDecode6LayerBasicBlock

class LArMatchPretrainingAutoEncoder(nn.Module):

    def __init__(self,ndimensions=2,
                 inputshape=(1024,3584),                 
                 input_nfeatures=1,
                 input_nplanes=3,
                 norm_layer='batchnorm'):
        """
        parameters
        -----------
        ndimensions [int]    number of spatial dimensions of input data, default=2
        inputshape  [tuple of int]  size of input tensor/image in (num of tick pixels, num of wire pixels), default=(1024,3456)
        input_nfeatures [int] number of features in the input tensor, default=1 (the image charge)
        """
        super(LArMatchPretrainingAutoEncoder,self).__init__()
        
        # INPUT LAYERS: converts torch tensor into Minkowski Sparse Tensor
        self.ninput_planes = input_nplanes
        
        # STEM
        stem_nfeatures = 16
        stem_nlayers = 3
        stem_layers = OrderedDict()
        if stem_nlayers==1:
            respath = ME.MinkowskiConvolution( input_nfeatures, stem_nfeatures, kernel_size=1, stride=1, dimension=ndimensions )
            if norm_layer=='instancenorm':            
                block   = BasicBlockInstanceNorm( input_nfeatures, stem_nfeatures, dimension=ndimensions, downsample=respath )
            elif norm_layer=='batchnorm':
                block   = BasicBlock( input_nfeatures, stem_nfeatures, dimension=ndimensions, downsample=respath )
            stem_layers["stem_layer0"] = block
        else:
            for istem in range(stem_nlayers):
                if istem==0:
                    respath = ME.MinkowskiConvolution( input_nfeatures, stem_nfeatures, kernel_size=1, stride=1, dimension=ndimensions )
                    if norm_layer=='instancenorm':
                        block   = BasicBlockInstanceNorm( input_nfeatures, stem_nfeatures, dimension=ndimensions, downsample=respath )
                    elif norm_layer=='batchnorm':
                        block   = BasicBlock( input_nfeatures, stem_nfeatures, dimension=ndimensions, downsample=respath )                    
                else:
                    if norm_layer=='instancenorm':
                        block   = BasicBlockInstanceNorm( stem_nfeatures, stem_nfeatures, dimension=ndimensions  )
                    elif norm_layer=='batchnorm':
                        block   = BasicBlock( stem_nfeatures, stem_nfeatures, dimension=ndimensions  )                    
                stem_layers["stem_layer%d"%(istem)] = block
            
        self.stem = nn.Sequential(stem_layers)

        # RESIDUAL UNET FOR FEATURE CONSTRUCTION
        if norm_layer=="instancenorm":
            self.encoder = MinkEncode6LayerInstance( in_channels=stem_nfeatures, out_channels=stem_nfeatures, D=2 )
            self.decoder = MinkDecode6LayerInstance( in_channels=stem_nfeatures, out_channels=stem_nfeatures, D=2 )
        elif norm_layer=="batchnorm":
            self.encoder = MinkEncode6LayerBasicBlock( in_channels=stem_nfeatures, out_channels=stem_nfeatures, D=2 )
            self.decoder = MinkAEDecode6LayerBasicBlock( in_channels=stem_nfeatures, out_channels=1, D=2 )
        else:
            raise ValueError("unrecognized norm_layer value: ",norm_layer)
            

        # feature to pixel predict layer

    def forward(self,input_wireplane_sparsetensors, matchtriplets, batch_size ):
        # run the encoder
        xencoder_v = []
        for p,x_input in enumerate(input_wireplane_sparsetensors):
            #print(x_input)            
            x = self.stem(x_input)
            x_encode = self.encoder(x)
            # the output of x_encode is a list with the output activations of each layer
            xencoder_v.append(x_encode)

        # now we copy the weights of the encoder to the decoder?
        
            
            x_decode = self.decoder(x_encode)
            #print("------------------------------------------------------------")
            #print("output features plane[",p,"] ",x_decode.shape)
            #print(x_decode)
            #print("------------------------------------------------------------")
            x_feat_v.append( x_decode )
        


