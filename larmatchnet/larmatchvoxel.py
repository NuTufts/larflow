from __future__ import print_function
from collections import OrderedDict
import torch
import torch.nn as nn
import sparseconvnet as scn

import MinkowskiEngine as ME
from resnetinstance_block import BasicBlockInstanceNorm
from minkunet import MinkUNet34B

class LArMatchVoxel(nn.Module):

    def __init__(self,dimension=3,
                 input_feats=3,
                 num_scales=4,
                 num_layers_per_scale=4,
                 features_per_scale=[16,16,32,64],
                 classifier_nfeatures=[32,32] ):
        """
        parameters
        -----------
        ndimensions [int]    number of spatial dimensions of input data, default=2
        inputshape  [tuple of int]  size of input tensor/image in (num of tick pixels, num of wire pixels), default=(1024,3456)
        input_nfeatures [int] number of features in the input tensor, default=1 (the image charge)
        stem_nfeatures [int] number of features in the stem layers, also controls num of features in unet layers, default=16
        features_per_layer [int] number of channels in the resnet layers that follow unet layers, default=16
        classifier_nfeatures [tuple of int] number of channels per hidden layer of larmatch classification network, default=[32,32]
        leakiness [float] leakiness of LeakyRelu activiation layers, default=0.01
        ninput_planes [int] number of input image planes, default=3
        use_unet [bool] if true, use UNet layers, default=True
        nresnet_blocks [int] number of resnet blocks if not using unet, DEPRECATED, default=10
        """
        super(LArMatchVoxel,self).__init__()

        # stem
        stem_features = 16
        stem_nlayers = 3
        stem_layers = OrderedDict()
        if stem_nlayers==1:
            respath = ME.MinkowskiConvolution( input_feats, 32, kernel_size=1, stride=1, dimension=dimension )
            block   = BasicBlockInstanceNorm( input_feats, 32, dimension=dimension, downsample=respath )
            stem_layers["stem_layer0"] = block
        else:
            for istem in range(stem_nlayers):
                if istem==0:
                    respath = ME.MinkowskiConvolution( input_feats, stem_features, kernel_size=1, stride=1, dimension=dimension )
                    block   = BasicBlockInstanceNorm( input_feats, stem_features, dimension=dimension, downsample=respath )
                else:
                    block   = BasicBlockInstanceNorm( stem_features, stem_features, dimension=dimension  )
                stem_layers["stem_layer%d"%(istem)] = block
            
        self.stem = nn.Sequential(stem_layers)
        self.unet = MinkUNet34B(stem_features,stem_features,D=dimension)
        self.dropout = ME.MinkowskiDropout()

        # classifier
        final_vec_nfeats = stem_features
        lm_class_layers = OrderedDict()
        for i,nfeat in enumerate(classifier_nfeatures):
            if i==0:
                lm_class_layers["lmclassifier_layer%d"%(i)] = ME.MinkowskiConvolution( final_vec_nfeats, nfeat, kernel_size=1, stride=1, dimension=dimension )
            else:
                lm_class_layers["lmclassifier_layer%d"%(i)] = ME.MinkowskiConvolution( classifier_nfeatures[i-1], nfeat, kernel_size=1, stride=1, dimension=dimension )
            lm_class_layers["lmclassifier_norm%d"%(i)] = ME.MinkowskiInstanceNorm(nfeat)
            lm_class_layers["lmclassifier_relu%d"%(i)] = ME.MinkowskiReLU(inplace=True)
        lm_class_layers["lmclassifier_out"] = ME.MinkowskiConvolution( classifier_nfeatures[-1], 2, kernel_size=1, stride=1, dimension=dimension )
        self.lm_classifier = nn.Sequential( lm_class_layers )

    def init_weights(self):
        for n,layers in enumerate(self.scale_layers):
            for op in layers:
                if type(op) is ME.MinkowskiConvolutionTranspose:
                    print("freeze ",n,": ",op)
                    op.kernel.requires_grad = False                    
                    op.kernel.fill_(1.0)

    def build_pooling_net(self, num_scales ):
        
        self.num_scale = num_scales

        # define N parallel tracks to build up feature representations        
        self.pooling = []
        self.scale_layers = nn.ModuleList()
        final_vec_nfeats = 0
        for n in range(self.num_scale):
            if n==0:
                self.pooling.append(None)
            else:
                self.pooling.append( ME.MinkowskiAvgPooling(stride=2,kernel_size=2,dimension=3) )
                
            scaleseq = OrderedDict()
            
            # add feature generation
            final_vec_nfeats += features_per_scale[n]
            for nl in range(num_layers_per_scale):
                if nl==0:
                    # first layer must change nfeatures on residual path
                    respath = ME.MinkowskiConvolution( input_feats, features_per_scale[n], kernel_size=1, stride=1, dimension=dimension )
                    block = BasicBlockInstanceNorm( input_feats, features_per_scale[n], dimension=dimension, downsample=respath )
                else:
                    block = BasicBlockInstanceNorm( features_per_scale[n], features_per_scale[n], dimension=dimension )

                scaleseq["scale%d_resnet%d"%(n,nl)] = block
                
            # now upsample to original resolution
            for i_up in range(n):
                upsample = ME.MinkowskiConvolutionTranspose( features_per_scale[n],
                                                             features_per_scale[n],
                                                             kernel_size=2,
                                                             stride=2,
                                                             dimension=dimension )
                scaleseq["scale%d_up%d"%(n,i_up)] = upsample
                #norm = ME.MinkowskiInstanceNorm(features_per_scale[n])                
                #scalesqe.add( norm )
            self.scale_layers.append( nn.Sequential(scaleseq) )
        
        
    def forward(self,xinput):
        x   = self.stem(xinput)
        x   = self.unet(x)
        x   = self.dropout(x)
        out = self.lm_classifier( x )
        return out

    def forward_pool_pyramid(self,xinput):
        xinput_last = xinput
        scale_out = []
        for n,pool in enumerate(self.pooling):
            if pool is None:
                xscale = xinput_last
            else:
                xscale = pool(xinput_last)
                
            #print(f"xinput[scale={n}]: ",xscale.shape)
            sout = self.scale_layers[n](xscale)
            print(f"sout[scale={n}]: ",sout.shape," ",sout.F[:10])
            scale_out.append(sout)
            xinput_last = xscale
        

