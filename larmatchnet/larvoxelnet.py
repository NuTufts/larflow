from __future__ import print_function
from collections import OrderedDict
import torch
import torch.nn as nn
import sparseconvnet as scn

import MinkowskiEngine as ME
from resnetinstance_block import BasicBlockInstanceNorm
from minkencodedecode import MinkEncode34C,MinkDecode34C
from larmatch_keypoint_classifier import LArMatchKeypointClassifier

class LArVoxelMultiDecoder(nn.Module):

    def __init__(self,dimension=3,
                 input_feats=3,
                 num_scales=4,
                 num_layers_per_scale=4,
                 features_per_scale=[16,16,32,64],
                 classifier_nfeatures=[32,32],
                 run_ssnet=True,
                 run_kplabel=True):
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
        super(LArVoxelMultiDecoder,self).__init__()

        self.encoder = MinkEncode34C(in_channels=input_feats, out_channels=1, D=dimension)

        # each classifier has a decoder
        lm_out = OrderedDict()
        lm_out["lm_decoder"]    = MinkDecode34C(in_channels=input_feats, out_channels=32, D=dimension)        
        lm_out["lm_dropout"]    = ME.MinkowskiDropout(0.5)
        lm_out["lm_classifier"] = ME.MinkowskiConvolution( 32, 2, kernel_size=1, stride=1, dimension=dimension )
        self.larmatch_out = nn.Sequential( lm_out )

        # ssnet
        self.run_ssnet = run_ssnet
        if self.run_ssnet:
            ssnet_out = OrderedDict()
            ssnet_out["ssnet_decoder"] = MinkDecode34C(in_channels=input_feats, out_channels=32, D=dimension)        
            ssnet_out["ssnet_dropout"] = ME.MinkowskiDropout(0.5)
            ssnet_out["ssnet_classifier"] = ME.MinkowskiConvolution( 32, 7, kernel_size=1, stride=1, dimension=dimension )
            self.ssnet_out = nn.Sequential( ssnet_out )

        # keypoint
        self.run_kplabel = run_kplabel
        if self.run_kplabel:
            kp_out = OrderedDict()
            kp_out["kp_decoder"] = MinkDecode34C(in_channels=input_feats, out_channels=32, D=dimension)        
            kp_out["kp_dropout"] = ME.MinkowskiDropout(0.5)
            kp_out["kplabel_classifier0_conv"] = ME.MinkowskiConvolution( 32, 32, kernel_size=1, stride=1, dimension=dimension )
            kp_out["kplabel_classifier0_bn"]   = ME.MinkowskiInstanceNorm(32)
            kp_out["kplabel_classifier0_relu"] = ME.MinkowskiReLU(inplace=True)
            kp_out["kplabel_classifier1_conv"] = ME.MinkowskiConvolution( 32, 32, kernel_size=1, stride=1, dimension=dimension )
            kp_out["kplabel_classifier1_bn"]   = ME.MinkowskiInstanceNorm(32)
            kp_out["kplabel_classifier1_relu"] = ME.MinkowskiReLU(inplace=True)
            kp_out["kplabel_regression_conv"]  = ME.MinkowskiConvolution( 32, 6, kernel_size=1, stride=1, dimension=dimension )
            kp_out["kplabel_sigmoid"]          = ME.MinkowskiSigmoid()
            self.kp_out = nn.Sequential( kp_out )
        
    def forward(self,xinput):
        encoder_out   = self.encoder(xinput)
        lm_out = self.larmatch_out( encoder_out )
        if self.run_ssnet:
            #ssnet_out = self.ssnet_out( encoder_out )
            ssnet_out = torch.transpose( self.ssnet_out( encoder_out ).F, 1,0 ).unsqueeze(0)
        else:
            ssnet_out = None

        if self.run_kplabel:
            kplabel_out = torch.transpose( self.kp_out( encoder_out ).F, 1,0 ).unsqueeze(0)
        else:
            kplabel_out = None
        return {"larmatch":lm_out,"ssnet":ssnet_out,"kplabel":kplabel_out}


if __name__ == "__main__":

    model = LArVoxelMultiDecoder()
    print(model)
