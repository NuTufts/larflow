from __future__ import print_function
from collections import OrderedDict
import torch
import torch.nn as nn
import sparseconvnet as scn

import MinkowskiEngine as ME
from .resnetinstance_block import BasicBlockInstanceNorm
from .minkencodedecode import MinkEncode34C,MinkDecode34C

class LArVoxelClassifier(nn.Module):

    def __init__(self,input_feats=3,
                 embed_dim=512,
                 out_num_classes=6):
        """
        parameters
        -----------
        """
        super(LArVoxelClassifier,self).__init__()
        self.D = 3
        self.encoder = MinkEncode34C(in_channels=input_feats, out_channels=1, D=self.D)

        print(self.encoder.PLANES)
        
        totchannels = self.encoder.INIT_DIM
        for nch in self.encoder.PLANES:
            totchannels += nch
        print("tot channels: ",totchannels)
        
        multiscale_layers = []
        multiscale_layers.append(
            nn.Sequential(
                ME.MinkowskiConvolution(
                    totchannels,
                    embed_dim//4,
                    kernel_size=3,
                    stride=2,
                    dimension=self.D,
                ),
                ME.MinkowskiInstanceNorm(embed_dim//4),
                ME.MinkowskiLeakyReLU()))
        multiscale_layers.append(
            nn.Sequential(
                ME.MinkowskiConvolution(
                    embed_dim//4,
                    embed_dim//2,
                    kernel_size=3,
                    stride=2,
                    dimension=self.D,
                ),
                ME.MinkowskiInstanceNorm(embed_dim//2),
                ME.MinkowskiLeakyReLU() ))
        multiscale_layers.append(
            nn.Sequential(
                ME.MinkowskiConvolution(
                    embed_dim//2,
                    embed_dim,
                    kernel_size=3,
                    stride=2,
                    dimension=self.D,
                ),
                ME.MinkowskiInstanceNorm(embed_dim),
                ME.MinkowskiLeakyReLU(),
            ) )

        self.multiscale_conv = nn.Sequential(*multiscale_layers)
        self.global_max_pool = ME.MinkowskiGlobalMaxPooling()
        self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()

        self.final = nn.Sequential(
            self.get_mlp_block(embed_dim * 2, 512),
            ME.MinkowskiDropout(),
            self.get_mlp_block(512, 512),
            ME.MinkowskiLinear(512, out_num_classes, bias=True),
        )

    def get_mlp_block(self, in_channel, out_channel):
        return nn.Sequential(
            ME.MinkowskiLinear(in_channel, out_channel, bias=False),
            ME.MinkowskiInstanceNorm(out_channel),
            ME.MinkowskiLeakyReLU(),
        )
    
    def forward(self, xinput : ME.TensorField ):        
        xsparse = xinput.sparse()
        encoder_out   = self.encoder(xsparse)

        # we slice, which broadcasts pooled features back to original resolution?
        y = [ x.slice(xinput) for x in encoder_out ]
        y = ME.cat(y).sparse()
        
        y = self.multiscale_conv(y)
        y = ME.cat( [self.global_max_pool(y),self.global_avg_pool(y)] )
        y = self.final(y)
        
        return y

