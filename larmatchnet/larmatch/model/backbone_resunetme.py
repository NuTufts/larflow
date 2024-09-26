import torch
import torch.nn as nn
import MinkowskiEngine as ME
from .minkencodedecode import MinkEncodeBase,MinkDecodeBase
from MinkowskiEngine.modules.resnet_block import BasicBlock
from .resnetinstance_block import BasicBlockInstanceNorm

""" 
Implementations of different Residual UNet backbones for larmatch
"""

class MinkEncode6LayerInstance(MinkEncodeBase):
    BLOCK = BasicBlockInstanceNorm
    NORM  = ME.MinkowskiInstanceNorm
    LAYERS = ( 1,  1,  1,   1,   1,   1)
    PLANES = (16, 32, 64, 128, 256, 512)
    INIT_DIM = 16

class MinkDecode6LayerInstance(MinkDecodeBase):
    BLOCK = BasicBlockInstanceNorm
    NORM  = ME.MinkowskiInstanceNorm
    IN_PLANES = (16, 32, 64, 128, 256, 512)    
    LAYERS = (1, 1, 1, 1, 1, 1)    
    PLANES = (256, 128, 64, 32, 16, 16)
    INIT_DIM = 16

class MinkEncode6LayerBasicBlock(MinkEncodeBase):
    BLOCK = BasicBlock
    NORM  = ME.MinkowskiBatchNorm
    LAYERS = ( 1,  1,  1,   1,   1,   1)
    PLANES = (16, 32, 64, 128, 256, 512)
    INIT_DIM = 16

class MinkDecode6LayerBasicBlock(MinkDecodeBase):
    BLOCK = BasicBlock
    NORM  = ME.MinkowskiBatchNorm    
    IN_PLANES = (16, 32, 64, 128, 256, 512)    
    LAYERS = (1, 1, 1, 1, 1, 1)    
    PLANES = (256, 128, 64, 32, 16, 16)
    INIT_DIM = 16
    
class MEResUNet6Layer(nn.Module):
    """
    Residual UNet built using Minkowski engine
    """
    def __init__(self,in_channels=1, out_channels=16, D=2, norm='batchnorm'):
        super(MinkEncodeDecodeUNet34,self).__init__()
        if norm=='batchnorm':
            self.encoder = MinkEncode6LayerBasicBlock(in_channels=in_channels, out_channels=out_channels, D=D)
            self.decoder = MinkDecode6LayerBasicBlock(in_channels=in_channels, out_channels=out_channels, D=D)
        else:
            self.encoder = MinkEncode6LayerInstance(in_channels=in_channels, out_channels=out_channels, D=D)
            self.decoder = MinkDecode6LayerInstance(in_channels=in_channels, out_channels=out_channels, D=D)

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
        

    

