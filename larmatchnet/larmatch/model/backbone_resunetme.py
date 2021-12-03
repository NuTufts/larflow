import torch
import torch.nn as nn
from .minkencodedecode import MinkEncodeBase,MinkDecodeBase
from .resnetinstance_block import BasicBlockInstanceNorm

""" 
Implementations of different Residual UNet backbones for larmatch
"""

class MinkEncode6Layer(MinkEncodeBase):
    BLOCK = BasicBlockInstanceNorm
    LAYERS = ( 1,  1,  1,   1,   1,   1)
    PLANES = (16, 32, 64, 128, 256, 512)
    INIT_DIM = 16

class MinkDecode6Layer(MinkDecodeBase):
    BLOCK = BasicBlockInstanceNorm
    IN_PLANES = (16, 32, 64, 128, 256, 512)    
    LAYERS = (1, 1, 1, 1, 1, 1)    
    PLANES = (256, 128, 64, 32, 16, 16)
    INIT_DIM = 16

class MEResUNet6Layer(nn.Module):
    """
    Residual UNet built using Minkowski engine
    """
    def __init__(self,in_channels=1, out_channels=16, D=2):
        super(MinkEncodeDecodeUNet34,self).__init__()    
        self.encoder = MinkEncode6Layer(in_channels=in_channels, out_channels=out_channels, D=D)
        self.decoder = MinkDecode6Layer(in_channels=in_channels, out_channels=out_channels, D=D)

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
        

    

