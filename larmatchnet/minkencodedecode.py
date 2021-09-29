# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import torch
import torch.nn as nn
from torch.optim import SGD

import MinkowskiEngine as ME
from resnetinstance_block import BasicBlockInstanceNorm
from resnet import ResNetBase


class MinkEncodeBase(ResNetBase):
    BLOCK = None
    PLANES = None
    DILATIONS = (1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 512, 1024)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling
    # initialize_coords
    def __init__(self, in_channels=3, out_channels=5, D=3):
        ResNetBase.__init__(self, in_channels, out_channels, D)

    def network_initialization(self, in_channels, out_channels, D):
        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM

        # Encoder Stem
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=5, dimension=D)

        self.bn0 = ME.MinkowskiInstanceNorm(self.inplanes)

        nlayers = len( self.LAYERS )

        for ilayer in range(nlayers):

            conv  = ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)                
            bn    = ME.MinkowskiInstanceNorm(self.inplanes)
            block = self._make_layer(self.BLOCK, self.PLANES[ilayer], self.LAYERS[ilayer])
            setattr(self,"layer%02d_convs2"%(ilayer),conv)
            setattr(self,"layer%02d_bn"%(ilayer),bn)
            setattr(self,"layer%02d_block"%(ilayer),block)

        self.relu = ME.MinkowskiReLU(inplace=True)        
            

    def forward(self, x):

        # stem
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        stemout = self.relu(out)
        
        layerout = [stemout]
        for ilayer in range( len(self.LAYERS) ):
            #print("Encoder layer-%d"%(ilayer))
            conv  = getattr(self, "layer%02d_convs2"%(ilayer))
            bn    = getattr(self, "layer%02d_bn"%(ilayer))
            block = getattr(self, "layer%02d_block"%(ilayer))
            out = conv(layerout[-1])

            out = bn(out)
            out = self.relu(out)
            out_px = block(out)
            layerout.append(out_px)

        return layerout

class MinkDecodeBase(ResNetBase):
    BLOCK = None
    PLANES = None
    DILATIONS = (1, 1, 1, 1, 1, 1)
    LAYERS    = (2, 2, 2, 2, 2, 2)
    IN_PLANES = ( 32,  64, 128, 256, 512, 1024 ) 
    PLANES    = (512, 256, 128,  64,  32, 32 )    
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling
    # initialize_coords
    def __init__(self, in_channels, out_channels, D=3):
        ResNetBase.__init__(self, in_channels, out_channels, D)

    def network_initialization(self, in_channels, out_channels, D):
        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM

        nlayers = len( self.LAYERS )
        
        for ilayer in range(nlayers):
            
            convtr = ME.MinkowskiConvolutionTranspose(self.IN_PLANES[-1-ilayer], self.PLANES[ilayer],
                                                      kernel_size=2, stride=2, dimension=D)
            bntr   = ME.MinkowskiInstanceNorm(self.PLANES[ilayer])

            if ilayer+1<nlayers:
                self.inplanes = self.IN_PLANES[-2-ilayer] + self.PLANES[ilayer] * self.BLOCK.expansion
            else:
                # last layer use stem out
                self.inplanes = self.INIT_DIM + self.PLANES[ilayer] * self.BLOCK.expansion
            block  = self._make_layer(self.BLOCK, self.PLANES[ilayer], self.LAYERS[ilayer])
            setattr(self,"decode_layer%02d_convtrs2"%(ilayer),convtr)
            setattr(self,"decode_layer%02d_bntr"%(ilayer),bntr)
            setattr(self,"decode_layer%02d_block"%(ilayer),block)

        self.final = ME.MinkowskiConvolution(
            self.PLANES[-1] * self.BLOCK.expansion,
            out_channels,
            kernel_size=1,
            bias=True,
            dimension=D)
            
        self.relu = ME.MinkowskiReLU(inplace=True)        
            

    def forward(self, encoder_output):
        
        out = encoder_output[-1]
        for ilayer in range( len(self.LAYERS) ):
            #print("decoder layer-%d"%(ilayer))            
            convtr  = getattr(self, "decode_layer%02d_convtrs2"%(ilayer))
            bn      = getattr(self, "decode_layer%02d_bntr"%(ilayer))
            block   = getattr(self, "decode_layer%02d_block"%(ilayer))

            out = convtr(out)
            out = bn(out)            
            out = self.relu(out)

            out = ME.cat(out,encoder_output[-2-ilayer])
            out = block(out)

        out = self.final(out)
        return out
    

class MinkEncode14(MinkEncodeBase):
    BLOCK = BasicBlockInstanceNorm
    LAYERS = (1, 1, 1, 1, 1, 1)

class MinkEncode34(MinkEncodeBase):
    BLOCK = BasicBlockInstanceNorm    
    LAYERS = (2, 3, 4, 6, 8, 8)

class MinkEncode14A(MinkEncode14):
    PLANES = (32, 64, 128, 256, 256, 512)

class MinkEncode34C(MinkEncode34):
    PLANES = (16, 32, 64, 128, 256, 512)
    INIT_DIM = 16

class MinkDecode14(MinkDecodeBase):
    BLOCK = BasicBlockInstanceNorm
    LAYERS = (1, 1, 1, 1, 1)

class MinkDecode34C(MinkDecodeBase):
    BLOCK = BasicBlockInstanceNorm
    IN_PLANES = (16, 32, 64, 128, 256, 512)    
    LAYERS = (1, 1, 1, 1, 1, 1)    
    PLANES = (256, 128, 64, 32, 16, 16)
    INIT_DIM = 16

class MinkDecode14A(MinkDecode14):    
    IN_PLANES = (32, 64, 128, 256, 256, 512)
    PLANES    = (256, 256, 128, 64, 32 )

class MinkEncodeDecodeUNet14(nn.Module):
    def __init__(self,in_channels=3, out_channels=5, D=3):
        super(MinkEncodeDecodeUNet34,self).__init__()    
        self.encoder = MinkEncode34C(in_channels=in_channels, out_channels=out_channels, D=D)
        self.decoder = MinkDecode34C(in_channels=in_channels, out_channels=out_channels, D=D)
        

if __name__ == '__main__':
    import sys
    #from tests.python.common import data_loader
    # loss and network    
    criterion = nn.CrossEntropyLoss()
    #net = MinkEncode34C(in_channels=3, out_channels=5, D=2)
    encoder = MinkEncode14A(in_channels=3, out_channels=5, D=3)    
    print(encoder)
    decoder = MinkDecode14A(in_channels=3, out_channels=5, D=3)
    print(decoder)
    net = MinkEncodeDecodeUNet14(in_channels=3,out_channels=5,D=3)
    print(net)

    # a data loader must return a tuple of coords, features, and labels.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    
    #encoder = encoder.to(device)
    #decoder = decoder.to(device)


    sys.exit(0)

    for i in range(10):
        optimizer.zero_grad()

        # Get new data
        coords, feat, label = data_loader(is_classification=False)
        input = ME.SparseTensor(feat, coordinates=coords, device=device)
        label = label.to(device)

        # Forward
        output = net(input)

        # Loss
        loss = criterion(output.F, label)
        print('Iteration: ', i, ', Loss: ', loss.item())

        # Gradient
        loss.backward()
        optimizer.step()

    # Saving and loading a network
    torch.save(net.state_dict(), 'test.pth')
    net.load_state_dict(torch.load('test.pth'))
