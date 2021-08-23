# This is the script for the LM net itself (Python class)

import torch
import torch.nn as nn
import numpy as np

import sparseconvnet as scn

class LightModelNet(nn.Module):

#    def __init__(self,dimension):
    def __init__(self,dimension, showSizes, nPlanes=1):
#    def __init__(self, dimension, reps, nPlanes=1, nin_features, nout_features, showSizes=True):

        """
        description of parameters
        --------------------------
        dimension [int]    number of spatial dimensions of input data. this should be 3 for now (?)
        reps [int]         number of number of residual modules per layer
        nPlanes [int]      number of input planes. This shoudl be 1 (for now)
        nin_features [int]: number of features in the first convolutional layer
        nout_features [int]: number of features that feed into the regression layer
        showSizes [bool]: if True, print sizes while running forward
        """
        super(LightModelNet,self).__init__()

        # Input size: #############################
        
        """
        self.dimension = dimension
        self.reps = reps
        self.nPlanes = nPlanes
        self.nin_features = nin_features
        self.nout_features = nout_features
        self.showSizes = showSizes
        """
        ######## start of copy

        m = scn.Sequential()
#        nPlanes = 1

        # blockType, n is number of features outputted, reps, stride
        layers = [('b',16,2,1),('b',16,2,1)]

        # Residual block
        def residual(nIn, nOut, stride):
            if stride > 1:
                return scn.Convolution(dimension, nIn, nOut, 3, stride, False)
            elif nIn != nOut:
                return scn.NetworkInNetwork(nIn, nOut, False)
            else:
                return scn.Identity()

        for blockType, n, reps, stride in layers:
            for rep in range(reps):
                if blockType[0] == 'b':  # basic block
                    if rep == 0:
                        m.add(scn.BatchNormReLU(nPlanes))
                        m.add(
                        scn.ConcatTable().add(
                        scn.Sequential().add(
                        scn.SubmanifoldConvolution(dimension, nPlanes, n, 3, False) if stride == 1 else scn.Convolution(dimension, nPlanes, n, 3, stride, False)) .add(
                        scn.BatchNormReLU(n)) .add(
                        scn.SubmanifoldConvolution(dimension,n,n,3,False))) .add(
                        residual(nPlanes, n, stride)))
                    else:
                        m.add(
                        scn.ConcatTable().add(
                        scn.Sequential().add(
                        scn.BatchNormReLU(nPlanes)) .add(
                        scn.SubmanifoldConvolution(dimension, nPlanes, n, 3, False)) .add(
                        scn.BatchNormReLU(n)) .add(
                        scn.SubmanifoldConvolution(dimension, n, n, 3, False))) .add(
                        scn.Identity()))
                nPlanes = n
                m.add(scn.AddTable())
            m.add(scn.BatchNormReLU(nPlanes))
        
        self.m = m
    
        self.dimension = dimension
        self._showSizes = showSizes
        self.mode = 0
        
        self.input = scn.InputLayer(self.dimension, (64, 64, 256),self.mode)
        self.output = scn.SparseToDense(self.dimension, 16)
        self.linear = nn.Linear(16*64*64*256, 32)
        
    def forward( self, coord_t, feat_t):
        if self._showSizes:
            print("coord_t ",coord_t.shape)
            print("feat_t ",feat_t.shape)
        x=(coord_t,feat_t, 8) # batchsize =1
        print("call input")
        x=self.input(x)
        print("call m")
        x=self.m(x)
        print("call output")
        x=self.output(x)
        print("after output")
        x=x.view(x.size(0), -1) # have to flatten first
        x=self.linear(x)
        return x

# to run this script itself:
if __name__ == "__main__":
    
#    dimlen = 16
    net = LightModelNet(3, True)

    print("Just printing the net model: ")
    print(net)
