import torch
import torch.nn as nn
import numpy as np

import sparseconvnet as scn

class LightModelNet(nn.Module):

    def __init__(self,ndimensions):

        """
        description of parameters
        --------------------------
        ndimensions [int]    number of spatial dimensions of input data, default=2
        """

        super(LightModelNet,self).__init__()

        # number of dimensions, input shape
        self.ndimensions = ndimensions

        def forward( self, coord_t):

if __name__ == "__main__":

#    dimlen = 16
    net = LightModelNet(3)

    print net
