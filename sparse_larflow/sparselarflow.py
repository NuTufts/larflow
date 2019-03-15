import os,sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sparseconvnet as scn
import time
import math
import numpy as np

#data.init(-1,24,24*8,16)
#dimension = 3
#reps = 1 #Conv block repetition factor
#m = 32   #Unet number of features
#nPlanes = [m, 2*m, 3*m, 4*m, 5*m] #UNet number of features per level
class SparseEncoder(nn.Module):
    def __init__( self, name, nreps, ninputchs, chs_per_layer ):
        nn.Module.__init__(self)
        self.chs_per_layer = chs_per_layer
        self.inputchs = ninputchs
        self.residual_blocks = True
        self.dimension = 2
        self._layers = []
        self._name = name
        
        for ilayer,noutputchs in enumerate(self.chs_per_layer):
            if ilayer>0:
                ninchs = self.chs_per_layer[ilayer-1]
            else:
                ninchs = self.inputchs
            layer = self.make_encoder_layer(ninchs,noutputchs,nreps)
            self._layers.append(layer)
            setattr(self,"%s_enclayer%d"%(name,ilayer),layer)

    def block(self, m, a, b, leakiness=0.01):
        """
        append to the sequence:
        produce output of [identity,3x3+3x3] then add together

        inputs
        ------
        m: scn.Sequential module (modified)
        a: number of input channels
        b: number of output channels
        """        
        if self.residual_blocks: #ResNet style blocks
            m.add(scn.ConcatTable()
                  .add(scn.Identity() if a == b else scn.NetworkInNetwork(a, b, False))
                  .add(scn.Sequential()
                    .add(scn.BatchNormLeakyReLU(a,leakiness=leakiness))
                    .add(scn.SubmanifoldConvolution(self.dimension, a, b, 3, False))
                    .add(scn.BatchNormLeakyReLU(b,leakiness=leakiness))
                    .add(scn.SubmanifoldConvolution(self.dimension, b, b, 3, False)))
             ).add(scn.AddTable())

    def make_encoder_layer(self,ninputchs,noutputchs,nreps,
                           leakiness=0.01,downsample=[2, 2]):
        """
        inputs
        ------
        ninputchs: number of features going into layer
        noutputchs: number of features output by layer
        """
        encode_blocks = scn.Sequential()
        for iblock in xrange(nreps):
            self.block(encode_blocks,ninputchs,ninputchs)

        m = scn.Sequential()
        m.add(encode_blocks)
        m.add(scn.BatchNormLeakyReLU(ninputchs,leakiness=leakiness))
        m.add(scn.Convolution(self.dimension, ninputchs, noutputchs,
                              downsample[0], downsample[1], False))
        return m

    def forward(self,x):
        self.layerout = []
        for ilayer,layer in enumerate(self._layers):
            if ilayer==0:
                inputx = x
            else:
                inputx = self.layerout[-1]
            out = layer(inputx)
            print "%s_enclayerout[%d]: "%(self._name,ilayer),inputx.features.shape,inputx.spatial_size,"-->",out.features.shape,out.spatial_size
            self.layerout.append( out )
        return self.layerout
        

class SparseLArFlow(nn.Module):
    def __init__(self, inputshape, reps, nfeatures, nplanes):
        nn.Module.__init__(self)

        # set parameters
        self.dimensions = 2 # not playing with 3D for now

        # input shape: LongTensor, tuple, or list. Handled by InputLayer
        # size of each spatial dimesion
        self.inputshape = inputshape
        if len(self.inputshape)!=self.dimensions:
            raise ValueError("expected inputshape to contain size of 2 dimensions only. given %d values"%(len(self.inputshape)))
        
        # mode variable: how to deal with repeated data
        self.mode = 0

        # nfeatures
        self.nfeatures = nfeatures

        # plane structure
        self.nPlanes = [ self.nfeatures*2**(n+1) for n in xrange(nplanes) ]

        # repetitions (per plane)
        self.reps = reps
        
        # residual blocks
        self.residual_blocks = True

        # need encoder for both source and target
        # then cat tensor
        # and produce one decoder for flow, another decoder for visibility
        
        # model:
        # input
        self.src_inputlayer  = scn.InputLayer(self.dimensions, self.inputshape, mode=self.mode)
        self.tar1_inputlayer = scn.InputLayer(self.dimensions, self.inputshape, mode=self.mode)
        self.tar2_inputlayer = scn.InputLayer(self.dimensions, self.inputshape, mode=self.mode)        
        
        # stem
        self.src_stem  = scn.SubmanifoldConvolution(self.dimensions, 1, self.nfeatures, 3, False)
        self.tar1_stem = scn.SubmanifoldConvolution(self.dimensions, 1, self.nfeatures, 3, False)
        self.tar2_stem = scn.SubmanifoldConvolution(self.dimensions, 1, self.nfeatures, 3, False) 

        # encoders
        self.source_encoder  = SparseEncoder( "src",  self.reps, self.nfeatures, self.nPlanes )
        self.target1_encoder = SparseEncoder( "tar1", self.reps, self.nfeatures, self.nPlanes )
        self.target2_encoder = SparseEncoder( "tar2", self.reps, self.nfeatures, self.nPlanes )
        
        # concat
        self.join_enclayers = []
        for ilayer in xrange(len(self.nPlanes)):
            self.join_enclayers.append(scn.JoinTable())
            setattr(self,"join_enclayers%d"%(ilayer),self.join_enclayers[ilayer])

        
    def forward(self, coord_t, src_feat_t, tar1_feat_t, tar2_feat_t, batchsize ):
        srcx = ( coord_t, src_feat_t,  batchsize )
        tar1 = ( coord_t, tar1_feat_t, batchsize )
        tar2 = ( coord_t, tar2_feat_t, batchsize )

        # source encoder
        srcx = self.src_inputlayer(srcx)
        srcx = self.src_stem(srcx)
        srcout_v = self.source_encoder(srcx)
        
        tar1 = self.tar1_inputlayer(tar1)
        tar1 = self.tar1_stem(tar1)
        tar1out_v = self.target1_encoder(tar1)

        tar2 = self.tar1_inputlayer(tar2)
        tar2 = self.tar1_stem(tar2)
        tar2out_v = self.target2_encoder(tar2)

        self.joinout = []
        for src,tar1,tar2,joiner in zip(srcout_v,tar1out_v,tar2out_v,self.join_enclayers):
            self.joinout.append( joiner( (src,tar1,tar2) ) )
        
        return self.joinout

if __name__ == "__main__":

    model = SparseLArFlow( (256,256), 2, 16, 5 )
    print model

    # random 100 points from a hypothetical 256x256
    xcoords = np.zeros( (100,2), dtype=np.int )
    xcoords[:,0] = np.random.randint( 0, 256, 100 )
    xcoords[:,1] = np.random.randint( 0, 256, 100 )
    srcx = np.random.random( (100,1) ).astype(np.float32)
    tar1 = np.random.random( (100,1) ).astype(np.float32)
    tar2 = np.random.random( (100,1) ).astype(np.float32)    

    coord_t = torch.from_numpy(xcoords)
    src_feats_t  = torch.from_numpy(srcx)
    tar1_feats_t = torch.from_numpy(tar1)
    tar2_feats_t = torch.from_numpy(tar2)
    batchsize = 1
    
    print "coord-shape: ",coord_t.shape
    print "src feats-shape: ",src_feats_t.shape    
    out_v = model( coord_t, src_feats_t, tar1_feats_t, tar2_feats_t, batchsize )
    for i,out in enumerate(out_v):
        print "layerout[%d]: "%(i),"out=[",out.features.shape,out.spatial_size,"]"
