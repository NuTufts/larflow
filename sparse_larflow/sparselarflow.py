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


            if ilayer+1<len(self.chs_per_layer):
                layer = self.make_encoder_layer(ninchs,noutputchs,nreps)
            else:
                #layer = self.make_encoder_layer(ninchs,noutputchs,nreps, downsample=None)
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
            if downsample is not None:
                self.block(encode_blocks,ninputchs,ninputchs)
            else:
                if iblock==0:
                    self.block(encode_blocks,ninputchs,noutputchs)
                else:
                    self.block(encode_blocks,noutputchs,noutputchs)

        m = scn.Sequential()
        m.add(encode_blocks)
        if downsample is not None:
            m.add(scn.BatchNormLeakyReLU(ninputchs,leakiness=leakiness))
            m.add(scn.Convolution(self.dimension, ninputchs, noutputchs,
                                  downsample[0], downsample[1], False))

        return m

    def forward(self,x):
        layerout = []
        for ilayer,layer in enumerate(self._layers):
            if ilayer==0:
                inputx = x
            else:
                inputx = layerout[-1]
            out = layer(inputx)
            print "[%s] Encode Layer[%d]: "%(self._name,ilayer),inputx.features.shape,inputx.spatial_size,"-->",out.features.shape,out.spatial_size
            layerout.append( out )
        return layerout

class SparseDecoder(nn.Module):
    def __init__( self, name, nreps, inputchs_per_layer, outputchs_per_layer ):
        nn.Module.__init__(self)
        self.inputchs_per_layer  = inputchs_per_layer
        self.outputchs_per_layer = outputchs_per_layer
        self.residual_blocks = True
        self.dimension = 2
        self._deconv_layers = []
        self._res_layers = []
        self._joiner = []
        self._name = name
        print self.inputchs_per_layer
        print self.outputchs_per_layer
        for ilayer,(ninputchs,noutputchs) in enumerate(zip(self.inputchs_per_layer,self.outputchs_per_layer)):
            if ilayer+1<len(outputchs_per_layer):
                islast = False
            else:
                islast = True
            deconv,res,joiner = self.make_decoder_layer(ilayer,ninputchs,noutputchs,nreps,islast=islast)
            self._deconv_layers.append(deconv)
            self._res_layers.append(res)
            self._joiner.append(joiner)

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

    def make_decoder_layer(self,ilayer,ninputchs,noutputchs,nreps,
                           leakiness=0.01,downsample=[2, 2],islast=False):
        """
        defines two layers: 
          1) the deconv layer pre-concat 
          2) residual blocks post-concat
        
        inputs
        ------
        ninputchs: number of features going into layer
        noutputchs: number of features output by layer
        """

        # resnet block
        decode_blocks = scn.Sequential()
        for iblock in xrange(nreps):
            if iblock==0:
                self.block(decode_blocks,ninputchs,2*noutputchs)
            else:
                self.block(decode_blocks,2*noutputchs,2*noutputchs)
        setattr(self,"decodeblocks%d"%(ilayer),decode_blocks)

        # deconv
        m = scn.Sequential()
        m.add(scn.BatchNormLeakyReLU(2*noutputchs,leakiness=leakiness))
        m.add(scn.Deconvolution(self.dimension, 2*noutputchs, noutputchs,
                                downsample[0], downsample[1], False))
        setattr(self,"deconv%d"%(ilayer),m)
        print "DecoderLayer[",ilayer,"] inputchs[",ninputchs," -> resout[",2*noutputchs,"] -> deconv output[",noutputchs,"]"


        if not islast:
            # joiner for skip connections        
            joiner = scn.JoinTable()
            setattr(self,"skipjoin%d"%(ilayer),joiner)
        else:
            joiner = None
        return m,decode_blocks,joiner

    def forward(self,encoder_layers):
        layerout = None
        for ilayer,(deconvl,resl,joiner) in enumerate(zip(self._deconv_layers,self._res_layers,self._joiner)):
            if ilayer==0:
                # first layer, input tensor comes from last encoder layer
                inputx = encoder_layers[-(1+ilayer)]
            else:
                # otherwise, get the last tensor that was created
                inputx = layerout
            #print "decoder-input[",ilayer,"] input=",inputx.features.shape,inputx.spatial_size

            # res conv
            out = resl(inputx)
            #print "decode-res[",ilayer,"]: outdecode=",out.features.shape,out.spatial_size

            # upsample
            out = deconvl(out)
            #print "decode-deconv[",ilayer,"]: outdecode=",out.features.shape,out.spatial_size

            # concat if not last layer, for last layer, joiner will be None
            if joiner is not None:
                # get next encoder layer
                catlayer  = encoder_layers[-(2+ilayer)]
                #print "decode-concat[",ilayer,"]: appending=",catlayer.features.shape,catlayer.spatial_size

                # concat
                out   = joiner( (out,catlayer) )
            
            print "[%s] Decode Layer[%d]: "%(self._name,ilayer),inputx.features.shape,inputx.spatial_size,"-->",out.features.shape,out.spatial_size            
            layerout = out
        return layerout
    

class SparseLArFlow(nn.Module):
    def __init__(self, inputshape, reps, nin_features, nout_features, nplanes):
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
        self.nfeatures = nin_features
        self.nout_features = nout_features

        # plane structure
        self.nPlanes = [ self.nfeatures*2**(n+1) for n in xrange(nplanes) ]
        print self.nPlanes

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

        # calculate decoder planes
        self.decode_layers_inchs  = []
        self.decode_layers_outchs = []
        for ilayer,enc_outchs in enumerate(reversed(self.nPlanes)):
            self.decode_layers_inchs.append( 4*enc_outchs if ilayer>0 else 3*enc_outchs )
            self.decode_layers_outchs.append( self.nPlanes[-(1+ilayer)]/2 )
        print "decode in chs: ",self.decode_layers_inchs
        print "decode out chs: ",self.decode_layers_outchs

        # decoders
        self.flow1_decoder = SparseDecoder( "flow1", self.reps,
                                            self.decode_layers_inchs,
                                            self.decode_layers_outchs )
        self.flow2_decoder = SparseDecoder( "flow2", self.reps,
                                            self.decode_layers_inchs,
                                            self.decode_layers_outchs )

        # last deconv concat
        self.flow1_concat = scn.JoinTable()
        self.flow2_concat = scn.JoinTable()        
        
        # final feature set convolution
        flow_resblock_inchs = 3*self.nfeatures + self.decode_layers_outchs[-1]
        self.flow1_resblock = scn.Sequential()
        for iblock in xrange(self.reps):
            if iblock==0:
                self.block(self.flow1_resblock,flow_resblock_inchs,self.nout_features)
            else:
                self.block(self.flow1_resblock,self.nout_features,self.nout_features)
        self.flow2_resblock = scn.Sequential()
        for iblock in xrange(self.reps):
            if iblock==0:
                self.block(self.flow2_resblock,flow_resblock_inchs,self.nout_features)
            else:
                self.block(self.flow2_resblock,self.nout_features,self.nout_features)

        # regression layer
        self.flow1_out = scn.SubmanifoldConvolution(self.dimensions,self.nout_features,1,1,True)
        self.flow2_out = scn.SubmanifoldConvolution(self.dimensions,self.nout_features,1,1,True)



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
                    .add(scn.SubmanifoldConvolution(self.dimensions, a, b, 3, False))
                    .add(scn.BatchNormLeakyReLU(b,leakiness=leakiness))
                    .add(scn.SubmanifoldConvolution(self.dimensions, b, b, 3, False)))
             ).add(scn.AddTable())        
        
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

        # concat features from all three planes
        joinout = []
        for _src,_tar1,_tar2,_joiner in zip(srcout_v,tar1out_v,tar2out_v,self.join_enclayers):
            joinout.append( _joiner( (_src,_tar1,_tar2) ) )

        # Flow 1: src->tar1
        # ------------------
        # use 3-plane features to make flow features
        flow1 = self.flow1_decoder( joinout )
        
        # concat stem out with decoder out
        flow1 = self.flow1_concat( (flow1,srcx,tar1,tar2) )

        # last feature conv layer
        flow1 = self.flow1_resblock( flow1 )

        # finally, 1x1 conv layer from features to flow value
        flow1 = self.flow1_out( flow1 )

        # Flow 2: src->tar1
        # ------------------
        # use 3-plane features to make flow features
        flow2 = self.flow2_decoder( joinout )
        
        # concat stem out with decoder out
        flow2 = self.flow2_concat( (flow2,srcx,tar1,tar2) )

        # last feature conv layer
        flow2 = self.flow2_resblock( flow2 )

        # finally, 1x1 conv layer from features to flow value
        flow2 = self.flow2_out( flow2 )
        
        return flow1,flow2

if __name__ == "__main__":

    nrows = 1024
    ncols = 3456
    nrows = 512
    ncols = 832
    sparsity = 0.01
    #device = torch.device("cpu")
    device = torch.device("cuda")
    ntrials = 10
    batchsize = 1
    
    model = SparseLArFlow( (nrows,ncols), 2, 16, 16, 4 ).to(device)
    model.eval()
    #print model

    npts = int(nrows*ncols*sparsity)
    print "for (%d,%d) and average sparsity of %.3f, expected npts=%d"%(nrows,ncols,sparsity,npts)

    # random points from a hypothetical (nrows x ncols) image
    dtforward = 0
    for itrial in xrange(ntrials):
        xcoords = np.zeros( (npts,2), dtype=np.int )
        xcoords[:,0] = np.random.randint( 0, nrows, npts )
        xcoords[:,1] = np.random.randint( 0, ncols, npts )
        srcx = np.random.random( (npts,1) ).astype(np.float32)
        tar1 = np.random.random( (npts,1) ).astype(np.float32)
        tar2 = np.random.random( (npts,1) ).astype(np.float32)    
        
        coord_t = torch.from_numpy(xcoords).to(device)
        src_feats_t  = torch.from_numpy(srcx).to(device)
        tar1_feats_t = torch.from_numpy(tar1).to(device)
        tar2_feats_t = torch.from_numpy(tar2).to(device)

        tforward = time.time()
        print "coord-shape: ",coord_t.shape
        print "src feats-shape: ",src_feats_t.shape    
        out1,out2 = model( coord_t, src_feats_t, tar1_feats_t, tar2_feats_t, batchsize )
        dtforward += time.time()-tforward
        print "modelout: flow1=[",out1.features.shape,out1.spatial_size,"]"
        
    print "ave. forward time over %d trials: "%(ntrials),dtforward/ntrials," secs"
