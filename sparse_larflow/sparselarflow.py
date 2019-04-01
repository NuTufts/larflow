import os,sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sparseconvnet as scn
import time
import math
import numpy as np

def residual_block(m, a, b, leakiness=0.01, dimensions=2):
    """
    append to a sequence module:
    produce output of [identity,3x3+3x3] then add together

    inputs
    ------
    m [scn.Sequential module] network to add layers to
    a [int]: number of input channels
    b [int]: number of output channels
    leakiness [float]: leakiness of ReLU activations
    dimensions [int]: dimensions of input sparse tensor

    modifies
    --------
    m: adds layers
    """
    m.add(scn.ConcatTable()
          .add(scn.Identity() if a == b else scn.NetworkInNetwork(a, b, False))
          .add(scn.Sequential()
               .add(scn.BatchNormLeakyReLU(a,leakiness=leakiness))
               .add(scn.SubmanifoldConvolution(dimensions, a, b, 3, False))
               .add(scn.BatchNormLeakyReLU(b,leakiness=leakiness))
               .add(scn.SubmanifoldConvolution(dimensions, b, b, 3, False)))
    ).add(scn.AddTable())

def create_resnet_layer(nreps, ninputchs, noutputchs,
                        downsample=[2,2]):
    """
    creates a layer formed by a repetition of residual blocks

    inputs
    ------
    nreps [int] number of times to repeat residula block
    ninputchs [int] input features to layer
    noutputchs [int] output features from layer

    outputs
    -------
    [scn.Sequential] module with residual blocks
    """
    m = scn.Sequential()
    for iblock in xrange(nreps):
        if iblock==0:
            # in first repitition we change
            # number of features from input to output
            residual_block(m,ninputchs,noutputchs)
        else:
            # other repitions we do not change number of features
            residual_block(m,noutputchs,noutputchs)
    return m


class SparseEncoder(nn.Module):
    """
    The encoder involves a series of layers which
    1) first apply a series of residual convolution blocks (using 3x3 kernels)
    2) then apply a strided convolution to downsample the image
       right now, a standard stride of 2 is used
    """

    def __init__( self, name, nreps, ninputchs, chs_per_layer):
        nn.Module.__init__(self)
        """
        inputs
        ------
        name [str] to name the encoder
        nreps [int] number of times residual block is repeated per layer
        ninputchs [int] input channels to entire encoder module
        chs_per_layer [list of int] output channels for each layer
        """
        # store variables and configuration
        self.chs_per_layer = chs_per_layer
        self.inputchs = ninputchs
        self.residual_blocks = True # always uses residual blocks
        self.dimension = 2 # always for 2D images
        self._layers = []  # stores layers inside this module
        self._name = name  # name of this instance
        self._verbose = False

        # create the individual layers
        for ilayer,noutputchs in enumerate(self.chs_per_layer):
            if ilayer>0:
                # use last num output channels
                # in previous layer for num input ch
                ninchs = self.chs_per_layer[ilayer-1]
            else:
                # for first layer, use the provided number of chanels
                ninchs = self.inputchs

            # create the encoder layer
            layer = self.make_encoder_layer(ninchs,noutputchs,nreps)
            # store it
            self._layers.append(layer)
            # create an attribute for the module so pytorch
            # can know the components in this module
            setattr(self,"%s_enclayer%d"%(name,ilayer),layer)

    def set_verbose(self,verbose=True):
        self._verbose = verbose

    def make_encoder_layer(self,ninputchs,noutputchs,nreps,
                           leakiness=0.01,downsample=[2, 2]):
        """
        inputs
        ------
        ninputchs [int]: number of features going into layer
        noutputchs [int]: number of features output by layer
        nreps [int]: number of times residual modules repeated
        leakiness [int]: leakiness of LeakyReLU layers
        downsample [length 2 list of int]: stride in [height,width] dims

        outputs
        -------
        scn.Sequential module with resnet and downsamping layers
        """
        encode_blocks = create_resnet_layer(nreps,ninputchs,noutputchs,
                                            downsample=downsample)
        if downsample is not None:
            # if we specify downsize factor for each dimension, we apply
            # it to the output of the residual layers
            encode_blocks.add(scn.BatchNormLeakyReLU(noutputchs,leakiness=leakiness))
            encode_blocks.add(scn.Convolution(self.dimension, noutputchs, noutputchs,
                                              downsample[0], downsample[1], False))
        return encode_blocks

    def forward(self,x):
        """
        run the layers on the input

        inputs
        ------
        x [scn.SparseManifoldTensor]: input tensor

        outputs
        -------
        list of scn.SparseManifoldTensor: one tensor for each
           encoding layer. all returned for use in skip connections
           in the SparseDecoding modules
        """
        layerout = []
        for ilayer,layer in enumerate(self._layers):
            if ilayer==0:
                inputx = x
            else:
                inputx = layerout[-1]
            out = layer(inputx)
            if self._verbose:
                print "[%s] Encode Layer[%d]: "%(self._name,ilayer),
                print inputx.features.shape,inputx.spatial_size,
                print "-->",out.features.shape,out.spatial_size
            layerout.append( out )
        return layerout

class SparseDecoder(nn.Module):
    """
    SparseDecoder class features layers containing:
    1) sequence of residual convolution modules
    2) upsampling using convtranspose
    3) skip connection from concating tensors
    """
    def __init__( self, name, nreps, inputchs_per_layer, outputchs_per_layer ):
        nn.Module.__init__(self)
        """
        inputs
        ------
        name [str]
        nreps [int]
        inputchs_per_layer  [list of ints]
        outputchs_per_layer [list of ints]
        """
        self.inputchs_per_layer  = inputchs_per_layer
        self.outputchs_per_layer = outputchs_per_layer
        self.residual_blocks = True
        self.dimension = 2
        self._deconv_layers = []
        self._joiner = []
        self._name = name
        self._verbose = False

        layer_chs = zip(self.inputchs_per_layer,self.outputchs_per_layer)
        for ilayer,(ninputchs,noutputchs) in enumerate(layer_chs):
            if ilayer+1<len(outputchs_per_layer):
                islast = False
            else:
                islast = True
            deconv,joiner = self.make_decoder_layer(ilayer,ninputchs,
                                                    noutputchs,nreps,
                                                    islast=islast)
            self._deconv_layers.append(deconv)
            self._joiner.append(joiner)

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
        decode_blocks = create_resnet_layer(nreps,ninputchs,2*noutputchs,
                                            downsample=downsample)

        # deconv
        decode_blocks.add(scn.BatchNormLeakyReLU(2*noutputchs,leakiness=leakiness))
        decode_blocks.add(scn.Deconvolution(self.dimension, 2*noutputchs, noutputchs,
                                            downsample[0], downsample[1], False))
        setattr(self,"deconv%d"%(ilayer),decode_blocks)
        if self._verbose:
            print "DecoderLayer[",ilayer,"] inputchs[",ninputchs,
            print " -> resout[",2*noutputchs,"] -> deconv output[",noutputchs,"]"


        if not islast:
            # joiner for skip connections
            joiner = scn.JoinTable()
            setattr(self,"skipjoin%d"%(ilayer),joiner)
        else:
            joiner = None
        return decode_blocks,joiner

    def set_verbose(self,verbose=True):
        self._verbose=verbose

    def forward(self,encoder_layers):
        """
        inputs
        ------

        outputs
        -------
        """
        layerout = None
        for ilayer,(deconvl,joiner) in enumerate(zip(self._deconv_layers,self._joiner)):
            if ilayer==0:
                # first layer, input tensor comes from last encoder layer
                inputx = encoder_layers[-(1+ilayer)]
            else:
                # otherwise, get the last tensor that was created
                inputx = layerout
            #print "decoder-input[",ilayer,"] input=",inputx.features.shape,inputx.spatial_size

            # residual + upsample
            out = deconvl(inputx)
            #print "decode-deconv[",ilayer,"]: outdecode=",out.features.shape,out.spatial_size

            # concat if not last layer, for last layer, joiner will be None
            if joiner is not None:
                # get next encoder layer
                catlayer  = encoder_layers[-(2+ilayer)]
                #print "decode-concat[",ilayer,"]: appending=",
                #print catlayer.features.shape,catlayer.spatial_size

                # concat
                out   = joiner( (out,catlayer) )

            if self._verbose:
                print "[%s] Decode Layer[%d]: "%(self._name,ilayer),
                print inputx.features.shape,inputx.spatial_size,
                print "-->",out.features.shape,out.spatial_size
            layerout = out
        return layerout


class SparseLArFlow(nn.Module):
    """
    Sparse Submanifold implementation of LArFlow
    """
    avail_flows = ['y2u','y2v']
    
    def __init__(self, inputshape, reps, nin_features, nout_features, nplanes,
                 flowdirs=['y2u','y2v'],
                 features_per_layer=None,
                 show_sizes=False,
                 share_encoder_weights=False):
        nn.Module.__init__(self)
        """
        inputs
        ------
        inputshape [list of int]: dimensions of the matrix or image
        reps [int]: number of residual modules per layer (for both encoder and decoder)
        nin_features [int]: number of features in the first convolutional layer
        nout_features [int]: number of features that feed into the regression layer
        nplanes [int]: the depth of the U-Net
        flowdirs [list of str]: which flow directions to implement, if two (y2u+y2v)
                       then we must process all three planes and produce two flow predictions. 
                       if one, then only two planes are processed by encoder, and one flow predicted.
        share_encoder_weights [bool]: if True, share the weights for the encoder
        features_per_layer [list of int]: if provided, defines the feature size of each layer depth. 
                       if None, calculated automatically.
        show_sizes [bool]: if True, print sizes while running forward
        """
        # set parameters
        self.dimensions = 2 # not playing with 3D for now

        # input shape: LongTensor, tuple, or list. Handled by InputLayer
        # size of each spatial dimesion
        self.inputshape = inputshape
        if len(self.inputshape)!=self.dimensions:
            raise ValueError("expected inputshape to contain size of 2 dimensions only."
                             +"given %d values"%(len(self.inputshape)))

        # mode variable: how to deal with repeated data
        self.mode = 0

        # for debug, show sizes of layers/planes
        self._show_sizes = show_sizes

        # nfeatures
        self.nfeatures = nin_features
        self.nout_features = nout_features

        # plane structure
        if features_per_layer is None:
            self.nPlanes = [ self.nfeatures*2**(n+1) for n in xrange(nplanes) ]
        else:
            if ( type(features_per_layer) is list
                 and len(features_per_layer)==nplanes
                 and type(features_per_layer[0]) is int):
                self.nPlanes = features_per_layer
            else:
                raise ValueError("features_per_layer should be a list of int with number of elements equalling 'nplanes' argument")
        if self._show_sizes:
            print "Features per plane/layer: ",self.nPlanes

        # repetitions (per plane)
        self.reps = reps

        # residual blocks
        self.residual_blocks = True

        # need encoder for both source and target
        # then cat tensor
        # and produce one decoder for flow, another decoder for visibility

        # set which flows to run
        for flowdir in flowdirs:
            if flowdir not in SparseLArFlow.avail_flows:
                raise ValueError("flowdir={} not available. Allowed flows: {}".format(flowdir,SparseLArFlow.avail_flows))
        self.flowdirs = flowdirs
        self.nflows   = len(self.flowdirs)

        # do we share weights
        self._share_encoder_weights = share_encoder_weights

        # model:
        # input
        self.src_inputlayer  = scn.InputLayer(self.dimensions, self.inputshape, mode=self.mode)
        self.tar1_inputlayer = scn.InputLayer(self.dimensions, self.inputshape, mode=self.mode)
        if self.nflows==2:
            self.tar2_inputlayer = scn.InputLayer(self.dimensions, self.inputshape, mode=self.mode)

        # stem
        self.src_stem  = scn.SubmanifoldConvolution(self.dimensions, 1, self.nfeatures, 3, False)
        if not self._share_encoder_weights:
            # if not sharing weights, producer separate stems
            self.tar1_stem = scn.SubmanifoldConvolution(self.dimensions, 1, self.nfeatures, 3, False)
            if self.nflows==2:
                self.tar2_stem = scn.SubmanifoldConvolution(self.dimensions, 1, self.nfeatures, 3, False)

        # encoders
        self.source_encoder  = SparseEncoder( "src",  self.reps, self.nfeatures, self.nPlanes )
        if not self._share_encoder_weights:
            # if not sharing weights, add additional encoders
            self.target1_encoder = SparseEncoder( "tar1", self.reps, self.nfeatures, self.nPlanes )
            if self.nflows==2:
                self.target2_encoder = SparseEncoder( "tar2", self.reps, self.nfeatures, self.nPlanes )
        if self._show_sizes:
            self.source_encoder.set_verbose(True)

        # concat        
        self.join_enclayers = []
        for ilayer in xrange(len(self.nPlanes)):
            self.join_enclayers.append(scn.JoinTable())
            setattr(self,"join_enclayers%d"%(ilayer),self.join_enclayers[ilayer])

        # calculate decoder planes
        self.decode_layers_inchs  = []
        self.decode_layers_outchs = []
        for ilayer,enc_outchs in enumerate(reversed(self.nPlanes)):
            # for input, we expect features from encoder + output features
            enc_nfeatures = (self.nflows+1)*enc_outchs
            dec_nfeatures = 0 if ilayer==0 else self.decode_layers_outchs[-1]
            self.decode_layers_inchs.append( enc_nfeatures+dec_nfeatures )
            self.decode_layers_outchs.append( enc_nfeatures  )
        if self._show_sizes:
            print "decoder layers input  chs: ",self.decode_layers_inchs
            print "decoder layers output chs: ",self.decode_layers_outchs

        # decoders
        self.flow1_decoder = SparseDecoder( "flow1", self.reps,
                                            self.decode_layers_inchs,
                                            self.decode_layers_outchs )
        if self.nflows==2:
            self.flow2_decoder = SparseDecoder( "flow2", self.reps,
                                                self.decode_layers_inchs,
                                                self.decode_layers_outchs )
        if self._show_sizes:
            self.flow1_decoder.set_verbose(True)

        # last deconv concat
        self.flow1_concat = scn.JoinTable()
        if self.nflows==2:
            self.flow2_concat = scn.JoinTable()

        # final feature set convolution
        flow_resblock_inchs = (self.nflows+1)*self.nfeatures + self.decode_layers_outchs[-1]
        self.flow1_resblock = create_resnet_layer(self.reps,
                                                  flow_resblock_inchs,self.nout_features)
        if self.nflows==2:
            self.flow2_resblock = create_resnet_layer(self.reps,
                                                      flow_resblock_inchs,self.nout_features)

        # regression layer
        self.flow1_out = scn.SubmanifoldConvolution(self.dimensions,self.nout_features,1,1,True)
        if self.nflows==2:
            self.flow2_out = scn.SubmanifoldConvolution(self.dimensions,self.nout_features,1,1,True)


    def forward(self, coord_t, src_feat_t, tar1_feat_t, tar2_feat_t, batchsize):
        """
        run the network

        inputs
        ------
        coord_flow1_t [ (N,3) Torch Tensor ]: list of (row,col,batchid) N pix coordinates
        src_feat_t  [ (N,) torch tensor ]: list of pixel values for source image
        tar1_feat_t [ (N,) torch tensor ]: list of pixel values for target 1 image
        tar2_feat_t [ (N,) torch tensor ]: list of pixel values for target 2 image. provide NULL if nflows==1
        batchsize [int]: batch size

        outputs
        -------
        [ (N,) torch tensor ] flow values to target 1
        [ (N,) torch tensor ] flow values to target 2
        """
        srcx = ( coord_t, src_feat_t,  batchsize )
        tar1 = ( coord_t, tar1_feat_t, batchsize )
        if self.nflows==2:
            tar2 = ( coord_t, tar2_feat_t, batchsize )

        # source input, stem, encoder
        srcx = self.src_inputlayer(srcx)
        if self._show_sizes:
            print "input[src]: ",srcx.features.shape
        
        srcx = self.src_stem(srcx)
        if self._show_sizes:
            print "stem[src]: ",srcx.features.shape
        
        srcout_v = self.source_encoder(srcx)
        if self._show_sizes:
            for ienc,srcplane in enumerate(srcout_v):
                print "[",ienc,"] ",srcplane.features.shape

        # target input
        tar1 = self.tar1_inputlayer(tar1)
        if self.nflows==2:
            tar2 = self.tar2_inputlayer(tar2)
            
        if not self._share_encoder_weights:
            # separate encoders
            tar1 = self.tar1_stem(tar1)
            tar1out_v = self.target1_encoder(tar1)
            
            if self.nflows==2:
                tar2 = self.tar2_stem(tar2)
                tar2out_v = self.target2_encoder(tar2)
        else:
            # shared weights for encoder
            tar1 = self.src_stem(tar1)
            tar1out_v = self.source_encoder(tar1)
            
            if self.nflows==2:
                tar2 = self.src_stem(tar2)
                tar2out_v = self.source_encoder(tar2)
            

        # concat features from all three planes
        joinout = []
        if self.nflows==1:
            # merge 2 encoder outputs
            for _src,_tar1,_joiner in zip(srcout_v,tar1out_v,self.join_enclayers):
                joinout.append( _joiner( (_src,_tar1) ) )
        elif self.nflows==2:
            # merge 3 encoder outputs
            for _src,_tar1,_tar2,_joiner in zip(srcout_v,tar1out_v,tar2out_v,self.join_enclayers):
                joinout.append( _joiner( (_src,_tar1,_tar2) ) )
        else:
            raise ValueError("number of flows={} not supported".format(self.nflows))

        # Flow 1: src->tar1
        # ------------------
        # use 3-plane features to make flow features
        flow1 = self.flow1_decoder( joinout )

        # concat stem out with decoder out
        if self.nflows==1:
            flow1 = self.flow1_concat( (flow1,srcx,tar1) )
        elif self.nflows==2:
            flow1 = self.flow1_concat( (flow1,srcx,tar1,tar2) )

        # last feature conv layer
        flow1 = self.flow1_resblock( flow1 )

        # finally, 1x1 conv layer from features to flow value
        flow1 = self.flow1_out( flow1 )

        # Flow 2: src->tar1
        # ------------------
        if self.nflows==2:
            # use 3-plane features to make flow features
            flow2 = self.flow2_decoder( joinout )
            
            # concat stem out with decoder out
            flow2 = self.flow2_concat( (flow2,srcx,tar1,tar2) )
            
            # last feature conv layer
            flow2 = self.flow2_resblock( flow2 )
            
            # finally, 1x1 conv layer from features to flow value
            flow2 = self.flow2_out( flow2 )
        else:
            flow2 = None

        return flow1,flow2

if __name__ == "__main__":
    """
    here we test/debug the network and losses here
    we can use a random matrix mimicing our sparse lartpc images
      or actual images from the loader.
    """

    nrows     = 1024
    ncols     = 3456
    sparsity  = 0.01
    #device    = torch.device("cpu")
    device    = torch.device("cuda")
    ntrials   = 1
    batchsize = 1
    use_random_data = False
    test_loss = False
    run_w_grad = False
    ENABLE_PROFILER=True
    PROF_USE_CUDA=True

    # two-flow input
    #flowdirs = ['y2u','y2v'] # two flows
    #inputfile    = "../testdata/mcc9mar_bnbcorsika/larcv_mctruth_ee881c25-aeca-4c92-9622-4c21f492db41.root"    
    #inputfile = "out_sparsified.root"
    #producer_name = "larflow"

    # one-flow input: Y2U
    flowdirs = ['y2u']
    inputfile = "out_sparsified_y2u.root"
    producer_name = "larflow_y2u"

    # one-flow input: Y2V
    #flowdirs = ['y2v']
    #inputfile = "out_sparsified_y2v.root"
    #producer_name = "larflow_y2v"    

    ninput_features  = 16
    noutput_features = 16
    nplanes = 5
    nfeatures_per_layer = [16,16,32,32,64]

    model = SparseLArFlow( (nrows,ncols), 2, ninput_features, noutput_features,
                           nplanes, features_per_layer=nfeatures_per_layer,
                           show_sizes=True,
                           flowdirs=flowdirs ).to(device)
    model.eval()
    #print model

    npts = int(nrows*ncols*sparsity)
    print "for (%d,%d) and average sparsity of %.3f, expected npts=%d"%(nrows,ncols,sparsity,npts)

    if not use_random_data:
        from larcv import larcv
        from sparselarflowdata import load_larflow_larcvdata
        nworkers     = 3
        tickbackward = True
        #ro_products  = ( ("wiremc",larcv.kProductImage2D),
        #                 ("larflow",larcv.kProductImage2D) )
        ro_products = None
        dataloader   = load_larflow_larcvdata( "larflowsparsetest", inputfile,
                                               batchsize, nworkers,
                                               nflows=len(flowdirs),
                                               producer_name=producer_name,
                                               tickbackward=tickbackward,
                                               readonly_products=ro_products )

    if test_loss:
        from loss_sparse_larflow import SparseLArFlow3DConsistencyLoss
        consistency_loss = SparseLArFlow3DConsistencyLoss(1024, 3456,
                                                          larcv_version=1,
                                                          calc_consistency=False)

    # random points from a hypothetical (nrows x ncols) image
    dtforward = 0
    dtdata = 0
    for itrial in xrange(ntrials):

        # random data
        tdata = time.time()
        if use_random_data:
            # Create random data
            xcoords = np.zeros( (npts,2), dtype=np.int )
            xcoords[:,0] = np.random.randint( 0, nrows, npts )
            xcoords[:,1] = np.random.randint( 0, ncols, npts )
            srcx = np.random.random( (npts,1) ).astype(np.float32)
            tar1 = np.random.random( (npts,1) ).astype(np.float32)
            tar2 = np.random.random( (npts,1) ).astype(np.float32)

            coord_t   = torch.from_numpy(xcoords).to(device)
            srcpix_t  = torch.from_numpy(srcx).to(device)
            tarpix_flow1_t = torch.from_numpy(tar1).to(device)
            tarpix_flow2_t = torch.from_numpy(tar2).to(device)
        else:
            # Get data from file
            datadict = dataloader.get_tensor_batch(device)
            coord_t = datadict["coord"]
            srcpix_t = datadict["src"]
            tarpix_flow1_t = datadict["tar1"]
            truth_flow1_t  = datadict["flow1"]
            if len(flowdirs)==2:
                tarpix_flow2_t = datadict["tar2"]            
                truth_flow2_t  = datadict["flow2"]
            else:
                tarpix_flow2_t = None
                truth_flow2_t  = None

        dtdata += time.time()-tdata

        tforward = time.time()
        print "coord-shape: flow1=",coord_t.shape
        print "src feats-shape: ",srcpix_t.shape
        print "tarpix_flow1 feats-shape: ",tarpix_flow1_t.shape
        if tarpix_flow2_t is not None:
            print "tarpix_flow2 feats-shape: ",tarpix_flow2_t.shape
        if truth_flow1_t is not None:
            print "truth flow1: ",truth_flow1_t.shape
        if truth_flow2_t is not None:
            print "truth flow2: ",truth_flow2_t.shape
        with torch.autograd.profiler.profile(enabled=ENABLE_PROFILER,use_cuda=PROF_USE_CUDA) as prof:
            # workup
            with torch.set_grad_enabled(run_w_grad):
                predict1_t,predict2_t = model.forward( coord_t, srcpix_t,
                                                       tarpix_flow1_t, tarpix_flow2_t,
                                                       batchsize )
            #with torch.autograd.profiler.emit_nvtx():
            #    predict1_t,predict2_t = model( coord_t, srcpix_t,
            #                                   tarpix_flow1_t, tarpix_flow2_t,
            #                                   batchsize )
                
        dtforward += time.time()-tforward

        if test_loss:
            tloss = time.time()
            loss = consistency_loss(coord_t, predict1_t, predict2_t,
                                    truth_flow1_t, truth_flow2_t)
            print "loss: ",loss.detach().cpu().item()

        #print "modelout: flow1=[",out1.features.shape,out1.spatial_size,"]"

    print "ave. data time o/ %d trials: %.2f secs"%(ntrials,dtdata/ntrials)
    print "ave. forward time o/ %d trials: %.2f secs"%(ntrials,dtforward/ntrials)
    if ENABLE_PROFILER:
        profout = open('profileout_cuda.txt','w')
        print>>profout,prof
