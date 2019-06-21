import os,sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sparseconvnet as scn
import time
import math
import numpy as np
from utils_sparselarflow import create_resnet_layer
from sparseencoder import SparseEncoder
from sparsedecoder import SparseDecoder

class SparseLArFlow(nn.Module):
    """
    Sparse Submanifold implementation of LArFlow
    """
    avail_flows = ['y2u','y2v']
    
    def __init__(self, inputshape, reps, nin_features, nout_features, nplanes,
                 flowdirs=['y2u','y2v'],
                 predict_classvec=False,
                 home_gpu=None,
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
        nout_features [int]: number of features that feed into the classification/regression layer
        nplanes [int]: the depth of the U-Net
        flowdirs [list of str]: which flow directions to implement, if two (y2u+y2v)
                       then we must process all three planes and produce two flow predictions. 
                       if one, then only two planes are processed by encoder, and one flow predicted.
        share_encoder_weights [bool]: if True, share the weights for the encoder
        features_per_layer [list of int]: if provided, defines the feature size of each layer depth. 
                       if None, calculated automatically.
        show_sizes [bool]: if True, print sizes while running forward
        predict_pixenum [bool]: if True, we predict a class vector representing what target column matches
        """
        # set parameters
        self.dimensions = 2 # not playing with 3D for now
        self.home_gpu = home_gpu
        self.predict_classvec = predict_classvec

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

        if 'y2u' in self.flowdirs:
            self.tar1_inputlayer = scn.InputLayer(self.dimensions, self.inputshape, mode=self.mode)
        else:
            self.tar1_inputlayer = None

        if 'y2v' in self.flowdirs:
            self.tar2_inputlayer = scn.InputLayer(self.dimensions, self.inputshape, mode=self.mode)
        else:
            self.tar2_inputlayer = None

        # stem
        self.src_stem  = scn.SubmanifoldConvolution(self.dimensions, 1, self.nfeatures, 3, False)
        if not self._share_encoder_weights:
            # if not sharing weights, producer separate stems
            if 'y2u' in self.flowdirs:
                self.tar1_stem = scn.SubmanifoldConvolution(self.dimensions, 1, self.nfeatures, 3, False)
            else:
                self.tar1_stem = None

            if 'y2v' in self.flowdirs:
                self.tar2_stem = scn.SubmanifoldConvolution(self.dimensions, 1, self.nfeatures, 3, False)
            else:
                self.tar2_stem = None

        # encoders
        self.source_encoder  = SparseEncoder( "src",  self.reps, self.nfeatures, self.nPlanes )
        if not self._share_encoder_weights:
            # if not sharing weights, add additional encoders
            if 'y2u' in self.flowdirs:
                self.target1_encoder = SparseEncoder( "tar1", self.reps, self.nfeatures, self.nPlanes )
            else:
                self.target1_encoder = None

            if 'y2v' in self.flowdirs:
                self.target2_encoder = SparseEncoder( "tar2", self.reps, self.nfeatures, self.nPlanes )
            else:
                self.target2_encoder = None

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
        if 'y2u' in self.flowdirs:
            self.flow1_decoder = SparseDecoder( "flow1", self.reps,
                                                self.decode_layers_inchs,
                                                self.decode_layers_outchs )
        else:
            self.flow1_decoder = None

        if 'y2v' in self.flowdirs:
            self.flow2_decoder = SparseDecoder( "flow2", self.reps,
                                                self.decode_layers_inchs,
                                                self.decode_layers_outchs )
        else:
            self.flow2_decoder = None
            
        if self._show_sizes:
            for fd in [self.flow1_decoder,self.flow2_decoder]:
                if fd is not None:
                    fd.set_verbose(True)

        # last deconv concat
        if 'y2u' in self.flowdirs:
            self.flow1_concat = scn.JoinTable()
        else:
            self.flow1_concat = None

        if 'y2v' in self.flowdirs:
            self.flow2_concat = scn.JoinTable()
        else:
            self.flow2_concat = None

        # final feature set convolution
        flow_resblock_inchs = (self.nflows+1)*self.nfeatures + self.decode_layers_outchs[-1]
        if 'y2u' in self.flowdirs:
            self.flow1_resblock = create_resnet_layer(self.reps,
                                                      flow_resblock_inchs,self.nout_features)
        else:
            self.flow1_resblock = None

        if 'y2v' in self.flowdirs:
            self.flow2_resblock = create_resnet_layer(self.reps,
                                                      flow_resblock_inchs,self.nout_features)
        else:
            self.flow2_resblock = None


        # OUTPUT LAYER
        if self.predict_classvec:
            if 'y2u' in self.flowdirs:
                self.flow1_out = scn.SubmanifoldConvolution(self.dimensions,self.nout_features,self.inputshape[1],1,True)
            else:
                self.flow1_out = None

            if 'y2v' in self.flowdirs:
                self.flow2_out = scn.SubmanifoldConvolution(self.dimensions,self.nout_features,self.inputshape[1],1,True)
            else:
                self.flow2_out = None
        else:
            # regression layer
            if 'y2u' in self.flowdirs:
                self.flow1_out = scn.SubmanifoldConvolution(self.dimensions,self.nout_features,1,1,True)
            else:
                self.flow1_out = None

            if 'y2v' in self.flowdirs:
                self.flow2_out = scn.SubmanifoldConvolution(self.dimensions,self.nout_features,1,1,True)
            else:
                self.flow2_out = None


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
        run_y2u = 'y2u' in self.flowdirs
        run_y2v = 'y2v' in self.flowdirs

        srcx = ( coord_t, src_feat_t,  batchsize )
        if run_y2u:
            tar1 = ( coord_t, tar1_feat_t, batchsize )
        if run_y2v:
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
        if run_y2u:
            tar1 = self.tar1_inputlayer(tar1)
            if self._show_sizes:
                print "input[tar1]: ",tar1.features.shape

        if run_y2v:
            tar2 = self.tar2_inputlayer(tar2)
            if self._show_sizes:
                print "input[tar2]: ",tar2.features.shape
            
            
        if not self._share_encoder_weights:
            # separate encoders
            if run_y2u:
                tar1 = self.tar1_stem(tar1)
                tar1out_v = self.target1_encoder(tar1)
            
            if run_y2v:
                tar2 = self.tar2_stem(tar2)
                tar2out_v = self.target2_encoder(tar2)
        else:
            # shared weights for encoder
            if run_y2u:
                tar1 = self.src_stem(tar1)
                tar1out_v = self.source_encoder(tar1)
            if run_y2v:
                tar2 = self.src_stem(tar2)
                tar2out_v = self.source_encoder(tar2)

        # concat features from all three planes
        joinout = []
        if self.nflows==1:
            # merge 2 encoder outputs
            if run_y2u:
                for _src,_tar1,_joiner in zip(srcout_v,tar1out_v,self.join_enclayers):
                    joinout.append( _joiner( (_src,_tar1) ) )
            elif run_y2v:
                for _src,_tar2,_joiner in zip(srcout_v,tar2out_v,self.join_enclayers):
                    joinout.append( _joiner( (_src,_tar2) ) )
                
        elif self.nflows==2:
            # merge 3 encoder outputs
            for _src,_tar1,_tar2,_joiner in zip(srcout_v,tar1out_v,tar2out_v,self.join_enclayers):
                joinout.append( _joiner( (_src,_tar1,_tar2) ) )
        else:
            raise ValueError("number of flows={} not supported".format(self.nflows))

        # Flow 1: src->tar1
        # ------------------
        # use 3-plane features to make flow features
        if run_y2u:
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

        else:
            flow1 = None

        # Flow 2: src->tar1
        # ------------------
        if run_y2v:
            flow2 = self.flow2_decoder( joinout )

            # concat stem out with decoder out
            if self.nflows==1:
                flow2 = self.flow2_concat( (flow2,srcx,tar2) )
            elif self.nflows==2:
                flow2 = self.flow2_concat( (flow2,srcx,tar1,tar2) )

            # last feature conv layer
            flow2 = self.flow2_resblock( flow2 )

            # finally, 1x1 conv layer from features to flow value
            flow2 = self.flow2_out( flow2 )

        else:
            flow2 = None

        if self.home_gpu is not None:
            if run_y2u:
                flow1.features = flow1.features.to( torch.device("cuda:%d"%(self.home_gpu)) )
            if run_y2v:
                flow2.features = flow2.features.to( torch.device("cuda:%d"%(self.home_gpu)) )

        return flow1,flow2

if __name__ == "__main__":

    from sparselarflowdata import load_larflow_larcvdata
    
    """
    here we test/debug the network and losses here
    we can use a random matrix mimicing our sparse lartpc images
      or actual images from the loader.
    """

    nrows     = 512
    ncols     = 832
    sparsity  = 0.01
    device    = torch.device("cpu")
    #device    = torch.device("cuda")
    ntrials   = 1
    batchsize = 1
    use_random_data = False
    test_loss  = True
    run_w_grad = False
    ENABLE_PROFILER=False
    PROF_USE_CUDA=True

    # two-flow input
    #flowdirs = ['y2u','y2v'] # two flows
    #inputfile    = "../testdata/mcc9mar_bnbcorsika/larcv_mctruth_ee881c25-aeca-4c92-9622-4c21f492db41.root"    
    #inputfile = "out_sparsified.root"
    #producer_name = "larflow"

    # one-flow input: Y2U
    #flowdirs = ['y2u']
    #inputfile = "out_sparsified_y2u.root"
    #producer_name = "larflow_y2u"

    # one-flow input: Y2V
    #flowdirs = ['y2v']
    #inputfile = "out_sparsified_y2v.root"
    #producer_name = "larflow_y2v"

    # dual-flow input:
    flowdirs = ['y2u','y2v']
    inputfile = "/home/taritree/data/sparselarflow/larflow_sparsify_cropped_valid_v5.root"

    ninput_features  = 16
    noutput_features = 16
    nplanes = 6
    predict_classvec = True
    producer_name = "sparsecropdual"
    nfeatures_per_layer = [16,16,16,16,16,16]

    model = SparseLArFlow( (nrows,ncols), 2, ninput_features, noutput_features,
                           nplanes, features_per_layer=nfeatures_per_layer,
                           predict_classvec=predict_classvec,
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
        ro_products  = None
        dataloader   = load_larflow_larcvdata( "larflowsparsetest", inputfile,
                                               batchsize, nworkers,
                                               nflows=len(flowdirs),
                                               producer_name=producer_name,
                                               tickbackward=tickbackward,
                                               readonly_products=ro_products )

    if test_loss:
        from loss_sparse_larflow import SparseLArFlow3DConsistencyLoss
        consistency_loss = SparseLArFlow3DConsistencyLoss(nrows, ncols,
                                                          larcv_version=1,
                                                          calc_consistency=False,
                                                          predict_classvec=predict_classvec)

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
                
        dtforward += time.time()-tforward
        print "output: flow1=",predict1_t.features.shape," predict2_t=",predict2_t.features.shape

        if test_loss:
            tloss = time.time()
            loss,flow1loss,flow2loss = consistency_loss(coord_t, predict1_t, predict2_t,
                                                        truth_flow1_t, truth_flow2_t)
            print "loss: tot=",loss.detach().cpu().item()
            if flow1loss is not None:
                print "      flow1=",flow1loss.detach().cpu().item()
            if flow2loss is not None:
                print "      flow2=",flow2loss.detach().cpu().item()                

        #print "modelout: flow1=[",out1.features.shape,out1.spatial_size,"]"

    print "ave. data time o/ %d trials:    %.2f secs"%(ntrials,dtdata/ntrials)
    print "ave. forward time o/ %d trials: %.2f secs"%(ntrials,dtforward/ntrials)
    if ENABLE_PROFILER:
        profout = open('profileout_cuda.txt','w')
        print>>profout,prof
