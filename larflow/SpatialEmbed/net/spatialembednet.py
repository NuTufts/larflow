from collections import OrderedDict
import torch
import torch.nn as nn
import sparseconvnet as scn
from utils_sparselarflow import residual_block, create_resnet_layer, resnet_encoding_layers, resnet_decoding_layers
import numpy as np

class SpatialEmbedNet(nn.Module):

    def __init__(self,ndimensions,inputshape,
                 input_nfeatures=1,
                 stem_nfeatures=16,
                 num_unet_layers=5,
                 nclasses=5,
                 leakiness=0.001 ):
        """
        parameters
        -----------
        ndimensions [int]    number of spatial dimensions of input data, default=2
        inputshape  [tuple of int]  size of input tensor/image in (num of tick pixels, num of wire pixels), default=(1024,3456)
        input_nfeatures [int] number of features in the input tensor, default=1 (the image charge)
        stem_nfeatures [int] number of features in the stem layers, also controls num of features in unet layers, default=16
        features_per_layer [int] number of channels in the resnet layers that follow unet layers, default=16
        classifier_nfeatures [tuple of int] number of channels per hidden layer of larmatch classification network, default=[32,32]
        leakiness [float] leakiness of LeakyRelu activiation layers, default=0.01
        ninput_planes [int] number of input image planes, default=3
        use_unet [bool] if true, use UNet layers, default=True
        """
        super(SpatialEmbedNet,self).__init__()

        # number of dimensions
        self.ndimensions = ndimensions
        
        # INPUT LAYERS: converts torch tensor into scn.SparseMatrix
        self.inputlayer  = scn.InputLayer(ndimensions,inputshape,mode=0)

        # STEM
        self.stem = scn.Sequential() 
        self.stem.add( scn.SubmanifoldConvolution(ndimensions, input_nfeatures, stem_nfeatures, 3, False ) )
        #self.stem.add( scn.BatchNormLeakyReLU(stem_nfeatures,leakiness=leakiness) )

        feat_per_layers = [stem_nfeatures*2,
                           stem_nfeatures*3]
        feat_per_layers = []
        for n in range(num_unet_layers):
            feat_per_layers = [(2+n)*stem_nfeatures]

        nlayers = len(feat_per_layers)
        
        self.encoding_layers = resnet_encoding_layers( stem_nfeatures, feat_per_layers, 2, dimensions=3 )
        self.embed_layers,self.embed_up = resnet_decoding_layers( stem_nfeatures, feat_per_layers, 2, dimensions=3 )
        self.seed_layers,self.seed_up   = resnet_decoding_layers( stem_nfeatures, feat_per_layers, 2, dimensions=3 )

        # register the layers into the module
        for i,en in enumerate(self.encoding_layers):
            setattr(self,"encoder_%d"%(i),en)
        for i,(conv,up) in enumerate(zip(self.embed_layers,self.embed_up)):
            setattr(self,"embed_conv_%d"%(i),conv)
            setattr(self,"embed_upsample_%d"%(i),up)
        for i,(conv,up) in enumerate(zip(self.seed_layers,self.seed_up)):
            setattr(self,"seed_conv_%d"%(i),conv)
            setattr(self,"seed_upsample_%d"%(i),up)

        # make skip connections that feed into embed and seed decoders
        self.embed_skip = []
        self.seed_skip  = []
        for i in range(len(self.encoding_layers)):
            self.embed_skip.append( scn.Identity() )
            self.seed_skip.append( scn.Identity() )            
            setattr(self,"embed_skip_%d"%(i),self.embed_skip[-1])
            setattr(self,"seed_skip_%d"%(i),self.seed_skip[-1])

        # seed output, each pixel produces score for each class
        self.seed_out = scn.Sequential()
        residual_block(self.seed_out,stem_nfeatures*2,stem_nfeatures,leakiness=leakiness)
        #residual_block(self.seed_out,stem_nfeatures,stem_nfeatures,leakiness=leakiness)
        residual_block(self.seed_out,stem_nfeatures,nclasses,leakiness=leakiness)

        # a cat function
        self.cat = scn.JoinTable()

        # instance/embed out, each pixel needs 3 dimension shift and 1 sigma
        self.embed_out = scn.Sequential()
        residual_block(self.embed_out,stem_nfeatures*2,stem_nfeatures,leakiness=leakiness)
        #residual_block(self.embed_out,stem_nfeatures,stem_nfeatures,leakiness=leakiness)
        residual_block(self.embed_out,stem_nfeatures,4,leakiness=leakiness)

        self.embed_sparse2dense = scn.OutputLayer(ndimensions)
        self.seed_sparse2dense  = scn.OutputLayer(ndimensions)
        

    def forward( self, coord_t, feat_t, device, verbose=False ):
        """
        """
        if verbose:
            print "[larmatch::make feature vectors] "
            print "  coord=",coord_t.shape," feat=",feat_t.shape
            print "  cuda: coord=",coord_t.is_cuda," feat=",feat_t.is_cuda  

        # Stem
        x = self.inputlayer( (coord_t, feat_t) )
        x = self.stem( x )
        if verbose: print "  stem=",x.features.shape

        # unet
        # must save for each encoder layer for skip connections
        x_encode = [ x ]
        for i,enlayer in enumerate(self.encoding_layers):
            y = enlayer(x_encode[-1])        
            if verbose: print "  encoder[",i,"]: ",y.features.shape            
            x_encode.append( enlayer(x_encode[-1]) )

        # embed decoder
        if verbose: print " len(x_encode): ",len(x_encode)
        
        decode_out = {"embed":[],"seed":[]}
        for name,conv_layers,up_layers,skip_layers in [("embed",self.embed_layers,self.embed_up,self.embed_skip),
                                                       ("seed",self.seed_layers,self.seed_up,self.seed_skip)]:
            decode_input = x_encode[-1]
            for i,(conv,up) in enumerate(zip(conv_layers,up_layers)):
                # up sample the input
                if verbose: print " ",name,"-decoder[",i,"]-input: ",decode_input.features.shape,decode_input.features.is_cuda
                x = up(decode_input)
                
                if verbose: print " ",name,"-decoder[",i,"]-upsample: ",x.features.shape
                if verbose: print " ",name,"-decoder[",i,"]-encode source[",-2-i,"]: ",x_encode[-2-i].features.shape
                x_skip = skip_layers[i](x_encode[-2-i])
                if verbose: print " ",name,"-decoder[",i,"]-skip: ",x_skip.features.shape
                x = self.cat( (x,x_skip) )
                if verbose: print " ",name,"-decoder[",i,"]-skipcat: ",x.features.shape
                x = conv(x)
                decode_out[name].append(x)
                decode_input = x
                if verbose: print " ",name,"-decoder[",i,"]-conv: ",decode_input.features.shape
            if verbose: print " ",name,"-out: isname=",torch.isnan(decode_out[name][-1].features.detach()).sum(),torch.isinf(decode_out[name][-1].features.detach()).sum()

        x_embed = self.embed_out(decode_out["embed"][-1])
        if verbose: print "embed-out: ",x_embed.features.shape," num-nan=",torch.isnan(x_embed.features.detach()).sum()
        
        x_seed  = self.seed_out(decode_out["seed"][-1])
        if verbose: print "seed-out:  ",x_seed.features.shape," num-nan=",torch.isnan(x_seed.features.detach()).sum()

        # go back to dense array now
        x_embed = self.embed_sparse2dense(x_embed)
        x_seed  = self.seed_sparse2dense(x_seed)

        # normalize x,y,z shifts within [-1,1]
        x_embed_shift = torch.tanh( x_embed[:,0:self.ndimensions] )
        # sigma output is predicting ln(0.5/sigma^2)
        x_embed_sigma = x_embed[:,self.ndimensions:]
        x_embed_out = torch.cat( [x_embed_shift,x_embed_sigma], dim=1 )

        # normalize seed map output between [0,1]
        x_seed  = torch.sigmoid( x_seed )
        
        return x_embed_out,x_seed
                                


if __name__ == "__main__":

    dimlen = 16
    net = SpatialEmbedNet(3, (dimlen,dimlen,dimlen), input_nfeatures=3, nclasses=1,
                          stem_nfeatures=16)

    print net
    
    nsamples = 50
    coord_np = np.zeros( (nsamples,4), dtype=np.int64 )
    coord_np[:,0] = np.random.randint(0, high=dimlen, size=nsamples, dtype=np.int64)
    coord_np[:,1] = np.random.randint(0, high=dimlen, size=nsamples, dtype=np.int64)
    coord_np[:,2] = np.random.randint(0, high=dimlen, size=nsamples, dtype=np.int64)
    feat_np = np.zeros( (nsamples,3), dtype=np.float32 )
    for i in range(3):
        feat_np[:,i] = np.random.rand(nsamples)

    coord_t = torch.from_numpy(coord_np)
    feat_t  = torch.from_numpy(feat_np)
    embed,seed = net( coord_t, feat_t, verbose=True )
    print "embed shape: ",embed.shape
    print "seed shape: ",seed.shape
