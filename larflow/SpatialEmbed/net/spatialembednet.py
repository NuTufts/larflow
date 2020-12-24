from collections import OrderedDict
import torch
import torch.nn as nn
import sparseconvnet as scn
from utils_sparselarflow import residual_block, create_resnet_layer, resnet_encoding_layers, resnet_decoding_layers
import numpy as np
from loss_spatialembed import SpatialEmbedLoss

class SpatialEmbedNet(nn.Module):

    def __init__(self,ndimensions,inputshape,
                 input_nfeatures=1,
                 stem_nfeatures=16,
                 num_unet_layers=5,
                 nclasses=7,
                 nsigma=3,
                 embedout_shapes=1,
                 leakiness=0.001,
                 smooth_inference=False ):
        """
        parameters
        -----------
        ndimensions [int]    number of spatial dimensions of input data, default=2
        inputshape  [tuple of int]  size of input tensor/image in (num of tick pixels, num of wire pixels), default=(1024,3456)
        stem_nfeatures [int] number of features in the stem layers, also controls num of features in unet layers, default=16
        num_unet_layers [int] depth of unet
        nclasses [int] number of particle type classes, default=5 {1:electron,2:gamma,3:pi0,4:muon,5:Kaon,6:pion,7:proton}
        classifier_nfeatures [tuple of int] number of channels per hidden layer of larmatch classification network, default=[32,32]
        leakiness [float] leakiness of LeakyRelu activiation layers, default=0.01
        ninput_planes [int] number of input image planes, default=3
        use_unet [bool] if true, use UNet layers, default=True
        """
        super(SpatialEmbedNet,self).__init__()

        # number of dimensions, input shape
        self.ndimensions = ndimensions
        self.input_shape = inputshape
        self.nsigma = nsigma
        self.nclasses = nclasses
        self.embedout_shapes = embedout_shapes
        self.smooth_inference = smooth_inference
        
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

        # ENCODER
        self.encoding_layers = resnet_encoding_layers( stem_nfeatures, feat_per_layers, 2, dimensions=3 )
        # register the layers into the module
        for i,en in enumerate(self.encoding_layers):
            setattr(self,"encoder_%d"%(i),en)

        # SEED OUTPUT/CLASS PREDICTION
        self.seed_layers,self.seed_up   = resnet_decoding_layers( stem_nfeatures, feat_per_layers, 2, dimensions=3 )
        for i,(conv,up) in enumerate(zip(self.seed_layers,self.seed_up)):
            setattr(self,"seed_conv_%d"%(i),conv)
            setattr(self,"seed_upsample_%d"%(i),up)
        # make skip connections that feed into embed and seed decoders
        self.seed_skip  = []
        for i in range(len(self.encoding_layers)):        
            self.seed_skip.append( scn.Identity() )            
            setattr(self,"seed_skip_%d"%(i),self.seed_skip[-1])
        # seed output, each pixel produces score for each class
        self.seed_out = scn.Sequential()
        residual_block(self.seed_out,stem_nfeatures*2,stem_nfeatures,leakiness=leakiness)
        residual_block(self.seed_out,stem_nfeatures,stem_nfeatures,leakiness=leakiness)
        self.seed_out.add( scn.BatchNormLeakyReLU(stem_nfeatures,leakiness=leakiness) )
        self.seed_out.add( scn.SubmanifoldConvolution(ndimensions, stem_nfeatures, nclasses, 3, True) )        
        self.seed_sparse2dense  = scn.OutputLayer(ndimensions)
        # At inference, we want to smooth things out
        self.seed_smooth_layers = []
        self.seed_smooth_layers.append( scn.InputLayer(ndimensions,inputshape,mode=0) )
        self.seed_smooth_layers.append( scn.SubmanifoldConvolution(ndimensions,nclasses,nclasses,9,False,groups=nclasses) )
        self.seed_smooth_layers.append( scn.OutputLayer(ndimensions) )
        with torch.no_grad():        
            self.seed_smooth_layers[1].weight.fill_(1.0)
        
                
        # EMBEDDING OUTPUT
        if False:
            """ same layers output embedding for all particle shapes, kept for backwards compat
            """
            self.embed_classes = [0]
            self.embed_layers,self.embed_up = resnet_decoding_layers( stem_nfeatures, feat_per_layers, 2, dimensions=3 )
            for i,(conv,up) in enumerate(zip(self.embed_layers,self.embed_up)):
                setattr(self,"embed_conv_%d"%(i),conv)
                setattr(self,"embed_upsample_%d"%(i),up)
            # make skip connections that feed into embed and seed decoders
            self.embed_skip = []
            for i in range(len(self.encoding_layers)):
                self.embed_skip.append( scn.Identity() )
                setattr(self,"embed_skip_%d"%(i),self.embed_skip[-1])
        else:
            """ different embedding output for track/shower topologies """
            self.embed_classes = [0]*self.embedout_shapes
            self.embed_layers = {}
            self.embed_up    = {}            
            self.embed_skip  = {}
            self.embed_out_v = {}
            self.embed_sparse2dense_v = {}
            for ishape in range(self.embedout_shapes):
                # embedding decoder
                embed_layers,embed_up = resnet_decoding_layers( stem_nfeatures, feat_per_layers, 2, dimensions=3 )
                self.embed_layers[ishape] = embed_layers
                self.embed_up[ishape] = embed_up
                for i,(conv,up) in enumerate(zip(self.embed_layers[ishape],self.embed_up[ishape])):
                    if self.embedout_shapes>1:
                        setattr(self,"embed_shape%d_conv_%d"%(ishape,i),conv)
                        setattr(self,"embed_shape%d_upsample_%d"%(ishape,i),up)
                    else:
                        """ backwards compat. want to deprecate """
                        setattr(self,"embed_conv_%d"%(i),conv)
                        setattr(self,"embed_upsample_%d"%(i),up)                        
                    
                # make skip connections that feed into embed decoders
                self.embed_skip[ishape] = []
                for i in range(len(self.encoding_layers)):
                    self.embed_skip[ishape].append( scn.Identity() )
                    if self.embedout_shapes>1:
                        setattr(self,"embed_shape%d_skip_%d"%(ishape,i),self.embed_skip[ishape][-1])
                    else:
                        setattr(self,"embed_skip_%d"%(i),self.embed_skip[ishape][-1])
                        
                # instance/embed out, each pixel needs 3 dimension shift and 1 sigma
                self.embed_out_v[ishape] = scn.Sequential()
                residual_block(self.embed_out_v[ishape],stem_nfeatures*2,stem_nfeatures,leakiness=leakiness)
                self.embed_out_v[ishape].add( scn.BatchNormLeakyReLU(stem_nfeatures,leakiness=leakiness) )
                self.embed_out_v[ishape].add( scn.SubmanifoldConvolution(ndimensions, stem_nfeatures, 3+nsigma, 3, True) )
                if self.embedout_shapes>1:
                    setattr(self,"embed_shape%d_out"%(ishape),self.embed_out_v[ishape])
                else:
                    setattr(self,"embed_out",self.embed_out_v[ishape])

                # sparse to dense
                self.embed_sparse2dense_v[ishape] = scn.OutputLayer(ndimensions)
                if self.embedout_shapes>1:
                    setattr(self,"embed_shape%d_sparse2dense"%(ishape),self.embed_sparse2dense_v[ishape])
                else:
                    setattr(self,"embed_sparse2dense",self.embed_sparse2dense_v[ishape])

        # a cat function
        self.cat = scn.JoinTable()


    def init_embedout(self,nsigma=1):
        """
        trick: the margin term likes to blow up, so we initial to a better spot
        from paper's code repo
        with torch.no_grad():
            output_conv = self.decoders[0].output_conv
            print('initialize last layer with size: ',
                  output_conv.weight.size())

            output_conv.weight[:, 0:2, :, :].fill_(0)
            output_conv.bias[0:2].fill_(0)

            output_conv.weight[:, 2:2+n_sigma, :, :].fill_(0)
            output_conv.bias[2:2+n_sigma].fill_(1)
        """
        with torch.no_grad():
            for ishape in range(self.embedout_shapes):
                print self.embed_out_v[ishape][-1]
                for n,p in self.embed_out_v[ishape][-1].named_parameters():
                    print n,": pars: ",p.shape
                self.embed_out_v[ishape][-1].weight.fill_(0)
                self.embed_out_v[ishape][-1].bias[0:3].fill_(0)
                self.embed_out_v[ishape][-1].bias[3:].fill_(0.0)
            
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
        
        decode_out = {"seed":[]}
        decode_list = [("seed",self.seed_layers,self.seed_up,self.seed_skip)]
        for ishape in range(self.embedout_shapes):
            decode_list.append( ("embed-shape%d"%(ishape),self.embed_layers[ishape],
                                 self.embed_up[ishape],self.embed_skip[ishape]) )
            decode_out["embed-shape%d"%(ishape)] = []
            
        for name,conv_layers,up_layers,skip_layers in decode_list:
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

        # FINISH EMBED OUT
        x_embed_out = {}
        for ishape in range(self.embedout_shapes):
            x_embed = self.embed_out_v[ishape](decode_out["embed-shape%d"%(ishape)][-1])
            if verbose: print "embed-shape%d-out: "%(ishape),x_embed.features.shape," num-nan=",torch.isnan(x_embed.features.detach()).sum()
            # sparse to dense            
            x_embed = self.embed_sparse2dense_v[ishape](x_embed)
            # normalize x,y,z shifts within [-1,1]            
            x_embed_shift = torch.tanh( x_embed[:,0:self.ndimensions] )
            # margin output is predicting ln(0.5/sigma^2)
            x_embed_margin = x_embed[:,self.ndimensions:]
            x_embed_out[ishape] = torch.cat( [x_embed_shift,x_embed_margin], dim=1 )
            
            
        x_seed  = self.seed_out(decode_out["seed"][-1])
        if verbose: print "seed-out:  ",x_seed.features.shape," num-nan=",torch.isnan(x_seed.features.detach()).sum()
        x_seed  = self.seed_sparse2dense(x_seed)
        # normalize seed map output between [0,1]
        x_seed = torch.sigmoid( x_seed ) # remove this?

        if self.smooth_inference:
            with torch.no_grad():                    
                x_seed[x_seed<0.5] = 0.0
                print "[inference-only] x_seed-dense: ",x_seed.shape
                x_seed = self.seed_smooth_layers[0]( (coord_t,x_seed) )
                x_seed = self.seed_smooth_layers[1]( x_seed )
                x_seed = self.seed_smooth_layers[2]( x_seed )

        return x_embed_out,x_seed

    def make_clusters(self,coord_t,embed_v,seed_t,verbose=False,sigma_scale=5.0,seed_threshold=0.5):

        loss = SpatialEmbedLoss( self.input_shape, sigma_scale=sigma_scale )

        batch_clusters = []

        with torch.no_grad():
            nbatches = torch.max(coord_t[:,3])+1

            fcoord_t = coord_t.to(torch.float32)
            fcoord_t[:,0] /= float(self.input_shape[0])
            fcoord_t[:,1] /= float(self.input_shape[1])
            fcoord_t[:,2] /= float(self.input_shape[2])
            
            for ib in range(nbatches):
                if verbose: print "== BATCH[",ib,"] ==================="                
                bmask = coord_t[:,3].eq(ib)

                coord_b = fcoord_t[bmask,:]
                embed_b = [embed_t[ishape][bmask,:] for ishape in range(self.embedout_shapes)]
                seed_b  = seed_t[bmask,:]
                
                # calc embeded position
                spembed_b = [coord_b[:,0:3]+embed_b[ishape][:,0:3] for ishape in range(self.embedout_shapes)]# coordinate + shift
                sigma_b   = [torch.exp(embed_b[ishape][:,3:3+self.nsigma]) for ishape in range(self.embedout_shapes)]
                nvoxels_b = spembed_b[0].shape[0]
                
                cluster_id_all = torch.zeros( (nvoxels_b,self.nclasses), dtype=torch.int )
                
                for c in range(self.nclasses):
                    seed_c = seed_b[:,c]
                    cluster_id = cluster_id_all[:,c]
                    
                    if verbose: print "batch coord: ",coord_b.shape
                    if verbose: print "batch seed: ",seed_c.shape
                    
                    maxseed_value = seed_c[torch.argmax( seed_c )]
                    unused = cluster_id.eq(0)                
                    nvoxel_unused = unused.sum()
                    currentid = 1
                    if verbose: print "starting maxseed value class[",c,"]: ",maxseed_value

                    while maxseed_value>seed_threshold and nvoxel_unused>0:

                        if verbose: print "// form CLASS[",c,"] cluster[",currentid,"] //"
                        mask = unused.to(torch.float)
                        if verbose: print "mask: #unused=",mask.sum()," shape=",mask.shape
                        seed_u    = seed_c*mask

                        maxseed_idx   = torch.argmax( seed_u )
                        maxseed_value = seed_u[maxseed_idx]
                        maxseed_shape = loss.class_to_shape[c]

                        if verbose: print "maxseed index: ",maxseed_idx.item()," value=",maxseed_value.item()
                        centroid = spembed_b[loss.class_to_shape[c]][maxseed_idx,:] # (1,3)
                        if verbose: print "centroid: ",centroid
                        s = 1.0e-6+sigma_b[loss.class_to_shape[c]][maxseed_idx,:] # (1,3)
                        if verbose: print "maxseed sigma: ",s
                        dist = torch.pow(spembed_b[loss.class_to_shape[c]]-centroid,2)
                        if verbose:
                            d = torch.sum(dist,1)*mask
                            print "average dist^2: ",d.sum()/mask.sum()," (min,max)=(",d[mask>0.5].min(),",",d[mask>0.5].max(),")"
                        dist = torch.sum( dist*s, 1 )
                        gaus = torch.exp(-sigma_scale*dist)*mask
                        if verbose: print "gaus: ",gaus.shape
                        if verbose: print "num inside margin: ",(gaus>0.5).sum()
                    
                        cluster_id[gaus>0.5] = currentid
                        currentid += 1

                        unused = cluster_id.eq(0)
                        nvoxel_unused = unused.sum()
                        if nvoxel_unused>0:
                            maxseed_value = seed_c[unused].max()
                            if verbose:
                                print "remaining maxseed value: ",maxseed_value
                                if False:
                                    print "[next] nunused",nvoxel_unused
                                    raw_input()
                        else:
                            maxseed_value = 0.

                        # end of while loop
                    
                    # end of class loop
                    
                batch_clusters.append( (cluster_id_all,spembed_b,seed_b) )
                #end of batch loop
        
        return batch_clusters

    def make_clusters2(self,coord_t,embed_t,seed_t,verbose=False,sigma_scale=5.0,seed_threshold=0.5):
        # we have a copy of the loss module to get the functions that give us the embedding and seed values
        loss = SpatialEmbedLoss( self.input_shape, sigma_scale=sigma_scale )
        batch_clusters = []
        with torch.no_grad():
            batch_size = coord_t[:,3].max()+1

            fcoord_t = coord_t.to(torch.float32)
            fcoord_t[:,0] /= float(self.input_shape[0])
            fcoord_t[:,1] /= float(self.input_shape[1])
            fcoord_t[:,2] /= float(self.input_shape[2])

            for b in range(batch_size):
                if verbose: print "==== BATCH ",b," =================="
                bmask = coord_t[:,3].eq(b)
                coord_b = fcoord_t[bmask,:]
                embed_b = [embed_t[ishape][bmask,:] for ishape in range(self.embedout_shapes)]                            
                seed_b  = seed_t[bmask,:]

                # calc embeded position
                spembed_b = [coord_b[:,0:3]+embed_b[ishape][:,0:3] for ishape in range(self.embedout_shapes)]# coordinate + shift
                sigma_b   = [torch.exp(embed_b[ishape][:,3:3+self.nsigma]) for ishape in range(self.embedout_shapes)]                            
                nvoxels_b = spembed_b[0].shape[0]

                # location where we accumulate cluster results
                cluster_id_all = torch.zeros( (nvoxels_b,self.nclasses), dtype=torch.int ).to( embed_t[0].device )

                icluster = 0
                maxval = 1.0
                while maxval>0.5:
                    unused_pixels = cluster_id_all[:,0].eq(0)
                    maxval = 0.0
                    maxclass = 0
                    maxarg = 0
                    for c in range(self.nclasses):
                        class_maxarg = torch.argmax(seed_b[:,c]*unused_pixels.float())
                        class_maxval = seed_b[class_maxarg,c]
                        if class_maxval > maxval:
                            maxval = class_maxval
                            maxclass = c
                            maxarg = class_maxarg
                    if maxval<0.5:
                        break
                    maxshape = loss.class_to_shape[maxclass]

                    # we cluster using the class
                    if verbose or True: print "FORM CLUSTER[",icluster,"] max-class=",maxclass+1," ///////"
                    if verbose or True: print " max-voxel-index: ",maxarg
                    if verbose or True: print " max-voxel-value: ",maxval
                    if verbose or True: print " max-voxel-shape: ",maxshape
                    centroid_i = spembed_b[maxshape][maxarg,:].view( (1,3) )
                    sigma_i    = sigma_b[maxshape][maxarg,:].view( (1,3) )
                    if verbose or True: print " centroid: ",centroid_i
                    if verbose or True: print " sigma: ",sigma_i
                    if verbose or True: print " spembed_b: ",spembed_b[maxshape].shape
                    prob = torch.exp( -sigma_scale*torch.sum( torch.pow( spembed_b[maxshape]-centroid_i, 2 )*sigma_i, 1 ) )*(unused_pixels.float())
                    # no cluster made
                    if (prob>0.5).sum()==0:
                        break
                    cluster_id_all[ prob>0.5, 0 ] = icluster+1
                    icluster += 1
                batch_clusters.append( (cluster_id_all,spembed_b[0],seed_b) )
                
        return batch_clusters


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
