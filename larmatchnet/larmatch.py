from __future__ import print_function
from collections import OrderedDict
import torch
import torch.nn as nn
import sparseconvnet as scn
from utils_sparselarflow import create_resnet_layer

class LArMatch(nn.Module):

    def __init__(self,ndimensions=2,inputshape=(1024,3456),
                 input_nfeatures=1,
                 stem_nfeatures=16,
                 features_per_layer=16,                 
                 classifier_nfeatures=[32,32],
                 leakiness=0.01,
                 ninput_planes=3,
                 use_unet=False,
                 nresnet_blocks=10 ):
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
        nresnet_blocks [int] number of resnet blocks if not using unet, DEPRECATED, default=10
        """
        super(LArMatch,self).__init__()

        self.use_unet = use_unet
        
        # INPUT LAYERS: converts torch tensor into scn.SparseMatrix
        self.ninput_planes = ninput_planes        
        self.source_inputlayer  = scn.InputLayer(ndimensions,inputshape,mode=0)
        self.target1_inputlayer = scn.InputLayer(ndimensions,inputshape,mode=0)
        self.target2_inputlayer = scn.InputLayer(ndimensions,inputshape,mode=0)

        # STEM
        self.stem = scn.Sequential() 
        self.stem.add( scn.SubmanifoldConvolution(ndimensions, input_nfeatures, stem_nfeatures, 3, False ) )
        self.stem.add( scn.BatchNormLeakyReLU(stem_nfeatures,leakiness=leakiness) )
        if self.use_unet:
            self.stem.add( scn.SubmanifoldConvolution(ndimensions, stem_nfeatures, stem_nfeatures, 3, False ) )
            self.stem.add( scn.BatchNormLeakyReLU(stem_nfeatures,leakiness=leakiness) )
            self.stem.add( scn.SubmanifoldConvolution(ndimensions, stem_nfeatures, stem_nfeatures, 3, False ) )

        # UNET BLOCK
        if self.use_unet:
            self.unet_layers = scn.UNet( 2, 2,
                                         [stem_nfeatures,
                                          stem_nfeatures*2,
                                          stem_nfeatures*3,
                                          stem_nfeatures*4,
                                          stem_nfeatures*5],
                                         residual_blocks=True,
                                         downsample=[2, 2] )

        # RESNET BLOCK
        if not self.use_unet:
            self.resnet_layers = create_resnet_layer(nresnet_blocks, stem_nfeatures, features_per_layer, leakiness=leakiness )
        else:
            self.resnet_layers = create_resnet_layer(1,  stem_nfeatures, features_per_layer, leakiness=leakiness )
            self.resnet_layers.add( scn.BatchNormLeakyReLU(stem_nfeatures,leakiness=leakiness) )

        # OUTPUT FEATURES
        self.nfeatures = features_per_layer
        self.feature_layer = scn.SubmanifoldConvolution(ndimensions, features_per_layer, self.nfeatures, 1, True )

        # from there, we move back into a tensor
        self.source_outlayer  = scn.OutputLayer(ndimensions)
        self.target1_outlayer = scn.OutputLayer(ndimensions)
        self.target2_outlayer = scn.OutputLayer(ndimensions)

        # CLASSIFER: MATCH/NO-MATCH
        classifier_layers = OrderedDict()
        classifier_layers["class0conv"] = torch.nn.Conv1d(self.ninput_planes*features_per_layer,classifier_nfeatures[0],1)
        #classifier_layers["class0bn"]   = torch.nn.BatchNorm1d        
        classifier_layers["class0relu"] = torch.nn.ReLU()
        for ilayer,nfeats in enumerate(classifier_nfeatures[1:]):
            classifier_layers["class%dconv"%(ilayer+1)] = torch.nn.Conv1d(nfeats,nfeats,1)
            #classifier_layers["class0bn"]   = torch.nn.BatchNorm1d                    
            classifier_layers["class%drelu"%(ilayer+1)] = torch.nn.ReLU()
        classifier_layers["classout"] = torch.nn.Conv1d(nfeats,1,1)
        #classifier_layers["sigmoid"]  = torch.nn.Sigmoid()
        self.classifier = torch.nn.Sequential( classifier_layers )
        

    def forward_features( self, coord_plane0_t, plane0_feat_t,
                          coord_plane1_t, plane1_feat_t,
                          coord_plane2_t, plane2_feat_t,
                          batchsize, verbose=False ):

        """
        run the feature generating portion of network only. 
        get feature vector at each pixel coordinate.
        expects three planes

        input
        -----
        coord_plane0_t [torch tensor (N_0,3)] list of N pixel coordinates (row,col,batch)
        plane0_feat_t  [torch tensor (N_0,1)] list of pixel values for each N pixel coordinate
        coord_plane1_t [torch tensor (N_1,3)] list of N pixel coordinates (row,col,batch)
        plane1_feat_t  [torch tensor (N_1,1)] list of pixel values for each N pixel coordinate
        coord_plane2_t [torch tensor (N_2,3)] list of N pixel coordinates (row,col,batch)
        plane2_feat_t  [torch tensor (N_2,1)] list of pixel values for each N pixel coordinate        

        output
        ------
        tuple containing 3 feature tensors [torch tensors with shapes ((N_0,C), (N_1,C), (N_2,C))]
        """
        if verbose:
            print("[larmatch::make feature vectors] ")
            print("  coord[plane0]=",coord_plane0_t.shape," feat[plane0]=",plane0_feat_t.shape)
            print("  coord[plane1]=",coord_plane1_t.shape," feat[plane1]=",plane1_feat_t.shape)
            print("  coord[plane2]=",coord_plane2_t.shape," feat[plane2]=",plane2_feat_t.shape)

        # Form input tuples for input layers
        # adds torch tensors to SparseConvTensor object
        input_plane0 = ( coord_plane0_t, plane0_feat_t, batchsize )
        input_plane1 = ( coord_plane1_t, plane1_feat_t, batchsize )
        input_plane2 = ( coord_plane2_t, plane2_feat_t, batchsize )

        # generate plane0 features
        xplane0 = self.source_inputlayer(input_plane0)
        xplane0 = self.stem( xplane0 )
        if verbose: print("  stem[plane0]=",xplane0.features.shape)
        if self.use_unet: xplane0 =self.unet_layers(xplane0)            
        if verbose: print("  unet[plane0]=",xplane0.features.shape)
        xplane0 = self.resnet_layers( xplane0 )
        if verbose: print("  resnet[plane0]=",xplane0.features.shape)
        xplane0 = self.feature_layer( xplane0 )
        if verbose: print("  feature[plane0]=",xplane0.features.shape)
        
        xplane1 = self.target1_inputlayer(input_plane1)
        xplane1 = self.stem( xplane1 )
        if verbose: print("  stem[plane1]=",xplane1.features.shape)
        if self.use_unet: xplane1 =self.unet_layers(xplane1)
        if verbose: print("  unet[plane1]=",xplane1.features.shape)
        xplane1 = self.resnet_layers( xplane1 )
        if verbose: print("  resnet[plane1]=",xplane1.features.shape)
        xplane1 = self.feature_layer( xplane1 )
        if verbose: print("  feature[plane1]=",xplane1.features.shape)

        xplane2 = self.target2_inputlayer(input_plane2)
        xplane2 = self.stem( xplane2 )
        if verbose: print("  stem[plane2]=",xplane2.features.shape)
        if self.use_unet: xplane2 =self.unet_layers(xplane2)
        if verbose: print("  unet[plane2]=",xplane2.features.shape)
        xplane2 = self.resnet_layers( xplane2 )
        if verbose: print("  resnet[plane2]=",xplane2.features.shape)
        xplane2 = self.feature_layer( xplane2 )
        if verbose: print("  feature[plane2]=",xplane2.features.shape)

        # extracts torch tensors from SparseConvTensor object        
        xplane0  = self.source_outlayer( xplane0 )
        xplane1 = self.target1_outlayer( xplane1 )
        xplane2 = self.target2_outlayer( xplane2 )

        return xplane0,xplane1,xplane2
                                
    def extract_features(self, feat_u_t, feat_v_t, feat_y_t, index_t, npts, DEVICE, verbose=False ):
        """ 
        take in index list and concat the triplet feature vector.
        the feature vectors are those produced by the forward_feature method.
        The information of which feature vectors to combine are in index_t.
        The information for index_t is made using the larflow::PrepMatchTriplets class.
        
        inputs
        ------
        feat_u_t [torch tensor shape (N_u,C)] u-plane feature tensor containing C-length feature vector for N_u pixel coordinates.
        feat_v_t [torch tensor shape (N_v,C)] v-plane feature tensor containing C-length feature vector for N_v pixel coordinates.
        feat_y_t [torch tensor shape (N_y,C)] y-plane feature tensor containing C-length feature vector for N_y pixel coordinates.
        index_t  [torch tensor shape (N_m,3)] N_m triplets containing indices to feat_u_t, feat_v_t, feat_y_t that should be combined
        npts [int] number of points in index_t to evaluate
        DEVICE [torch device] device to put output tensors
        verbose [bool] print tensor shape information, default=False

        outputs
        --------
        feature vector for spacepoint triplet [torch tensor shape (1,3C,npts)]
        """
        if verbose:
            print("[larmatch::extract_features]")
            print("  index-shape=",index_t.shape,)
            print(" feat-u-shape=",feat_u_t.shape)
            print(" feat-v-shape=",feat_v_t.shape)
            print(" feat-y-shape=",feat_y_t.shape)

        feats   = [ feat_u_t, feat_v_t, feat_y_t ]
        for f in feats:
            f = f.to(DEVICE)
        feats_t = [ torch.index_select( feats[x], 0, index_t[:npts,x] ) for x in range(0,3) ]
        if verbose:
            print("  index-selected feats_t[0]=",feats_t[0].shape)
        
        veclen = feats_t[0].shape[1]+feats_t[1].shape[1]+feats_t[2].shape[1]
        catvec = torch.cat( (feats_t[0],feats_t[1],feats_t[2]), dim=1 )
        if verbose: print("  concat out: ",catvec.shape)
        matchvec = torch.transpose( catvec, 1, 0 ).reshape(1,veclen,npts)
        if verbose: print("  output-triplet-tensor: ",matchvec.shape)
            

        return matchvec
    
    def classify_triplet(self,triplet_feat_t):
        """
        classify triplet of (u,v,y) wire plane pixel locations as being a true or false position.
        use information from concat feature vectors.

        inputs:
        triplet_feat_t [torch tensor (1,3C,N)] concat spacepoint feature tensor, output of extract_features
        """
        pred = self.classifier(triplet_feat_t)
        return pred

