from __future__ import print_function
from collections import OrderedDict
import torch
import torch.nn as nn
import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
from .backbone_resunetme import MinkEncode6Layer, MinkDecode6Layer
from .resnetinstance_block import BasicBlockInstanceNorm
from .larmatch_spacepoint_classifier import LArMatchSpacepointClassifier
from .larmatch_ssnet_classifier import LArMatchSSNetClassifier
from .larmatch_keypoint_classifier import LArMatchKeypointClassifier
from .larmatch_kpshift_regressor   import LArMatchKPShiftRegressor
from .larmatch_affinityfield_regressor import LArMatchAffinityFieldRegressor

class LArMatchMinkowski(nn.Module):

    def __init__(self,ndimensions=2,
                 inputshape=(1024,3584),                 
                 input_nfeatures=1,
                 input_nplanes=3,
                 run_lm=True,
                 run_ssnet=True,
                 run_kp=True,
                 num_ssnet_classes=7,
                 num_kp_classes=6):
        """
        parameters
        -----------
        ndimensions [int]    number of spatial dimensions of input data, default=2
        inputshape  [tuple of int]  size of input tensor/image in (num of tick pixels, num of wire pixels), default=(1024,3456)
        input_nfeatures [int] number of features in the input tensor, default=1 (the image charge)
        """
        super(LArMatchMinkowski,self).__init__()
        
        # INPUT LAYERS: converts torch tensor into Minkowski Sparse Tensor
        self.ninput_planes = input_nplanes
        
        # STEM
        stem_nfeatures = 16
        stem_nlayers = 3
        stem_layers = OrderedDict()
        if stem_nlayers==1:
            respath = ME.MinkowskiConvolution( input_nfeatures, stem_nfeatures, kernel_size=1, stride=1, dimension=ndimensions )
            block   = BasicBlockInstanceNorm( input_nfeatures, stem_nfeatures, dimension=ndimensions, downsample=respath )
            #block   = BasicBlock( input_nfeatures, stem_nfeatures, dimension=ndimensions, downsample=respath )            
            stem_layers["stem_layer0"] = block
        else:
            for istem in range(stem_nlayers):
                if istem==0:
                    respath = ME.MinkowskiConvolution( input_nfeatures, stem_nfeatures, kernel_size=1, stride=1, dimension=ndimensions )
                    block   = BasicBlockInstanceNorm( input_nfeatures, stem_nfeatures, dimension=ndimensions, downsample=respath )
                    #block   = BasicBlock( input_nfeatures, stem_nfeatures, dimension=ndimensions, downsample=respath )                    
                else:
                    block   = BasicBlockInstanceNorm( stem_nfeatures, stem_nfeatures, dimension=ndimensions  )
                    #block   = BasicBlock( stem_nfeatures, stem_nfeatures, dimension=ndimensions  )                    
                stem_layers["stem_layer%d"%(istem)] = block
            
        self.stem = nn.Sequential(stem_layers)

        # RESIDUAL UNET FOR FEATURE CONSTRUCTION
        self.encoder = MinkEncode6Layer( in_channels=stem_nfeatures, out_channels=stem_nfeatures, D=2 )
        self.decoder = MinkDecode6Layer( in_channels=stem_nfeatures, out_channels=stem_nfeatures, D=2 )

        # sparse to dense operation
        self.sparse_to_dense = [ ME.MinkowskiToFeature() for p in range(input_nplanes) ]

        # DROPOUT ON FEATURE LAYER
        #self.dropout = ME.MinkowskiDropout()

        # TASK HEADS
        self.run_lm      = run_lm
        self.run_ssnet   = run_ssnet
        self.run_kplabel = run_kp
        self.run_paf     = False
        self.use_kp_bn   = False
        
        # CLASSIFERS
        if self.run_lm:      self.lm_classifier = LArMatchSpacepointClassifier( num_input_feats=stem_nfeatures*3 )
        if self.run_ssnet:   self.ssnet_head    = LArMatchSSNetClassifier(features_per_layer=stem_nfeatures,num_classes=num_ssnet_classes)
        if self.run_kplabel: self.kplabel_head  = LArMatchKeypointClassifier(features_per_layer=stem_nfeatures,nclasses=num_kp_classes,use_bn=self.use_kp_bn)
        if self.run_paf:     self.affinity_head = LArMatchAffinityFieldRegressor(layer_nfeatures=[8,8,8],input_features=features_per_layer)
        

    def forward( self, input_wireplane_sparsetensors, matchtriplets, batch_size ):

        # check input
        
        # we push through each sparse image through the stem and backbone (e.g. unet)
        x_feat_v = []
        for p,x_input in enumerate(input_wireplane_sparsetensors):
            #print(x_input)            
            x = self.stem(x_input)
            x_encode = self.encoder(x)
            x_decode = self.decoder(x_encode)
            #print(x_decode.shape)
            x_feat_v.append( x_decode )

        # then we have to extract a feature tensor
        batch_spacepoint_feat = self.extract_features(x_feat_v, matchtriplets, batch_size )

        # we pass the features through the different classifiers
        batch_output = []
        for b,spacepoint_feat in enumerate(batch_spacepoint_feat):
            output = {}            
            x = spacepoint_feat.unsqueeze(0)

            if self.run_lm:
                #print("batch ",b," spacepoint feats: ",x.shape)
                output["lm"] = self.lm_classifier( x )

            if self.run_ssnet:
                output["ssnet"] = self.ssnet_head( x )

            if self.run_kplabel:
                output["kp"] = self.kplabel_head( x )
            
            batch_output.append( output )

        return batch_output
                                        
    def extract_features(self, feat_v, index_t, batch_size, verbose=False ):
        """ 
        take in index list and concat the triplet feature vector.
        the feature vectors are those produced by the forward_feature method.
        The information of which feature vectors to combine are in index_t.
        The information for index_t is made using the larflow::PrepMatchTriplets class.
        
        inputs
        ------
        feat_v []
        index_t  [torch tensor shape (N_m,3)] N_m triplets containing indices to feat_u_t, feat_v_t, feat_y_t that should be combined
        npts [int] number of points in index_t to evaluate
        DEVICE [torch device] device to put output tensors
        verbose [bool] print tensor shape information, default=False

        outputs
        --------
        feature vector for spacepoint triplet [torch tensor shape (1,3C,npts)]
        """

        #for x in feat_v:
        #    print(x.decomposition_permutations)
        
        plane_feat_v = [ self.sparse_to_dense[p](x) for p,x in enumerate(feat_v) ]
        #for x in plane_feat_v:
        #    print("sparse to dense out: ",x.shape)

        batch_feats = []            
        for b in range(batch_size):
            batch_triplets = index_t[b]
            batch_spacepoint_v = []
            for p,x in enumerate(feat_v):
                batch_indices = x.decomposition_permutations[b]
                batch_plane_feat = torch.index_select( plane_feat_v[p], 0, batch_indices )
                #print("batch[%d]_plane[%d]_feat: "%(b,p),batch_plane_feat.shape)
                spacepoint_feat = torch.index_select( batch_plane_feat, 0, batch_triplets[:,p] )
                #print("batch[%d]_plane[%d] spacepoint_feat: "%(b,p),spacepoint_feat.shape)
                batch_spacepoint_v.append( spacepoint_feat )
            spacepoint_feats_t = torch.transpose( torch.cat( batch_spacepoint_v, dim=1 ), 1, 0 )
            #print("batch[%d] spacepoint_feats_t: "%(b),spacepoint_feats_t.shape)
            batch_feats.append( spacepoint_feats_t )
            
        return batch_feats
    
