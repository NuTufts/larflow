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
                 stem_nfeatures = 16,
                 stem_nlayers = 3,                 
                 run_lm=True,
                 run_ssnet=True,
                 run_kp=True,
                 norm_layer='instance',
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
        stem_layers = OrderedDict()
        if stem_nlayers==1:
            respath = ME.MinkowskiConvolution( input_nfeatures, stem_nfeatures, kernel_size=1, stride=1, dimension=ndimensions )
            block   = BasicBlockInstanceNorm( input_nfeatures, stem_nfeatures, dimension=ndimensions, downsample=respath, norm_layer=norm_layer )
            #block   = BasicBlock( input_nfeatures, stem_nfeatures, dimension=ndimensions, downsample=respath )            
            stem_layers["stem_layer0"] = block
        else:
            for istem in range(stem_nlayers):
                if istem==0:
                    respath = ME.MinkowskiConvolution( input_nfeatures, stem_nfeatures, kernel_size=1, stride=1, dimension=ndimensions )
                    block   = BasicBlockInstanceNorm( input_nfeatures, stem_nfeatures, dimension=ndimensions, downsample=respath, norm_layer=norm_layer )
                    #block   = BasicBlock( input_nfeatures, stem_nfeatures, dimension=ndimensions, downsample=respath )                    
                else:
                    block   = BasicBlockInstanceNorm( stem_nfeatures, stem_nfeatures, dimension=ndimensions, norm_layer=norm_layer  )
                    #block   = BasicBlock( stem_nfeatures, stem_nfeatures, dimension=ndimensions  )                    
                stem_layers["stem_layer%d"%(istem)] = block
            
        self.stem = nn.Sequential(stem_layers)

        # RESIDUAL UNET FOR FEATURE CONSTRUCTION
        self.encoder = MinkEncode6Layer( in_channels=stem_nfeatures, out_channels=stem_nfeatures, D=2, norm_layer=norm_layer )
        self.decoder = MinkDecode6Layer( in_channels=stem_nfeatures, out_channels=stem_nfeatures, D=2, norm_layer=norm_layer )

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
        if self.run_lm:      self.lm_classifier = LArMatchSpacepointClassifier( num_input_feats=stem_nfeatures*3, norm_layer=norm_layer )
        if self.run_ssnet:   self.ssnet_head    = LArMatchSSNetClassifier(features_per_layer=stem_nfeatures,num_classes=num_ssnet_classes,norm_layer=norm_layer)
        if self.run_kplabel: self.kplabel_head  = LArMatchKeypointClassifier(features_per_layer=stem_nfeatures,nclasses=num_kp_classes,norm_layer=norm_layer)
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
            #print("------------------------------------------------------------")
            #print("output features plane[",p,"] ",x_decode.shape)
            #print(x_decode)
            #print("------------------------------------------------------------")
            x_decode = ME.cat(x_decode,x_input)
            #print("------------------------------------------------------------")
            #print("decoder out + input cat")
            #print(x_decode)
            #print("------------------------------------------------------------")            
            x_feat_v.append( x_decode )

        # then we have to extract a feature tensor
        batch_spacepoint_feat   = self.extract_features(x_feat_v, matchtriplets, batch_size )
        #with torch.no_grad():
        #    for b in range(batch_spacepoint_feat.shape[0]):
        #        spacepoint_feat = batch_spacepoint_feat[b]
        #        print("--------------------------------------------------------")
        #        print("extracted features batch[",b,"]_spacepoint_feat")            
        #        print(spacepoint_feat.shape)
        #    print("--------------------------------------------------------")            

        # we pass the features through the different classifiers
        output = {}        
        if self.run_lm:
            #print("batch ",b," spacepoint feats: ",x.shape)
            output["lm"] = self.lm_classifier( batch_spacepoint_feat )

        if self.run_ssnet:
            output["ssnet"] = self.ssnet_head( batch_spacepoint_feat )

        if self.run_kplabel:
            output["kp"] = self.kplabel_head( batch_spacepoint_feat )
            
        return output
                                        
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

        #for p,x in enumerate(feat_v):
        #    print("----------------------------------------")
        #    print("feat_v[",p,"] decomposition_permutations")
        #    print(x.decomposition_permutations)
        #    print("----------------------------------------")        
        
        plane_feat_v = [ self.sparse_to_dense[p](x) for p,x in enumerate(feat_v) ]
        #for p,x in enumerate(plane_feat_v):
        #    print("-------------------------------------------------------")
        #    print("sparse to dense out plane[",p,"]: ",x.shape)
        #    print(x)
        #print("---------------------------------------------------------")

        batch_feats = []            
        for b in range(batch_size):
            batch_triplets = index_t[b]
            batch_spacepoint_v = []

            for p,x in enumerate(plane_feat_v):
                #print("----------------------------------")
                #batch_indices,batch_feats = x.coordinates_and_features_at(b)
                #batch_decomp = x.decomposition_permutations[b]
                #print("(extract) batch indices: ",batch_indices.shape)
                #print(batch_indices)
                #print("(extract) batch feats: ",batch_feats.shape)
                #print(batch_feats)
                #print("(extract) batch_triplets: ",batch_triplets[:,p].shape)
                #print(batch_triplets[:,p])                
                #batch_plane_feat  = torch.index_select( batch_feats,   0, batch_decomp )
                #batch_plane_coord = torch.index_select( batch_indices, 0, batch_decomp )
                #print("(extract) batch_plane_coord: ",batch_plane_coord.shape)
                #print(batch_plane_coord)                
                #batch_plane_feat = x.features_at(b)
                #print("(extract) batch[%d]_plane[%d]_feat: "%(b,p),batch_plane_feat.shape)                
                #print(batch_plane_feat)                
                #spacepoint_feat = torch.index_select( batch_plane_feat, 0, batch_triplets[:,p] )
                spacepoint_feat = torch.index_select( x, 0, batch_triplets[:,p] )
                #print("(extract) batch[%d]_plane[%d] spacepoint_feat: "%(b,p),spacepoint_feat.shape)
                #print( spacepoint_feat )
                batch_spacepoint_v.append( spacepoint_feat )
            spacepoint_feats_t = torch.transpose( torch.cat( batch_spacepoint_v, dim=1 ), 1, 0 )
            #print("------------------------------------------------------------")
            #print("(extract) batch[%d] spacepoint_feats_t: "%(b),spacepoint_feats_t.shape)
            #print(spacepoint_feats_t)
            #print("------------------------------------------------------------")            
            batch_feats.append( spacepoint_feats_t.unsqueeze(0) )
        
        # contact
        batch_feats_cat = torch.cat( batch_feats, dim=0 )
            
        return batch_feats_cat
