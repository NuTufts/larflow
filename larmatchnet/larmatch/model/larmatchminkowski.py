from __future__ import print_function
from collections import OrderedDict
import torch
import torch.nn as nn
import MinkowskiEngine as ME
from .backbone_resunetme import MinkEncode6Layer, MinkDecode6Layer
from .resnetinstance_block import BasicBlockInstanceNorm
from .larmatch_spacepoint_classifier import LArMatchSpacepointClassifier
from .larmatch_ssnet_classifier import LArMatchSSNetClassifier
from .larmatch_keypoint_classifier import LArMatchKeypointClassifier
from .larmatch_kpshift_regressor   import LArMatchKPShiftRegressor
from .larmatch_affinityfield_regressor import LArMatchAffinityFieldRegressor

class LArMatchMinkowski(nn.Module):

    def __init__(self,ndimensions=2,inputshape=(1024,3584),                 
                 input_nfeatures=1,input_nplanes=3):
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
            stem_layers["stem_layer0"] = block
        else:
            for istem in range(stem_nlayers):
                if istem==0:
                    respath = ME.MinkowskiConvolution( input_nfeatures, stem_nfeatures, kernel_size=1, stride=1, dimension=ndimensions )
                    block   = BasicBlockInstanceNorm( input_nfeatures, stem_nfeatures, dimension=ndimensions, downsample=respath )
                else:
                    block   = BasicBlockInstanceNorm( stem_nfeatures, stem_nfeatures, dimension=ndimensions  )
                stem_layers["stem_layer%d"%(istem)] = block
            
        self.stem = nn.Sequential(stem_layers)

        # RESIDUAL UNET FOR FEATURE CONSTRUCTION
        self.encoder = MinkEncode6Layer( in_channels=stem_nfeatures, out_channels=stem_nfeatures, D=2 )
        self.decoder = MinkDecode6Layer( in_channels=stem_nfeatures, out_channels=stem_nfeatures, D=2 )

        # sparse to dense operation
        self.sparse_to_dense = [ ME.MinkowskiToFeature() for p in range(input_nplanes) ]

        # DROPOUT ON FEATURE LAYER
        self.dropout = ME.MinkowskiDropout()
        
        # CLASSIFERS
        self.lm_classifier = LArMatchSpacepointClassifier( num_input_feats=stem_nfeatures*3 )

        # OTHER TASK HEADS
        self.run_ssnet   = False
        self.run_kplabel = False
        self.run_paf     = False
        self.use_kp_bn   = True
        if self.run_ssnet:   self.ssnet_head    = LArMatchSSNetClassifier(features_per_layer=stem_nfeatures,num_classes=num_ssnet_classes)
        if self.run_kplabel: self.kplabel_head  = LArMatchKeypointClassifier(features_per_layer=stem_nfeatures,nclasses=num_kp_classes,use_bn=self.use_kp_bn)
        if self.run_paf:     self.affinity_head = LArMatchAffinityFieldRegressor(layer_nfeatures=[8,8,8],input_features=features_per_layer)
        

    def forward( self, input_wireplane_sparsetensors, matchtriplets, orig_coords, batch_size ):

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
        batch_spacepoint_feat = self.extract_features(x_feat_v, matchtriplets, orig_coords, batch_size )

        # we pass the features through the different classifiers
        batch_output = []
        for b,spacepoint_feat in enumerate(batch_spacepoint_feat):
            output = {}            
            x = spacepoint_feat.unsqueeze(0)
            print("batch ",b," spacepoint feats: ",x.shape)
            lm_pred = self.lm_classifier( x )
            output["lm"] = lm_pred

            batch_output.append( output )
        
        return batch_output
                                        
    def extract_features(self, feat_v, index_t, orig_coords, batch_size, verbose=False ):
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
            print("batch[%d] spacepoint_feats_t: "%(b),spacepoint_feats_t.shape)
            batch_feats.append( spacepoint_feats_t )
            
        return batch_feats
            
    # def classify_triplet(self,triplet_feat_t):
    #     """
    #     classify triplet of (u,v,y) wire plane pixel locations as being a true or false position.
    #     use information from concat feature vectors.

    #     inputs:
    #     triplet_feat_t [torch tensor (1,3C,N)] concat spacepoint feature tensor, output of extract_features
    #     """
    #     pred = self.classifier(triplet_feat_t)
    #     return pred

    # def forward(self,coord_t, feat_t, match_t, npairs, device, verbose=False):
    #     """
    #     all pass, from input to head outputs for select match indices
    #     """
    #     if type(coord_t) is not list:
    #         raise ValueError("argument `coord_t` should be list of torch tensors")
    #     nplanes = len(coord_t)
    #     if len(feat_t)!=nplanes:
    #         raise ValueError("number of tensors in coord_t and feat_t do not match")
    #     if nplanes!=3:
    #         raise ValueError("number of coord_t and feat_t is not 3. (current limitation)")


    #     feat_u_t, feat_v_t, feat_y_t = self.forward_features( coord_t[0], feat_t[0],
    #                                                           coord_t[1], feat_t[1],
    #                                                           coord_t[2], feat_t[2], 1,
    #                                                           verbose=verbose )
        
    #     feat_triplet_t = self.extract_features( feat_u_t, feat_v_t, feat_y_t,
    #                                             match_t, npairs,
    #                                             device, verbose=verbose )

    #     # evaluate larmatch match classifier
    #     match_pred_t = self.classify_triplet( feat_triplet_t )
    #     match_pred_t = match_pred_t.reshape( (match_pred_t.shape[-1]) )
    #     if verbose: print("[larmatch] match-pred=",match_pred_t.shape)

    #     outdict = {"match":match_pred_t}

    #     if self.run_ssnet:
    #         ssnet_pred_t = self.ssnet_head.forward( feat_triplet_t )
    #         ssnet_pred_t = ssnet_pred_t.reshape( (ssnet_pred_t.shape[1],ssnet_pred_t.shape[2]) )
    #         ssnet_pred_t = torch.transpose( ssnet_pred_t, 1, 0 )
    #         outdict["ssnet"] = ssnet_pred_t
    #         if verbose: print("[larmatch] ssnet-pred=",ssnet_pred_t.shape)

    #     if self.run_kplabel:
    #         kplabel_pred_t = self.kplabel_head.forward( feat_triplet_t )
    #         kplabel_pred_t = kplabel_pred_t.reshape( (kplabel_pred_t.shape[1], kplabel_pred_t.shape[2]) )
    #         kplabel_pred_t = torch.transpose( kplabel_pred_t, 1, 0 )
    #         outdict["kplabel"] = kplabel_pred_t
    #         if verbose: print("[larmatch train] kplabel-pred=",kplabel_pred_t.shape)
        
    #     if self.run_kpshift:
    #         kpshift_pred_t = self.kpshift_head.forward( feat_triplet_t )
    #         kpshift_pred_t = kpshift_pred_t.reshape( (kpshift_pred_t.shape[1],kpshift_pred_t.shape[2]) )
    #         kpshift_pred_t = torch.transpose( kpshift_pred_t, 1, 0 )
    #         outdict["kpshift"] = kpshift_pred_t
    #         if verbose: print("[larmatch train] kpshift-pred=",kpshift_pred_t.shape)


    #     if self.run_paf:
    #         paf_pred_t = self.affinity_head.forward( feat_triplet_t )
    #         paf_pred_t = paf_pred_t.reshape( (paf_pred_t.shape[1],paf_pred_t.shape[2]) )
    #         paf_pred_t = torch.transpose( paf_pred_t, 1, 0 )
    #         outdict["paf"] = paf_pred_t
    #         if verbose: print("[larmatch train]: paf pred=",paf_pred_t.shape)


    #     return outdict
            
    # def extract_features(self, feat_u_t, feat_v_t, feat_y_t, index_t, npts, DEVICE, verbose=False ):
    #     """ 
    #     take in index list and concat the triplet feature vector.
    #     the feature vectors are those produced by the forward_feature method.
    #     The information of which feature vectors to combine are in index_t.
    #     The information for index_t is made using the larflow::PrepMatchTriplets class.
        
    #     inputs
    #     ------
    #     feat_u_t [torch tensor shape (N_u,C)] u-plane feature tensor containing C-length feature vector for N_u pixel coordinates.
    #     feat_v_t [torch tensor shape (N_v,C)] v-plane feature tensor containing C-length feature vector for N_v pixel coordinates.
    #     feat_y_t [torch tensor shape (N_y,C)] y-plane feature tensor containing C-length feature vector for N_y pixel coordinates.
    #     index_t  [torch tensor shape (N_m,3)] N_m triplets containing indices to feat_u_t, feat_v_t, feat_y_t that should be combined
    #     npts [int] number of points in index_t to evaluate
    #     DEVICE [torch device] device to put output tensors
    #     verbose [bool] print tensor shape information, default=False

    #     outputs
    #     --------
    #     feature vector for spacepoint triplet [torch tensor shape (1,3C,npts)]
    #     """
    #     if verbose:
    #         print("[larmatch::extract_features]")
    #         print("  index-shape=",index_t.shape,)
    #         print(" feat-u-shape=",feat_u_t.shape)
    #         print(" feat-v-shape=",feat_v_t.shape)
    #         print(" feat-y-shape=",feat_y_t.shape)

    #     feats   = [ feat_u_t, feat_v_t, feat_y_t ]
    #     for f in feats:
    #         f = f.to(DEVICE)
    #     feats_t = [ torch.index_select( feats[x], 0, index_t[:npts,x] ) for x in range(0,3) ]
    #     if verbose:
    #         print("  index-selected feats_t[0]=",feats_t[0].shape)
        
    #     veclen = feats_t[0].shape[1]+feats_t[1].shape[1]+feats_t[2].shape[1]
    #     catvec = torch.cat( (feats_t[0],feats_t[1],feats_t[2]), dim=1 )
    #     if verbose: print("  concat out: ",catvec.shape)
    #     matchvec = torch.transpose( catvec, 1, 0 ).reshape(1,veclen,npts)
    #     if verbose: print("  output-triplet-tensor: ",matchvec.shape)
            

    #     return matchvec
    
        

