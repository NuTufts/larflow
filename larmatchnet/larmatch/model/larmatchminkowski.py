from __future__ import print_function
from collections import OrderedDict
import torch
import torch.nn as nn
import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
from .backbone_resunetme import MinkEncode6LayerInstance, MinkDecode6LayerInstance, MinkEncode6LayerBasicBlock, MinkDecode6LayerBasicBlock
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
                 run_paf=True,
                 num_ssnet_classes=5,
                 num_kp_classes=6,
                 use_kp_bn=True,
                 use_feature_dropout=False,
                 norm_layer='batchnorm'):
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
            if norm_layer=='instancenorm':            
                block   = BasicBlockInstanceNorm( input_nfeatures, stem_nfeatures, dimension=ndimensions, downsample=respath )
            elif norm_layer=='batchnorm':
                block   = BasicBlock( input_nfeatures, stem_nfeatures, dimension=ndimensions, downsample=respath )
            stem_layers["stem_layer0"] = block
        else:
            for istem in range(stem_nlayers):
                if istem==0:
                    respath = ME.MinkowskiConvolution( input_nfeatures, stem_nfeatures, kernel_size=1, stride=1, dimension=ndimensions )
                    if norm_layer=='instancenorm':
                        block   = BasicBlockInstanceNorm( input_nfeatures, stem_nfeatures, dimension=ndimensions, downsample=respath )
                    elif norm_layer=='batchnorm':
                        block   = BasicBlock( input_nfeatures, stem_nfeatures, dimension=ndimensions, downsample=respath )                    
                else:
                    if norm_layer=='instancenorm':
                        block   = BasicBlockInstanceNorm( stem_nfeatures, stem_nfeatures, dimension=ndimensions  )
                    elif norm_layer=='batchnorm':
                        block   = BasicBlock( stem_nfeatures, stem_nfeatures, dimension=ndimensions  )                    
                stem_layers["stem_layer%d"%(istem)] = block
            
        self.stem = nn.Sequential(stem_layers)

        # RESIDUAL UNET FOR FEATURE CONSTRUCTION
        if norm_layer=="instancenorm":
            self.encoder = MinkEncode6LayerInstance( in_channels=stem_nfeatures, out_channels=stem_nfeatures, D=2 )
            self.decoder = MinkDecode6LayerInstance( in_channels=stem_nfeatures, out_channels=stem_nfeatures, D=2 )
        elif norm_layer=="batchnorm":
            self.encoder = MinkEncode6LayerBasicBlock( in_channels=stem_nfeatures, out_channels=stem_nfeatures, D=2 )
            self.decoder = MinkDecode6LayerBasicBlock( in_channels=stem_nfeatures, out_channels=stem_nfeatures, D=2 )
        else:
            raise ValueError("unrecognized norm_layer value: ",norm_layer)
            

        # sparse to dense operation
        self.sparse_to_dense = [ ME.MinkowskiToFeature() for p in range(input_nplanes) ]

        # DROPOUT ON FEATURE LAYER
        self.use_feature_dropout = use_feature_dropout
        if self.use_feature_dropout:
            self.dropout = ME.MinkowskiDropout()
        else:
            self.dropout = None

        # TASK HEADS
        self.run_lm      = run_lm
        self.run_ssnet   = run_ssnet
        self.run_kplabel = run_kp
        self.run_paf     = run_paf
        self.use_kp_bn   = use_kp_bn

        # For the tasks per spacepoint, we run several MLPs that use a feature vector
        # made by concatenating three feature vectors, one from each of the pixels from the three wire planes.
        spacepoint_nfeatures = stem_nfeatures*3
        
        
        # CLASSIFERS
        if self.run_lm:      self.lm_classifier = LArMatchSpacepointClassifier( num_input_feats=spacepoint_nfeatures )
        if self.run_ssnet:   self.ssnet_head    = LArMatchSSNetClassifier(features_per_layer=stem_nfeatures,num_classes=num_ssnet_classes)
        if self.run_kplabel: self.kplabel_head  = LArMatchKeypointClassifier(features_per_layer=stem_nfeatures,nclasses=num_kp_classes,use_bn=self.use_kp_bn)
        if self.run_paf:     self.affinity_head = LArMatchAffinityFieldRegressor(layer_nfeatures=[8,8,8],input_features=stem_nfeatures)

        # custom weight initialization
        self._init_weights()
        

    def _init_weights(self):
        # initialize the weights from the various subcomponents of the model
        for module in self.stem.modules():
            if isinstance(module, ME.MinkowskiConvolution):
                nn.init.kaiming_normal_(module.kernel, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias,0.0)
        for module in self.encoder.modules():
            if isinstance(module, ME.MinkowskiConvolution):
                nn.init.kaiming_normal_(module.kernel, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:                
                    nn.init.constant_(module.bias,0.0)
        for module in self.decoder.modules():
            if isinstance(module, ME.MinkowskiConvolution):
                nn.init.kaiming_normal_(module.kernel, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:                
                    nn.init.constant_(module.bias,0.0)
                
        if self.run_lm:
            self.lm_classifier._init_weights()
        if self.run_ssnet:
            self.ssnet_head._init_weights()
        if self.run_kplabel:
            self.kplabel_head._init_weights()
        if self.run_paf:
            self.affinity_head._init_weights()
             
            
    
    def forward( self, input_wireplane_sparsetensors, matchtriplets, query_v, batch_size ):

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
            x_feat_v.append( x_decode )

        if self.use_feature_dropout:
            for p in range(len(x_feat_v)):
                x_feat_v[p] = self.dropout( x_feat_v[p] )

        # then we have to extract a feature tensor
        batch_spacepoint_feat = self.extract_features(x_feat_v, matchtriplets, query_v, batch_size )
            
        #for b,spacepoint_feat in enumerate(batch_spacepoint_feat):
        #    print("--------------------------------------------------------")
        #    print("extracted features batch[",b,"]_spacepoint_feat")            
        #    print(spacepoint_feat)
        #print("--------------------------------------------------------")            

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

            if self.run_paf:
                output['paf'] = self.affinity_head( x )
            
            batch_output.append( output )

        return batch_output
                                        
    def extract_features(self, feat_v, index_t, query_v, batch_size, verbose=False ):
        """ 
        take in index list and concat the triplet feature vector.
        the feature vectors are those produced by the forward_feature method.
        The information of which feature vectors to combine are in index_t.
        The information for index_t is made using the larflow::PrepMatchTriplets class.
        
        inputs
        ------
        feat_v [] a list of SparseTensor, output of model
        index_t  [torch tensor shape (N_m,3)] N_m triplets containing indices to feat_u_t, feat_v_t, feat_y_t that should be combined
        npts [int] number of points in index_t to evaluate
        DEVICE [torch device] device to put output tensors
        verbose [bool] print tensor shape information, default=False

        outputs
        --------
        feature vector for spacepoint triplet [torch tensor shape (1,3C,npts)]
        """

        spacepoint_feat_v = [ feat_v[p].features_at_coordinates( query_v[p] ) for p in range(3) ]
        #for p in range(3):
        #    print("plane[",p,"] spacepoint_feat_v: ",spacepoint_feat_v[p].shape)

        # the feature tensor covers the whole batch
        #plane_feat_v = [ self.sparse_to_dense[p](x) for p,x in enumerate(feat_v) ]
        #for p,x in enumerate(plane_feat_v):
        #    print("-------------------------------------------------------")
        #    print("sparse to dense out plane[",p,"]: ",x.shape)
        #print("---------------------------------------------------------")

        batch_feats = []     
        bstart = 0      
        for b in range(batch_size):
            batch_triplets = index_t[b]
            batch_spacepoint_v = []

            npts = batch_triplets.shape[0]

            #plane_feat_v = [ feat_v[p].features_at(batch_index=b) for p in range(3) ]

            for p,x in enumerate(spacepoint_feat_v):
                #print("----------------------------------")
                #print("plane[",p,"] feat: ",x.shape)
                batch_spacepoint_v.append( x[bstart:bstart+npts] )
            spacepoint_feats_t = torch.transpose( torch.cat( batch_spacepoint_v, dim=1 ), 1, 0 )
            #print("------------------------------------------------------------")
            #print("(extract) batch[%d] spacepoint_feats_t: "%(b),spacepoint_feats_t.shape)
            bstart += npts
            #print(spacepoint_feats_t)
            #print("------------------------------------------------------------")            
            batch_feats.append( spacepoint_feats_t )
            
        return batch_feats

    def get_unet_params(self):
        unet_params = []
        unet_params += self.stem.parameters()
        unet_params += self.encoder.parameters()
        unet_params += self.decoder.parameters()
        for layer in self.sparse_to_dense:
            unet_params += layer.parameters()
        if self.dropout is not None:
            unet_params += self.dropout.parameters()
        return unet_params

    def get_head_params(self):
        head_params = []
        if self.run_lm: head_params += self.lm_classifier.parameters()
        if self.run_ssnet: head_params += self.ssnet_head.parameters()
        if self.run_kplabel: head_params += self.kplabel_head.parameters()
        if self.run_paf: head_params +=  self.affinity_head.parameters()
        return head_params
