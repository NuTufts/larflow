import os,sys
import torch
import torch.nn
import sparseconvnet as scn

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


def create_resnet_layer(nreps, ninputchs,noutputchs,leakiness=0.01):
                        
    """
    creates a layer formed by a repetition of residual blocks

    inputs
    ------
    nreps [int] number of times to repeat residual block
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
            residual_block(m,ninputchs, noutputchs, leakiness=leakiness)
        else:
            # other repitions we do not change number of features
            residual_block(m,noutputchs, noutputchs, leakiness=leakiness)
    return m

def resnet_encoding_layers( input_nfeatures, features_per_layer, reps, dimensions=2, downsample=[2,2], leakiness=0.001 ):
    """
    we generate layers of successive resnet blocks, where all the last layer applying a strided convolution
    """

    nfeats = [input_nfeatures] + features_per_layer
    
    # Encoding
    conv_layers = []
    nlayers = len(features_per_layer)
    for i in range(nlayers):
        m = scn.Sequential()                    
        m.add(scn.BatchNormLeakyReLU(nfeats[i], leakiness=leakiness))
        m.add(scn.Convolution(dimensions, nfeats[i], nfeats[i+1],
                              downsample[0], downsample[1], False) )

        for _ in range(reps):
            residual_block(m, nfeats[i+1], nfeats[i+1],
                           leakiness=leakiness, dimensions=dimensions)
        conv_layers.append( m )
        
    return conv_layers

def resnet_decoding_layers( unet_input_nfeats, encoder_nfeatures_per_layer, reps,
                            dimensions=3, upsample=[2,2], leakiness=0.001 ):
    # Decoding layers
    nlayers = len(encoder_nfeatures_per_layer)
    feats = [unet_input_nfeats] + encoder_nfeatures_per_layer
    feats.reverse()
    
    conv_layers = []
    up_layers = []
    for i in range(nlayers):

        input_feats = feats[i]
        if i>0:
            input_feats = 2*feats[i]
        skip_feats  = feats[i+1]

        # upsample but reduce feats
        m_up = scn.Sequential()
        m_up.add( scn.BatchNormLeakyReLU(input_feats,leakiness=leakiness))
        m_up.add( scn.Deconvolution(dimensions, input_feats, skip_feats,
                                    upsample[0], upsample[1], True) )
        up_layers.append(m_up)

        m_conv = scn.Sequential()
        for _ in range(reps):            
            residual_block(m_conv, 2*skip_feats, 2*skip_feats,
                           leakiness=leakiness, dimensions=dimensions)
        conv_layers.append( m_conv )
    return conv_layers,up_layers
    
