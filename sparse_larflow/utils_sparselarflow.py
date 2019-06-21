import os,sys

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
    
