import os,sys

import torch
from sparselarflow import SparseLArFlow

def load_models(name, device="cpu", IMAGE_HEIGHT=512, IMAGE_WIDTH=832,
                weight_file=None,weight_dict=None):
    """
    This function builds models with specified parameters.
    """
    
    if name=="dualflow_v1":
        imgdims = 2
        ninput_features  = 16
        noutput_features = 16
        nplanes = 5
        nfeatures_per_layer = [16,16,32,32,64]
        flowdirs = ['y2u','y2v']
    
        model = SparseLArFlow( (IMAGE_HEIGHT,IMAGE_WIDTH), imgdims,
                               ninput_features, noutput_features,
                               nplanes, features_per_layer=nfeatures_per_layer,
                               home_gpu=None,
                               flowdirs=flowdirs,show_sizes=False).to(torch.device(device))

    else:
        raise ValueError("Model name, {}, not recognized".format(name))

    if weight_file is not None:
        # load weights if provided
        
        if weight_dict is None:
            # create default weight mapping to CPU
            weight_dict = {}
            for x in xrange(10):
                weight_dict["cuda:%d"%(x)] = "cpu"

        checkpoint = torch.load( weight_file, map_location=weight_dict )
        model.load_state_dict( checkpoint["state_dict"] )

    return model

