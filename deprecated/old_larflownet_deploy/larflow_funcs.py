# larflow_funcs: utility functions used by the various analysis scripts

# python builtins
import os,sys,time
from collections import OrderedDict

# numpy
import numpy as np

# ROOT/larcv
import ROOT as rt
from larcv import larcv

# pytorch
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F

# larflow
LARFLOW_MODEL_DIR=None
if "LARFLOW_BASEDIR" in os.environ:
    LARFLOW_MODEL_DIR=os.environ["LARFLOW_BASEDIR"]+"/models"
    if LARFLOW_MODEL_DIR not in os.environ:
        sys.path.append(LARFLOW_MODEL_DIR)
else:
    sys.path.append("../models")

from larflow_uresnet import LArFlowUResNet


# load model with weights from checkpoints

def load_model( checkpointfile, deviceid=0, checkpointgpu=0, use_half=False ):

    model = LArFlowUResNet(inplanes=32,input_channels=1,showsizes=False,use_visi=False)
    if use_half:
        model = model.half()

    # stored parameters know what gpu ID they were on
    # if we change gpuids, we need to force the map here
    map_location=None
    if deviceid=="cpu":
        map_location={"cuda:%d"%(checkpointgpu):"cpu"}
    elif type(deviceid) is int and deviceid!=checkpointgpu:
        map_location={"cuda:%d"%(checkpointgpu):"cuda:%d"%(deviceid)}
    
    checkpoint = torch.load( checkpointfile, map_location=map_location )
    # check for data_parallel checkpoint which has "module" prefix
    from_data_parallel = False
    for k,v in checkpoint["state_dict"].items():
        if "module." in k:
            from_data_parallel = True
            break

    if from_data_parallel:
        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        checkpoint["state_dict"] = new_state_dict
    if use_half:
        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k] = v.half()
        checkpoint["state_dict"] = new_state_dict

    model.load_state_dict(checkpoint["state_dict"])

    return model

def unpack_checkpoint( checkpointfile ):
    checkpoint = torch.load( checkpointfile, map_location=map_location )
    print type(checkpoint)
    for k,v in checkpoint.items():
        print k


