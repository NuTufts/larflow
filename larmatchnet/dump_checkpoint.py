import os,sys

# torch
import torch
import torch.nn as nn

CHECKPOINT_MAP_LOCATIONS={}
for i in xrange(10):
    CHECKPOINT_MAP_LOCATIONS["cuda:%d"%(i)] = "cpu"
    
CHECKPOINT_FILE = sys.argv[1]

checkpoint = torch.load( CHECKPOINT_FILE, map_location=CHECKPOINT_MAP_LOCATIONS ) # load weights to gpuid
for dname,d in checkpoint.items():
    if "state" in dname:
        state_dict = checkpoint[dname]
        for name,arr in state_dict.items():
            print name
            if "running_mean" in name:
                print arr
            if "running_var" in name:
                print arr
